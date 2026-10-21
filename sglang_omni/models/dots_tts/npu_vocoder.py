# SPDX-License-Identifier: Apache-2.0
"""Graph-capturable convolutions for dots.tts-owned AudioVAE instances."""

from __future__ import annotations

import logging
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang_omni.models.dots_tts.compat import import_dots_tts

import_dots_tts()

from dots_tts.modules.backbone.layers import ConvTranspose1d
from dots_tts.modules.vocoder.alias_free_filter import LowPassFilter1d
from dots_tts.modules.vocoder.alias_free_resample import UpSample1d

logger = logging.getLogger(__name__)


def decoder_forward(module: nn.Module, latents: torch.Tensor) -> torch.Tensor:
    """Keep decoder precision separate from the FP32 projection and LSTM."""
    decoder_dtype = next(module.parameters()).dtype
    return type(module).forward(module, latents.to(dtype=decoder_dtype)).float()


def convolution(
    waveform: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    stride: int = 1,
    padding: int | str = 0,
    dilation: int = 1,
    groups: int = 1,
) -> torch.Tensor:
    # note: Singleton spatial dimensions select capturable aclnn convolution.
    spatial_padding = padding if isinstance(padding, str) else (0, padding)
    return F.conv2d(
        waveform[:, :, None, :],
        weight[:, :, None, :],
        bias,
        stride=(1, stride),
        padding=spatial_padding,
        dilation=(1, dilation),
        groups=groups,
    )[:, :, 0, :]


def convolution_forward(
    module: nn.Conv1d,
    waveform: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    if module.padding_mode != "zeros":
        waveform = F.pad(
            waveform,
            module._reversed_padding_repeated_twice,  # noqa: leading-underscore  # Torch padding contract
            mode=module.padding_mode,
        )
        padding = 0
    elif isinstance(module.padding, str):
        padding = module.padding
    else:
        padding = module.padding[0]
    return convolution(
        waveform,
        weight,
        bias,
        stride=module.stride[0],
        padding=padding,
        dilation=module.dilation[0],
        groups=module.groups,
    )


def transposed_convolution_forward(
    module: ConvTranspose1d, waveform: torch.Tensor
) -> torch.Tensor:
    waveform = F.conv_transpose2d(
        waveform[:, :, None, :],
        module.weight[:, :, None, :],
        module.bias,
        stride=(1, module.stride),
        padding=(0, module.padding[0]),
        output_padding=(0, module.output_padding[0]),
        dilation=(1, module.dilation[0]),
        groups=module.groups,
    )[:, :, 0, :]
    if module.causal:
        return waveform[:, :, : -module.stride]
    else:
        return waveform


def upsample_forward(module: UpSample1d, waveform: torch.Tensor) -> torch.Tensor:
    channels = waveform.shape[1]
    # note: NPU zero replicate padding corrupts long waveforms.
    if module.pad:
        waveform = F.pad(waveform, (module.pad, module.pad), mode="replicate")
    else:
        pass
    if module.fixed_filter:
        weight = module.filter.expand(channels, -1, -1)
    else:
        weight = module.filter
    # note: Grouped FIR phases avoid slow depthwise transpose convolution.
    full_samples = (waveform.shape[-1] - 1) * module.stride + module.kernel_size
    phase_taps = (module.kernel_size + module.stride - 1) // module.stride
    weight = weight.reshape(channels, module.kernel_size)
    extra_taps = phase_taps * module.stride - module.kernel_size
    if extra_taps:
        weight = F.pad(weight, (0, extra_taps))
    else:
        pass
    weight = (
        weight.reshape(channels, phase_taps, module.stride)
        .transpose(1, 2)
        .flip(-1)
        .reshape(channels * module.stride, 1, phase_taps)
    )
    trim_causal_tail = module.causal and module.kernel_size > module.stride
    if phase_taps > 1:
        # note: Causal cropping discards all right-padded FIR outputs.
        right_padding = 0 if trim_causal_tail else phase_taps - 1
        waveform = F.pad(waveform, (phase_taps - 1, right_padding))
    else:
        pass
    phases = convolution(waveform, weight, groups=channels)
    phases = phases.reshape(waveform.shape[0], channels, module.stride, -1)
    waveform = module.ratio * phases.transpose(-1, -2).flatten(-2)[..., :full_samples]
    if trim_causal_tail:
        return waveform
    elif module.causal:
        return waveform[..., : -(module.kernel_size - module.stride)]
    else:
        return waveform[..., module.pad_left : -module.pad_right]


def lowpass_forward(module: LowPassFilter1d, waveform: torch.Tensor) -> torch.Tensor:
    channels = waveform.shape[1]
    if module.padding:
        waveform = F.pad(
            waveform, (module.pad_left, module.pad_right), mode=module.padding_mode
        )
    else:
        pass
    if module.fixed_filter:
        weight = module.filter.expand(channels, -1, -1)
    else:
        weight = module.filter
    return convolution(waveform, weight, stride=module.stride, groups=channels)


def adapt_audio_vae_convolutions(audio_vae: nn.Module) -> None:
    """Keep module types, parameters and causal wrappers on the owned model."""
    if any(
        parameter.device.type == "npu" for parameter in audio_vae.parameters()
    ) or any(buffer.device.type == "npu" for buffer in audio_vae.buffers()):
        # note: Internal formats select uncapturable aclop Conv2D kernels.
        torch.get_device_module("npu").config.allow_internal_format = False
        logger.info("dots.tts AudioVAE disabled process-wide NPU internal formats")
    else:
        pass
    for module in audio_vae.modules():
        if isinstance(module, nn.Conv1d):
            module._conv_forward = MethodType(
                convolution_forward, module
            )  # noqa: leading-underscore  # Preserve causal forward wrappers
        elif isinstance(module, ConvTranspose1d):
            module.forward = MethodType(transposed_convolution_forward, module)
        elif isinstance(module, UpSample1d):
            module.forward = MethodType(upsample_forward, module)
        elif isinstance(module, LowPassFilter1d):
            module.forward = MethodType(lowpass_forward, module)
        else:
            pass
