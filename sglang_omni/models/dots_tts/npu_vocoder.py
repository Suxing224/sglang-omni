# SPDX-License-Identifier: Apache-2.0
"""Preserve AudioVAE upsampling without zero-width NPU replicate padding."""

from __future__ import annotations

from types import MethodType

import torch
import torch.nn.functional as F

from sglang_omni.models.dots_tts.compat import import_dots_tts

import_dots_tts()

from dots_tts.modules.vocoder.alias_free_resample import UpSample1d


def upsample_forward(module: UpSample1d, waveform: torch.Tensor) -> torch.Tensor:
    channels = waveform.shape[1]
    if module.pad:
        waveform = F.pad(waveform, (module.pad, module.pad), mode="replicate")
    else:
        pass
    weight = (
        module.filter.expand(channels, -1, -1) if module.fixed_filter else module.filter
    )
    waveform = module.ratio * F.conv_transpose1d(
        waveform, weight, stride=module.stride, groups=channels
    )
    if module.causal:
        return waveform[..., : -(module.kernel_size - module.stride)]
    else:
        return waveform[..., module.pad_left : -module.pad_right]


def adapt_audio_vae_upsampling(audio_vae: torch.nn.Module) -> None:
    """Keep the original filters and crop while skipping no-op padding."""
    for module in audio_vae.modules():
        if isinstance(module, UpSample1d):
            module.forward = MethodType(upsample_forward, module)
        else:
            pass
