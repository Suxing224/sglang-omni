# SPDX-License-Identifier: Apache-2.0
"""Per-layer convolution caches for incremental dots.tts AudioVAE decoding."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from functools import partial
from typing import Literal, NamedTuple

import torch

from sglang_omni.models.dots_tts.codec import DotsAudioCodec

LayerKind = Literal["pre", "up", "act", "conv", "conv_res"]


class CacheLayerSpec(NamedTuple):
    kind: LayerKind
    module: torch.nn.Module
    piece_index: int
    offset: int
    channels: int
    length: int
    stride: int
    save_input: bool


class ConvCacheDecoder:
    """Decode new segments using seeded per-layer left context."""

    def __init__(self, codec: DotsAudioCodec) -> None:
        inference = codec.inference
        decoder = inference.vocoder.decoder
        self.decoder = decoder
        self.lookahead = int(
            inference._decoder_stream_lookahead()
        )  # noqa: leading-underscore  # AudioVAE API spelling
        specs: list[CacheLayerSpec] = []
        offset = 0

        def add(
            kind: LayerKind,
            module: torch.nn.Module,
            channels: int,
            length: int,
            *,
            stride: int = 1,
            save_input: bool = False,
        ) -> CacheLayerSpec:
            nonlocal offset
            assert length > 0
            spec = CacheLayerSpec(
                kind, module, len(specs), offset, channels, length, stride, save_input
            )
            specs.append(spec)
            offset += channels * length
            return spec

        def activation_left_context(activation: torch.nn.Module) -> int:
            return int(
                inference._activation_left_context(activation)
            )  # noqa: leading-underscore  # AudioVAE API spelling

        self.pre = add(
            "pre",
            decoder.conv_pre,
            int(decoder.conv_pre.in_channels),
            2 * self.lookahead,
        )
        stages: list[tuple[list[CacheLayerSpec], list[list[CacheLayerSpec]]]] = []
        for stage_index in range(int(decoder.num_upsamples)):
            upsample_specs: list[CacheLayerSpec] = []
            for upsample in decoder.ups[stage_index]:
                stride = (
                    int(upsample.stride[0])
                    if isinstance(upsample.stride, tuple)
                    else int(upsample.stride)
                )
                upsample_specs.append(
                    add("up", upsample, int(upsample.in_channels), 1, stride=stride)
                )
            stage_start = stage_index * int(decoder.num_kernels)
            stage_blocks = decoder.resblocks[
                stage_start : stage_start + int(decoder.num_kernels)
            ]
            block_specs: list[list[CacheLayerSpec]] = []
            for block in stage_blocks:
                acts1, acts2 = block.activations[::2], block.activations[1::2]
                sublayers: list[CacheLayerSpec] = []
                for conv1, conv2, act1, act2 in zip(
                    block.convs1, block.convs2, acts1, acts2, strict=True
                ):
                    sublayers.append(
                        add(
                            "act",
                            act1,
                            int(act1.act.in_features),
                            activation_left_context(act1),
                            save_input=True,
                        )
                    )
                    sublayers.append(
                        add(
                            "conv",
                            conv1,
                            int(conv1.in_channels),
                            int(conv1.dilation[0]) * (int(conv1.kernel_size[0]) - 1),
                        )
                    )
                    sublayers.append(
                        add(
                            "act",
                            act2,
                            int(act2.act.in_features),
                            activation_left_context(act2),
                        )
                    )
                    sublayers.append(
                        add(
                            "conv_res",
                            conv2,
                            int(conv2.in_channels),
                            int(conv2.dilation[0]) * (int(conv2.kernel_size[0]) - 1),
                        )
                    )
                block_specs.append(sublayers)
            stages.append((upsample_specs, block_specs))
        self.stages = stages
        self.post = (
            add(
                "act",
                decoder.activation_post,
                int(decoder.activation_post.act.in_features),
                activation_left_context(decoder.activation_post),
            ),
            add(
                "conv",
                decoder.conv_post,
                int(decoder.conv_post.in_channels),
                int(decoder.conv_post.dilation[0])
                * (int(decoder.conv_post.kernel_size[0]) - 1),
            ),
        )
        self.specs = specs
        self.flat_size = int(offset)
        self.use_tanh = bool(decoder.h.get("use_tanh_at_final", True))
        # note (codex): Seed only when the window fills every layer's left context.
        rate = 1
        margin = 2 * self.lookahead
        for spec in specs:
            if spec.kind != "pre":
                margin = max(margin, self.lookahead + -(-spec.length // rate))
            else:
                pass
            if spec.kind == "up":
                rate *= spec.stride
            else:
                pass
        self.warmup_frames = int(margin)

    def new_cache_buffer(self, num_slots: int) -> torch.Tensor:
        ref = next(self.decoder.parameters())
        return torch.zeros(
            int(num_slots), self.flat_size, device=ref.device, dtype=ref.dtype
        )

    def run_layer(
        self,
        spec: CacheLayerSpec,
        x: torch.Tensor,
        flat_cache: torch.Tensor,
        pieces: list[torch.Tensor | None],
        block_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = int(x.shape[0])
        new_frames = int(x.shape[-1])
        cache_row = flat_cache[
            :, spec.offset : spec.offset + spec.channels * spec.length
        ].view(batch, spec.channels, spec.length)
        layer_input = torch.cat([cache_row, x], dim=-1)
        pieces[spec.piece_index] = layer_input[..., -spec.length :]
        if spec.kind == "pre":
            layer_output = spec.module(layer_input)
            return layer_output[..., self.lookahead : self.lookahead + new_frames]
        elif spec.kind == "up":
            return spec.module(layer_input)[..., -new_frames * spec.stride :]
        elif spec.kind == "conv_res":
            assert block_input is not None
            return spec.module(layer_input)[..., -new_frames:] + block_input
        else:
            return spec.module(layer_input)[..., -new_frames:]

    @contextlib.contextmanager
    def capture_spec_inputs(self) -> Iterator[dict[int, torch.Tensor]]:
        """Collect eager layer inputs to seed warmed slots."""
        captured: dict[int, torch.Tensor] = {}

        def capture_input(
            module: torch.nn.Module,
            layer_inputs: tuple[torch.Tensor, ...],
            *,
            piece_index: int,
        ) -> None:
            captured[piece_index] = layer_inputs[0].detach()

        handles = [
            spec.module.register_forward_pre_hook(
                partial(capture_input, piece_index=spec.piece_index)
            )
            for spec in self.specs
        ]
        try:
            yield captured
        finally:
            for handle in handles:
                handle.remove()

    def seed_from_captured(
        self,
        flat_row: torch.Tensor,
        captured: dict[int, torch.Tensor],
        row: int,
        valid_frames: int,
    ) -> None:
        """Seed one warmed slot at each layer's lookahead-adjusted input rate."""
        rate = 1
        for spec in self.specs:
            end = (
                valid_frames * rate
                if spec.kind == "pre"
                else (valid_frames - self.lookahead) * rate
            )
            start = end - spec.length
            if start < 0:
                raise RuntimeError(
                    "dots.tts conv-cache seed slice underflow: "
                    f"kind={spec.kind} rate={rate} length={spec.length} "
                    f"valid_frames={valid_frames}; raise warmup_frames"
                )
            else:
                pass
            piece = captured[spec.piece_index][int(row), :, start:end]
            flat_row[spec.offset : spec.offset + spec.channels * spec.length] = (
                piece.reshape(-1).to(dtype=flat_row.dtype)
            )
            if spec.kind == "up":
                rate *= spec.stride
            else:
                pass

    @torch.no_grad()
    def decode_segment(
        self,
        latents: torch.Tensor,
        flat_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode a warmed [B, C, N] latent segment and update its [B, F] cache."""
        batch = int(latents.shape[0])
        pieces: list[torch.Tensor | None] = [None] * len(self.specs)
        x = self.run_layer(self.pre, latents, flat_cache, pieces)
        for upsample_specs, block_specs in self.stages:
            for upsample_spec in upsample_specs:
                x = self.run_layer(upsample_spec, x, flat_cache, pieces)
            merged: torch.Tensor | None = None
            for sublayers in block_specs:
                block_input: torch.Tensor | None = None
                branch = x
                for spec in sublayers:
                    if spec.save_input:
                        block_input = branch
                    else:
                        pass
                    branch = self.run_layer(
                        spec, branch, flat_cache, pieces, block_input
                    )
                merged = branch if merged is None else merged + branch
            assert merged is not None
            x = merged / len(block_specs)
        x = self.run_layer(self.post[0], x, flat_cache, pieces)
        x = self.run_layer(self.post[1], x, flat_cache, pieces)
        assert all(piece is not None for piece in pieces)
        new_flat_cache = torch.cat(
            [piece.reshape(batch, -1) for piece in pieces if piece is not None], dim=-1
        )
        if self.use_tanh:
            audio = torch.tanh(x)
        else:
            audio = torch.clamp(x, min=-1.0, max=1.0)
        return audio.float(), new_flat_cache


__all__ = ["ConvCacheDecoder"]
