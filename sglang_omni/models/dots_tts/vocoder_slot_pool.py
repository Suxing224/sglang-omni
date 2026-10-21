# SPDX-License-Identifier: Apache-2.0
"""Per-slot causal AudioVAE streaming state for dots.tts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang_omni.models.dots_tts.vocoder_conv_cache import ConvCacheDecoder
from sglang_omni.models.dots_tts.vocoder_cuda_graph import DotsVocoderGraphRunner

if TYPE_CHECKING:
    from dots_tts.modules.vocoder.vocoder_inference import VocoderInference
else:
    pass


def append_decoder_input_per_row(
    decoder_input: torch.Tensor,
    window: torch.Tensor,
    valid_frames: torch.Tensor,
) -> torch.Tensor:
    """Append [B, C, T] into [B, C, W] with per-row valid_frames [B].

    Same geometry as upstream _append_stream_decoder_input_tensor, but rows may
    have different ages (upstream takes a scalar valid count).
    """
    if window.dtype != decoder_input.dtype:
        window = window.to(dtype=decoder_input.dtype)
    else:
        pass
    if valid_frames.ndim != 1 or int(valid_frames.shape[0]) != int(window.shape[0]):
        raise ValueError(
            "valid_frames must have shape [batch], "
            f"got {tuple(valid_frames.shape)} for window batch {int(window.shape[0])}"
        )
    else:
        pass
    chunk_size = int(decoder_input.size(-1))
    window_size = int(window.size(-1))
    if chunk_size >= window_size:
        raise ValueError(
            "decoder window size "
            f"{window_size} must be larger than chunk_size {chunk_size}."
        )
    else:
        pass
    batch = int(window.shape[0])
    positions = torch.arange(
        window_size, device=window.device, dtype=valid_frames.dtype
    )
    clipped_valid = valid_frames.clamp(min=0, max=window_size)
    combined = torch.cat(
        [
            window,
            decoder_input.new_zeros(batch, window.size(1), chunk_size),
        ],
        dim=-1,
    )
    insert_index = clipped_valid.unsqueeze(1) + torch.arange(
        chunk_size, device=window.device, dtype=valid_frames.dtype
    ).view(1, -1)
    combined.scatter_(
        -1,
        insert_index.unsqueeze(1).expand_as(decoder_input),
        decoder_input,
    )
    new_valid = (clipped_valid + chunk_size).clamp(max=window_size)
    start = (clipped_valid + chunk_size - window_size).clamp(min=0)
    gather_index = (start.unsqueeze(1) + positions.view(1, -1)).clamp(
        max=combined.size(-1) - 1
    )
    gathered = combined.gather(
        -1,
        gather_index.unsqueeze(1).expand(batch, window.size(1), window_size),
    )
    mask = (positions.view(1, -1) < new_valid.unsqueeze(1)).to(dtype=window.dtype)
    return gathered * mask.unsqueeze(1)


class DotsVocoderSlotPool:
    """Independent causal rows over one AudioVAE; equal-T steps run as one eager batch.

    Upstream VocoderStreamState keeps scalar total_frames / emitted_frames, so
    batch_size=N is lockstep-only. This pool owns per-slot counters instead.
    """

    def __init__(
        self,
        inference: "VocoderInference",
        *,
        num_slots: int,
        chunk_size: int,
        latent_dim: int,
        conv_cache_decoder: ConvCacheDecoder | None = None,
    ) -> None:
        if num_slots < 1:
            raise ValueError(f"num_slots must be >= 1, got {num_slots}")
        else:
            pass
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        else:
            pass
        self.inference = inference
        self.graph_runner: DotsVocoderGraphRunner | None = None
        self.prefix_graph_runners: dict[int, DotsVocoderGraphRunner] = {}
        self.conv_cache_decoder = conv_cache_decoder
        self.conv_cache_graph_runner: DotsVocoderGraphRunner | None = None
        self.num_slots = int(num_slots)
        self.chunk_size = int(chunk_size)
        # note (guozhihao-224): probe shapes through the public stream-state
        # constructor so we stay aligned with upstream window/LSTM layout.
        probe = inference.init_stream_state(batch_size=1, chunk_size=self.chunk_size)
        hidden_h, hidden_c = probe.lstm_hidden
        window = probe.decoder.window
        layers, _, hidden = hidden_h.shape
        _, channels, window_size = window.shape
        self.latent_dim = int(latent_dim)
        self.window_size = int(window_size)
        self.lookahead = int(
            inference._decoder_stream_lookahead()
        )  # noqa: leading-underscore  # AudioVAE API spelling
        self.hop_size = int(inference.vocoder.hop_size)
        self.lstm_h = hidden_h.new_zeros(int(layers), self.num_slots, int(hidden))
        self.lstm_c = hidden_c.new_zeros(int(layers), self.num_slots, int(hidden))
        self.window = window.new_zeros(self.num_slots, int(channels), self.window_size)
        if self.conv_cache_decoder is None:
            self.conv_cache_flat: torch.Tensor | None = None
        else:
            self.conv_cache_flat = self.conv_cache_decoder.new_cache_buffer(
                self.num_slots
            )
        self.total_frames = [0] * self.num_slots
        self.emitted_frames = [0] * self.num_slots
        self.free_slots = list(reversed(range(self.num_slots)))
        self.in_use: set[int] = set()

    @property
    def device(self) -> torch.device:
        return self.window.device

    def new_step_inputs(
        self, batch: int, frames: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Zeroed (packed, hidden_h, hidden_c, window, valid) laid out for forward()."""
        layers, _, hidden = self.lstm_h.shape
        _, channels, window_size = self.window.shape
        lstm_kwargs = {"dtype": self.lstm_h.dtype, "device": self.device}
        return (
            torch.zeros(batch, self.latent_dim, frames, **lstm_kwargs),
            torch.zeros(layers, batch, hidden, **lstm_kwargs),
            torch.zeros(layers, batch, hidden, **lstm_kwargs),
            torch.zeros(
                batch,
                channels,
                window_size,
                dtype=self.window.dtype,
                device=self.device,
            ),
            torch.zeros(batch, dtype=torch.int64, device=self.device),
        )

    def new_conv_cache_step_inputs(
        self, batch: int, frames: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(packed, hidden_h, hidden_c, flat_cache) input layout."""
        assert self.conv_cache_decoder is not None
        assert self.conv_cache_flat is not None
        layers, _, hidden = self.lstm_h.shape
        lstm_kwargs = {"dtype": self.lstm_h.dtype, "device": self.device}
        return (
            torch.zeros(batch, self.latent_dim, frames, **lstm_kwargs),
            torch.zeros(layers, batch, hidden, **lstm_kwargs),
            torch.zeros(layers, batch, hidden, **lstm_kwargs),
            self.conv_cache_decoder.new_cache_buffer(batch),
        )

    def acquire(self) -> int:
        if not self.free_slots:
            # note (guozhihao-224): slot exhaustion is an admission failure, not a
            # cue to resize the pool under a live request.
            raise RuntimeError(
                "dots.tts streaming vocoder admission failed: ran out of slots "
                f"(num_slots={self.num_slots})"
            )
        else:
            pass
        slot = int(self.free_slots.pop())
        self.reset_slot(slot)
        self.in_use.add(slot)
        return slot

    def release(self, slot: int) -> None:
        slot = int(slot)
        if slot not in self.in_use:
            return
        else:
            pass
        self.reset_slot(slot)
        self.in_use.remove(slot)
        self.free_slots.append(slot)

    @torch.no_grad()
    def step(self, slot_latents: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        """One uniform-T eager step. slot -> [1, T, C] in, slot -> wav out."""
        if not slot_latents:
            return {}
        else:
            pass
        step_lengths = {int(latents.shape[1]) for latents in slot_latents.values()}
        if len(step_lengths) != 1:
            raise ValueError(
                "dots.tts streaming step requires a uniform latent length, "
                f"got {sorted(step_lengths)}"
            )
        else:
            pass
        (step_t,) = step_lengths
        if step_t < 1:
            raise ValueError("dots.tts streaming step length must be positive")
        else:
            pass
        if step_t >= self.window_size:
            raise ValueError(
                f"streaming step length {step_t} must be < window size "
                f"{self.window_size}"
            )
        else:
            pass
        slots = [int(slot) for slot in slot_latents]
        for slot in slots:
            if slot not in self.in_use:
                raise RuntimeError(
                    f"dots.tts streaming step referenced free slot {slot}"
                )
            else:
                pass
            latents = slot_latents[slot]
            if latents.ndim != 3 or int(latents.shape[0]) != 1:
                raise ValueError(
                    "slot latents must have shape [1, frames, latent_dim], "
                    f"got {tuple(latents.shape)} for slot {slot}"
                )
            else:
                pass

        slot_index = torch.tensor(slots, device=self.window.device, dtype=torch.long)
        # note (guozhihao-224): upstream stream kernels take channel-major
        # latents [B, C, T]; Omni chunks arrive as [1, T, C].
        packed = torch.cat(
            [slot_latents[slot].transpose(1, 2).contiguous() for slot in slots],
            dim=0,
        )
        hidden_h = self.lstm_h.index_select(1, slot_index).contiguous()
        hidden_c = self.lstm_c.index_select(1, slot_index).contiguous()
        window = self.window.index_select(0, slot_index).contiguous()
        valid = torch.tensor(
            [min(self.total_frames[slot], self.window_size) for slot in slots],
            device=window.device,
            dtype=torch.int64,
        )

        self.inference._validate_stream_latents(
            packed
        )  # noqa: leading-underscore  # AudioVAE API spelling
        if self.conv_cache_decoder is not None:
            return self.step_mixed(
                slots,
                slot_index,
                packed,
                hidden_h,
                hidden_c,
                window,
                valid,
                step_t,
            )
        else:
            pass
        audio_window, hidden_h, hidden_c, new_window = self.forward_window_step(
            slots, packed, hidden_h, hidden_c, window, valid
        )

        self.lstm_h[:, slot_index, :] = hidden_h
        self.lstm_c[:, slot_index, :] = hidden_c
        self.window[slot_index] = new_window
        out: dict[int, torch.Tensor] = {}
        for row, slot in enumerate(slots):
            self.total_frames[slot] += step_t
            out[slot] = self.slice_audio(slot, audio_window[row : row + 1], final=False)
        return out

    def forward_window_step(
        self,
        slots: list[int],
        packed: torch.Tensor,
        hidden_h: torch.Tensor,
        hidden_c: torch.Tensor,
        window: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        step_frames = int(packed.shape[-1])
        graph_runner = self.graph_runner
        required_frames = max(self.total_frames[slot] + step_frames for slot in slots)
        for decoder_frames, prefix_runner in sorted(self.prefix_graph_runners.items()):
            if (
                required_frames <= decoder_frames
                and (len(slots), step_frames) in prefix_runner.captured_keys
            ):
                graph_runner = prefix_runner
                break
            else:
                pass
        replayed = (
            None
            if graph_runner is None
            else graph_runner.run((packed, hidden_h, hidden_c, window, valid))
        )
        return (
            self.forward(packed, hidden_h, hidden_c, window, valid)
            if replayed is None
            else replayed
        )

    @torch.no_grad()
    def step_mixed(
        self,
        slots: list[int],
        slot_index: torch.Tensor,
        packed: torch.Tensor,
        hidden_h: torch.Tensor,
        hidden_c: torch.Tensor,
        window: torch.Tensor,
        valid: torch.Tensor,
        step_frames: int,
    ) -> dict[int, torch.Tensor]:
        """Use window decoding until each row can seed its layer caches."""
        assert self.conv_cache_decoder is not None
        warmup = self.conv_cache_decoder.warmup_frames
        warm_rows = [
            row for row, slot in enumerate(slots) if self.total_frames[slot] < warmup
        ]
        hot_rows = [
            row for row, slot in enumerate(slots) if self.total_frames[slot] >= warmup
        ]
        out: dict[int, torch.Tensor] = {}
        if hot_rows:
            row_index = torch.tensor(
                hot_rows, device=slot_index.device, dtype=torch.long
            )
            out.update(
                self.step_conv_cache(
                    [slots[row] for row in hot_rows],
                    slot_index[row_index],
                    packed[row_index],
                    hidden_h[:, row_index],
                    hidden_c[:, row_index],
                    step_frames,
                )
            )
        else:
            pass
        if warm_rows:
            row_index = torch.tensor(
                warm_rows, device=slot_index.device, dtype=torch.long
            )
            out.update(
                self.step_warm(
                    [slots[row] for row in warm_rows],
                    slot_index[row_index],
                    packed[row_index],
                    hidden_h[:, row_index],
                    hidden_c[:, row_index],
                    window[row_index],
                    valid[row_index],
                    step_frames,
                )
            )
        else:
            pass
        return out

    @torch.no_grad()
    def step_warm(
        self,
        slots: list[int],
        slot_index: torch.Tensor,
        packed: torch.Tensor,
        hidden_h: torch.Tensor,
        hidden_c: torch.Tensor,
        window: torch.Tensor,
        valid: torch.Tensor,
        step_frames: int,
    ) -> dict[int, torch.Tensor]:
        """Window-path step for pre-warmup rows, seeding rows that cross."""
        assert self.conv_cache_decoder is not None
        assert self.conv_cache_flat is not None
        warmup = self.conv_cache_decoder.warmup_frames
        crossing = any(
            self.total_frames[slot] + step_frames >= warmup for slot in slots
        )
        captured: dict[int, torch.Tensor] | None = None
        if crossing:
            # note: Seed from eager hooks once, using only valid frames.
            decoder_frames = max(
                min(self.total_frames[slot] + step_frames, self.window_size)
                for slot in slots
            )
            with self.conv_cache_decoder.capture_spec_inputs() as hooked:
                audio_window, hidden_h, hidden_c, new_window = self.forward(
                    packed,
                    hidden_h,
                    hidden_c,
                    window,
                    valid,
                    decoder_window_frames=decoder_frames,
                )
            captured = hooked
        else:
            audio_window, hidden_h, hidden_c, new_window = self.forward_window_step(
                slots, packed, hidden_h, hidden_c, window, valid
            )
        self.lstm_h[:, slot_index, :] = hidden_h
        self.lstm_c[:, slot_index, :] = hidden_c
        self.window[slot_index] = new_window
        out: dict[int, torch.Tensor] = {}
        for row, slot in enumerate(slots):
            self.total_frames[slot] += step_frames
            out[slot] = self.slice_audio(slot, audio_window[row : row + 1], final=False)
            if self.total_frames[slot] >= warmup:
                assert captured is not None
                valid_frames = min(self.total_frames[slot], self.window_size)
                self.conv_cache_decoder.seed_from_captured(
                    self.conv_cache_flat[slot], captured, row, valid_frames
                )
            else:
                pass
        return out

    @torch.no_grad()
    def forward(
        self,
        packed: torch.Tensor,
        hidden_h: torch.Tensor,
        hidden_c: torch.Tensor,
        window: torch.Tensor,
        valid: torch.Tensor,
        *,
        decoder_window_frames: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # note (lennox): CUDA graph capture boundary; keep slot bookkeeping and
        # anything host-side out of it.
        inference = self.inference
        # note (guozhihao-224): call VocoderInference private eager helpers so
        # rows can age independently; stream_step's scalar counters are
        # lockstep-only. Expect breakage if upstream renames these.
        decoder_input, (hidden_h, hidden_c) = (
            inference._decode_stream_latents(  # noqa: leading-underscore  # AudioVAE API spelling
                packed, (hidden_h, hidden_c)
            )
        )
        new_window = append_decoder_input_per_row(decoder_input, window, valid)
        decoder_window = (
            new_window
            if decoder_window_frames is None
            else new_window[..., :decoder_window_frames].contiguous()
        )
        audio_window = inference._decode_stream_window(
            decoder_window
        )  # noqa: leading-underscore  # AudioVAE API spelling
        return audio_window, hidden_h, hidden_c, new_window

    @torch.no_grad()
    def forward_conv_cache(
        self,
        packed: torch.Tensor,
        hidden_h: torch.Tensor,
        hidden_c: torch.Tensor,
        flat_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # note (lennox): device-graph capture boundary; keep slot bookkeeping and
        # anything host-side out of it.
        assert self.conv_cache_decoder is not None
        decoder_input, (hidden_h, hidden_c) = (
            self.inference._decode_stream_latents(  # noqa: leading-underscore  # AudioVAE API spelling
                packed, (hidden_h, hidden_c)
            )
        )
        audio, new_flat_cache = self.conv_cache_decoder.decode_segment(
            decoder_input, flat_cache
        )
        return audio, hidden_h, hidden_c, new_flat_cache

    @torch.no_grad()
    def step_conv_cache(
        self,
        slots: list[int],
        slot_index: torch.Tensor,
        packed: torch.Tensor,
        hidden_h: torch.Tensor,
        hidden_c: torch.Tensor,
        step_frames: int,
    ) -> dict[int, torch.Tensor]:
        """Decode warmed rows and emit each slot's unseen stable suffix."""
        assert self.conv_cache_decoder is not None
        assert self.conv_cache_flat is not None
        flat_cache = self.conv_cache_flat.index_select(0, slot_index).contiguous()
        graph_runner = self.conv_cache_graph_runner
        replayed = (
            None
            if graph_runner is None
            else graph_runner.run((packed, hidden_h, hidden_c, flat_cache))
        )
        audio, hidden_h, hidden_c, new_flat_cache = (
            self.forward_conv_cache(packed, hidden_h, hidden_c, flat_cache)
            if replayed is None
            else replayed
        )
        self.lstm_h[:, slot_index, :] = hidden_h
        self.lstm_c[:, slot_index, :] = hidden_c
        self.conv_cache_flat[slot_index] = new_flat_cache
        out: dict[int, torch.Tensor] = {}
        for row, slot in enumerate(slots):
            self.total_frames[slot] += step_frames
            stable_end = max(0, self.total_frames[slot] - self.lookahead)
            emit_frames = stable_end - self.emitted_frames[slot]
            if emit_frames <= 0:
                out[slot] = audio.new_zeros((1, 1, 0))
            else:
                if emit_frames > step_frames:
                    raise RuntimeError(
                        "dots.tts conv-cache step emission exceeds the step length: "
                        f"slot {slot} emits {emit_frames} frames from a "
                        f"{step_frames}-frame step"
                    )
                else:
                    pass
                out[slot] = audio[
                    row : row + 1, :, (step_frames - emit_frames) * self.hop_size :
                ]
                self.emitted_frames[slot] = stable_end
        return out

    @torch.no_grad()
    def flush(self, slot: int) -> torch.Tensor:
        slot = int(slot)
        if slot not in self.in_use:
            raise RuntimeError(f"dots.tts streaming flush referenced free slot {slot}")
        else:
            pass
        if (
            self.conv_cache_decoder is not None
            and self.total_frames[slot] >= self.conv_cache_decoder.warmup_frames
        ):
            return self.flush_conv_cache(slot)
        else:
            pass
        audio_window = self.inference._decode_stream_window(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            self.window[slot : slot + 1]
        )
        return self.slice_audio(slot, audio_window, final=True)

    @torch.no_grad()
    def flush_conv_cache(self, slot: int) -> torch.Tensor:
        """Drain the lookahead tail: zero frames play the window right zero-pad."""
        assert self.conv_cache_decoder is not None
        assert self.conv_cache_flat is not None
        total = self.total_frames[slot]
        emit_frames = total - self.emitted_frames[slot]
        if emit_frames <= 0:
            return torch.zeros(1, 1, 0, dtype=torch.float32, device=self.device)
        else:
            pass
        zero_latents = self.conv_cache_flat.new_zeros(
            1,
            int(self.conv_cache_decoder.decoder.conv_pre.in_channels),
            self.lookahead,
        )
        audio, new_flat_cache = self.conv_cache_decoder.decode_segment(
            zero_latents,
            self.conv_cache_flat[slot : slot + 1],
        )
        self.conv_cache_flat[slot : slot + 1] = new_flat_cache
        self.emitted_frames[slot] = total
        return audio[..., (self.lookahead - emit_frames) * self.hop_size :]

    def reset_slot(self, slot: int) -> None:
        self.lstm_h[:, slot].zero_()
        self.lstm_c[:, slot].zero_()
        self.window[slot].zero_()
        if self.conv_cache_flat is not None:
            self.conv_cache_flat[slot].zero_()
        else:
            pass
        self.total_frames[slot] = 0
        self.emitted_frames[slot] = 0

    def slice_audio(
        self, slot: int, audio_window: torch.Tensor, *, final: bool
    ) -> torch.Tensor:
        # note (guozhihao-224): mirrors VocoderInference._slice_stream_audio_window
        # with per-slot total/emitted counters instead of the upstream scalars.
        total = self.total_frames[slot]
        emitted = self.emitted_frames[slot]
        stable_end = total if final else max(0, total - self.lookahead)
        if stable_end <= emitted:
            return audio_window.new_zeros((audio_window.size(0), 1, 0))
        else:
            pass

        valid_frames = min(total, self.window_size)
        window_start = total - valid_frames
        if emitted < window_start:
            raise RuntimeError(
                "Decoder stream window is too short for fixed-graph decoding."
            )
        else:
            pass
        local_start = emitted - window_start
        local_end = stable_end - window_start
        sample_start = local_start * self.hop_size
        sample_end = local_end * self.hop_size
        self.emitted_frames[slot] = stable_end
        return audio_window[..., sample_start:sample_end]


__all__ = ["DotsVocoderSlotPool", "append_decoder_input_per_row"]
