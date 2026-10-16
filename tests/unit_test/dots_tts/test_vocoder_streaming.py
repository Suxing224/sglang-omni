# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.dots_tts.vocoder import DotsTTSStreamingVocoder
from sglang_omni.models.dots_tts.vocoder_slot_pool import (
    DotsVocoderSlotPool,
    append_decoder_input_per_row,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem


class RecordingSlotPool:
    def __init__(self, *, num_slots: int = 4) -> None:
        self.num_slots = num_slots
        self.free = list(reversed(range(num_slots)))
        self.in_use: set[int] = set()
        self.steps: list[dict[int, torch.Tensor]] = []
        self.flushes: list[int] = []

    def acquire(self) -> int:
        if not self.free:
            raise RuntimeError(
                f"dots.tts streaming vocoder admission failed: ran out of slots "
                f"(num_slots={self.num_slots})"
            )
        slot = self.free.pop()
        self.in_use.add(slot)
        return slot

    def release(self, slot: int) -> None:
        slot = int(slot)
        if slot not in self.in_use:
            return
        self.in_use.remove(slot)
        self.free.append(slot)

    def step(self, slot_latents: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        self.steps.append(
            {slot: latents.clone() for slot, latents in slot_latents.items()}
        )
        return {slot: torch.full((1, 1, 8), float(slot + 1)) for slot in slot_latents}

    def flush(self, slot: int) -> torch.Tensor:
        self.flushes.append(int(slot))
        return torch.full((1, 1, 4), float(slot + 1))


class FakeInference:
    """Minimal VocoderInference surface for DotsVocoderSlotPool unit tests."""

    def __init__(self, *, latent_dim: int = 5, hop_size: int = 2) -> None:
        self.vocoder = SimpleNamespace(
            hop_size=hop_size,
            h=SimpleNamespace(latent_dim=latent_dim, causal=True),
        )
        self.batch_steps: list[tuple[int, int]] = []
        self.latent_dim = latent_dim
        self.hop_size = hop_size

    def init_stream_state(self, *, batch_size: int, chunk_size: int):
        window = torch.zeros(batch_size, self.latent_dim, chunk_size + 4)
        hidden = torch.zeros(1, batch_size, 8)
        return SimpleNamespace(
            lstm_hidden=(hidden, hidden.clone()),
            decoder=SimpleNamespace(window=window, chunk_size=chunk_size),
        )

    def _decoder_stream_lookahead(
        self,
    ) -> int:  # noqa: leading-underscore  # upstream name
        return 1

    def _validate_stream_latents(
        self, latents: torch.Tensor
    ) -> None:  # noqa: leading-underscore  # upstream name
        if latents.ndim != 3 or int(latents.shape[1]) != self.latent_dim:
            raise ValueError(f"bad latents {tuple(latents.shape)}")

    def _decode_stream_latents(
        self, latents, hidden
    ):  # noqa: leading-underscore  # upstream name
        batch, channels, frames = latents.shape
        self.batch_steps.append((batch, frames))
        decoder_input = latents.clone()
        return decoder_input, (hidden[0].clone(), hidden[1].clone())

    def _decode_stream_window(
        self, window: torch.Tensor
    ) -> torch.Tensor:  # noqa: leading-underscore  # upstream name
        return torch.zeros(
            window.size(0), 1, window.size(-1) * self.hop_size, dtype=window.dtype
        )


def make_codec(*, latent_dim: int = 5, patch_size: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        inference=FakeInference(latent_dim=latent_dim),
        lock=threading.RLock(),
        sample_rate=48000,
        patch_size=patch_size,
        latent_dim=latent_dim,
        device=torch.device("cpu"),
        hop_size=2,
    )


def patch(value: float = 0.0, *, frames: int = 3, dim: int = 5) -> torch.Tensor:
    return torch.full((1, frames, dim), value)


def test_append_decoder_input_per_row_matches_scalar_when_ages_equal() -> None:
    window = torch.zeros(2, 2, 6)
    window[0, :, :2] = 1
    window[1, :, :2] = 1
    decoder_input = torch.full((2, 2, 2), 3.0)
    valid = torch.tensor([2, 2], dtype=torch.int64)
    out = append_decoder_input_per_row(decoder_input, window, valid)
    assert out.shape == window.shape
    assert torch.equal(out[0], out[1])


def test_append_decoder_input_keeps_independent_row_ages() -> None:
    window = torch.zeros(2, 1, 6)
    window[0, :, :1] = 1
    window[1, :, :4] = 2
    decoder_input = torch.tensor([[[7.0, 8.0]], [[9.0, 10.0]]])
    valid = torch.tensor([1, 4], dtype=torch.int64)
    out = append_decoder_input_per_row(decoder_input, window, valid)
    assert out[0, 0, :3].tolist() == [1.0, 7.0, 8.0]
    assert out[1, 0, :6].tolist() == [2.0, 2.0, 2.0, 2.0, 9.0, 10.0]


def test_slot_pool_batches_equal_t_and_preserves_independent_counters() -> None:
    inference = FakeInference()
    pool = DotsVocoderSlotPool(inference, num_slots=4, chunk_size=6, latent_dim=5)
    s0 = pool.acquire()
    s1 = pool.acquire()
    older = torch.ones(1, 3, 5)
    newer = torch.full((1, 3, 5), 2.0)

    # note (guozhihao-224): age s0 alone so total_frames diverge before the
    # shared step.
    pool.step({s0: older})
    assert inference.batch_steps == [(1, 3)]

    pool.step({s0: older, s1: newer})
    assert inference.batch_steps[-1] == (2, 3)
    assert pool.total_frames[s0] == 6
    assert pool.total_frames[s1] == 3

    pool.release(s0)
    reused = pool.acquire()
    assert reused == s0
    assert pool.total_frames[reused] == 0


def test_slot_pool_rejects_mixed_step_lengths() -> None:
    pool = DotsVocoderSlotPool(FakeInference(), num_slots=2, chunk_size=6, latent_dim=5)
    a = pool.acquire()
    b = pool.acquire()
    with pytest.raises(ValueError, match="uniform latent length"):
        pool.step({a: torch.zeros(1, 2, 5), b: torch.zeros(1, 3, 5)})


def test_streaming_coalesces_equal_t_requests_into_one_pool_step() -> None:
    pool = RecordingSlotPool()
    vocoder = DotsTTSStreamingVocoder(
        make_codec(),
        optimize=True,
        merge_steps=2,
        max_batch_size=4,
        stream_slots=4,
        slot_pool=pool,
    )
    assert vocoder.can_batch_stream_chunks is True
    assert vocoder.stream_chunk_batch_max == 4

    for request_id in ("a", "b"):
        state = vocoder.create_stream_state(request_id)
        vocoder.stream_states[request_id] = state
        vocoder.ingest(request_id, state, patch(1.0 if request_id == "a" else 2.0))

    participants = vocoder.select_step_participants()
    assert {request_id for request_id, _ in participants} == {"a", "b"}
    plan = vocoder.build_step_plan(participants)
    assert plan.take_patches == 1
    assert len(plan.slot_latents) == 2
    decoded = vocoder.run_step(participants, plan)

    assert len(pool.steps) == 1
    assert len(pool.steps[0]) == 2
    assert set(decoded) == {"a", "b"}


def test_select_step_participants_respects_max_batch_size() -> None:
    pool = RecordingSlotPool(num_slots=8)
    vocoder = DotsTTSStreamingVocoder(
        make_codec(),
        optimize=True,
        merge_steps=2,
        max_batch_size=2,
        stream_slots=8,
        slot_pool=pool,
    )
    assert vocoder.stream_chunk_batch_max == 2

    for request_id in ("a", "b", "c", "d"):
        state = vocoder.create_stream_state(request_id)
        vocoder.stream_states[request_id] = state
        vocoder.ingest(request_id, state, patch(float(ord(request_id))))

    participants = vocoder.select_step_participants()
    assert len(participants) == 2
    plan = vocoder.build_step_plan(participants)
    vocoder.run_step(participants, plan)
    assert len(pool.steps[0]) == 2

    # note (guozhihao-224): pump drains the backlog across capped steps.
    remaining = vocoder.select_step_participants()
    assert len(remaining) == 2


def test_stream_chunk_batch_cap_follows_max_batch_size_not_slots() -> None:
    vocoder = DotsTTSStreamingVocoder(
        make_codec(),
        optimize=True,
        max_batch_size=8,
        stream_slots=1,
        slot_pool=RecordingSlotPool(num_slots=1),
    )
    assert vocoder.stream_chunk_batch_max == 8
    assert vocoder.stream_slots == 1


def test_select_step_participants_defers_fresh_small_groups() -> None:
    pool = RecordingSlotPool()
    vocoder = DotsTTSStreamingVocoder(
        make_codec(),
        optimize=True,
        merge_steps=2,
        max_batch_size=4,
        stream_slots=4,
        stream_group_wait_ms=100,
        slot_pool=pool,
    )
    state = vocoder.create_stream_state("a")
    vocoder.stream_states["a"] = state
    vocoder.ingest("a", state, patch(1.0))
    assert state.ready_since_ns is not None

    # A freshly-ready group below the cap waits for more arrivals.
    assert vocoder.select_step_participants() == []
    # Once aged past the wait bound it steps immediately.
    state.ready_since_ns -= 200_000_000
    assert [request_id for request_id, _ in vocoder.select_step_participants()] == ["a"]


def test_select_step_participants_full_group_steps_immediately() -> None:
    pool = RecordingSlotPool()
    vocoder = DotsTTSStreamingVocoder(
        make_codec(),
        optimize=True,
        merge_steps=2,
        max_batch_size=4,
        stream_slots=4,
        stream_group_wait_ms=100,
        slot_pool=pool,
    )
    for request_id in ("a", "b", "c", "d"):
        state = vocoder.create_stream_state(request_id)
        vocoder.stream_states[request_id] = state
        vocoder.ingest(request_id, state, patch(float(ord(request_id))))

    participants = vocoder.select_step_participants()
    assert len(participants) == 4


def test_group_wait_resets_after_draining_pending_patches() -> None:
    pool = RecordingSlotPool()
    vocoder = DotsTTSStreamingVocoder(
        make_codec(),
        optimize=True,
        merge_steps=2,
        max_batch_size=4,
        stream_slots=4,
        stream_group_wait_ms=100,
        slot_pool=pool,
    )
    fresh = vocoder.create_stream_state("fresh")
    vocoder.stream_states["fresh"] = fresh
    vocoder.ingest("fresh", fresh, patch(1.0))
    fresh.ready_since_ns -= 200_000_000
    participants = vocoder.select_step_participants()
    plan = vocoder.build_step_plan(participants)
    vocoder.run_step(participants, plan)
    # The step drained the only pending patch, so the state is no longer ready
    # and its age marker resets for the next accumulation cycle.
    assert fresh.ready_since_ns is None
    assert vocoder.select_step_participants() == []


def test_streaming_groups_by_exact_frame_count() -> None:
    pool = RecordingSlotPool()
    vocoder = DotsTTSStreamingVocoder(
        make_codec(),
        optimize=True,
        merge_steps=2,
        stream_slots=4,
        stream_group_wait_ms=0,
        slot_pool=pool,
    )
    early = vocoder.create_stream_state("early")
    steady = vocoder.create_stream_state("steady")
    vocoder.stream_states["early"] = early
    vocoder.stream_states["steady"] = steady

    vocoder.ingest("early", early, patch(1.0))
    # note (guozhihao-224): past the first-two-patch fast path so take_patches
    # diverges from early.
    steady.received_patches = 3
    steady.pending = [patch(2.0), patch(3.0)]
    steady.slot = pool.acquire()

    participants = vocoder.select_step_participants()
    frames = {vocoder.step_frames(state) for _, state in participants}
    assert len(frames) == 1
    plan = vocoder.build_step_plan(participants)
    assert len({int(t.shape[1]) for t in plan.slot_latents.values()}) == 1


def test_stream_done_flushes_and_releases_slot() -> None:
    pool = RecordingSlotPool()
    vocoder = DotsTTSStreamingVocoder(
        make_codec(),
        optimize=True,
        merge_steps=2,
        slot_pool=pool,
    )
    state = vocoder.create_stream_state("req")
    vocoder.ingest("req", state, patch(1.0))
    slot = state.slot
    assert slot is not None
    waveform = vocoder.decode_delta("req", state, is_final=True)
    assert waveform is not None
    assert state.slot is None
    assert pool.flushes == [slot]
    assert slot not in pool.in_use


def test_stream_done_drains_large_backlog_in_merge_step_pieces() -> None:
    # Group-wait deferral can leave more pending patches than one window
    # allows; the final drain must split them like the coalesced pump does.
    pool = RecordingSlotPool()
    vocoder = DotsTTSStreamingVocoder(
        make_codec(),
        optimize=True,
        merge_steps=2,
        slot_pool=pool,
    )
    state = vocoder.create_stream_state("req")
    for value in (1.0, 2.0, 3.0, 4.0, 5.0):
        vocoder.ingest("req", state, patch(value))
    waveform = vocoder.decode_delta("req", state, is_final=True)
    assert waveform is not None
    frames = [int(latents.shape[1]) for step in pool.steps for latents in step.values()]
    assert frames == [6, 6, 3]
    assert state.pending == []


def test_release_stream_resources_returns_slot() -> None:
    pool = RecordingSlotPool()
    vocoder = DotsTTSStreamingVocoder(
        make_codec(),
        optimize=False,
        slot_pool=pool,
    )
    state = vocoder.create_stream_state("req")
    vocoder.ingest("req", state, patch())
    slot = state.slot
    vocoder.release_stream_resources("req", state)
    assert state.slot is None
    assert slot not in pool.in_use


def test_slot_exhaustion_raises_clear_admission_error() -> None:
    pool = RecordingSlotPool(num_slots=1)
    vocoder = DotsTTSStreamingVocoder(
        make_codec(),
        optimize=True,
        stream_slots=1,
        slot_pool=pool,
    )
    first = vocoder.create_stream_state("a")
    vocoder.ingest("a", first, patch())
    second = vocoder.create_stream_state("b")
    with pytest.raises(RuntimeError, match="ran out of slots"):
        vocoder.ingest("b", second, patch())


def test_on_stream_chunk_batch_uses_pool_not_compiled_stream_step() -> None:
    pool = RecordingSlotPool()
    codec = make_codec()
    vocoder = DotsTTSStreamingVocoder(
        codec,
        optimize=True,
        merge_steps=2,
        stream_group_wait_ms=0,
        slot_pool=pool,
    )
    payload_state = vocoder.create_stream_state("req")
    vocoder.stream_states["req"] = payload_state
    vocoder.stream_payloads["req"] = SimpleNamespace(
        request_id="req",
        request=SimpleNamespace(params={"stream": True}),
        data={},
    )

    item = StreamItem(
        chunk_id=0,
        data=patch(1.0),
        from_stage="latent_engine",
        metadata={"modality": "audio_latents", "stream": True},
    )
    vocoder.on_stream_chunk_batch([("req", item)])

    assert pool.steps, "expected coalesced pool step"
    assert len(pool.steps[0]) == 1


def test_decode_delta_non_final_is_a_no_op() -> None:
    pool = RecordingSlotPool()
    vocoder = DotsTTSStreamingVocoder(
        make_codec(),
        optimize=True,
        merge_steps=2,
        slot_pool=pool,
    )
    state = vocoder.create_stream_state("req")
    vocoder.ingest("req", state, patch(1.0))
    assert vocoder.decode_delta("req", state, is_final=False) is None
    assert state.pending
    assert state.slot is not None
    assert not pool.steps


class LatentWaveformInference(FakeInference):
    def _decode_stream_window(
        self, window: torch.Tensor
    ) -> torch.Tensor:  # noqa: leading-underscore  # AudioVAE interface
        return window.sum(dim=1, keepdim=True).repeat_interleave(self.hop_size, dim=-1)


@pytest.mark.parametrize("valid_frame_counts", [(0, 0), (0, 3), (3, 3)])
def test_prefix_decoder_preserves_full_state_and_stable_audio(
    valid_frame_counts: tuple[int, int],
) -> None:
    inference = LatentWaveformInference()
    pool = DotsVocoderSlotPool(inference, num_slots=2, chunk_size=12, latent_dim=5)
    generator = torch.Generator().manual_seed(42)
    packed = torch.randn(2, 5, 3, generator=generator)
    hidden = torch.randn(1, 2, 8, generator=generator)
    window = torch.zeros(2, 5, pool.window_size)
    for row, valid_frames in enumerate(valid_frame_counts):
        window[row, :, :valid_frames] = torch.randn(
            5, valid_frames, generator=generator
        )
    valid = torch.tensor(valid_frame_counts, dtype=torch.int64)
    expected_audio, *expected_states = pool.forward(
        packed, hidden, hidden.clone(), window, valid
    )
    prefix_frames = max(valid_frame_counts) + packed.shape[-1]
    actual_audio, *actual_states = pool.forward(
        packed,
        hidden,
        hidden.clone(),
        window,
        valid,
        decoder_window_frames=prefix_frames,
    )
    for actual_state, expected_state in zip(
        actual_states, expected_states, strict=True
    ):
        torch.testing.assert_close(actual_state, expected_state)
    for row, valid_frames in enumerate(valid_frame_counts):
        stable_samples = (
            valid_frames + packed.shape[-1] - pool.lookahead
        ) * pool.hop_size
        torch.testing.assert_close(
            actual_audio[row, ..., :stable_samples],
            expected_audio[row, ..., :stable_samples],
        )
    assert actual_states[-1].shape == window.shape
