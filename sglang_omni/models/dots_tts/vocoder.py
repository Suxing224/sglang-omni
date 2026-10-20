# SPDX-License-Identifier: Apache-2.0
"""dots.tts AudioVAE adapters for Omni's shared vocoder schedulers."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from functools import partial

import torch

from sglang_omni.models.dots_tts.codec import DotsAudioCodec
from sglang_omni.models.dots_tts.payload_types import DotsTTSState, load_dots_tts_state
from sglang_omni.models.dots_tts.vocoder_conv_cache import ConvCacheDecoder
from sglang_omni.models.dots_tts.vocoder_cuda_graph import DotsVocoderGraphRunner
from sglang_omni.models.dots_tts.vocoder_slot_pool import DotsVocoderSlotPool
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase
from sglang_omni.scheduling.vocoder_base import BatchVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload

_LENGTH_BUCKET_FRAMES = 32
logger = logging.getLogger(__name__)


class DotsTTSBatchVocoder(BatchVocoderBase[DotsTTSState, torch.Tensor]):
    def __init__(self, codec: DotsAudioCodec) -> None:
        self.codec = codec
        self.logged_batch = False

    def prepare_item(self, payload: StagePayload) -> tuple[DotsTTSState, torch.Tensor]:
        state = load_dots_tts_state(payload)
        if state.generated_latents is None:
            raise RuntimeError("dots.tts vocoder received no generated latents")
        else:
            pass
        return state, state.generated_latents

    async def decode_batch(
        self, items: list[tuple[DotsTTSState, torch.Tensor]]
    ) -> list[tuple[torch.Tensor, int]]:
        outputs: list[tuple[torch.Tensor, int] | None] = [None] * len(items)
        buckets: dict[int, list[tuple[int, torch.Tensor]]] = {}
        for index, (_state, latents) in enumerate(items):
            self.validate_latents(latents)
            frames = int(latents.shape[1])
            bucket = (frames + _LENGTH_BUCKET_FRAMES - 1) // _LENGTH_BUCKET_FRAMES
            buckets.setdefault(bucket, []).append((index, latents))

        bucket_sizes = [len(bucket_items) for bucket_items in buckets.values()]
        if not self.logged_batch and max(bucket_sizes, default=0) > 1:
            logger.info(
                "dots.tts AudioVAE batched decode is active: requests=%d "
                "bucket_sizes=%s",
                len(items),
                bucket_sizes,
            )
            self.logged_batch = True
        else:
            pass

        with self.codec.lock:
            for bucket_items in buckets.values():
                frame_counts = [int(latents.shape[1]) for _, latents in bucket_items]
                max_frames = max(frame_counts)
                if len(bucket_items) == 1:
                    padded = bucket_items[0][1].to(self.codec.device)
                elif min(frame_counts) == max_frames:
                    padded = torch.cat(
                        [latents for _, latents in bucket_items], dim=0
                    ).to(self.codec.device)
                else:
                    padded = bucket_items[0][1].new_zeros(
                        len(bucket_items),
                        max_frames,
                        self.codec.latent_dim,
                        device=self.codec.device,
                    )
                    for row, ((_index, latents), frames) in enumerate(
                        zip(bucket_items, frame_counts, strict=True)
                    ):
                        padded[row, :frames].copy_(latents[0].to(self.codec.device))

                waveform_batch = self.codec.inference.decode_latents(padded)
                if waveform_batch.shape[0] != len(bucket_items):
                    raise RuntimeError(
                        "dots.tts AudioVAE returned an unexpected batch size: "
                        f"expected {len(bucket_items)}, got {waveform_batch.shape[0]}"
                    )
                else:
                    pass
                for row, (index, latents) in enumerate(bucket_items):
                    valid_samples = int(latents.shape[1]) * self.codec.hop_size
                    outputs[index] = (
                        waveform_batch[row : row + 1, ..., :valid_samples],
                        self.codec.sample_rate,
                    )

        if any(output is None for output in outputs):
            raise RuntimeError("dots.tts AudioVAE did not produce every batch result")
        else:
            pass
        return [output for output in outputs if output is not None]

    def validate_latents(self, latents: torch.Tensor) -> None:
        if latents.ndim != 3 or latents.shape[0] != 1:
            raise ValueError(
                "dots.tts latents must have shape [1, frames, latent_dim], "
                f"got {tuple(latents.shape)}"
            )
        else:
            pass
        if latents.shape[1] == 0:
            raise ValueError("dots.tts latents must contain at least one frame")
        else:
            pass
        if latents.shape[2] != self.codec.latent_dim:
            raise ValueError(
                f"dots.tts latent_dim must be {self.codec.latent_dim}, "
                f"got {latents.shape[2]}"
            )
        else:
            pass

    def store_result(
        self,
        payload: StagePayload,
        state: DotsTTSState,
        wav: torch.Tensor,
        sample_rate: int,
    ) -> StagePayload:
        data: dict[str, bytes | list[int] | str | int | dict[str, int | float]] = dict(
            audio_waveform_payload(
                wav,
                sample_rate=sample_rate,
                modality="audio",
                source_hint="dots.tts",
            )
        )
        usage = build_usage(state)
        if usage is not None:
            data["usage"] = usage
        else:
            pass
        payload.data = data
        return payload

    async def decode_payload(self, payload: StagePayload) -> StagePayload:
        state, latents = self.prepare_item(payload)
        [(wav, sample_rate)] = await self.decode_batch([(state, latents)])
        return self.store_result(payload, state, wav, sample_rate)


@dataclass
class DotsStreamState:
    slot: int | None = None
    pending: list[torch.Tensor] = field(default_factory=list)
    received_patches: int = 0
    ready_since_ns: int | None = None


@dataclass(frozen=True)
class DotsCoalescedStepPlan:
    take_patches: int
    slot_latents: dict[int, torch.Tensor]


class DotsTTSStreamingVocoder(
    StreamingVocoderBase[DotsStreamState, DotsCoalescedStepPlan]
):
    """Streaming AudioVAE with slot-pooled equal-T batched eager decode."""

    can_batch_stream_chunks = True
    stream_chunk_batch_distinct_requests = True

    def __init__(
        self,
        codec: DotsAudioCodec,
        *,
        optimize: bool,
        enable_streaming_fast_start: bool,
        enable_streaming_audio_vae_cuda_graph: bool = True,
        enable_streaming_conv_cache: bool = False,
        merge_steps: int = 4,
        max_batch_size: int = 4,
        max_batch_wait_ms: int = 2,
        stream_slots: int = 16,
        stream_chunk_batch_max: int | None = None,
        stream_group_wait_ms: float = 0.0,
        slot_pool: DotsVocoderSlotPool | None = None,
    ) -> None:
        if merge_steps < 1:
            raise ValueError("dots.tts vocoder merge_steps must be positive")
        else:
            pass
        if max_batch_size < 1:
            raise ValueError("dots.tts vocoder max_batch_size must be positive")
        else:
            pass
        if max_batch_wait_ms < 0:
            raise ValueError("dots.tts vocoder max_batch_wait_ms must be non-negative")
        else:
            pass
        if stream_slots < 1:
            raise ValueError("dots.tts vocoder stream_slots must be positive")
        else:
            pass
        if stream_chunk_batch_max is not None and stream_chunk_batch_max < 1:
            raise ValueError("dots.tts vocoder stream_chunk_batch_max must be positive")
        else:
            pass
        if stream_group_wait_ms < 0:
            raise ValueError(
                "dots.tts vocoder stream_group_wait_ms must be non-negative"
            )
        else:
            pass
        self.codec = codec
        self.optimize = bool(optimize)
        self.enable_streaming_fast_start = enable_streaming_fast_start
        self.merge_steps = int(merge_steps) if optimize else 1
        self.stream_slots = int(stream_slots)
        self.enable_streaming_audio_vae_cuda_graph = bool(
            enable_streaming_audio_vae_cuda_graph
        )
        self.enable_streaming_conv_cache = bool(enable_streaming_conv_cache)
        self.batch_vocoder = DotsTTSBatchVocoder(codec)
        self.slot_pool = slot_pool
        # note (guozhihao-224): coalesce width follows max_batch_size only;
        # stream_slots is admission capacity and must not redefine the batch cap.
        # stream_chunk_batch_max overrides the streaming cap alone so widening the
        # coalesce never changes the non-streaming batch path.
        self.stream_chunk_batch_max = (
            int(max_batch_size)
            if stream_chunk_batch_max is None
            else int(stream_chunk_batch_max)
        )
        # note (codex): Small groups become eligible after this coalescing delay.
        self.stream_group_wait_ms = float(stream_group_wait_ms)
        super().__init__(
            self.batch_vocoder.decode_payload,
            batch_compute_fn=self.batch_vocoder.decode_payloads,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            sample_rate=codec.sample_rate,
            stream_source_hint="dots.tts",
            stream_input_modality="audio_latents",
        )

    def ensure_slot_pool(self) -> DotsVocoderSlotPool:
        """Allocate the slot pool at executor setup (not the first live chunk)."""
        with self.codec.lock:
            return self.get_or_create_pool_locked()

    def validate_non_streaming_payload(self, payload: StagePayload) -> None:
        _state, latents = self.batch_vocoder.prepare_item(payload)
        self.batch_vocoder.validate_latents(latents)

    def create_stream_state(self, request_id: str) -> DotsStreamState:
        del request_id
        return DotsStreamState()

    def validate_chunk(
        self,
        request_id: str,
        state: DotsStreamState,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        del request_id, state
        if codes.ndim != 3 or codes.shape[0] != 1:
            raise ValueError(
                "dots.tts latent chunks must have shape [1, frames, latent_dim]"
            )
        else:
            pass
        if codes.shape[-1] != self.codec.latent_dim:
            raise ValueError(
                f"dots.tts latent_dim must be {self.codec.latent_dim}, "
                f"got {codes.shape[-1]}"
            )
        else:
            pass
        return codes.to(self.codec.device)

    def ingest(
        self, request_id: str, state: DotsStreamState, codes: torch.Tensor
    ) -> None:
        del request_id
        state.pending.append(codes)
        state.received_patches += 1
        self.ensure_slot(state)
        if state.ready_since_ns is None and self.pending_ready(state):
            state.ready_since_ns = time.perf_counter_ns()
        else:
            pass

    def should_decode(self, state: DotsStreamState, *, is_final: bool) -> bool:
        # note (guozhihao-224): coalesced serving uses select_step_participants;
        # this gate remains for direct decode_delta / stream-done callers.
        if is_final:
            return bool(state.pending)
        else:
            pass
        return self.pending_ready(state)

    def decode_delta(
        self, request_id: str, state: DotsStreamState, *, is_final: bool
    ) -> torch.Tensor | None:
        """Stream-done drain: pending -> slot step, flush, release.

        Steady chunks are emitted by the coalesced run_step pump.
        """
        del request_id
        if not is_final:
            return None
        else:
            pass
        chunks: list[torch.Tensor] = []
        with self.codec.lock:
            pool = self.get_or_create_pool_locked()
            if state.slot is None and state.pending:
                state.slot = pool.acquire()
            else:
                pass
            if state.slot is not None and state.pending:
                # note (codex): drain in merge_steps pieces like the coalesced
                # pump does; group-wait deferral lets pending grow past one
                # window (step length must stay < window size), so a single
                # take-all step is not always legal.
                while state.pending:
                    take = min(self.merge_steps, len(state.pending))
                    patches, state.pending = state.pending[:take], state.pending[take:]
                    # note (db-ol): compiled stream_step cudagraph trees corrupt the
                    # backbone decode graph replay in this process, see issue 1392;
                    # the slot pool stays on the eager kernels (#1395).
                    chunk = pool.step({state.slot: torch.cat(patches, dim=1)})[
                        state.slot
                    ]
                    if chunk.numel():
                        chunks.append(chunk)
                    else:
                        pass
            else:
                pass
            if state.slot is not None:
                tail = pool.flush(state.slot)
                pool.release(state.slot)
                state.slot = None
                if tail.numel():
                    chunks.append(tail)
                else:
                    pass
            else:
                pass
        if not chunks:
            return None
        else:
            pass
        return torch.cat(chunks, dim=-1)

    def final_result_data(
        self, request_id: str, payload: StagePayload, state: DotsStreamState
    ) -> dict[str, str | int | dict[str, int | float]]:
        del request_id, state
        tts_state = load_dots_tts_state(payload)
        result: dict[str, str | int | dict[str, int | float]] = {
            "modality": "audio",
            "sample_rate": self.codec.sample_rate,
        }
        usage = build_usage(tts_state)
        if usage is not None:
            result["usage"] = usage
        else:
            pass
        return result

    def fallback_full_decode(
        self, request_id: str, payload: StagePayload, state: DotsStreamState
    ) -> torch.Tensor | None:
        del request_id, state
        tts_state = load_dots_tts_state(payload)
        if tts_state.generated_latents is None:
            return None
        else:
            pass
        with self.codec.lock:
            return self.codec.inference.decode_latents(
                tts_state.generated_latents.to(self.codec.device)
            )

    def release_stream_resources(self, request_id: str, state: DotsStreamState) -> None:
        del request_id
        if state.slot is None:
            return
        else:
            pass
        with self.codec.lock:
            self.get_or_create_pool_locked().release(state.slot)
        state.slot = None

    def has_ready_work(self) -> bool:
        # note (codex): Expired groups must drain even while the producer is idle.
        with self.state_lock:
            return bool(self.select_step_participants())

    def select_step_participants(self) -> list[tuple[str, DotsStreamState]]:
        slotted = [
            (request_id, state)
            for request_id, state in self.stream_state_items()
            if state.slot is not None and self.pending_ready(state)
        ]
        if not slotted:
            return []
        else:
            pass
        # note (guozhihao-224): exact-T groups only; padding would change AudioVAE
        # convolution boundaries. Cap with max_batch_size so one pool.step does
        # not grow to stream_slots under high concurrency.
        by_frames: dict[int, list[tuple[str, DotsStreamState]]] = {}
        for entry in slotted:
            frames = self.step_frames(entry[1])
            by_frames.setdefault(frames, []).append(entry)
        wait_ns = int(self.stream_group_wait_ms * 1e6)
        now = time.perf_counter_ns()
        eligible: list[list[tuple[str, DotsStreamState]]] = []
        for entries in by_frames.values():
            if len(entries) >= self.stream_chunk_batch_max:
                eligible.append(entries)
                continue
            else:
                pass
            if any(
                state.ready_since_ns is None or now - state.ready_since_ns >= wait_ns
                for _, state in entries
            ):
                eligible.append(entries)
            else:
                pass
        if not eligible:
            return []
        else:
            pass
        return max(eligible, key=len)[: self.stream_chunk_batch_max]

    def build_step_plan(
        self, participants: list[tuple[str, DotsStreamState]]
    ) -> DotsCoalescedStepPlan:
        take_patches = self.take_patches(participants[0][1])
        slot_latents: dict[int, torch.Tensor] = {}
        for _, state in participants:
            if state.slot is None:
                raise RuntimeError("dots.tts coalesced step is missing a vocoder slot")
            else:
                pass
            if self.take_patches(state) != take_patches:
                raise RuntimeError("dots.tts coalesced step mixed unequal patch counts")
            else:
                pass
            patches = state.pending[:take_patches]
            latents = torch.cat(patches, dim=1)
            if int(latents.shape[1]) != take_patches * self.codec.patch_size:
                raise RuntimeError(
                    "dots.tts coalesced step latent frames "
                    f"{int(latents.shape[1])} != "
                    f"{take_patches * self.codec.patch_size}"
                )
            else:
                pass
            slot_latents[state.slot] = latents
        return DotsCoalescedStepPlan(
            take_patches=take_patches,
            slot_latents=slot_latents,
        )

    def run_step(
        self,
        participants: list[tuple[str, DotsStreamState]],
        plan: DotsCoalescedStepPlan,
    ) -> dict[str, torch.Tensor]:
        with self.codec.lock:
            decoded = self.get_or_create_pool_locked().step(plan.slot_latents)
        out: dict[str, torch.Tensor] = {}
        for request_id, state in participants:
            del state.pending[: plan.take_patches]
            if not self.pending_ready(state):
                state.ready_since_ns = None
            else:
                pass
            if state.slot is None:
                continue
            else:
                pass
            waveform = decoded[state.slot]
            if waveform.numel():
                out[request_id] = waveform
            else:
                pass
        return out

    def get_or_create_pool_locked(self) -> DotsVocoderSlotPool:
        # Caller must hold self.codec.lock (RLock).
        if self.slot_pool is None:
            # note (codex): the conv-cache decoder only pays off where the
            # full-window step is the pacer; other devices keep the window path.
            use_conv_cache = (
                self.optimize
                and self.enable_streaming_conv_cache
                and self.codec.device.type == "npu"
            )
            pool = DotsVocoderSlotPool(
                self.codec.inference,
                num_slots=self.stream_slots,
                chunk_size=self.codec.patch_size * self.merge_steps,
                latent_dim=self.codec.latent_dim,
                conv_cache_decoder=(
                    ConvCacheDecoder(self.codec) if use_conv_cache else None
                ),
            )
            capture_graphs = (
                self.enable_streaming_audio_vae_cuda_graph
                and pool.device.type in {"cuda", "npu"}
            )
            if pool.conv_cache_decoder is not None:
                pool.conv_cache_graph_runner = (
                    self.capture_conv_cache_step_graphs(pool)
                    if capture_graphs
                    else None
                )
            else:
                pass
            # note (codex): rows below warmup_frames still take the window path
            # on the conv-cache pool, so the window graphs stay wired either way.
            pool.graph_runner = (
                self.capture_step_graphs(pool) if capture_graphs else None
            )
            if pool.device.type == "npu" and pool.graph_runner is not None:
                patch_frames = self.codec.patch_size
                merged_frames = patch_frames * self.merge_steps
                decoder_shapes = {
                    patch_frames: patch_frames,
                    patch_frames * 2: patch_frames,
                    patch_frames * 2 + merged_frames: merged_frames,
                    patch_frames * 2 + merged_frames * 2: merged_frames,
                }
                for decoder_frames, latent_frames in decoder_shapes.items():
                    if decoder_frames < pool.window_size:
                        pool.prefix_graph_runners[decoder_frames] = (
                            self.capture_step_graphs(
                                pool, decoder_shape=(decoder_frames, latent_frames)
                            )
                        )
                    else:
                        pass
            else:
                pass
            self.slot_pool = pool
            logger.info(
                f"dots.tts streaming vocoder slot pool ready: "
                f"slots={self.stream_slots} merge_steps={self.merge_steps} "
                f"chunk_size={self.codec.patch_size * self.merge_steps} "
                f"conv_cache={pool.conv_cache_decoder is not None}"
            )
        else:
            pass
        return self.slot_pool

    def capture_step_graphs(
        self,
        pool: DotsVocoderSlotPool,
        *,
        decoder_shape: tuple[int, int] | None = None,
    ) -> DotsVocoderGraphRunner:
        if decoder_shape is None:
            forward = pool.forward
            capture_keys = self.cuda_graph_capture_keys()
        else:
            decoder_frames, latent_frames = decoder_shape
            forward = partial(pool.forward, decoder_window_frames=decoder_frames)
            capture_keys = [
                capture_key
                for capture_key in self.cuda_graph_capture_keys()
                if capture_key[1] == latent_frames
            ]
        runner = DotsVocoderGraphRunner(
            forward=forward,
            new_inputs=pool.new_step_inputs,
            device=pool.device,
        )
        runner.capture(capture_keys)
        return runner

    def capture_conv_cache_step_graphs(
        self,
        pool: DotsVocoderSlotPool,
    ) -> DotsVocoderGraphRunner:
        runner = DotsVocoderGraphRunner(
            forward=pool.forward_conv_cache,
            new_inputs=pool.new_conv_cache_step_inputs,
            device=pool.device,
        )
        runner.capture(self.cuda_graph_capture_keys())
        return runner

    def cuda_graph_capture_keys(self) -> list[tuple[int, int]]:
        # note (lennox): only single-request final drains use intermediate lengths.
        return [
            (batch, self.codec.patch_size * patches)
            for batch in range(
                1, min(self.stream_chunk_batch_max, self.stream_slots) + 1
            )
            for patches in range(1, self.merge_steps + 1)
            if batch == 1 or patches in (1, self.merge_steps)
        ]

    @property
    def cuda_graph_count(self) -> int:
        pool = self.slot_pool
        if pool is None:
            return 0
        else:
            pass
        count = 0
        if pool.conv_cache_graph_runner is not None:
            count += len(pool.conv_cache_graph_runner.captured_keys)
        else:
            pass
        runner = pool.graph_runner
        if runner is None:
            return count
        else:
            pass
        return (
            count
            + len(runner.captured_keys)
            + sum(
                len(prefix_runner.captured_keys)
                for prefix_runner in pool.prefix_graph_runners.values()
            )
        )

    def ensure_slot(self, state: DotsStreamState) -> None:
        if state.slot is not None:
            return
        else:
            pass
        with self.codec.lock:
            if state.slot is None:
                state.slot = self.get_or_create_pool_locked().acquire()
            else:
                pass

    def pending_ready(self, state: DotsStreamState) -> bool:
        if not state.pending:
            return False
        else:
            pass
        if self.enable_streaming_fast_start and state.received_patches <= 2:
            return True
        else:
            pass
        return len(state.pending) >= self.merge_steps

    def take_patches(self, state: DotsStreamState) -> int:
        if self.enable_streaming_fast_start and state.received_patches <= 2:
            return 1
        else:
            pass
        return min(self.merge_steps, len(state.pending))

    def step_frames(self, state: DotsStreamState) -> int:
        return self.take_patches(state) * self.codec.patch_size


__all__ = ["DotsTTSBatchVocoder", "DotsTTSStreamingVocoder"]
