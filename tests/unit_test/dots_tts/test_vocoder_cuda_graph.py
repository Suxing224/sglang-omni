# SPDX-License-Identifier: Apache-2.0
"""Graph replay and incremental AudioVAE decoding regression tests."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sglang_omni.models.dots_tts.compat import import_dots_tts
from sglang_omni.models.dots_tts.npu_vocoder import adapt_audio_vae_convolutions
from sglang_omni.models.dots_tts.vocoder import DotsTTSStreamingVocoder
from sglang_omni.models.dots_tts.vocoder_conv_cache import ConvCacheDecoder
from sglang_omni.models.dots_tts.vocoder_cuda_graph import (
    DotsVocoderGraphRunner,
    capture_device_graph,
)
from sglang_omni.models.dots_tts.vocoder_slot_pool import DotsVocoderSlotPool
from sglang_omni.platforms import current_platform

import_dots_tts()

from dots_tts.modules.backbone.layers import Conv1d, ConvTranspose1d
from dots_tts.modules.vocoder.alias_free_filter import LowPassFilter1d
from dots_tts.modules.vocoder.alias_free_resample import UpSample1d
from dots_tts.modules.vocoder.bigvgan import Decoder
from dots_tts.modules.vocoder.vocoder_inference import VocoderInference

LATENT_DIM = 5
HIDDEN = 8
HOP = 2
PATCH = 4
MERGE = 4
CHUNK = PATCH * MERGE
NUM_SLOTS = 6
MODEL_MAX_GENERATE_PATCHES = 500
MODEL_PATCH_FRAMES = 4
MODEL_HOP_SAMPLES = 1920
REFERENCE_CHUNK_SAMPLES = 65536


@pytest.mark.accelerator
@pytest.mark.parametrize(
    "channels, samples, ratio, kernel_size, fixed_filter, causal",
    [
        (channels, samples, ratio, kernel_size, fixed_filter, causal)
        for channels, samples, ratio, kernel_size in (
            (48, 61440, 2, 12),
            (24, 122880, 2, 12),
            (3, 5, 3, 17),
            (3, 1, 2, 11),
        )
        for fixed_filter in (False, True)
        for causal in (False, True)
    ]
    + [
        (
            24,
            MODEL_MAX_GENERATE_PATCHES * MODEL_PATCH_FRAMES * MODEL_HOP_SAMPLES,
            2,
            12,
            False,
            True,
        )
    ],
)
@torch.inference_mode()
def test_long_upsampling_matches_filter_definition_after_graph_replay(
    graph_device: torch.device,
    channels: int,
    samples: int,
    ratio: int,
    kernel_size: int,
    fixed_filter: bool,
    causal: bool,
) -> None:
    module = UpSample1d(
        channels=channels,
        causal=causal,
        fixed_filter=fixed_filter,
        ratio=ratio,
        kernel_size=kernel_size,
    )
    generator = torch.Generator().manual_seed(17)
    module.filter.add_(torch.randn(module.filter.shape, generator=generator) * 0.02)
    filter_weights = module.filter.detach().expand(channels, -1, -1).clone()
    module.to(graph_device)
    if graph_device.type == "npu":
        # note (codex): Reproduce the format policy enabled by engine initialization.
        torch.get_device_module(graph_device).config.allow_internal_format = True
        adapt_audio_vae_convolutions(module)
    else:
        pass
    waveform = torch.zeros(1, channels, samples, device=graph_device)
    device_module = torch.get_device_module(graph_device)
    capture_stream = device_module.Stream(device=graph_device)
    capture_stream.wait_stream(device_module.current_stream(graph_device))
    with device_module.stream(capture_stream):
        module(waveform)
        module(waveform)
    capture_stream.synchronize()
    with capture_device_graph(
        device=graph_device,
        graph_pool=device_module.graph_pool_handle(),
        capture_stream=capture_stream,
    ) as graph:
        actual = module(waveform)
    device_module.current_stream(graph_device).wait_stream(capture_stream)
    generator = torch.Generator().manual_seed(42)
    for _ in range(2):
        reference_waveform = (
            torch.randn(1, channels, samples, generator=generator) * 0.2
        )
        reference_waveform[..., [0, samples // 2, samples - 1]] = 1.0
        padded = F.pad(reference_waveform, (module.pad, module.pad), mode="replicate")
        inserted = F.pad(padded.unsqueeze(-1), (0, module.stride - 1)).flatten(-2)
        convolution_windows = F.pad(
            inserted, (module.kernel_size - 1, module.kernel_size - module.stride)
        ).unfold(-1, module.kernel_size, 1)
        # note (codex): Bound the FIR product without shortening the waveform.
        expected = convolution_windows.new_empty(convolution_windows.shape[:-1])
        for sample_start in range(0, expected.shape[-1], REFERENCE_CHUNK_SAMPLES):
            sample_slice = slice(sample_start, sample_start + REFERENCE_CHUNK_SAMPLES)
            expected[..., sample_slice] = module.ratio * (
                convolution_windows[..., sample_slice, :]
                * filter_weights.flip(-1).unsqueeze(0)
            ).sum(-1)
        if module.causal:
            expected = expected[..., : -(module.kernel_size - module.stride)]
        else:
            expected = expected[..., module.pad_left : -module.pad_right]
        waveform.copy_(reference_waveform)
        graph.replay()
        observed = actual.cpu()
        assert torch.isfinite(observed).all()
        torch.testing.assert_close(observed, expected, rtol=1e-3, atol=3e-4)


class NonlinearInference:
    def __init__(self, device: torch.device, *, seed: int = 0) -> None:
        self.device = device
        self.dtype = torch.float32
        self.lookahead = 1
        self.left_context = 3
        self.vocoder = SimpleNamespace(hop_size=HOP)
        generator = torch.Generator().manual_seed(seed)
        self.drive = torch.rand(1, 1, HIDDEN, generator=generator).to(device)

    def init_stream_state(self, *, batch_size: int, chunk_size: int):
        window = torch.zeros(
            batch_size,
            LATENT_DIM,
            chunk_size + self.lookahead + self.left_context,
            device=self.device,
            dtype=self.dtype,
        )
        hidden = torch.zeros(1, batch_size, HIDDEN, device=self.device)
        return SimpleNamespace(
            lstm_hidden=(hidden, hidden.clone()),
            decoder=SimpleNamespace(window=window, chunk_size=chunk_size),
        )

    def _decoder_stream_lookahead(
        self,
    ) -> int:  # noqa: leading-underscore  # AudioVAE interface
        return self.lookahead

    def _validate_stream_latents(
        self, latents: torch.Tensor
    ) -> None:  # noqa: leading-underscore  # AudioVAE interface
        if latents.ndim != 3 or int(latents.shape[1]) != LATENT_DIM:
            raise ValueError(f"bad latents {tuple(latents.shape)}")
        else:
            pass

    def _decode_stream_latents(
        self, latents, hidden
    ):  # noqa: leading-underscore  # AudioVAE interface
        hidden_h, hidden_c = hidden
        latents = latents.float()
        drive = latents.mean(dim=(1, 2))
        hidden_h = torch.tanh(hidden_h * 0.9 + drive[None, :, None] * self.drive)
        hidden_c = hidden_c + latents.abs().mean(dim=(1, 2))[None, :, None]
        gain = 1.0 + hidden_h.sum(dim=(0, 2))
        return (latents * gain[:, None, None]).to(self.dtype), (hidden_h, hidden_c)

    def _decode_stream_window(
        self, window: torch.Tensor
    ) -> torch.Tensor:  # noqa: leading-underscore  # AudioVAE interface
        mixed = torch.tanh(window.sum(dim=1).cumsum(dim=-1))
        return mixed.repeat_interleave(HOP, dim=-1).unsqueeze(1)


def make_pool(device: torch.device, *, graph_keys=None) -> DotsVocoderSlotPool:
    inference = NonlinearInference(device)
    pool = DotsVocoderSlotPool(
        inference, num_slots=NUM_SLOTS, chunk_size=CHUNK, latent_dim=LATENT_DIM
    )
    if graph_keys is not None:
        runner = DotsVocoderGraphRunner(
            forward=pool.forward,
            new_inputs=pool.new_step_inputs,
            device=pool.device,
        )
        runner.capture(graph_keys)
        pool.graph_runner = runner
    else:
        pass
    return pool


# (slots stepped together, frames): batch sizes 1..4 with uneven row ages,
# plus single-row final drains at both intermediate frame counts.
SCENARIO = [
    ([0], PATCH),
    ([0, 1], PATCH),
    ([0, 1, 2, 3], CHUNK),
    ([1, 3], CHUNK),
    ([2], PATCH),
    ([0, 1, 2], CHUNK),
    ([3, 0, 2, 1], CHUNK),
    ([1, 2, 3], PATCH),
    ([0], 2 * PATCH),
    ([1], 3 * PATCH),
]


def scenario_inputs(device: torch.device) -> list[dict[int, torch.Tensor]]:
    generator = torch.Generator().manual_seed(1234)
    steps = []
    for slots, frames in SCENARIO:
        steps.append(
            {
                slot: torch.randn(1, frames, LATENT_DIM, generator=generator).to(device)
                for slot in slots
            }
        )
    return steps


def run_scenario(pool: DotsVocoderSlotPool, steps) -> list[dict[int, torch.Tensor]]:
    for _ in range(4):
        pool.acquire()
    outputs = [pool.step(step) for step in steps]
    outputs.append({slot: pool.flush(slot) for slot in range(4)})
    return outputs


def assert_same_outputs(
    reference: list[dict[int, torch.Tensor]],
    candidate: list[dict[int, torch.Tensor]],
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> None:
    assert len(reference) == len(candidate)
    for step_index, (expected, actual) in enumerate(
        zip(reference, candidate, strict=True)
    ):
        assert expected.keys() == actual.keys()
        for slot in expected:
            torch.testing.assert_close(
                actual[slot],
                expected[slot],
                rtol=rtol,
                atol=atol,
                msg=lambda message: f"step {step_index} slot {slot}: {message}",
            )


@pytest.mark.accelerator
def test_replay_preserves_temporary_inputs_across_caller_streams(
    graph_device: torch.device,
) -> None:
    device_module = torch.get_device_module(graph_device)
    steps = scenario_inputs(graph_device) * 20
    reference = run_scenario(make_pool(graph_device), steps)
    pool = make_pool(graph_device, graph_keys=ALL_KEYS)
    caller_streams = [device_module.Stream(device=graph_device) for _ in range(2)]
    setup_stream = device_module.current_stream(graph_device)

    def replay_steps() -> list[dict[int, torch.Tensor]]:
        with device_module.device(graph_device):
            current_stream = device_module.current_stream(graph_device)
            current_stream.wait_stream(setup_stream)
            for _ in range(4):
                pool.acquire()
            previous_stream = current_stream
            outputs: list[dict[int, torch.Tensor]] = []
            for step_index, step in enumerate(steps):
                caller_stream = caller_streams[step_index % len(caller_streams)]
                caller_stream.wait_stream(previous_stream)
                with device_module.stream(caller_stream):
                    temporary_inputs = {
                        slot: latents.clone() for slot, latents in step.items()
                    }
                    outputs.append(pool.step(temporary_inputs))
                    temporary_inputs.clear()
                    for latents in step.values():
                        torch.empty_like(latents).fill_(1234.0)
                previous_stream = caller_stream
            current_stream.wait_stream(previous_stream)
            outputs.append({slot: pool.flush(slot) for slot in range(4)})
            return outputs

    with ThreadPoolExecutor(max_workers=1) as executor:
        actual = executor.submit(replay_steps).result()
    assert_same_outputs(reference, actual)


@pytest.fixture(params=["cuda", "npu"])
def graph_device(request: pytest.FixtureRequest) -> torch.device:
    device_type = request.param
    if current_platform.device_type != device_type:
        pytest.skip(
            f"requires {device_type}; this host uses {current_platform.device_type}"
        )
    else:
        pass
    device_module = torch.get_device_module(device_type)
    if not device_module.is_available():
        pytest.skip(f"{device_type} is unavailable")
    else:
        pass
    return torch.device(device_type, device_module.current_device())


ALL_KEYS = [
    (1, 4),
    (1, 8),
    (1, 12),
    (1, 16),
    (2, 4),
    (2, 16),
    (3, 4),
    (3, 16),
    (4, 4),
    (4, 16),
]


@pytest.mark.accelerator
@torch.inference_mode()
def test_real_convolutions_capture_and_replay_long_causal_waveforms(
    graph_device: torch.device,
) -> None:
    torch.manual_seed(42)
    original = torch.nn.Sequential(
        Conv1d(4, 4, 3, dilation=2, groups=4, causal=True),
        ConvTranspose1d(4, 4, 4, stride=2, groups=4, causal=True),
        UpSample1d(channels=4, causal=True, fixed_filter=True),
        LowPassFilter1d(channels=4, causal=True, fixed_filter=False, stride=2),
    ).to(graph_device)
    candidate = deepcopy(original)
    if graph_device.type == "npu":
        adapt_audio_vae_convolutions(candidate)
    else:
        pass
    waveform = torch.randn(2, 4, 2300, device=graph_device)
    device_module = torch.get_device_module(graph_device)
    capture_stream = device_module.Stream(device=graph_device)
    capture_stream.wait_stream(device_module.current_stream(graph_device))
    with device_module.stream(capture_stream):
        candidate(waveform)
        candidate(waveform)
    capture_stream.synchronize()
    with capture_device_graph(
        device=graph_device,
        graph_pool=device_module.graph_pool_handle(),
        capture_stream=capture_stream,
    ) as graph:
        actual = candidate(waveform)
    device_module.current_stream(graph_device).wait_stream(capture_stream)
    for _ in range(3):
        waveform.normal_()
        expected = original(waveform)
        graph.replay()
        device_module.synchronize()
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)
    graph.reset()


@pytest.mark.accelerator
@pytest.mark.parametrize(
    "captured_keys",
    [ALL_KEYS, [(2, PATCH), (4, CHUNK)]],
    ids=["all-shapes", "eager-fallback"],
)
def test_graph_step_matches_eager_across_aging_rows(
    graph_device: torch.device, captured_keys: list[tuple[int, int]]
) -> None:
    steps = scenario_inputs(graph_device)
    reference = run_scenario(make_pool(graph_device), steps)
    pool = make_pool(graph_device, graph_keys=captured_keys)
    assert pool.graph_runner.captured_keys == captured_keys
    assert_same_outputs(reference, run_scenario(pool, steps))
    hits = sum((len(slots), frames) in captured_keys for slots, frames in SCENARIO)
    assert pool.graph_runner.replays == hits
    assert pool.graph_runner.misses == len(SCENARIO) - hits


def make_streaming_vocoder(
    device: torch.device,
    *,
    inference: NonlinearInference | TinyDecoderInference | None = None,
    **kwargs: bool | int,
) -> DotsTTSStreamingVocoder:
    codec = SimpleNamespace(
        inference=NonlinearInference(device) if inference is None else inference,
        lock=threading.RLock(),
        sample_rate=48000,
        patch_size=PATCH,
        latent_dim=LATENT_DIM,
        device=device,
        hop_size=HOP,
    )
    return DotsTTSStreamingVocoder(
        codec,
        **{
            "optimize": True,
            "enable_streaming_fast_start": True,
            **kwargs,
        },
    )


def test_step_graphs_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    vocoder = make_streaming_vocoder(
        torch.device("cpu"), enable_streaming_audio_vae_cuda_graph=False
    )
    monkeypatch.setattr(
        DotsVocoderSlotPool, "device", property(lambda self: torch.device("cuda"))
    )
    assert vocoder.ensure_slot_pool().graph_runner is None


@pytest.mark.parametrize("device_type", ["xpu"])
def test_default_step_graphs_preserve_non_cuda_eager_path(
    device_type: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    vocoder = make_streaming_vocoder(torch.device("cpu"))
    monkeypatch.setattr(
        DotsVocoderSlotPool, "device", property(lambda self: torch.device(device_type))
    )
    assert vocoder.ensure_slot_pool().graph_runner is None


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, ALL_KEYS),
        ({"stream_slots": 1}, [(1, 4), (1, 8), (1, 12), (1, 16)]),
        ({"max_batch_size": 2}, [(1, 4), (1, 8), (1, 12), (1, 16), (2, 4), (2, 16)]),
        ({"optimize": False}, [(1, 4), (2, 4), (3, 4), (4, 4)]),
    ],
)
def test_capture_keys_match_serving_shapes(
    kwargs: dict[str, bool | int], expected: list[tuple[int, int]]
) -> None:
    vocoder = make_streaming_vocoder(torch.device("cpu"), **kwargs)
    assert vocoder.cuda_graph_capture_keys() == expected


@pytest.mark.accelerator
def test_default_vocoder_captures_serving_shapes(graph_device: torch.device) -> None:
    vocoder = make_streaming_vocoder(graph_device)
    pool = vocoder.ensure_slot_pool()
    assert pool.graph_runner.captured_keys == ALL_KEYS
    assert vocoder.ensure_slot_pool() is pool


@pytest.mark.accelerator
@pytest.mark.parametrize("failure_call", [1, 3], ids=["warmup", "capture"])
def test_capture_failure_aborts_pool_setup(
    graph_device: torch.device, monkeypatch: pytest.MonkeyPatch, failure_call: int
) -> None:
    vocoder = make_streaming_vocoder(graph_device)
    decode_window = (
        vocoder.codec.inference._decode_stream_window
    )  # noqa: leading-underscore  # AudioVAE interface
    calls = 0

    def fail_forward(window: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise RuntimeError("injected AudioVAE failure")
        else:
            return decode_window(window)

    monkeypatch.setattr(vocoder.codec.inference, "_decode_stream_window", fail_forward)
    with pytest.raises(RuntimeError, match="injected AudioVAE failure"):
        vocoder.ensure_slot_pool()
    assert vocoder.slot_pool is None


class AttrDict(dict):
    """Attribute access plus dict.get, matching the upstream vocoder config type."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


TINY_DECODER_CONFIG = AttrDict(
    latent_dim=LATENT_DIM,
    upsample_initial_channel=16,
    upsample_rates=[2, 2],
    upsample_kernel_sizes=[4, 4],
    resblock="1",
    resblock_kernel_sizes=[3, 7],
    resblock_dilation_sizes=[(1, 2, 3), (1, 2, 3)],
    activation="snakebeta",
    snake_logscale=False,
    causal=True,
    use_tanh_at_final=False,
)


class TinyDecoderInference(NonlinearInference):
    """Use the real tiny decoder with deterministic FP32 recurrent state."""

    def __init__(
        self, device: torch.device, *, dtype: torch.dtype = torch.float32
    ) -> None:
        decoder = Decoder(TINY_DECODER_CONFIG).to(device=device, dtype=dtype).eval()
        if device.type == "npu":
            adapt_audio_vae_convolutions(decoder)
        else:
            pass
        hop_size = 1
        for rate in TINY_DECODER_CONFIG["upsample_rates"]:
            hop_size *= int(rate)
        super().__init__(device, seed=7)
        self.vocoder = SimpleNamespace(decoder=decoder, hop_size=hop_size)
        self.dtype = dtype
        self.lookahead = (int(decoder.conv_pre.kernel_size[0]) - 1) // 2
        # note (codex): run the upstream left-context formula on a bare real
        # instance so the window geometry never drifts from production.
        probe = object.__new__(VocoderInference)
        probe.vocoder = self.vocoder
        self.left_context = int(
            probe._decoder_stream_left_context()
        )  # noqa: leading-underscore  # AudioVAE interface

    def _decode_stream_window(
        self, window: torch.Tensor
    ) -> torch.Tensor:  # noqa: leading-underscore  # AudioVAE interface
        return self.vocoder.decoder(window.to(dtype=self.dtype)).float()

    def _activation_left_context(
        self, activation: torch.nn.Module
    ) -> int:  # noqa: leading-underscore  # AudioVAE interface
        return int(
            VocoderInference._activation_left_context(activation)
        )  # noqa: leading-underscore  # AudioVAE interface


def make_decoder_pool(
    inference: TinyDecoderInference, *, use_cache: bool = True
) -> DotsVocoderSlotPool:
    return DotsVocoderSlotPool(
        inference,
        num_slots=NUM_SLOTS,
        chunk_size=CHUNK,
        latent_dim=LATENT_DIM,
        conv_cache_decoder=(
            ConvCacheDecoder(SimpleNamespace(inference=inference))
            if use_cache
            else None
        ),
    )


def conv_cache_scenario_steps(device: torch.device) -> list[dict[int, torch.Tensor]]:
    # note (codex): Mixed row ages and irregular flush tails exercise cache seeding.
    plan: list[tuple[tuple[int, ...], int]] = [((0,), 4), ((1,), 4), ((0, 1), 8)]
    for step_index in range(29):
        slots = (0, 1) if step_index % 4 else (0, 1, 2, 3)
        plan.append((slots, 16))
    plan.extend([((0, 1, 2), 12), ((1, 2, 3), 4), ((0, 1, 2, 3), 16)])
    generator = torch.Generator().manual_seed(2024)
    return [
        {
            slot: torch.randn(1, frames, LATENT_DIM, generator=generator).to(device)
            for slot in slots
        }
        for slots, frames in plan
    ]


def run_pool_scenario(
    pool: DotsVocoderSlotPool, steps: list[dict[int, torch.Tensor]]
) -> list[dict[int, torch.Tensor]]:
    outputs = run_scenario(pool, steps)
    device = steps[0][0].device
    for slot in (0, 1):
        pool.release(slot)
    reused_slots = [pool.acquire(), pool.acquire()]
    generator = torch.Generator().manual_seed(99)
    for _ in range(3):
        outputs.append(
            pool.step(
                {
                    slot: torch.randn(1, 16, LATENT_DIM, generator=generator).to(device)
                    for slot in reused_slots
                }
            )
        )
    outputs.append({slot: pool.flush(slot) for slot in reused_slots})
    return outputs


def run_window_cache_pair(
    device: torch.device, *, dtype: torch.dtype
) -> tuple[list[dict[int, torch.Tensor]], list[dict[int, torch.Tensor]]]:
    inference = TinyDecoderInference(device, dtype=dtype)
    steps = conv_cache_scenario_steps(device)
    pair: list[list[dict[int, torch.Tensor]]] = []
    for use_cache in (False, True):
        pool = make_decoder_pool(inference, use_cache=use_cache)
        pair.append(run_pool_scenario(pool, steps))
    return pair[0], pair[1]


def test_conv_cache_matches_window_path_on_cpu() -> None:
    reference, actual = run_window_cache_pair(torch.device("cpu"), dtype=torch.float32)
    assert_same_outputs(reference, actual, rtol=1e-4, atol=1e-5)


@pytest.mark.accelerator
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_conv_cache_matches_window_path_on_accelerator(
    graph_device: torch.device,
    dtype: torch.dtype,
) -> None:
    reference, actual = run_window_cache_pair(graph_device, dtype=dtype)
    if dtype == torch.float16:
        assert_same_outputs(reference, actual, rtol=2e-2, atol=2e-2)
    else:
        assert_same_outputs(reference, actual, rtol=1e-4, atol=1e-5)


@pytest.mark.accelerator
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_conv_cache_graph_step_is_bit_identical_to_eager(
    graph_device: torch.device,
    dtype: torch.dtype,
) -> None:
    inference = TinyDecoderInference(graph_device, dtype=dtype)
    steps = scenario_inputs(graph_device)
    reference = run_scenario(make_decoder_pool(inference), steps)
    pool = make_decoder_pool(inference)
    runner = DotsVocoderGraphRunner(
        forward=pool.forward_conv_cache,
        new_inputs=pool.new_conv_cache_step_inputs,
        device=pool.device,
    )
    runner.capture(ALL_KEYS)
    pool.conv_cache_graph_runner = runner
    actual = run_scenario(pool, steps)
    assert_same_outputs(reference, actual)
    # note (codex): rows below warmup_frames take the eager window path, so only
    # steps that start with at least one hot row replay a conv-cache graph.
    assert pool.conv_cache_decoder is not None
    warmup = pool.conv_cache_decoder.warmup_frames
    totals: dict[int, int] = {}
    hot_steps = 0
    for slots, frames in SCENARIO:
        if any(totals.get(slot, 0) >= warmup for slot in slots):
            hot_steps += 1
        else:
            pass
        for slot in slots:
            totals[slot] = totals.get(slot, 0) + frames
    assert 0 < hot_steps < len(SCENARIO)
    assert runner.replays == hot_steps
    assert runner.misses == 0


@pytest.mark.accelerator
def test_conv_cache_wiring_keeps_window_graphs_for_warmup(
    graph_device: torch.device,
) -> None:
    if graph_device.type != "npu":
        pytest.skip("conv cache is enabled on NPU only")
    else:
        pass
    vocoder = make_streaming_vocoder(
        graph_device,
        inference=TinyDecoderInference(graph_device),
        enable_streaming_conv_cache=True,
    )
    pool = vocoder.ensure_slot_pool()
    assert pool.conv_cache_decoder is not None
    assert pool.conv_cache_graph_runner is not None
    assert pool.conv_cache_graph_runner.captured_keys == ALL_KEYS
    # note (codex): pre-warmup rows still take the window path, so its graphs
    # stay wired exactly like the no-conv-cache pool.
    assert pool.graph_runner is not None
    assert pool.graph_runner.captured_keys == ALL_KEYS
    reference_pool = make_streaming_vocoder(
        graph_device, inference=vocoder.codec.inference
    ).ensure_slot_pool()
    assert (
        pool.prefix_graph_runners.keys() == reference_pool.prefix_graph_runners.keys()
    )
    for decoder_frames, prefix_runner in pool.prefix_graph_runners.items():
        assert (
            prefix_runner.captured_keys
            == reference_pool.prefix_graph_runners[decoder_frames].captured_keys
        )
    assert vocoder.cuda_graph_count == len(ALL_KEYS) * 2 + sum(
        len(prefix_runner.captured_keys)
        for prefix_runner in reference_pool.prefix_graph_runners.values()
    )


def test_conv_cache_stays_off_acceleratorless_devices() -> None:
    vocoder = make_streaming_vocoder(
        torch.device("cpu"), enable_streaming_conv_cache=True
    )
    pool = vocoder.ensure_slot_pool()
    assert pool.conv_cache_decoder is None
    assert pool.conv_cache_graph_runner is None
