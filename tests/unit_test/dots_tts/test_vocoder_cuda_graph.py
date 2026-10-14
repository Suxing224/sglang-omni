# SPDX-License-Identifier: Apache-2.0
"""Graph-vs-eager contract for the slot-pool AudioVAE step.

The fake inference below is deliberately nonlinear and state-dependent per
row, so a stale static buffer, a cross-row leak, or a wrong (B, T) lookup
changes the waveform instead of cancelling out.
"""

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
        expected = module.ratio * (
            convolution_windows * filter_weights.flip(-1).unsqueeze(0)
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
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.vocoder = SimpleNamespace(hop_size=HOP)
        generator = torch.Generator().manual_seed(0)
        self.drive = torch.rand(1, 1, HIDDEN, generator=generator).to(device)

    def init_stream_state(self, *, batch_size: int, chunk_size: int):
        window = torch.zeros(batch_size, LATENT_DIM, chunk_size + 4, device=self.device)
        hidden = torch.zeros(1, batch_size, HIDDEN, device=self.device)
        return SimpleNamespace(
            lstm_hidden=(hidden, hidden.clone()),
            decoder=SimpleNamespace(window=window, chunk_size=chunk_size),
        )

    def _decoder_stream_lookahead(
        self,
    ) -> int:  # noqa: leading-underscore  # AudioVAE interface
        return 1

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
        drive = latents.mean(dim=(1, 2))
        hidden_h = torch.tanh(hidden_h * 0.9 + drive[None, :, None] * self.drive)
        hidden_c = hidden_c + latents.abs().mean(dim=(1, 2))[None, :, None]
        gain = 1.0 + hidden_h.sum(dim=(0, 2))
        return latents * gain[:, None, None], (hidden_h, hidden_c)

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


def assert_same_outputs(reference, candidate) -> None:
    assert len(reference) == len(candidate)
    for step_index, (expected, actual) in enumerate(zip(reference, candidate)):
        assert expected.keys() == actual.keys()
        for slot in expected:
            assert torch.equal(expected[slot], actual[slot]), (
                f"step {step_index} slot {slot} differs: "
                f"max|delta|={(expected[slot] - actual[slot]).abs().max().item():.3e}"
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
def test_graph_step_is_bit_identical_to_eager_across_aging_rows(
    graph_device: torch.device,
) -> None:
    device = graph_device
    steps = scenario_inputs(device)
    reference = run_scenario(make_pool(device), steps)

    pool = make_pool(device, graph_keys=ALL_KEYS)
    assert pool.graph_runner.captured_keys == ALL_KEYS
    assert_same_outputs(reference, run_scenario(pool, steps))
    assert pool.graph_runner.replays == len(SCENARIO)
    assert pool.graph_runner.misses == 0


@pytest.mark.accelerator
def test_uncaptured_shape_falls_back_to_eager_bit_identically(
    graph_device: torch.device,
) -> None:
    device = graph_device
    steps = scenario_inputs(device)
    reference = run_scenario(make_pool(device), steps)

    captured = [(2, PATCH), (4, CHUNK)]
    pool = make_pool(device, graph_keys=captured)
    assert_same_outputs(reference, run_scenario(pool, steps))
    hits = sum(1 for slots, frames in SCENARIO if (len(slots), frames) in captured)
    assert pool.graph_runner.replays == hits
    assert pool.graph_runner.misses == len(SCENARIO) - hits


def make_streaming_vocoder(
    device: torch.device, **kwargs: bool | int
) -> DotsTTSStreamingVocoder:
    codec = SimpleNamespace(
        inference=NonlinearInference(device),
        lock=threading.RLock(),
        sample_rate=48000,
        patch_size=PATCH,
        latent_dim=LATENT_DIM,
        device=device,
        hop_size=HOP,
    )
    return DotsTTSStreamingVocoder(codec, **{"optimize": True, **kwargs})


def test_step_graphs_can_be_disabled() -> None:
    vocoder = make_streaming_vocoder(
        torch.device("cpu"), enable_streaming_audio_vae_cuda_graph=False
    )
    assert vocoder.ensure_slot_pool().graph_runner is None


@pytest.mark.parametrize("device_type", ["cpu", "xpu"])
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
