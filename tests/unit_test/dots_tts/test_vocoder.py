# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import sglang_omni.models.dots_tts.codec as codec_module
from sglang_omni.models.dots_tts.compat import import_dots_tts
from sglang_omni.models.dots_tts.npu_vocoder import (
    adapt_audio_vae_convolutions,
    decoder_forward,
)
from sglang_omni.models.dots_tts.payload_types import DotsTTSState
from sglang_omni.models.dots_tts.vocoder import (
    DotsTTSBatchVocoder,
    DotsTTSStreamingVocoder,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage

import_dots_tts()

from dots_tts.modules.backbone.layers import Conv1d, ConvTranspose1d
from dots_tts.modules.vocoder.alias_free_filter import LowPassFilter1d
from dots_tts.modules.vocoder.alias_free_resample import UpSample1d


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("fixed_filter", [True, False])
def test_audio_vae_convolution_adapters_preserve_waveforms(
    causal: bool, fixed_filter: bool
) -> None:
    torch.manual_seed(17)
    convolutions = torch.nn.ModuleList(
        [
            Conv1d(4, 4, 3, dilation=2, groups=4, causal=causal),
            ConvTranspose1d(4, 4, 4, stride=2, groups=4, causal=causal),
            UpSample1d(channels=4, causal=causal, fixed_filter=fixed_filter),
            LowPassFilter1d(
                channels=4, causal=causal, fixed_filter=fixed_filter, stride=2
            ),
            torch.nn.Conv1d(4, 8, 3, padding=2, padding_mode="reflect"),
        ]
    ).double()
    waveform = torch.randn(2, 4, 37, dtype=torch.float64)
    expected = [module(waveform) for module in convolutions]
    parameters = dict(convolutions.named_parameters())
    state_keys = tuple(convolutions.state_dict())
    module_types = [type(module) for module in convolutions]
    adapt_audio_vae_convolutions(convolutions)
    for module, reference in zip(convolutions, expected, strict=True):
        torch.testing.assert_close(module(waveform), reference, rtol=1e-12, atol=1e-12)
    assert tuple(convolutions.state_dict()) == state_keys
    assert [type(module) for module in convolutions] == module_types
    assert all(
        parameter is parameters[name]
        for name, parameter in convolutions.named_parameters()
    )


@pytest.mark.parametrize(
    "kernel_size, ratio", [(1, 2), (2, 2), (3, 2), (11, 2), (17, 3)]
)
@pytest.mark.parametrize("samples", [1, 37])
@pytest.mark.parametrize("fixed_filter", [False, True])
def test_causal_upsampling_preserves_transpose_convolution_crop(
    kernel_size: int, ratio: int, samples: int, fixed_filter: bool
) -> None:
    module = UpSample1d(
        channels=4,
        causal=True,
        fixed_filter=fixed_filter,
        ratio=ratio,
        kernel_size=kernel_size,
    ).double()
    generator = torch.Generator().manual_seed(23)
    with torch.no_grad():
        module.filter.copy_(torch.randn(module.filter.shape, generator=generator))
    waveform = torch.randn(2, 4, samples, dtype=torch.float64, generator=generator)
    expected = ratio * F.conv_transpose1d(
        waveform, module.filter.expand(4, -1, -1), stride=ratio, groups=4
    )
    expected = expected[..., : -(kernel_size - ratio)]
    adapt_audio_vae_convolutions(module)
    torch.testing.assert_close(module(waveform), expected, rtol=1e-12, atol=1e-12)


class FakeInference:
    def __init__(self, hop_size: int) -> None:
        self.hop_size = hop_size
        self.inputs: list[torch.Tensor] = []
        self.input_data_ptrs: list[int] = []

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        self.input_data_ptrs.append(latents.data_ptr())
        self.inputs.append(latents.clone())
        rows = []
        for row in latents:
            samples = row[:, 0].repeat_interleave(self.hop_size)
            rows.append(samples.unsqueeze(0))
        return torch.stack(rows)


def make_codec(*, latent_dim: int = 3, hop_size: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        device=torch.device("cpu"),
        latent_dim=latent_dim,
        hop_size=hop_size,
        sample_rate=48000,
        patch_size=4,
        lock=threading.RLock(),
        inference=FakeInference(hop_size),
    )


def make_latents(frames: int, value: float, *, latent_dim: int = 3) -> torch.Tensor:
    return torch.full((1, frames, latent_dim), value)


def decode(
    vocoder: DotsTTSBatchVocoder, latents: list[torch.Tensor]
) -> list[tuple[torch.Tensor, int]]:
    items = [(DotsTTSState(), item) for item in latents]
    return asyncio.run(vocoder.decode_batch(items))


def payload(
    request_id: str, frames: int, value: float, *, stream: bool = False
) -> StagePayload:
    state = DotsTTSState(generated_latents=make_latents(frames, value))
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="hello", params={"stream": stream}),
        data=state.to_dict(),
    )


def test_equal_length_inputs_use_one_audiovae_forward() -> None:
    codec = make_codec()
    vocoder = DotsTTSBatchVocoder(codec)
    outputs = decode(
        vocoder,
        [make_latents(16, 1), make_latents(16, 2), make_latents(16, 3)],
    )

    assert vocoder.logged_batch
    assert len(codec.inference.inputs) == 1
    assert codec.inference.inputs[0].shape == (3, 16, 3)
    assert [waveform.shape for waveform, _ in outputs] == [(1, 1, 32)] * 3


def test_mixed_length_bucket_pads_and_crops_each_output() -> None:
    codec = make_codec()
    outputs = decode(
        DotsTTSBatchVocoder(codec),
        [make_latents(17, 1), make_latents(31, 2)],
    )

    [padded] = codec.inference.inputs
    assert padded.shape == (2, 31, 3)
    assert torch.count_nonzero(padded[0, 17:]) == 0
    assert [waveform.shape[-1] for waveform, _ in outputs] == [34, 62]
    assert torch.all(outputs[0][0] == 1)
    assert torch.all(outputs[1][0] == 2)


def test_multiple_buckets_restore_original_request_order() -> None:
    codec = make_codec()
    outputs = decode(
        DotsTTSBatchVocoder(codec),
        [
            make_latents(33, 1),
            make_latents(16, 2),
            make_latents(40, 3),
            make_latents(8, 4),
        ],
    )

    assert [batch.shape[0] for batch in codec.inference.inputs] == [2, 2]
    assert [waveform[0, 0, 0].item() for waveform, _ in outputs] == [1, 2, 3, 4]


def test_single_input_preserves_batch_and_waveform_shapes() -> None:
    codec = make_codec()
    vocoder = DotsTTSBatchVocoder(codec)
    latents = make_latents(12, 5)
    [output] = decode(vocoder, [latents])

    assert not vocoder.logged_batch
    assert codec.inference.input_data_ptrs == [latents.data_ptr()]
    assert codec.inference.inputs[0].shape == (1, 12, 3)
    assert output[0].shape == (1, 1, 24)
    assert output[1] == 48000


@pytest.mark.parametrize(
    ("latents", "message"),
    [
        (torch.zeros(4, 3), "shape"),
        (torch.zeros(2, 4, 3), "shape"),
        (torch.zeros(1, 0, 3), "at least one frame"),
        (torch.zeros(1, 4, 2), "latent_dim"),
    ],
)
def test_invalid_latents_are_rejected(latents: torch.Tensor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        decode(DotsTTSBatchVocoder(make_codec()), [latents])


def test_streaming_vocoder_enables_payload_and_chunk_batching() -> None:
    codec = make_codec()
    scheduler = DotsTTSStreamingVocoder(
        codec,
        optimize=False,
    )

    assert scheduler.batch_fn is not None
    assert scheduler.max_batch_size == 4
    assert scheduler.stream_chunk_batch_max == 4
    assert scheduler.max_batch_wait_s == 0.002
    assert scheduler.can_batch_stream_chunks
    results = asyncio.run(
        scheduler.batch_fn([payload("a", 16, 1), payload("b", 16, 2)])
    )
    assert len(codec.inference.inputs) == 1
    assert [result.request_id for result in results] == ["a", "b"]
    assert scheduler.is_streaming_payload(payload("stream", 16, 1, stream=True))


def test_non_streaming_batch_isolates_invalid_payload() -> None:
    codec = make_codec()
    scheduler = DotsTTSStreamingVocoder(codec, optimize=False)
    invalid = payload("bad", 16, 2)
    invalid.data["generated_latents"] = torch.zeros(2, 4, 3)

    scheduler.handle_new_request_batch(
        [
            IncomingMessage("good-a", "new_request", payload("good-a", 16, 1)),
            IncomingMessage("bad", "new_request", invalid),
            IncomingMessage("good-b", "new_request", payload("good-b", 16, 3)),
        ]
    )

    outputs = [scheduler.outbox.get_nowait() for _ in range(3)]
    by_request = {output.request_id: output for output in outputs}
    assert by_request["bad"].type == "error"
    assert isinstance(by_request["bad"].data, ValueError)
    assert by_request["good-a"].type == "result"
    assert by_request["good-b"].type == "result"
    assert [item.shape for item in codec.inference.inputs] == [(2, 16, 3)]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_batch_size": 0}, "max_batch_size"),
        ({"max_batch_wait_ms": -1}, "max_batch_wait_ms"),
        ({"stream_slots": 0}, "stream_slots"),
    ],
)
def test_invalid_batch_config_is_rejected(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DotsTTSStreamingVocoder(make_codec(), optimize=False, **kwargs)


class TestVocoderFactorySignature:
    """The vocoder factory declares every kwarg it accepts.

    With a ``**kwargs`` catch-all, a mistyped ``factory.*`` key -- or a
    correctly spelled one the factory never reads -- would be swallowed
    silently; without it, the typed-kwargs check refuses it."""

    def test_an_unknown_factory_key_is_refused(self) -> None:
        stages = pytest.importorskip("sglang_omni.models.dots_tts.stages")
        from sglang_omni.config.runtime import apply_typed_stage_kwargs

        with pytest.raises(ValueError, match="stream_slotz"):
            apply_typed_stage_kwargs(
                stages.create_vocoder_executor,
                {},
                {"stream_slotz": 8},
                stage_name="vocoder",
            )

    def test_declared_kwargs_still_pass(self) -> None:
        stages = pytest.importorskip("sglang_omni.models.dots_tts.stages")
        from sglang_omni.config.runtime import apply_typed_stage_kwargs

        out = apply_typed_stage_kwargs(
            stages.create_vocoder_executor,
            {},
            {"stream_slots": 8, "enable_streaming_audio_vae_cuda_graph": True},
            stage_name="vocoder",
        )
        assert out == {"stream_slots": 8, "enable_streaming_audio_vae_cuda_graph": True}


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16])
def test_decoder_precision_adapter_casts_input_and_preserves_checkpoint(
    input_dtype: torch.dtype,
) -> None:
    decoder = torch.nn.Conv1d(4, 2, 3).half().eval()
    latent_frames = torch.randn(2, 4, 12).to(dtype=input_dtype)
    expected = decoder(latent_frames.half()).float()
    state_keys = tuple(decoder.state_dict())
    decoder.forward = MethodType(decoder_forward, decoder)
    actual = decoder(latent_frames)
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == torch.float32
    assert type(decoder) is torch.nn.Conv1d
    assert tuple(decoder.state_dict()) == state_keys


class DecoderPrecisionCodec:
    def __init__(self, checkpoint: str, *, device: str) -> None:
        self.checkpoint: str = checkpoint
        self.device: torch.device = torch.device(device)
        self.vocoder: SimpleNamespace = SimpleNamespace(
            decoder=torch.nn.Conv1d(4, 2, 3),
            post_proj=torch.nn.Conv1d(4, 4, 1),
        )


def test_codec_cache_keeps_decoder_precisions_independent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(codec_module, "resolve_checkpoint", str)
    monkeypatch.setattr(codec_module, "DotsAudioCodec", DecoderPrecisionCodec)
    monkeypatch.setattr(codec_module, "_CODEC_CACHE", {})
    reference = codec_module.load_dots_audio_codec(str(tmp_path), device="cpu")
    candidate = codec_module.load_dots_audio_codec(
        str(tmp_path), device="cpu", decoder_dtype=torch.float16
    )
    assert reference is not candidate
    assert reference.vocoder.decoder.weight.dtype == torch.float32
    assert candidate.vocoder.decoder.weight.dtype == torch.float16
    assert candidate.vocoder.post_proj.weight.dtype == torch.float32
    assert codec_module.load_dots_audio_codec(str(tmp_path), device="cpu") is reference
    assert (
        codec_module.load_dots_audio_codec(
            str(tmp_path), device="cpu", decoder_dtype=torch.float16
        )
        is candidate
    )
    output = candidate.vocoder.decoder(torch.randn(1, 4, 12))
    assert output.dtype == torch.float32
