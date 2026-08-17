# SPDX-License-Identifier: Apache-2.0
"""NPU platform adaptation tests for FishAudio S2-Pro."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic import (
    audio_decoder as fish_audio_decoder,
)


def test_npu_kvcache_attention_matches_dense_sdpa() -> None:
    bs, seq_q, n_heads, head_dim, cache_len = 2, 1, 4, 8, 11
    q = torch.randn(bs, seq_q, n_heads, head_dim)
    k = torch.randn(bs, seq_q, n_heads, head_dim)
    v = torch.randn(bs, seq_q, n_heads, head_dim)
    k_cache = torch.zeros(bs, cache_len, n_heads, head_dim)
    v_cache = torch.zeros(bs, cache_len, n_heads, head_dim)
    cache_position = 3

    out = fish_audio_decoder._npu_kvcache_attention(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k=k,
        v=v,
        cache_position=cache_position,
    )

    assert out.shape == q.shape
    assert torch.equal(k_cache[:, cache_position : cache_position + 1], k)
    assert torch.equal(v_cache[:, cache_position : cache_position + 1], v)

    # The implementation attends over the whole populated cache window, so the
    # reference must use the same window (zero-padded past positions included).
    cache_end = cache_position + 1
    q_t = q.transpose(1, 2)
    k_t = k_cache[:, :cache_end].transpose(1, 2)
    v_t = v_cache[:, :cache_end].transpose(1, 2)
    causal_mask = torch.tril(
        torch.ones(q_t.shape[2], k_t.shape[2], dtype=torch.bool),
        diagonal=k_t.shape[2] - q_t.shape[2],
    )
    ref = torch.nn.functional.scaled_dot_product_attention(
        q_t, k_t, v_t, attn_mask=causal_mask, is_causal=False
    ).transpose(1, 2)
    assert torch.allclose(out, ref, atol=1e-5)


def test_npu_kvcache_attention_attends_to_full_populated_prefix() -> None:
    bs, seq_q, n_heads, head_dim, cache_len = 1, 1, 4, 8, 11
    q = torch.randn(bs, seq_q, n_heads, head_dim)
    k = torch.randn(bs, seq_q, n_heads, head_dim)
    v = torch.randn(bs, seq_q, n_heads, head_dim)
    # Pre-populate the prefix: past positions must stay visible to the new
    # trailing query (torch's is_causal would only expose the first position).
    k_cache = torch.randn(bs, cache_len, n_heads, head_dim)
    v_cache = torch.randn(bs, cache_len, n_heads, head_dim)
    cache_position = 7

    out = fish_audio_decoder._npu_kvcache_attention(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k=k,
        v=v,
        cache_position=cache_position,
    )

    cache_end = cache_position + 1
    q_t = q.transpose(1, 2)
    k_t = k_cache[:, :cache_end].transpose(1, 2)
    v_t = v_cache[:, :cache_end].transpose(1, 2)
    causal_mask = torch.tril(
        torch.ones(q_t.shape[2], k_t.shape[2], dtype=torch.bool),
        diagonal=k_t.shape[2] - q_t.shape[2],
    )
    ref = torch.nn.functional.scaled_dot_product_attention(
        q_t, k_t, v_t, attn_mask=causal_mask, is_causal=False
    ).transpose(1, 2)

    assert torch.allclose(out, ref, atol=1e-5)


def test_npu_kvcache_attention_requires_kv_and_cache_position() -> None:
    q = torch.randn(1, 1, 4, 8)
    k_cache = torch.zeros(1, 11, 4, 8)
    v_cache = torch.zeros(1, 11, 4, 8)
    k = torch.randn(1, 1, 4, 8)

    with pytest.raises(ValueError, match="requires k and v"):
        fish_audio_decoder._npu_kvcache_attention(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            k=None,
            v=None,
            cache_position=0,
        )
    with pytest.raises(ValueError, match="requires cache_position"):
        fish_audio_decoder._npu_kvcache_attention(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            k=k,
            v=k,
            cache_position=-1,
        )


def test_fish_engine_builder_npu_defaults(monkeypatch) -> None:
    from sglang_omni.models.fishaudio_s2_pro import engine_builder as fish_engine

    fake_npu = SimpleNamespace(is_npu=lambda: True, device_type="npu")
    monkeypatch.setattr(fish_engine, "current_platform", fake_npu)

    builder = fish_engine.FishS2ProEngineBuilder(max_new_tokens=256, ras_window=16)
    builder.gpu_id = 0

    defaults = builder.generation_defaults(dtype="bfloat16")
    assert defaults["disable_cuda_graph"] is True
    assert defaults["enable_torch_compile"] is False
    assert defaults["dtype"] == "bfloat16"

    assert fish_engine._resolve_fast_ar_attention_backend(gpu_id=0) == "ascend"


def test_fish_engine_builder_cuda_defaults_unchanged(monkeypatch) -> None:
    from sglang_omni.models.fishaudio_s2_pro import engine_builder as fish_engine

    fake_cuda = SimpleNamespace(is_npu=lambda: False)
    monkeypatch.setattr(fish_engine, "current_platform", fake_cuda)
    monkeypatch.setattr(
        fish_engine,
        "get_visible_gpu_sm_version",
        lambda gpu_id: 90,
    )

    builder = fish_engine.FishS2ProEngineBuilder(max_new_tokens=256, ras_window=16)
    builder.gpu_id = 0

    defaults = builder.generation_defaults(dtype="bfloat16")
    assert defaults["disable_cuda_graph"] is False
    assert defaults["enable_torch_compile"] is True


def test_resolve_stage_device_uses_current_platform(monkeypatch) -> None:
    from sglang_omni.models.fishaudio_s2_pro import stages as fish_stages

    monkeypatch.setattr(fish_stages, "current_platform", SimpleNamespace(device_type="npu"))
    assert fish_stages._resolve_stage_device(0) == "npu:0"
    assert fish_stages._resolve_stage_device(None) == "cpu"

    monkeypatch.setattr(
        fish_stages, "current_platform", SimpleNamespace(device_type="cuda")
    )
    assert fish_stages._resolve_stage_device(0) == "cuda:0"


def test_s2pro_tts_engine_stage_device_is_platformized() -> None:
    from sglang_omni.models.fishaudio_s2_pro.config import S2ProPipelineConfig
    from sglang_omni.platforms import current_platform

    config = S2ProPipelineConfig(model_path="x")
    tts_stage = next(
        stage for stage in config.stages if stage.name == "tts_engine"
    )

    assert tts_stage.factory_args["device"] == f"{current_platform.device_type}:0"
