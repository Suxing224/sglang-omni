# SPDX-License-Identifier: Apache-2.0
"""FishAudio S2-Pro model support for sglang-omni."""

from sglang_omni.models.model_capabilities import ModelCapabilities

from . import config

# FishAudio owns compilation of its Fast-AR decoder layers. NPU keeps this
# path opt-in until its interaction with fused attention and NPU graph has
# production validation.

CAPABILITIES = ModelCapabilities(
    supports_reference_audio=True,
    supports_batch_vocoder=True,
    supports_streaming_vocoder=True,
    supports_cuda_graph=True,
    supports_torch_compile=True,
    supports_breakable_prefill_cuda_graph=False,
)

__all__ = ["CAPABILITIES", "config"]
