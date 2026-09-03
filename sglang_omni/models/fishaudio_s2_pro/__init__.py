# SPDX-License-Identifier: Apache-2.0
"""FishAudio S2-Pro model support for sglang-omni."""

from sglang_omni.models.model_capabilities import ModelCapabilities
from sglang_omni.platforms import current_platform

from . import config

# NPU keeps the Fast-AR compilation path disabled until its interaction with
# fused attention and NPU graph decode has production validation.
_supports_torch_compile = not current_platform.is_npu()

CAPABILITIES = ModelCapabilities(
    supports_reference_audio=True,
    supports_batch_vocoder=True,
    supports_streaming_vocoder=True,
    supports_cuda_graph=True,
    supports_torch_compile=_supports_torch_compile,
    supports_breakable_prefill_cuda_graph=False,
)

__all__ = ["CAPABILITIES", "config"]
