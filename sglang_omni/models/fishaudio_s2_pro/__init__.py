# SPDX-License-Identifier: Apache-2.0
"""FishAudio S2-Pro model support for sglang-omni."""

from sglang_omni.models.model_capabilities import ModelCapabilities
from sglang_omni.platforms import current_platform

from . import config

# CUDA graphs and the owned torch.compile path are validated on CUDA only;
# Ascend NPU runs eager with CUDA graphs disabled.
_supports_accelerator_graph = not current_platform.is_npu()

CAPABILITIES = ModelCapabilities(
    supports_reference_audio=True,
    supports_batch_vocoder=True,
    supports_streaming_vocoder=True,
    supports_cuda_graph=_supports_accelerator_graph,
    supports_torch_compile=_supports_accelerator_graph,
    supports_breakable_prefill_cuda_graph=False,
)

__all__ = ["CAPABILITIES", "config"]
