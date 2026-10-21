# SPDX-License-Identifier: Apache-2.0
"""Device graphs over the dots.tts slot-pool AudioVAE step."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, nullcontext
from typing import NamedTuple, Protocol

import torch

logger = logging.getLogger(__name__)
HIT_RATE_LOG_EVERY = 1024

StepInputs = tuple[torch.Tensor, ...]
StepOutputs = tuple[torch.Tensor, ...]


class ReplayableGraph(Protocol):
    def replay(self) -> None: ...

    def reset(self) -> None: ...


@contextmanager
def capture_device_graph(
    *, device: torch.device, graph_pool: tuple[int, int], capture_stream: torch.Stream
) -> Iterator[ReplayableGraph]:
    """Capture dots.tts without importing unrelated accelerator graph types."""
    if device.type == "cuda":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            cuda_graph=graph,
            pool=graph_pool,
            stream=capture_stream,
            capture_error_mode="thread_local",
        ):
            yield graph
    elif device.type == "npu":
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(
            npu_graph=graph,
            pool=graph_pool,
            stream=capture_stream,
            capture_error_mode="thread_local",
        ):
            yield graph
    else:
        raise RuntimeError(f"dots.tts has no graph backend for {device}")


class WindowStepForward(Protocol):
    def __call__(
        self,
        packed: torch.Tensor,
        hidden_h: torch.Tensor,
        hidden_c: torch.Tensor,
        window: torch.Tensor,
        valid: torch.Tensor,
    ) -> StepOutputs: ...


class ConvCacheStepForward(Protocol):
    def __call__(
        self,
        packed: torch.Tensor,
        hidden_h: torch.Tensor,
        hidden_c: torch.Tensor,
        flat_cache: torch.Tensor,
    ) -> StepOutputs: ...


StepForward = WindowStepForward | ConvCacheStepForward


class StepInputFactory(Protocol):
    def __call__(self, batch_size: int, frames: int) -> StepInputs: ...


class CapturedStepGraph(NamedTuple):
    graph: ReplayableGraph
    inputs: StepInputs
    outputs: StepOutputs
    capture_stream: torch.Stream


class DotsVocoderGraphRunner:
    """Replays one captured graph per (batch, frames) key; misses return None.

    Inputs are copied into static buffers, outputs are cloned out, so callers
    keep the eager step's value semantics and a replay never aliases the
    previous step's result.
    """

    def __init__(
        self,
        *,
        forward: StepForward,
        new_inputs: StepInputFactory,
        device: torch.device,
        warmup_iters: int = 2,
    ) -> None:
        self.forward = forward
        self.new_inputs = new_inputs
        self.device = device
        self.warmup_iters = int(warmup_iters)
        self.graphs: dict[tuple[int, int], CapturedStepGraph] = {}
        self.replays = 0
        self.misses = 0

    @property
    def captured_keys(self) -> list[tuple[int, int]]:
        return sorted(self.graphs)

    @torch.no_grad()
    def capture(self, keys: Iterable[tuple[int, int]]) -> None:
        if self.device.type not in {"cuda", "npu"}:
            raise RuntimeError(
                f"dots.tts vocoder has no graph backend for {self.device}"
            )
        else:
            pass
        # note (lennox): capture largest first so smaller graphs reuse freed blocks.
        ordered = sorted(
            {(int(b), int(t)) for b, t in keys} - set(self.graphs), reverse=True
        )
        device_module = torch.get_device_module(self.device)
        with device_module.device(self.device):
            current_stream = device_module.current_stream(self.device)
            capture_stream = device_module.Stream(device=self.device)
            capture_stream.wait_stream(current_stream)
            graph_pool = device_module.graph_pool_handle()
            for key in ordered:
                inputs = self.new_inputs(*key)
                with device_module.stream(capture_stream):
                    for _ in range(self.warmup_iters):
                        self.forward(*inputs)
                capture_stream.synchronize()
                with capture_device_graph(
                    device=self.device,
                    graph_pool=graph_pool,
                    capture_stream=capture_stream,
                ) as graph:
                    outputs = self.forward(*inputs)
                self.graphs[key] = CapturedStepGraph(
                    graph, inputs, outputs, capture_stream
                )
            current_stream.wait_stream(capture_stream)
            device_module.synchronize(self.device)
        logger.info(
            f"dots.tts streaming vocoder {self.device.type} graphs captured: {self.captured_keys}"
        )

    @torch.no_grad()
    def run(self, step_inputs: StepInputs) -> StepOutputs | None:
        packed = step_inputs[0]
        captured = self.graphs.get((int(packed.shape[0]), int(packed.shape[2])))
        if captured is None:
            self.misses += 1
            outputs = None
        else:
            device_module = torch.get_device_module(self.device)
            current_stream = device_module.current_stream(self.device)
            if self.device.type == "npu":
                # note: Keep NPU graph outputs ordered with their capture stream.
                captured.capture_stream.wait_stream(current_stream)
                for live_input in step_inputs:
                    live_input.record_stream(captured.capture_stream)
                execution_context = device_module.stream(captured.capture_stream)
            else:
                execution_context = nullcontext()
            with execution_context:
                for static_input, live_input in zip(
                    captured.inputs, step_inputs, strict=True
                ):
                    static_input.copy_(live_input)
                captured.graph.replay()
                self.replays += 1
                outputs = tuple(output.clone() for output in captured.outputs)
            if self.device.type == "npu":
                current_stream.wait_stream(captured.capture_stream)
                for tensor in outputs:
                    tensor.record_stream(current_stream)
            else:
                pass
        calls = self.replays + self.misses
        milestone = calls == 1 or calls % HIT_RATE_LOG_EVERY == 0
        logger.log(
            logging.INFO if milestone else logging.DEBUG,
            f"dots.tts streaming vocoder {self.device.type} graph replays={self.replays} "
            f"misses={self.misses} ({100.0 * self.replays / calls:.1f}% hit)",
        )
        return outputs


__all__ = ["DotsVocoderGraphRunner", "ReplayableGraph", "capture_device_graph"]
