"""TensorRT engine build (FP32 / FP16 / INT8-with-calibration) and a minimal PyTorch-interop runtime.

Uses the pip `tensorrt-cu12` wheels (no NVIDIA developer login needed).  The builder workspace is
sized from the shared-GPU reservation (GPU_GUARD_GB) so engine building stays inside it.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

from ..utils import gpu_guard_gb


def _trt():
    import tensorrt as trt

    return trt


class EntropyCalibrator:
    """INT8 entropy calibrator fed from an iterable of (B,3,S,S) float32 CPU tensors."""

    def __new__(cls, batches: Iterable[torch.Tensor], cache_file: str | Path):
        trt = _trt()

        class _Calib(trt.IInt8EntropyCalibrator2):
            def __init__(self):
                super().__init__()
                self.batches = iter(list(batches))
                self.first = None
                self.cache_file = Path(cache_file)
                self.dev = None

            def get_batch_size(self):
                return self.first.shape[0] if self.first is not None else 1

            def get_batch(self, names):
                try:
                    b = next(self.batches)
                except StopIteration:
                    return None
                if self.first is None:
                    self.first = b
                self.dev = b.cuda().contiguous()
                return [int(self.dev.data_ptr())]

            def read_calibration_cache(self):
                return self.cache_file.read_bytes() if self.cache_file.exists() else None

            def write_calibration_cache(self, cache):
                self.cache_file.write_bytes(bytes(cache))

        return _Calib()


def build_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    img_size: int,
    precision: str = "fp16",
    max_batch: int = 8,
    opt_batch: int = 8,
    workspace_gb: float | None = None,
    calib_batches: Iterable[torch.Tensor] | None = None,
) -> Path:
    trt = _trt()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        raise RuntimeError(
            "ONNX parse failed: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        )
    config = builder.create_builder_config()
    if workspace_gb is None:  # stay inside the shared-GPU reservation: half of it (minus context) for tactics
        g = gpu_guard_gb()
        workspace_gb = max(0.25, (g - 0.4) / 2) if g > 0 else 1.0
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * 2**30))
    if precision in ("fp16", "int8"):
        config.set_flag(trt.BuilderFlag.FP16)
    if precision == "int8":
        if calib_batches is None:
            raise ValueError("INT8 needs calibration batches")
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = EntropyCalibrator(calib_batches, Path(engine_path).with_suffix(".calib"))
    inp = network.get_input(0)
    profile = builder.create_optimization_profile()
    profile.set_shape(
        inp.name,
        (1, 3, img_size, img_size),
        (opt_batch, 3, img_size, img_size),
        (max_batch, 3, img_size, img_size),
    )
    config.add_optimization_profile(profile)
    if precision == "int8":
        config.set_calibration_profile(profile)
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("TensorRT engine build failed")
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(blob))
    return engine_path


class TRTModule:
    """Run a single-input/single-output engine on torch CUDA tensors (zero-copy, current stream)."""

    def __init__(self, engine_path: str | Path):
        trt = _trt()
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inp = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.out = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        self.device_memory_mib = (
            self.engine.device_memory_size_v2 / 2**20
            if hasattr(self.engine, "device_memory_size_v2")
            else float("nan")
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().float()
        self.ctx.set_input_shape(self.inp, tuple(x.shape))
        out = torch.empty(tuple(self.ctx.get_tensor_shape(self.out)), device=x.device, dtype=torch.float32)
        self.ctx.set_tensor_address(self.inp, int(x.data_ptr()))
        self.ctx.set_tensor_address(self.out, int(out.data_ptr()))
        if not self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        return out


def onnx_size_mb(path: str | Path) -> float:
    return Path(path).stat().st_size / 2**20


__all__ = ["TRTModule", "build_engine", "onnx_size_mb", "np"]
