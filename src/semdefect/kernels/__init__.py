"""GPU kernels: kNN top-k for memory banks and fused SEM preprocessing.

Three interchangeable backends - `torch` (reference), `cupy` (NVRTC JIT prototype) and `ext`
(nvcc AOT pybind11 module) - all compiled from the same `cuda/*.cuh` sources.
"""
