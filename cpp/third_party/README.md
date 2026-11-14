# Third-party code (fetched, not committed)

## TensorRT public headers (only needed for `-DSEMDEFECT_BUILD_TRT_RUNNER=ON`)

The pip wheels `tensorrt-cu12-libs==10.8.0.43` ship the runtime DLLs (`nvinfer_10.dll`, ...) but no
headers or import libraries, and the official TensorRT SDK zip needs an NVIDIA developer login.
The runner therefore uses the Apache-2.0 public headers from the open-source repository at the
matching tag and loads `nvinfer_10.dll` at run time (only `createInferRuntime_INTERNAL` is resolved;
everything else is virtual dispatch), so no import library is needed.

```bash
mkdir -p cpp/third_party/tensorrt/include
for f in NvInfer.h NvInferImpl.h NvInferLegacyDims.h NvInferPluginBase.h NvInferRuntime.h \
         NvInferRuntimeBase.h NvInferRuntimeCommon.h NvInferRuntimePlugin.h NvInferVersion.h; do
  curl -s -o cpp/third_party/tensorrt/include/$f \
    https://raw.githubusercontent.com/NVIDIA/TensorRT/v10.8.0/include/$f
done
```

The header version **must match** the wheel (`NV_TENSORRT_VERSION` is passed to
`createInferRuntime_INTERNAL`); an engine file is only loadable by the TensorRT version that built it.

With the full TensorRT SDK installed (e.g. on Linux/Colab), point `-DTENSORRT_INCLUDE_DIR=` at its
`include/` directory instead.

## GoogleTest

Fetched by CMake `FetchContent` (v1.15.2 zip from GitHub) into `cpp/build/_deps/` at configure time.
