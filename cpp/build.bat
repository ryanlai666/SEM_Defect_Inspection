@echo off
rem Build the C++/CUDA part on Windows without admin rights:
rem   MSVC from VS 2022 (vcvars64), nvcc 12.8 from the project-local conda env (.conda-cuda),
rem   CMake + Ninja + pybind11 from the project venv (.venv).
rem Usage:  cpp\build.bat [extra cmake -D options]      e.g.  cpp\build.bat -DSEMDEFECT_BUILD_TRT_RUNNER=ON
setlocal
set "ROOT=%~dp0.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"

if "%SEMDEFECT_VCVARS%"=="" set "SEMDEFECT_VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%SEMDEFECT_VCVARS%" (
  echo vcvars64.bat not found - set SEMDEFECT_VCVARS to your Visual Studio vcvars64.bat
  exit /b 1
)
call "%SEMDEFECT_VCVARS%" >nul || exit /b 1

set "CUDA_PATH=%ROOT%\.conda-cuda\Library"
if not exist "%CUDA_PATH%\bin\nvcc.exe" (
  echo nvcc not found in %CUDA_PATH% - create it with:
  echo   conda create -y -p .conda-cuda --override-channels -c nvidia/label/cuda-12.8.1 -c conda-forge cuda-nvcc cuda-cudart-dev cuda-cccl
  exit /b 1
)
set "PATH=%CUDA_PATH%\bin;%ROOT%\.venv\Scripts;%PATH%"

cmake -S "%ROOT%\cpp" -B "%ROOT%\cpp\build" -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CUDA_COMPILER="%CUDA_PATH%/bin/nvcc.exe" -DCUDAToolkit_ROOT="%CUDA_PATH%" ^
  -DPython_EXECUTABLE="%ROOT%\.venv\Scripts\python.exe" %* || exit /b 1
cmake --build "%ROOT%\cpp\build" --parallel || exit /b 1
if "%SEMDEFECT_SKIP_TESTS%"=="1" goto :done
rem GPU tests go through the optional shared-GPU guard (scripts\guarded.py, env GPU_GUARD); direct otherwise.
"%ROOT%\.venv\Scripts\python.exe" "%ROOT%\scripts\guarded.py" --gb 1.0 --note "semdefect gtest" -- ctest --test-dir "%ROOT%\cpp\build" --output-on-failure || exit /b 1
echo BUILD+TESTS OK
:done
