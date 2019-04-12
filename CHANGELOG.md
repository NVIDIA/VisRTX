# Changelog

## 0.1.4
##### 2019-04-12
- Volumetric absorption (e.g. Glass material)
- Added volume sample
- Buffer access fix

## 0.1.3
##### 2019-04-10
- Tone mapping bug fix
- EGL interop in samples (Linux only, run with parameter `egl`)
- New CMake options `VISRTX_SAMPLE_WITH_GLFW`, `VISRTX_SAMPLE_WITH_EGL`
- CMake OpenGL fixes

## 0.1.2
##### 2019-04-09
- Automatically enable all available GPUs (see note about `CUDA_VISIBLE_DEVICES` order in README)
- Faster BVH refit for dynamic geometry
- Batch upload launch parameters
- Fixed minimum stack size for OptiX 5.1
- CMake configuration install fix

## 0.1.1
##### 2019-04-05
- Compatibility fixes for GCC 5/6
- Multi-GPU fixes
- Better GLX/EGL/GLVND support (Removed GLEW dependency)

## 0.1.0
##### 2019-03-15
- Initial release
