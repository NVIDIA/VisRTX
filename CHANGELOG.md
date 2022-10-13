# Changelog

## 0.5.0
##### TBD
- Update to latest ANARI SDK v0.3.0
- Added support for `curve` geometry subtype
- Added support `image1D` sampler subtype
- Added support for `primitive.index` on `sphere` geometry
- Added support for missing array types for image samplers
- Improved robustness around object subtypes unknown to VisRTX
- Improved object leak warnings on device release
- Fix incorrect handling of managed arrays of objects

## 0.4.0
##### 2022-07-22
- Update to latest ANARI SDK v0.2.0
- All object info queries are now implemented
- Implement `VISRTX_ARRAY1D_DYNAMIC_REGION` extension for 1D arrays
- VisRTX now provides an installed header for custom VisRTX functionality
    - See [tutorial app](examples/tutorial) as an example
- Fix incorrect warning about different sized arrays for indexed triangles
- Fix missing BLAS updates occuring in certain cases
- Memory offload optimizations for arrays that are about to go out of scope
- Improved texture loading for OBJ scenes in sample `viewer` app

## 0.3.2
##### 2022-04-13
- Fix build issues found on certain platforms
- Fix crash when querying device properties in certain cases
- Fix occasional instability of PTX generation + embedding process

## 0.3.1
##### 2022-04-05
- Fix incorrect update of BVH when modifying some geometries

## 0.3.0
##### 2022-03-30
- Update to latest ANARI SDK v0.1.0 (please note new versioning scheme)
- Added support for `cone` geometry
- Improved warning messages in various places
- Fix incorrect handling of array deleters for captured arrays
- Fix bug in instancing code causing UB in some cases

## 0.2.0
##### 2022-03-01
- Completely new ANARI-based implementation (see README for details)

## 0.1.6
##### 2019-08-13
- Automatic ray epsilon handling
- Clipping planes
- Bug fixes (sampling, light visibility, device enumeration, VS 2013 support)

## 0.1.5
##### 2019-04-15
- Added Camera::SetImageRegion
- Priorities fix for nested volumetric materials

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
