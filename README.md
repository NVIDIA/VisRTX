# VisRTX

![VisRTX Teaser](teaser.png)

VisRTX is an experimental, scientific visualization-focused implementation of
the [Khronos ANARI standard](https://www.khronos.org/anari), and is developed by
the HPC Visualization Developer Technology team at NVIDIA.

VisRTX is designed to track ongoing developments of the ANARI standard and
provide usable extensions where possible. Prospective backend implementors of
ANARI are encouraged to use VisRTX as a much more complete example of a
GPU-accelerated, ray tracing based implementation of ANARI.

Note that the ANARI implementation of VisRTX is a complete rewrite from previous
versions. Please refer to the `v0.1.6` release of VisRTX for the previous
implementation.

Please do not hesitate to provide feedback by [opening an
issue](https://github.com/NVIDIA/VisRTX/issues/new)!

## Build + Install

VisRTX is supported on both Linux and Windows.

### Core ANARI Library

Building VisRTX requires the following:

- CMake 3.17+
- C++17 compiler
- NVIDIA Driver 470+
- CUDA 11.3.1+
- [OptiX 7.3+](https://developer.nvidia.com/rtx/ray-tracing/optix)
- [ANARI-SDK](https://github.com/KhronosGroup/ANARI-SDK)

Building VisRTX is done through invoking CMake on the source directory from a
stand alone build directory. This might look like

```bash
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=path/to/desired/install /path/to/visrtx/source
make
make install
```

The OptiX and ANARI-SDK dependencies can be found via placing their installation
locations on `CMAKE_PREFIX_PATH`, either as an environment variable or a CMake
variable.

The build will result in a single `libanari_library_visrtx` library that will
install to `${CMAKE_INSTALL_PREFIX}/lib`, and is usable with any ANARI app if
either it is installed to the same location as the ANARI-SDK or
`libanari_library_visrtx` is placed on `LD_LIBRARY_PATH` respectively.

### Provided Examples

VisRTX comes with a simple, single-file tutorial application that show how to
use VisRTX through the ANARI API. It is always enabled as it only requires the
ANARI SDK and compiles very quickly.

VisRTX also comes with an optional interactive example application that gives
application developers a sense of what VisRTX has to offer. To enable the
interactive example, simply turn on the `VISRTX_BUILD_INTERACTIVE_EXAMPLE`
option in your local CMake build.  This can be done with adding
`-DVISRTX_BUILD_INTERACTIVE_EXAMPLE=ON` to the CMake command above, or done with
either of the interactive CMake programs (`ccmake` or `cmake-gui`).

The interactive example requires [GLFW](https://www.glfw.org/) as an additional
dependency.

# Feature Overview

The following sections describes details of VisRTX's ANARI completeness,
provided extensions, and known missing features to add in the future.

## Queryable ANARI Extensions

The following extension strings will return true when queried with
`anariDeviceImplements()`. Note that all vendor extensions are subject to change
at any time.

#### "ANARI_KHR_STOCHASTIC_RENDERING"

This core extension indicates that frames will accumulate samples when
subsequent calls to `anariRenderFrame()` are made if no objects modifications
are made since the previously rendered frame.  Accumulation will not reset on
`anariSetParameter()`, but only if objects that have parameter changes have been
commited via `anariCommit()`.

Note that variance estimation and convergence progress properties are not yet
implemented.

#### "VISRTX_SAMPLER_COLOR_MAP"

This vendor extension indicates that an additional sampler subtype is available:
`"colorMap"`. This sampler takes the first channel of the input attribute and
applies a transfer function (normalization + color) to it. This sampler has the
following parameters in addition to the base ANARI sampler parameters:

| Name           | Type               | Default | Description                                                             |
|:---------------|:-------------------|--------:|:------------------------------------------------------------------------|
| color          | ARRAY1D of Color   |         | array to map sampled and clamped field values to color                  |
| color.position | ARRAY1D of FLOAT32 |         | optional array to position the elements of color values in `valueRange` |
| valueRange     | FLOAT32_BOX1       |  \[0,1\]| input attribute values are clamped to this range                        |

This sampler follows the same rules for the `color` and `color.position`
parameters as they are found on the `"scivis"` volume subtype. Values from this
sampler are output as `VEC3(r, g, b, 1)` to the input of the material where the
sampler used, where `r`, `g`, and `b` come from the values in `color`
respectively.

#### "VISRTX_CUDA_OUTPUT_BUFFERS"

This vendor extension indicates that raw CUDA GPU buffers from frame objects can
be mapped for applications which are already using CUDA. The following
additional channels can be mapped:

- `"colorGPU"`
- `"depthGPU"`

GPU pointers returned by `anariMapFrame()` are device pointers intended to be
kept on the device. Applications which desire to copy data from the device back
to the host should instead map the ordinary `color` and `depth` channels.

#### "VISRTX_TRIANGLE_ATTRIBUTE_INDEXING" (experimental)

This vendor extension indicates that additional attribute indexing is
available for the `triangle` geometry subtype. Specifically, the following
additional index arrays will be interpreted if set on the geometry:

- `vertex.color.index`
- `vertex.normal.index`
- `vertex.attribute0.index`
- `vertex.attribute1.index`
- `vertex.attribute2.index`
- `vertex.attribute3.index`

Each of these arrays must be of type `UINT32_VEC3` and is indexed per-triangle
on the geometry where each component indexes into the corresponding index array
that matches. For example, `vertex.color.index` for primitive 0 (first triangle)
will load values from `vertex.color` accordingly. For every `.index` array
present, the matching vertex array must also be present. All index values must
be within the size of the corresponding vertex array it accesses.

## Additional ANARI Parameter and Property Extensions

The following section describes what additional parameters and properties can be
used on various ANARI objects.

#### Device

The device itself can take a single `INT32` parameter `"cudaDevice"` to select
which CUDA GPU should be used for rendering. Once this value has been set _and_
the implementation has initialized CUDA for itself, then changing this to
another value will be ignored (a warning will tell you this if it happens). The
device will initialize CUDA for itself if any object gets created from the
device.

#### Frame

The following optional parameters are available to set on `ANARIFrame`:

| Name         | Type | Default | Description                                      |
|:-------------|:-----|--------:|:-------------------------------------------------|
| denoise      | BOOL |   false | enable the OptiX denoiser on the `color` channel |
| checkerboard | BOOL |   false | trade fewer samples per-frame for interactivity  |

The `checkerboard` parameter will sample subsets of the image at a faster rate,
while still converging to the same image, as the final set of samples taken for
each pixel ends up being the same. The pattern and method of which this feature
is implementented is subject to change, so applications which desire exact
sample counts should use the `numSamples` property described below.

The following properties are available to query on `ANARIFrame`:

| Name           | Type  | Description                                           |
|:---------------|:------|:------------------------------------------------------|
| numSamples     | INT32 | get the number of pixel samples currently accumulated |
| nextFrameReset | BOOL  | query whether the next frame will reset accumulation  |

The `numSamples` property is the lower bound of pixel samples taken when the
`checkerboard` parameter is enabled because not every pixel will have the same
number of samples accumulated.

The `nextFrameReset` property can give the application feedback for when
accumulation is about to reset in the next frame. When the property is queried
and the current frame is complete, all committed objects since the last
rendering operation will be internally updated (may be expensive).

#### Renderer

The ANARI specification does not have any required renderer subtypes devices
must provite, other than the existence of a `default` subtype. VisRTX provides
the following subtypes:

- `scivis` (default)
- `ao`
- `raycast`
- `debug`

All renderers share the following parameters:

| Name            | Type         | Default   | Description                                               |
|:----------------|:-------------|----------:|:----------------------------------------------------------|
| backgroundColor | FLOAT32_VEC4 | {1,1,1,1} | color of the background                                   |
| pixelSamples    | INT32        |         1 | number of samples taken per call to `anariRenderFrame()`  |

The `pixelSamples` parameter is equivalent to calling `anariRenderFrame()` N
times to reduce noise in the image.

The `debug` renderer is designed to help developers understand how VisRTX is
interpreting the scene it is rendering. This renderer uses a `STRING` parameter
named `"method"` to control which debugging views of the scene is used. The
following values are valid values:

| Method   | Description                                                |
|:---------|:-----------------------------------------------------------|
| primID   | visualize geometry primitive index                         |
| geomID   | visualize geometry index within a group                    |
| instID   | visualize instance index within the world                  |
| Ng       | visualize geometric normal                                 |
| uvw      | visualize geometry barycentric coordinates                 |
| istri    | show objects as green if they are HW accelerated triangles |
| isvol    | show objects as green if they are a volume                 |
| backface | show front facing primitives as green, red if back facing  |

The `debug` renderer can use a `_[method]` suffix on the subtype string to set
the default method. This can be a convenient alternative for applications to
switch between debug renderer views. For example, `"debug_Ng"` would initially
use the `Ng` method. The debug renderer with method suffixes are listed out as
complete subtypes by `anariGetObjectSubtypes()`.

Note that VisRTX renderers and their parameters are very much still in flux, so
applications should use ANARI object introspection query functions to get the
latest available parameters they can use. When the list of renderers and their
parameters stabilize over time, they will be documented here. VisRTX will always
keep the `default` renderer subtype as something usable without needing to
change parameters.

## Known Missing Core ANARI Features + Extensions

The following features are not yet implemented by VisRTX:

- Geometry: `curve`
- Light: `spot`, instancing
- Camera: `omnidirectional`, stereo rendering, direct transform parameter
- Sampler: `image1D`, `image3D`, in/out transforms on `image2D`
- Frame: variance property
- Sparse shared arrays (non-zero stride)
- Core extensions:
    - `ANARI_KHR_AREA_LIGHTS`
    - `ANARI_KHR_FRAME_COMPLETION_CALLBACK`
    - `ANARI_KHR_DEVICE_SYNCHRONIZATION`
    - `ANARI_KHR_TRANSFORMATION_MOTION_BLUR`
- Introspection for individual parameters of objects other than renderers

The following features have known to be incomplete:

- Arrays for image samplers only suport FLOAT32 type elements

For any found bugs in features that are implemented, please [open an
issue](https://github.com/NVIDIA/VisRTX/issues/new)!
