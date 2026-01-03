# VisRTX

VisRTX is an experimental, scientific visualization-focused implementation of
the [Khronos ANARI standard](https://www.khronos.org/anari).

VisRTX is designed to track ongoing developments of the ANARI standard and
provide usable extensions where possible. Prospective backend implementors of
ANARI are encouraged to use VisRTX as a much more complete example of a
GPU-accelerated, ray tracing based implementation of ANARI.

Note that the ANARI implementation of VisRTX is a complete rewrite from previous
versions. Please refer to the `v0.1.6` release of VisRTX for the previous
implementation.

## Build + Install

VisRTX is supported on both Linux and Windows.

### Core ANARI Library

Building VisRTX requires the following:

- CMake 3.17+
- C++17 compiler
- NVIDIA Driver 530+
- CUDA 12+
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

# RTX Device Feature Overview

The following sections describes details of VisRTX's ANARI completeness,
provided extensions, and known missing extensions to add in the future.

## Queryable ANARI Extensions

In addition to standard `ANARI_KHR` extensions, the following extensions are
also implemented in the `visrtx` device. Note that all extensions are subject to
change.

#### "ANARI_NV_ARRAY_CUDA" (experimental)

This extension indicates that applications can use pointers to CUDA device
memory when created shared and captured arrays when using VisRTX. All the normal
rules for shared and captured array data still apply.

#### "ANARI_NV_FRAME_BUFFERS_CUDA"

This extension indicates that raw CUDA GPU buffers from frame objects can be
mapped for applications which are already using CUDA. The following additional
channels can be mapped:

- `"colorCUDA"`
- `"depthCUDA"`

GPU pointers returned by `anariMapFrame()` are device pointers intended to be
kept on the device. Applications which desire to copy data from the device back
to the host should instead map the ordinary `color` and `depth` channels.

#### VISRTX_SPATIAL_FIELD_DATA_CENTERING

The `dataCentering` parameter controls how spatial field data is interpreted relative to the grid structure. 
This extension enables fine-grained control over whether data values represent quantities at grid vertices 
(node-centered) or at the center of grid cells (cell-centered).

**Supported Values:**
- `"node"`: Data is centered at grid vertices (node-centered). Each data value corresponds to a grid point.
- `"cell"`: Data is centered at cell centers. Each data value corresponds to the center of a voxel/cell.

> [!Note]
> The spatial extent of a StructuredRegular volume depends on the `dataCentering` parameter. 
> When set to `"node"`, the extent is `[origin, origin + (data.size - 1) × spacing]`.
> When set to `"cell"`, the extent is `[origin, origin + data.size × spacing]`.

#### VISRTX_SPATIAL_FIELD_REGION_OF_INTEREST

Allows clipping a spatial field to an application-defined box in object space.
Two optional parameters on `structuredRegular` and `nanovdb` fields define the
box; omitting either leaves the bound open.

**Parameters:**
- `roi` (`FLOAT32_BOX3`): the restriction box to apply (default: unbounded).

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

The following properties are available to query on `ANARIFrame`:

| Name           | Type  | Description                                           |
|:---------------|:------|:------------------------------------------------------|
| numSamples     | INT32 | get the number of pixel samples currently accumulated |
| nextFrameReset | BOOL  | query whether the next frame will reset accumulation  |

The `numSamples` property is the lower bound of pixel samples taken when the
`checkerboard` renderer parameter (see below) is enabled because not every pixel
will have the same number of samples accumulated.

The `nextFrameReset` property can give the application feedback for when
accumulation is about to reset in the next frame. When the property is queried
and the current frame is complete, all committed objects since the last
rendering operation will be internally updated (may be expensive).

## List of Implemented ANARI Extensions

The following extensions are either partially or fully implemented by VisRTX:

- `KHR_ARRAY1D_REGION`
- `KHR_CAMERA_DEPTH_OF_FIELD`
- `KHR_CAMERA_ORTHOGRAPHIC`
- `KHR_CAMERA_PERSPECTIVE`
- `KHR_DEVICE_SYNCHRONIZATION`
- `KHR_FRAME_ACCUMULATION`
- `KHR_FRAME_CHANNEL_PRIMITIVE_ID`
- `KHR_FRAME_CHANNEL_OBJECT_ID`
- `KHR_FRAME_CHANNEL_INSTANCE_ID`
- `KHR_FRAME_COMPLETION_CALLBACK`
- `KHR_GEOMETRY_CONE`
- `KHR_GEOMETRY_CURVE`
- `KHR_GEOMETRY_CYLINDER`
- `KHR_GEOMETRY_QUAD`
- `KHR_GEOMETRY_SPHERE`
- `KHR_GEOMETRY_TRIANGLE`
- `KHR_INSTANCE_TRANSFORM`
- `KHR_INSTANCE_TRANSFORM_ARRAY`
- `KHR_LIGHT_DIRECTIONAL`
- `KHR_LIGHT_HDRI`
- `KHR_LIGHT_POINT`
- `KHR_LIGHT_SPOT`
- `KHR_MATERIAL_MATTE`
- `KHR_MATERIAL_PHYSICALLY_BASED`
- `KHR_RENDERER_AMBIENT_LIGHT`
- `KHR_RENDERER_BACKGROUND_COLOR`
- `KHR_RENDERER_BACKGROUND_IMAGE`
- `KHR_SAMPLER_IMAGE1D`
- `KHR_SAMPLER_IMAGE2D`
- `KHR_SAMPLER_IMAGE3D`
- `KHR_SAMPLER_PRIMITIVE`
- `KHR_SAMPLER_TRANSFORM`
- `KHR_SPATIAL_FIELD_NANOVDB`
- `KHR_SPATIAL_FIELD_STRUCTURED_REGULAR`
- `KHR_VOLUME_TRANSFER_FUNCTION1D`
- `EXT_SAMPLER_COMPRESSED_IMAGE2D`
- `EXT_SAMPLER_COMPRESSED_FORMAT_BC123`
- `EXT_SAMPLER_COMPRESSED_FORMAT_BC45`
- `NV_ARRAY_CUDA`
- `NV_FRAME_BUFFERS_CUDA`
- `VISRTX_TRIANGLE_BACK_FACE_CULLING`
- `VISRTX_SPATIAL_FIELD_DATA_CENTERING`
- `VISRTX_SPATIAL_FIELD_REGION_OF_INTEREST`

For any found bugs in extensions that are implemented, please [open an
issue](https://github.com/NVIDIA/VisRTX/issues/new)!
