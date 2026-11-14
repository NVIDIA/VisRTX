// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "CudaRGBExpansion.h"
#include "gpu/gpu_decl.h"

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>

#include <cstdint>
#include <type_traits>

namespace visrtx {

namespace {

struct Surface
{
  ~Surface();
  cudaSurfaceObject_t object{0};
};

Surface::~Surface()
{
  if (object)
    cudaDestroySurfaceObject(object);
}

template <typename T>
struct RGBTexel
{
  const T *src;
  T pad;
  cudaSurfaceObject_t surface;
  cudaExtent extent;

  VISRTX_DEVICE void operator()(size_t texel) const;
};

template <typename T>
VISRTX_DEVICE void RGBTexel<T>::operator()(size_t texel) const
{
  using Pixel = std::conditional_t<std::is_same_v<T, float>,
      float4,
      std::conditional_t<sizeof(T) == 1,
          uchar4,
          std::conditional_t<sizeof(T) == 2, ushort4, uint4>>>;
  const Pixel rgba{src[3 * texel], src[3 * texel + 1], src[3 * texel + 2], pad};
  // Surface x coordinates are in bytes, not texels. Use the actual CUDA array
  // dimensions: VisRTX also represents a 1D image as a 2D array of height one.
  const int x = int((texel % extent.width) * sizeof(Pixel));
  if (extent.height == 0) {
    surf1Dwrite(rgba, surface, x);
  } else {
    const int y = int((texel / extent.width) % extent.height);
    if (extent.depth == 0)
      surf2Dwrite(rgba, surface, x, y);
    else
      surf3Dwrite(
          rgba, surface, x, y, int(texel / (extent.width * extent.height)));
  }
}

} // namespace

template <typename T>
cudaError_t expandRGBtoRGBADevice(const T *src, cudaArray_t dst, T pad)
{
  cudaChannelFormatDesc desc{};
  cudaExtent extent{};
  unsigned int flags = 0;
  auto error = cudaArrayGetInfo(&desc, &extent, &flags, dst);
  if (error != cudaSuccess)
    return error;
  if (!(flags & cudaArraySurfaceLoadStore))
    return cudaErrorInvalidValue;

  cudaResourceDesc resource{};
  resource.resType = cudaResourceTypeArray;
  resource.res.array.array = dst;
  Surface surface;
  error = cudaCreateSurfaceObject(&surface.object, &resource);
  if (error != cudaSuccess)
    return error;

  const size_t texels = extent.width * (extent.height ? extent.height : 1)
      * (extent.depth ? extent.depth : 1);
  const auto begin = thrust::make_counting_iterator<size_t>(0);
  // The synchronous policy finishes surface writes before the surface is
  // destroyed and before the caller samples the updated texture.
  // Thrust checks CUDA's last-error slot internally. Discard earlier,
  // unrelated synchronous errors so only this operation determines its status.
  cudaGetLastError();
  try {
    thrust::for_each(thrust::device,
        begin,
        begin + texels,
        RGBTexel<T>{src, pad, surface.object, extent});
  } catch (const thrust::system_error &e) {
    return e.code().category() == thrust::cuda_category()
        ? static_cast<cudaError_t>(e.code().value())
        : cudaErrorUnknown;
  } catch (...) {
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}

template cudaError_t expandRGBtoRGBADevice<float>(
    const float *, cudaArray_t, float);
template cudaError_t expandRGBtoRGBADevice<uint8_t>(
    const uint8_t *, cudaArray_t, uint8_t);
template cudaError_t expandRGBtoRGBADevice<uint16_t>(
    const uint16_t *, cudaArray_t, uint16_t);
template cudaError_t expandRGBtoRGBADevice<uint32_t>(
    const uint32_t *, cudaArray_t, uint32_t);

} // namespace visrtx
