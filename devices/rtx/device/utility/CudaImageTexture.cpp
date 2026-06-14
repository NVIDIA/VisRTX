/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "CudaImageTexture.h"
#include <cuda_runtime_api.h>
#include <driver_functions.h>
#include <driver_types.h>
#include <texture_types.h>
#include "utility/AnariTypeHelpers.h"

namespace visrtx {

// Function definitions ///////////////////////////////////////////////////////

int countCudaChannels(const cudaChannelFormatDesc &desc)
{
  int channels = 0;
  if (desc.x != 0)
    channels++;
  if (desc.y != 0)
    channels++;
  if (desc.z != 0)
    channels++;
  if (desc.w != 0)
    channels++;
  return channels;
}

cudaTextureAddressMode stringToAddressMode(const std::string &str)
{
  if (str == "repeat")
    return cudaAddressModeWrap;
  else if (str == "mirrorRepeat")
    return cudaAddressModeMirror;
  else if (str == "clampToBorder")
    return cudaAddressModeBorder;
  else
    return cudaAddressModeClamp;
}

void makeCudaArrayFloat(
    cudaArray_t &cuArray, int nc, const float *data, uvec3 size)
{
  if (!cuArray) {
    auto desc = cudaCreateChannelDesc(nc >= 1 ? 32 : 0,
        nc >= 2 ? 32 : 0,
        nc >= 3 ? 32 : 0,
        nc >= 4 ? 32 : 0,
        cudaChannelFormatKindFloat);

    cudaMalloc3DArray(&cuArray,
        &desc,
        make_cudaExtent(
            size.x, size.y <= 1 ? 0 : size.y, size.z <= 1 ? 0 : size.z));
  }

  cudaMemcpy3DParms p = {};
  p.dstArray = cuArray;
  p.srcPtr = make_cudaPitchedPtr(
      const_cast<float *>(data), size.x * nc * sizeof(float), size.x, size.y);
  p.srcPos = p.dstPos = make_cudaPos(0, 0, 0);
  p.extent = make_cudaExtent(size.x, size.y, size.z);
  p.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p);
}

void makeCudaArray(cudaArray_t &cuArray, const Array &array, uint32_t size)
{
  makeCudaArray(cuArray, array, uvec3(size, 1, 1));
}

void makeCudaArray(cudaArray_t &cuArray, const Array &array, uvec2 size)
{
  makeCudaArray(cuArray, array, uvec3(size, 1));
}

namespace {

// IEEE-754 half-precision encoding of 1.0 (FLOAT16 is stored as raw uint16_t).
constexpr uint16_t kHalfOne = 0x3C00;

// Expand a 3-channel image to 4 channels (CUDA has no 3-channel cudaArray),
// padding the added alpha so it samples as opaque/1.0. This is the only data
// transform applied to a texture; every other format is copied verbatim.
template <typename T>
static void expandRGBtoRGBA(const Array &array, void *dstRaw, T pad)
{
  const T *src = static_cast<const T *>(array.data());
  T *dst = static_cast<T *>(dstRaw);
  const size_t texels = array.totalSize();
  for (size_t i = 0; i < texels; ++i) {
    dst[4 * i + 0] = src[3 * i + 0];
    dst[4 * i + 1] = src[3 * i + 1];
    dst[4 * i + 2] = src[3 * i + 2];
    dst[4 * i + 3] = pad;
  }
}

// Build a cudaArray whose channel kind and bit depth mirror the ANARI element
// type, then copy the data verbatim. 8/16/32-bit unsigned and 16/32-bit float
// are kept in their native form; sRGB stays as raw 8-bit bytes (the sampler
// does sRGB->linear). The sole transform is 3->4 channel expansion, which CUDA
// forces. Integer arrays are read back with normalized read mode, so keeping
// them at native depth (not promoted to float) is both correct and compact.
static void buildNativeCudaArray(
    cudaArray_t &cuArray, const Array &array, uvec3 size)
{
  const ANARIDataType format = array.elementType();
  const int nc = numANARIChannels(format);
  const size_t compBytes = bytesPerChannel(format);
  const int bits = int(compBytes) * 8;
  const cudaChannelFormatKind kind = isFloat(format)
      ? cudaChannelFormatKindFloat
      : cudaChannelFormatKindUnsigned;
  const int storedNc = (nc == 3) ? 4 : nc;

  if (!cuArray) {
    auto desc = cudaCreateChannelDesc(storedNc >= 1 ? bits : 0,
        storedNc >= 2 ? bits : 0,
        storedNc >= 3 ? bits : 0,
        storedNc >= 4 ? bits : 0,
        kind);
    cudaMalloc3DArray(&cuArray,
        &desc,
        make_cudaExtent(size.x, size.y, size.z <= 1 ? 0 : size.z));
  }

  // Expand RGB->RGBA when needed; otherwise copy the host data straight in.
  std::vector<uint8_t> staging;
  const void *src = array.data();
  size_t srcChannels = size_t(nc);
  if (nc == 3) {
    staging.resize(array.totalSize() * 4 * compBytes);
    if (isFloat32(format))
      expandRGBtoRGBA<float>(array, staging.data(), 1.0f);
    else if (isFloat16(format))
      expandRGBtoRGBA<uint16_t>(array, staging.data(), kHalfOne);
    else if (compBytes == 4)
      expandRGBtoRGBA<uint32_t>(
          array, staging.data(), std::numeric_limits<uint32_t>::max());
    else if (compBytes == 2)
      expandRGBtoRGBA<uint16_t>(
          array, staging.data(), std::numeric_limits<uint16_t>::max());
    else
      expandRGBtoRGBA<uint8_t>(
          array, staging.data(), std::numeric_limits<uint8_t>::max());
    src = staging.data();
    srcChannels = 4;
  }

  cudaMemcpy3DParms p = {};
  p.dstArray = cuArray;
  p.srcPos = p.dstPos = make_cudaPos(0, 0, 0);
  p.srcPtr = make_cudaPitchedPtr(const_cast<void *>(src),
      size.x * srcChannels * compBytes,
      size.x,
      size.y);
  p.extent = make_cudaExtent(size.x, size.y, size.z < 1 ? 1 : size.z);
  p.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p);
}

} // namespace

void makeCudaArray(cudaArray_t &cuArray, const Array &array, uvec3 size)
{
  buildNativeCudaArray(cuArray, array, size);
}

cudaTextureObject_t makeCudaTextureObject(cudaArray_t cuArray,
    bool readModeNormalizedFloat,
    const std::string &filter,
    const std::string &wrap1,
    const std::string &wrap2,
    const std::string &wrap3,
    bool normalizedCoords,
    const vec4 &borderColor,
    bool sRGB)
{
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = stringToAddressMode(wrap1);
  texDesc.addressMode[1] = stringToAddressMode(wrap2);
  texDesc.addressMode[2] = stringToAddressMode(wrap3);
  texDesc.filterMode =
      filter == "nearest" ? cudaFilterModePoint : cudaFilterModeLinear;
  texDesc.readMode = readModeNormalizedFloat ? cudaReadModeNormalizedFloat
                                             : cudaReadModeElementType;
  texDesc.normalizedCoords = normalizedCoords;
  // Hardware sRGB->linear on sample for raw sRGB8 data (valid for 8-bit unorm).
  // Lets the texture stay in its native sRGB bytes instead of being linearized
  // into 8-bit on the CPU (which bands the dark end).
  texDesc.sRGB = sRGB;
  texDesc.borderColor[0] = borderColor.x;
  texDesc.borderColor[1] = borderColor.y;
  texDesc.borderColor[2] = borderColor.z;
  texDesc.borderColor[3] = borderColor.w;

  cudaTextureObject_t retval = {};
  cudaCreateTextureObject(&retval, &resDesc, &texDesc, nullptr);

  return retval;
}

void makeCudaCompressedTextureArray(cudaArray_t &cuArray,
    const uvec2 &size,
    const Array &array,
    const cudaChannelFormatKind channelFormatKind)
{
  assert(!cuArray);

  const ANARIDataType format = array.elementType();
  assert(format == ANARI_UINT8 || format == ANARI_INT8);

  // Create CUDA texture //
  cudaChannelFormatDesc desc;
  std::uint32_t blockWidth{};
  std::uint32_t bytesPerBlock{};

  switch (channelFormatKind) {
  case cudaChannelFormatKindUnsignedBlockCompressed1: {
    desc =
        cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed1>();
    blockWidth = 4;
    bytesPerBlock = 8;
    break;
  }
  case cudaChannelFormatKindUnsignedBlockCompressed1SRGB: {
    desc = cudaCreateChannelDesc<
        cudaChannelFormatKindUnsignedBlockCompressed1SRGB>();
    blockWidth = 4;
    bytesPerBlock = 8;
    break;
  }
  case cudaChannelFormatKindUnsignedBlockCompressed2: {
    desc =
        cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed2>();
    blockWidth = 4;
    bytesPerBlock = 16;
    break;
  }
  case cudaChannelFormatKindUnsignedBlockCompressed2SRGB: {
    desc = cudaCreateChannelDesc<
        cudaChannelFormatKindUnsignedBlockCompressed2SRGB>();
    blockWidth = 4;
    bytesPerBlock = 16;
    break;
  }
  case cudaChannelFormatKindUnsignedBlockCompressed3: {
    desc =
        cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed3>();
    blockWidth = 4;
    bytesPerBlock = 16;
    break;
  }
  case cudaChannelFormatKindUnsignedBlockCompressed3SRGB: {
    desc = cudaCreateChannelDesc<
        cudaChannelFormatKindUnsignedBlockCompressed3SRGB>();
    blockWidth = 4;
    bytesPerBlock = 16;
    break;
  }
  case cudaChannelFormatKindUnsignedBlockCompressed4: {
    desc =
        cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed4>();
    blockWidth = 4;
    bytesPerBlock = 8;
    break;
  }
  case cudaChannelFormatKindSignedBlockCompressed4: {
    desc = cudaCreateChannelDesc<cudaChannelFormatKindSignedBlockCompressed4>();
    blockWidth = 4;
    bytesPerBlock = 8;
    break;
  }
  case cudaChannelFormatKindUnsignedBlockCompressed5: {
    desc =
        cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed5>();
    blockWidth = 4;
    bytesPerBlock = 16;
    break;
  }
  case cudaChannelFormatKindSignedBlockCompressed5: {
    desc = cudaCreateChannelDesc<cudaChannelFormatKindSignedBlockCompressed5>();
    blockWidth = 4;
    bytesPerBlock = 16;
    break;
  }
  case cudaChannelFormatKindUnsignedBlockCompressed6H: {
    desc =
        cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed6H>();
    blockWidth = 4;
    bytesPerBlock = 16;
    break;
  }
  case cudaChannelFormatKindSignedBlockCompressed6H: {
    desc =
        cudaCreateChannelDesc<cudaChannelFormatKindSignedBlockCompressed6H>();
    blockWidth = 4;
    bytesPerBlock = 16;
    break;
  }
  case cudaChannelFormatKindUnsignedBlockCompressed7: {
    desc =
        cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed7>();
    blockWidth = 4;
    bytesPerBlock = 16;
    break;
  }
  case cudaChannelFormatKindUnsignedBlockCompressed7SRGB: {
    desc = cudaCreateChannelDesc<
        cudaChannelFormatKindUnsignedBlockCompressed7SRGB>();
    blockWidth = 4;
    bytesPerBlock = 16;
    break;
  }
  default:
    // Unknown format type
    return;
  };

  if (blockWidth == 0 || bytesPerBlock == 0) {
    return;
  }

  uint32_t widthInBlocks = (size.x + blockWidth - 1) / blockWidth;
  uint32_t heightInBlocks = (size.y + blockWidth - 1) / blockWidth;

  // Make sure the 3rd component is 0 so we allocate a 2D array. If 1 we
  // allocate a 3D array of depth 1 which is not the same and will not work with
  // the texture object.
  cudaMalloc3DArray(&cuArray, &desc, make_cudaExtent(size.x, size.y, 0));
  // cudaMalloc3DArray(&cuArray, &desc, make_cudaExtent(size.x, size.y, 0));

  cudaMemcpy3DParms p = {};
  p.dstArray = cuArray;
  p.srcPtr = make_cudaPitchedPtr(const_cast<void *>(array.data()),
      widthInBlocks * bytesPerBlock,
      widthInBlocks,
      heightInBlocks);
  // Compare to the extent above, we want the 3rd component to be 1 here so we
  // copy a full slice of data.
  p.extent = make_cudaExtent(size.x, size.y, 1); // extent;
  p.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p);
}

cudaTextureObject_t makeCudaCompressedTextureObject(cudaArray_t cuArray,
    const std::string &filter,
    const std::string &wrap1,
    const std::string &wrap2,
    const std::string &wrap3,
    bool normalizedCoords,
    const cudaChannelFormatKind channelFormatKind,
    const vec4 &borderColor)
{
  cudaResourceDesc resDesc{};
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;

  cudaTextureDesc texDesc{};
  texDesc.addressMode[0] = stringToAddressMode(wrap1);
  texDesc.addressMode[1] = stringToAddressMode(wrap2);
  texDesc.addressMode[2] = stringToAddressMode(wrap3);
  texDesc.filterMode =
      filter == "nearest" ? cudaFilterModePoint : cudaFilterModeLinear;
  texDesc.normalizedCoords = normalizedCoords;

  texDesc.borderColor[0] = borderColor.x;
  texDesc.borderColor[1] = borderColor.y;
  texDesc.borderColor[2] = borderColor.z;
  texDesc.borderColor[3] = borderColor.w;

  // Only explicit float type are to be read as element type. Others need to be
  // read as normalized floats.
  if (channelFormatKind != cudaChannelFormatKindUnsignedBlockCompressed6H
      && channelFormatKind != cudaChannelFormatKindSignedBlockCompressed6H) {
    texDesc.readMode = cudaReadModeNormalizedFloat;
  }

  // Correctly propagate sRGB information.
  switch (channelFormatKind) {
  case cudaChannelFormatKindUnsignedBlockCompressed1SRGB:
  case cudaChannelFormatKindUnsignedBlockCompressed2SRGB:
  case cudaChannelFormatKindUnsignedBlockCompressed3SRGB:
  case cudaChannelFormatKindUnsignedBlockCompressed7SRGB:
    texDesc.sRGB = true;
    break;
  default:
    break;
  }

  cudaTextureObject_t retval = {};

  cudaCreateTextureObject(&retval, &resDesc, &texDesc, nullptr);

  return retval;
}

cudaTextureObject_t makeCudaTextureObject1D(cudaArray_t cuArray,
    bool readModeNormalizedFloat,
    const std::string &filter,
    const std::string &wrap,
    const vec4 &borderColor,
    bool sRGB)
{
  return makeCudaTextureObject(cuArray,
      readModeNormalizedFloat,
      filter,
      wrap,
      wrap,
      wrap,
      true,
      borderColor,
      sRGB);
}

cudaTextureObject_t makeCudaTextureObject2D(cudaArray_t cuArray,
    bool readModeNormalizedFloat,
    const std::string &filter,
    const std::string &wrap1,
    const std::string &wrap2,
    const vec4 &borderColor,
    bool sRGB)
{
  return makeCudaTextureObject(cuArray,
      readModeNormalizedFloat,
      filter,
      wrap1,
      wrap2,
      wrap2,
      true,
      borderColor,
      sRGB);
}

cudaTextureObject_t makeCudaTextureObject3D(cudaArray_t cuArray,
    bool readModeNormalizedFloat,
    const std::string &filter,
    const std::string &wrap1,
    const std::string &wrap2,
    const std::string &wrap3,
    const vec4 &borderColor,
    bool sRGB)
{
  return makeCudaTextureObject(cuArray,
      readModeNormalizedFloat,
      filter,
      wrap1,
      wrap2,
      wrap3,
      true,
      borderColor,
      sRGB);
}

cudaTextureObject_t makeCudaTexelObject1D(cudaArray_t cuArray,
    bool readModeNormalizedFloat,
    const std::string &filter,
    const std::string &wrap,
    const vec4 &borderColor)
{
  return makeCudaTextureObject(cuArray,
      readModeNormalizedFloat,
      filter,
      wrap,
      wrap,
      wrap,
      false,
      borderColor,
      false);
}

cudaTextureObject_t makeCudaTexelObject2D(cudaArray_t cuArray,
    bool readModeNormalizedFloat,
    const std::string &filter,
    const std::string &wrap1,
    const std::string &wrap2,
    const vec4 &borderColor)
{
  return makeCudaTextureObject(cuArray,
      readModeNormalizedFloat,
      filter,
      wrap1,
      wrap2,
      wrap2,
      false,
      borderColor,
      false);
}

cudaTextureObject_t makeCudaTexelObject3D(cudaArray_t cuArray,
    bool readModeNormalizedFloat,
    const std::string &filter,
    const std::string &wrap1,
    const std::string &wrap2,
    const std::string &wrap3,
    const vec4 &borderColor)
{
  return makeCudaTextureObject(cuArray,
      readModeNormalizedFloat,
      filter,
      wrap1,
      wrap2,
      wrap3,
      false,
      borderColor,
      false);
}

cudaTextureObject_t makeCudaCompressedTextureObject1D(cudaArray_t cuArray,
    cudaChannelFormatKind channelFormatKind,
    const std::string &filter,
    const std::string &wrap,
    const vec4 &borderColor)
{
  return makeCudaCompressedTextureObject(
      cuArray, filter, wrap, wrap, wrap, true, channelFormatKind, borderColor);
}

cudaTextureObject_t makeCudaCompressedTextureObject2D(cudaArray_t cuArray,
    cudaChannelFormatKind channelFormatKind,
    const std::string &filter,
    const std::string &wrap1,
    const std::string &wrap2,
    const vec4 &borderColor)
{
  return makeCudaCompressedTextureObject(cuArray,
      filter,
      wrap1,
      wrap2,
      wrap2,
      true,
      channelFormatKind,
      borderColor);
}

cudaTextureObject_t makeCudaCompressedTextureObject3D(cudaArray_t cuArray,
    cudaChannelFormatKind channelFormatKind,
    const std::string &filter,
    const std::string &wrap1,
    const std::string &wrap2,
    const std::string &wrap3,
    const vec4 &borderColor)
{
  return makeCudaCompressedTextureObject(cuArray,
      filter,
      wrap1,
      wrap2,
      wrap3,
      true,
      channelFormatKind,
      borderColor);
}

cudaTextureObject_t makeCudaCompressedTexelObject1D(cudaArray_t cuArray,
    cudaChannelFormatKind channelFormatKind,
    const std::string &filter,
    const std::string &wrap,
    const vec4 &borderColor)
{
  return makeCudaCompressedTextureObject(
      cuArray, filter, wrap, wrap, wrap, false, channelFormatKind, borderColor);
}

cudaTextureObject_t makeCudaCompressedTexelObject2D(cudaArray_t cuArray,
    cudaChannelFormatKind channelFormatKind,
    const std::string &filter,
    const std::string &wrap1,
    const std::string &wrap2,
    const vec4 &borderColor)
{
  return makeCudaCompressedTextureObject(cuArray,
      filter,
      wrap1,
      wrap2,
      wrap2,
      false,
      channelFormatKind,
      borderColor);
}

cudaTextureObject_t makeCudaCompressedTexelObject3D(cudaArray_t cuArray,
    cudaChannelFormatKind channelFormatKind,
    const std::string &filter,
    const std::string &wrap1,
    const std::string &wrap2,
    const std::string &wrap3,
    const vec4 &borderColor)
{
  return makeCudaCompressedTextureObject(cuArray,
      filter,
      wrap1,
      wrap2,
      wrap3,
      false,
      channelFormatKind,
      borderColor);
}

} // namespace visrtx