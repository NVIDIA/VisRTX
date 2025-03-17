/*
 * Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Private helper functions ///////////////////////////////////////////////////

template <bool SRGB, typename T>
static uint8_t convertComponentUint8(T c)
{
  if constexpr (std::is_same_v<T, float>)
    return uint8_t(c * 255);
  else if constexpr (std::is_same_v<T, uint16_t>) {
    constexpr auto maxVal = float(std::numeric_limits<uint16_t>::max());
    return uint8_t((c / maxVal) * 255);
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    constexpr auto maxVal = float(std::numeric_limits<uint32_t>::max());
    return uint8_t((c / maxVal) * 255);
  } else if constexpr (SRGB) // uint8_t
    return uint8_t(glm::convertSRGBToLinear(vec1(c / 255.f)).x * 255);
  else // uint8_t, linear
    return c;
}

template <bool SRGB, typename T>
static float convertComponentFloat(T c)
{
  if constexpr (std::is_same_v<T, float>)
    return c;
  else if constexpr (std::is_same_v<T, uint16_t>) {
    constexpr auto maxVal = float(std::numeric_limits<uint16_t>::max());
    return c / maxVal;
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    constexpr auto maxVal = float(std::numeric_limits<uint32_t>::max());
    return c / maxVal;
  } else if constexpr (SRGB) // uint8_t
    return glm::convertSRGBToLinear(vec1(float(c) / 255.f)).x;
  else // uint8_t, linear
    return c / 255.f;
}

template <int IN_NC /*num components*/,
    typename IN_COMP_T /*component type*/,
    bool SRGB = false>
static void transformToStagingBufferUint8(
    const Array &image, uint8_t *stagingBuffer)
{
  auto *begin = (const IN_COMP_T *)image.data();
  auto *end = begin + (image.totalSize() * IN_NC);
  size_t out = 0;
  std::for_each(begin, end, [&](const IN_COMP_T &c) {
    stagingBuffer[out++] = convertComponentUint8<SRGB>(c);
    if constexpr (IN_NC == 3) {
      if (out % 4 == 3)
        stagingBuffer[out++] = 255;
    }
  });
}

template <int IN_NC /*num components*/,
    typename IN_COMP_T /*component type*/,
    bool SRGB = false>
static void transformToStagingBufferFloat(
    const Array &image, float *stagingBuffer)
{
  auto *begin = (const IN_COMP_T *)image.data();
  auto *end = begin + (image.totalSize() * IN_NC);
  size_t out = 0;
  std::for_each(begin, end, [&](const IN_COMP_T &c) {
    stagingBuffer[out++] = convertComponentFloat<SRGB>(c);
    if constexpr (IN_NC == 3) {
      if (out % 4 == 3)
        stagingBuffer[out++] = 1.f;
    }
  });
}

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
  else
    return cudaAddressModeClamp;
}

void makeCudaArrayUint8(cudaArray_t &cuArray, const Array &array, uvec2 size)
{
  makeCudaArrayUint8(cuArray, array, uvec3(size, 1));
}

void makeCudaArrayUint8(cudaArray_t &cuArray, const Array &array, uvec3 size)
{
  const ANARIDataType format = array.elementType();
  auto nc = numANARIChannels(format);

  // Create CUDA texture //

  std::vector<uint8_t> stagingBuffer(array.totalSize() * 4);

  if (nc == 4) {
    if (isFloat(format))
      transformToStagingBufferUint8<4, float>(array, stagingBuffer.data());
    else if (isFixed32(format))
      transformToStagingBufferUint8<4, uint32_t>(array, stagingBuffer.data());
    else if (isFixed16(format))
      transformToStagingBufferUint8<4, uint16_t>(array, stagingBuffer.data());
    else if (isFixed8(format))
      transformToStagingBufferUint8<4, uint8_t>(array, stagingBuffer.data());
    else if (isSrgb8(format))
      transformToStagingBufferUint8<4, uint8_t, true>(
          array, stagingBuffer.data());
  } else if (nc == 3) {
    if (isFloat(format))
      transformToStagingBufferUint8<3, float>(array, stagingBuffer.data());
    else if (isFixed32(format))
      transformToStagingBufferUint8<3, uint32_t>(array, stagingBuffer.data());
    else if (isFixed16(format))
      transformToStagingBufferUint8<3, uint16_t>(array, stagingBuffer.data());
    else if (isFixed8(format))
      transformToStagingBufferUint8<3, uint8_t>(array, stagingBuffer.data());
    else if (isSrgb8(format))
      transformToStagingBufferUint8<3, uint8_t, true>(
          array, stagingBuffer.data());
  } else if (nc == 2) {
    if (isFloat(format))
      transformToStagingBufferUint8<2, float>(array, stagingBuffer.data());
    else if (isFixed32(format))
      transformToStagingBufferUint8<2, uint32_t>(array, stagingBuffer.data());
    else if (isFixed16(format))
      transformToStagingBufferUint8<2, uint16_t>(array, stagingBuffer.data());
    else if (isFixed8(format))
      transformToStagingBufferUint8<2, uint8_t>(array, stagingBuffer.data());
    else if (isSrgb8(format))
      transformToStagingBufferUint8<2, uint8_t, true>(
          array, stagingBuffer.data());
  } else if (nc == 1) {
    if (isFloat(format))
      transformToStagingBufferUint8<1, float>(array, stagingBuffer.data());
    else if (isFixed32(format))
      transformToStagingBufferUint8<1, uint32_t>(array, stagingBuffer.data());
    else if (isFixed16(format))
      transformToStagingBufferUint8<1, uint16_t>(array, stagingBuffer.data());
    else if (isFixed8(format))
      transformToStagingBufferUint8<1, uint8_t>(array, stagingBuffer.data());
    else if (isSrgb8(format))
      transformToStagingBufferUint8<1, uint8_t, true>(
          array, stagingBuffer.data());
  }

  if (nc == 3)
    nc = 4;

  if (!cuArray) {
    auto desc = cudaCreateChannelDesc(nc >= 1 ? 8 : 0,
        nc >= 2 ? 8 : 0,
        nc >= 3 ? 8 : 0,
        nc >= 4 ? 8 : 0,
        cudaChannelFormatKindUnsigned);

    cudaMalloc3DArray(&cuArray,
        &desc,
        make_cudaExtent(size.x, size.y, size.z <= 1 ? 0 : size.z));
  }

  cudaMemcpy3DParms p = {};
  p.dstArray = cuArray;
  p.srcPtr = make_cudaPitchedPtr(
      stagingBuffer.data(), size.x * nc * sizeof(uint8_t), size.x, size.y);
  p.srcPos = p.dstPos = make_cudaPos(0, 0, 0);
  p.extent = make_cudaExtent(size.x, size.y, size.z);
  p.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p);
}

void makeCudaArrayFloat(cudaArray_t &cuArray, const Array &array, uvec2 size)
{
  makeCudaArrayFloat(cuArray, array, uvec3(size, 1));
}

void makeCudaArrayFloat(cudaArray_t &cuArray, const Array &array, uvec3 size)
{
  const ANARIDataType format = array.elementType();
  auto nc = numANARIChannels(format);

  // Create CUDA texture //

  std::vector<float> stagingBuffer(array.totalSize() * 4);

  if (nc == 4) {
    if (isFloat(format))
      transformToStagingBufferFloat<4, float>(array, stagingBuffer.data());
    else if (isFixed32(format))
      transformToStagingBufferFloat<4, uint32_t>(array, stagingBuffer.data());
    else if (isFixed16(format))
      transformToStagingBufferFloat<4, uint16_t>(array, stagingBuffer.data());
    else if (isFixed8(format))
      transformToStagingBufferFloat<4, uint8_t>(array, stagingBuffer.data());
    else if (isSrgb8(format))
      transformToStagingBufferFloat<4, uint8_t, true>(
          array, stagingBuffer.data());
  } else if (nc == 3) {
    if (isFloat(format))
      transformToStagingBufferFloat<3, float>(array, stagingBuffer.data());
    else if (isFixed32(format))
      transformToStagingBufferFloat<3, uint32_t>(array, stagingBuffer.data());
    else if (isFixed16(format))
      transformToStagingBufferFloat<3, uint16_t>(array, stagingBuffer.data());
    else if (isFixed8(format))
      transformToStagingBufferFloat<3, uint8_t>(array, stagingBuffer.data());
    else if (isSrgb8(format))
      transformToStagingBufferFloat<3, uint8_t, true>(
          array, stagingBuffer.data());
  } else if (nc == 2) {
    if (isFloat(format))
      transformToStagingBufferFloat<2, float>(array, stagingBuffer.data());
    else if (isFixed32(format))
      transformToStagingBufferFloat<2, uint32_t>(array, stagingBuffer.data());
    else if (isFixed16(format))
      transformToStagingBufferFloat<2, uint16_t>(array, stagingBuffer.data());
    else if (isFixed8(format))
      transformToStagingBufferFloat<2, uint8_t>(array, stagingBuffer.data());
    else if (isSrgb8(format))
      transformToStagingBufferFloat<2, uint8_t, true>(
          array, stagingBuffer.data());
  } else if (nc == 1) {
    if (isFloat(format))
      transformToStagingBufferFloat<1, float>(array, stagingBuffer.data());
    else if (isFixed32(format))
      transformToStagingBufferFloat<1, uint32_t>(array, stagingBuffer.data());
    else if (isFixed16(format))
      transformToStagingBufferFloat<1, uint16_t>(array, stagingBuffer.data());
    else if (isFixed8(format))
      transformToStagingBufferFloat<1, uint8_t>(array, stagingBuffer.data());
    else if (isSrgb8(format))
      transformToStagingBufferFloat<1, uint8_t, true>(
          array, stagingBuffer.data());
  }

  if (nc == 3)
    nc = 4;

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
      stagingBuffer.data(), size.x * nc * sizeof(float), size.x, size.y);
  p.srcPos = p.dstPos = make_cudaPos(0, 0, 0);
  p.extent = make_cudaExtent(size.x, size.y, size.z);
  p.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&p);
}

cudaTextureObject_t makeCudaTextureObject(cudaArray_t cuArray,
    bool readModeNormalizedFloat,
    const std::string &filter,
    const std::string &wrap1,
    const std::string &wrap2,
    const std::string &wrap3,
    bool normalizedCoords)
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
    const cudaChannelFormatKind channelFormatKind)
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

} // namespace visrtx