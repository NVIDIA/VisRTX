/*
 * Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

  auto desc = cudaCreateChannelDesc(nc >= 1 ? 8 : 0,
      nc >= 2 ? 8 : 0,
      nc >= 3 ? 8 : 0,
      nc >= 3 ? 8 : 0,
      cudaChannelFormatKindUnsigned);

  if (!cuArray)
    cudaMallocArray(&cuArray, &desc, size.x, size.y);
  cudaMemcpy2DToArray(cuArray,
      0,
      0,
      stagingBuffer.data(),
      size.x * nc * sizeof(uint8_t),
      size.x * nc * sizeof(uint8_t),
      size.y,
      cudaMemcpyHostToDevice);
}

void makeCudaArrayFloat(cudaArray_t &cuArray, const Array &array, uvec2 size)
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

  auto desc = cudaCreateChannelDesc(nc >= 1 ? 32 : 0,
      nc >= 2 ? 32 : 0,
      nc >= 3 ? 32 : 0,
      nc >= 3 ? 32 : 0,
      cudaChannelFormatKindFloat);

  if (!cuArray)
    cudaMallocArray(&cuArray, &desc, size.x, size.y);
  cudaMemcpy2DToArray(cuArray,
      0,
      0,
      stagingBuffer.data(),
      size.x * nc * sizeof(float),
      size.x * nc * sizeof(float),
      size.y,
      cudaMemcpyHostToDevice);
}

cudaTextureObject_t makeCudaTextureObject(cudaArray_t cuArray,
    bool readModeNormalizedFloat,
    const std::string &filter,
    const std::string &wrap1,
    const std::string &wrap2)
{
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = stringToAddressMode(wrap1);
  texDesc.addressMode[1] = stringToAddressMode(wrap2);
  texDesc.filterMode =
      filter == "nearest" ? cudaFilterModePoint : cudaFilterModeLinear;
  texDesc.readMode = readModeNormalizedFloat ? cudaReadModeNormalizedFloat
                                             : cudaReadModeElementType;
  texDesc.normalizedCoords = 1;

  cudaTextureObject_t retval = {};
  cudaCreateTextureObject(&retval, &resDesc, &texDesc, nullptr);

  return retval;
}

} // namespace visrtx