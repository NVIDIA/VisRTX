/*
 * Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ImageSamplerHelpers.h"

namespace visrtx {

bool isFloat(ANARIDataType format)
{
  switch (format) {
  case ANARI_FLOAT32_VEC4:
  case ANARI_FLOAT32_VEC3:
  case ANARI_FLOAT32_VEC2:
  case ANARI_FLOAT32:
    return true;
  default:
    break;
  }
  return false;
}

bool isFixed32(ANARIDataType format)
{
  switch (format) {
  case ANARI_UFIXED32_VEC4:
  case ANARI_UFIXED32_VEC3:
  case ANARI_UFIXED32_VEC2:
  case ANARI_UFIXED32:
    return true;
  default:
    break;
  }
  return false;
}

bool isFixed16(ANARIDataType format)
{
  switch (format) {
  case ANARI_UFIXED16_VEC4:
  case ANARI_UFIXED16_VEC3:
  case ANARI_UFIXED16_VEC2:
  case ANARI_UFIXED16:
    return true;
  default:
    break;
  }
  return false;
}

bool isFixed8(ANARIDataType format)
{
  switch (format) {
  case ANARI_UFIXED8_VEC4:
  case ANARI_UFIXED8_VEC3:
  case ANARI_UFIXED8_VEC2:
  case ANARI_UFIXED8:
    return true;
  default:
    break;
  }
  return false;
}

bool isSrgb8(ANARIDataType format)
{
  switch (format) {
  case ANARI_UFIXED8_RGBA_SRGB:
  case ANARI_UFIXED8_RGB_SRGB:
  case ANARI_UFIXED8_RA_SRGB:
  case ANARI_UFIXED8_R_SRGB:
    return true;
  default:
    break;
  }
  return false;
}

int numANARIChannels(ANARIDataType format)
{
  switch (format) {
  case ANARI_UFIXED8_RGBA_SRGB:
  case ANARI_UFIXED8_VEC4:
  case ANARI_UFIXED16_VEC4:
  case ANARI_UFIXED32_VEC4:
  case ANARI_FLOAT16_VEC4:
  case ANARI_FLOAT32_VEC4:
    return 4;
  case ANARI_UFIXED8_RGB_SRGB:
  case ANARI_UFIXED8_VEC3:
  case ANARI_UFIXED16_VEC3:
  case ANARI_UFIXED32_VEC3:
  case ANARI_FLOAT16_VEC3:
  case ANARI_FLOAT32_VEC3:
    return 3;
  case ANARI_UFIXED8_RA_SRGB:
  case ANARI_UFIXED8_VEC2:
  case ANARI_UFIXED16_VEC2:
  case ANARI_UFIXED32_VEC2:
  case ANARI_FLOAT16_VEC2:
  case ANARI_FLOAT32_VEC2:
    return 2;
  case ANARI_UFIXED8_R_SRGB:
  case ANARI_UFIXED8:
  case ANARI_UFIXED16:
  case ANARI_UFIXED32:
  case ANARI_FLOAT16:
  case ANARI_FLOAT32:
    return 1;
  default:
    break;
  }
  return 0;
}

int bytesPerChannel(ANARIDataType format)
{
  if (isFloat(format))
    return 4;
  else
    return 1;
}

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

} // namespace visrtx