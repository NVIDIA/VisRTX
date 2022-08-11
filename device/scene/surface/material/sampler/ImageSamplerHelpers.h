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

#pragma once

#include "Sampler.h"
// std
#include <array>

namespace visrtx {

template <int SIZE>
using texel_t = std::array<uint8_t, SIZE>;
using texel1 = texel_t<1>;
using texel2 = texel_t<2>;
using texel3 = texel_t<3>;
using texel4 = texel_t<4>;

// Helper functions ///////////////////////////////////////////////////////////

inline bool isFloat(ANARIDataType format)
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

inline int numANARIChannels(ANARIDataType format)
{
  switch (format) {
  case ANARI_UFIXED8_RGBA_SRGB:
  case ANARI_UFIXED8_VEC4:
  case ANARI_FLOAT32_VEC4:
    return 4;
  case ANARI_UFIXED8_RGB_SRGB:
  case ANARI_UFIXED8_VEC3:
  case ANARI_FLOAT16_VEC3:
  case ANARI_FLOAT32_VEC3:
    return 3;
  case ANARI_UFIXED8_VEC2:
  case ANARI_FLOAT16_VEC2:
  case ANARI_FLOAT32_VEC2:
    return 2;
  case ANARI_FLOAT32:
    return 1;
  default:
    break;
  }
  return 0;
}

inline int bytesPerChannel(ANARIDataType format)
{
  if (isFloat(format))
    return 4;
  else
    return 1;
}

inline int countCudaChannels(const cudaChannelFormatDesc &desc)
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

template <int SIZE, typename IN_VEC_T>
inline texel_t<SIZE> makeTexel(IN_VEC_T v)
{
  v *= 255;
  texel_t<SIZE> retval;
  auto *in = (float *)&v;
  for (int i = 0; i < SIZE; i++)
    retval[i] = uint8_t(in[i]);
  return retval;
}

inline cudaTextureAddressMode stringToAddressMode(const std::string &str)
{
  if (str == "repeat")
    return cudaAddressModeWrap;
  else if (str == "mirrorRepeat")
    return cudaAddressModeMirror;
  else
    return cudaAddressModeClamp;
}

} // namespace visrtx