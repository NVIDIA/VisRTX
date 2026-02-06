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

#pragma once

#include "../gpu/gpu_decl.h"
// anari
#include <anari/anari.h>

namespace visrtx {

constexpr bool isFloat32(ANARIDataType format)
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

constexpr bool isFloat64(ANARIDataType format)
{
  switch (format) {
  case ANARI_FLOAT64_VEC4:
  case ANARI_FLOAT64_VEC3:
  case ANARI_FLOAT64_VEC2:
  case ANARI_FLOAT64:
    return true;
  default:
    break;
  }
  return false;
}

constexpr bool isFloat(ANARIDataType format)
{
  return isFloat32(format) || isFloat64(format);
}

constexpr bool isFixed32(ANARIDataType format)
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

constexpr bool isFixed16(ANARIDataType format)
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

constexpr bool isFixed8(ANARIDataType format)
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

constexpr bool isSrgb8(ANARIDataType format)
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

constexpr bool isColor(ANARIDataType t)
{
  return isFloat(t) || isFixed32(t) || isFixed16(t) || isFixed8(t)
      || isSrgb8(t);
}

constexpr int numANARIChannels(ANARIDataType format)
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

constexpr int bytesPerChannel(ANARIDataType format)
{
  if (isFloat(format) || isFixed32(format))
    return 4;
  else if (isFixed16(format))
    return 2;
  else
    return 1;
}
} // namespace visrtx