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

#include "array/Array.h"
#include "utility/AnariTypeHelpers.h"
// std
#include <array>
#include <limits>
#include <type_traits>
// glm
#include <glm/gtc/color_space.hpp>

namespace visrtx {

template <int SIZE>
using texel_t = std::array<uint8_t, SIZE>;
using texel1 = texel_t<1>;
using texel2 = texel_t<2>;
using texel3 = texel_t<3>;
using texel4 = texel_t<4>;

template <int SIZE>
using byte_chunk_t = std::array<uint8_t, SIZE>;

int countCudaChannels(const cudaChannelFormatDesc &desc);
cudaTextureAddressMode stringToAddressMode(const std::string &str);

template <bool SRGB, typename T>
inline uint8_t convertComponent(T c)
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

template <int IN_NC /*num components*/,
    typename IN_COMP_T /*component type*/,
    bool SRGB = false>
inline void transformToStagingBuffer(Array &image, uint8_t *stagingBuffer)
{
  auto *begin = (IN_COMP_T *)image.data();
  auto *end = begin + (image.totalSize() * IN_NC);
  size_t out = 0;
  std::for_each(begin, end, [&](const IN_COMP_T &c) {
    stagingBuffer[out++] = convertComponent<SRGB>(c);
    if constexpr (IN_NC == 3) {
      if (out % 4 == 3)
        stagingBuffer[out++] = 255;
    }
  });
}

} // namespace visrtx