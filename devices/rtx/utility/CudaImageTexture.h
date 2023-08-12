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

void makeCudaArrayUint8(cudaArray_t &cuArray, const Array &array, uvec2 size);
void makeCudaArrayFloat(cudaArray_t &cuArray, const Array &array, uvec2 size);

cudaTextureObject_t makeCudaTextureObject(cudaArray_t cuArray,
    bool readModeNormalizedFloat,
    const std::string &filter,
    const std::string &wrap1 = "clampToEdge",
    const std::string &wrap2 = "clampToEdge");

} // namespace visrtx