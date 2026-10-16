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

#include <cuda_runtime.h>
#include <algorithm>
#include <anari/anari_cpp.hpp>
#include <cstdlib>
#include <vector>

// Capture device allocations so early PUBLIC release exercises array ownership.
// Ordinary inputs use their full range. Tests of logical slicing opt into
// padding explicitly; residency alone does not imply a nonzero range.
template <typename Container>
static void setTestArray(ANARIDevice device,
    anari::Geometry geometry,
    const char *name,
    const Container &values,
    bool onDevice,
    size_t offset = 0)
{
  using T = typename Container::value_type;
  std::vector<T> padded(values.size() + 2 * offset);
  std::copy(values.begin(), values.end(), padded.begin() + offset);
  anari::Array1D array;
  if (onDevice) {
    void *ptr = nullptr;
    if (cudaMalloc(&ptr, padded.size() * sizeof(T)) != cudaSuccess
        || cudaMemcpy(ptr,
               padded.data(),
               padded.size() * sizeof(T),
               cudaMemcpyHostToDevice)
            != cudaSuccess)
      std::exit(2);
    array = anariNewArray1D(
        device,
        ptr,
        [](const void *, const void *memory) {
          cudaFree(const_cast<void *>(memory));
        },
        nullptr,
        anari::ANARITypeFor<T>::value,
        padded.size());
  } else {
    array =
        anari::newArray1D(device, anari::ANARITypeFor<T>::value, padded.size());
    std::copy(padded.begin(), padded.end(), anari::map<T>(device, array));
    anari::unmap(device, array);
  }
  anari::setParameter(device, array, "begin", uint64_t(offset));
  anari::setParameter(device, array, "end", uint64_t(offset + values.size()));
  anari::commitParameters(device, array);
  anari::setAndReleaseParameter(device, geometry, name, array);
}
