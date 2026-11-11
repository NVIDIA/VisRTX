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

#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include "../../../device/geometry/PreparationResults.h"

namespace {
bool failDevice = false;
bool failHost = false;

template <typename T>
T next(const char *name)
{
  auto function = reinterpret_cast<T>(dlsym(RTLD_NEXT, name));
  if (!function)
    std::abort();
  return function;
}

using visrtx::GeometryResults;
using visrtx::PreparationResults;

// Writes one geometry's results as a kernel would.
bool write(PreparationResults &results,
    size_t position,
    float value,
    cudaStream_t stream)
{
  const GeometryResults written{
      {value, uint32_t(position)}, {-value, uint32_t(position) + 1}};
  return cudaMemcpyAsync(results.device(position),
             &written,
             sizeof(written),
             cudaMemcpyHostToDevice,
             stream)
      == cudaSuccess
      && cudaStreamSynchronize(stream) == cudaSuccess;
}

bool holds(const PreparationResults &results, size_t position, float value)
{
  const auto &read = results.host(position);
  return read.bounds.value == value && read.bounds.invalid == uint32_t(position)
      && read.area.value == -value
      && read.area.invalid == uint32_t(position) + 1;
}

// Reads back and waits, as finishGeometryPreparation() does.
bool readBackAndWait(
    PreparationResults &results, size_t count, cudaStream_t stream)
{
  return results.readback(count, stream) == cudaSuccess
      && cudaStreamSynchronize(stream) == cudaSuccess;
}
} // namespace

// Real failed allocations, so CUDA last-error state is set as in production.
extern "C" cudaError_t cudaMalloc(void **ptr, size_t bytes)
{
  if (failDevice) {
    failDevice = false;
    *ptr = nullptr;
    return next<decltype(&cudaMalloc)>("cudaMalloc")(ptr, size_t(-1));
  }
  return next<decltype(&cudaMalloc)>("cudaMalloc")(ptr, bytes);
}

extern "C" cudaError_t cudaMallocHost(void **ptr, size_t bytes)
{
  if (failHost) {
    failHost = false;
    *ptr = nullptr;
    return next<decltype(&cudaMallocHost)>("cudaMallocHost")(ptr, size_t(-1));
  }
  return next<decltype(&cudaMallocHost)>("cudaMallocHost")(ptr, bytes);
}

int main()
{
  constexpr size_t INITIAL = PreparationResults::MIN_CAPACITY;
  cudaStream_t stream{};
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)
    return 1;
  bool ok = true;
  {
    PreparationResults results;

    // Nothing reserved or written: readback transfers nothing.
    ok = readBackAndWait(results, 1, stream) && ok;

    // A failed first allocation of either side is reported, leaves no stale
    // error, and retries.
    for (bool *fail : {&failHost, &failDevice}) {
      *fail = true;
      ok = results.reserve(1, stream) != cudaSuccess && ok;
      ok = cudaGetLastError() == cudaSuccess && ok;
    }
    ok = results.reserve(1, stream) == cudaSuccess && ok;

    ok = write(results, 0, 1.f, stream) && ok;
    ok = readBackAndWait(results, 1, stream) && holds(results, 0, 1.f) && ok;

    // Failed growth keeps the existing storage and its results.
    ok = write(results, INITIAL - 1, 2.f, stream) && ok;
    for (bool *fail : {&failHost, &failDevice}) {
      *fail = true;
      ok = results.reserve(INITIAL + 1, stream) != cudaSuccess && ok;
      ok = cudaGetLastError() == cudaSuccess && ok;
    }
    ok = readBackAndWait(results, INITIAL, stream) && holds(results, 0, 1.f)
        && holds(results, INITIAL - 1, 2.f) && ok;

    // Growth keeps device results written before it, including at the last
    // position of the old capacity; never-written positions read back zero.
    ok = write(results, 0, 3.f, stream) && ok;
    ok = results.reserve(INITIAL + 1, stream) == cudaSuccess && ok;
    ok = write(results, INITIAL, 4.f, stream) && ok;
    ok = readBackAndWait(results, INITIAL + 1, stream) && ok;
    ok = holds(results, 0, 3.f) && holds(results, INITIAL - 1, 2.f)
        && holds(results, INITIAL, 4.f) && ok;
    const auto &unwritten = results.host(1);
    ok = unwritten.bounds.value == 0.f && unwritten.bounds.invalid == 0
        && unwritten.area.value == 0.f && unwritten.area.invalid == 0 && ok;

    // A rewrite after a readback reaches the host on the next readback, which
    // also transfers the untouched positions again.
    ok = write(results, 0, 5.f, stream) && ok;
    ok = readBackAndWait(results, INITIAL + 1, stream) && ok;
    ok = holds(results, 0, 5.f) && holds(results, INITIAL, 4.f) && ok;

    // Readback is limited to positions that have storage.
    ok = write(results, 0, 6.f, stream) && ok;
    ok = readBackAndWait(results, 4 * INITIAL, stream) && holds(results, 0, 6.f)
        && ok;
  }
  cudaStreamDestroy(stream);
  std::printf("preparation results %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
