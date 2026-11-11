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

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace visrtx {

struct PreparationResult
{
  float value;
  uint32_t invalid;
};

// One geometry's scalar preparation results.
struct GeometryResults
{
  PreparationResult bounds; // Epsilon scale and validity.
  PreparationResult area; // Total area; NaN when invalid.
};

// Frees pinned host memory from cudaMallocHost().
struct PinnedDeleter
{
  void operator()(void *memory) const;
};

// Scalar preparation results of the queued geometries, indexed by position in
// the device's preparation queue. Kernels write device storage; readback()
// copies the queued range to pinned host storage in one transfer. Device and
// host storage grow together to their high-water size and live until device
// teardown.
struct PreparationResults
{
  // Geometries; the first allocation holds this many.
  static constexpr size_t MIN_CAPACITY = 256;

  PreparationResults() = default;
  PreparationResults(const PreparationResults &) = delete;
  PreparationResults &operator=(const PreparationResults &) = delete;
  PreparationResults(PreparationResults &&) = delete;
  PreparationResults &operator=(PreparationResults &&) = delete;

  // Valid after a completed readback() covering `position`.
  const GeometryResults &host(size_t position) const;

  // Grows storage to hold `count` geometries, keeping earlier device results.
  // Growth after the first allocation synchronizes `stream`, the only stream
  // that writes results. Readback itself never allocates.
  cudaError_t reserve(size_t count, cudaStream_t stream);
  // Device results for a reserved position; marks them for the next readback.
  GeometryResults *device(size_t position);
  // Copies the first `count` geometries' results to the host if any were
  // written since the last readback. The caller synchronizes `stream` before
  // reading host() and before the next readback.
  cudaError_t readback(size_t count, cudaStream_t stream);

 private:
  struct DeviceDeleter
  {
    void operator()(void *memory) const;
  };
  std::unique_ptr<GeometryResults, DeviceDeleter> m_device;
  std::unique_ptr<GeometryResults, PinnedDeleter> m_host;
  size_t m_capacity{0}; // Geometries, on both sides.
  bool m_unread{false}; // Device results written since the last readback.
};

} // namespace visrtx
