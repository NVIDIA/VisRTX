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

#include "PreparationResults.h"

#include <algorithm>
#include <cassert>
#include <limits>

namespace visrtx {
namespace {

// Returns 0 when `count` geometries cannot be addressed.
size_t grownCapacity(size_t current, size_t count)
{
  constexpr size_t LIMIT =
      std::numeric_limits<size_t>::max() / sizeof(GeometryResults);
  if (count > LIMIT)
    return 0;
  const size_t doubled = current <= LIMIT / 2 ? 2 * current : LIMIT;
  return std::max({PreparationResults::MIN_CAPACITY, doubled, count});
}

// Returns `error`, or the stream's completion error if `error` is success.
cudaError_t completeStream(cudaError_t error, cudaStream_t stream)
{
  const auto completion = cudaStreamSynchronize(stream);
  return error == cudaSuccess ? completion : error;
}

} // namespace

void PinnedDeleter::operator()(void *memory) const
{
  cudaFreeHost(memory);
}

void PreparationResults::DeviceDeleter::operator()(void *memory) const
{
  cudaFree(memory);
}

const GeometryResults &PreparationResults::host(size_t position) const
{
  assert(position < m_capacity);
  return m_host.get()[position];
}

cudaError_t PreparationResults::reserve(size_t count, cudaStream_t stream)
{
  if (count <= m_capacity)
    return cudaSuccess;
  const size_t capacity = grownCapacity(m_capacity, count);
  if (capacity == 0)
    return cudaErrorInvalidValue;
  const size_t bytes = capacity * sizeof(GeometryResults);
  void *memory = nullptr;
  auto error = cudaMallocHost(&memory, bytes);
  std::unique_ptr<GeometryResults, PinnedDeleter> host(
      static_cast<GeometryResults *>(memory));
  memory = nullptr;
  if (error == cudaSuccess)
    error = cudaMalloc(&memory, bytes);
  std::unique_ptr<GeometryResults, DeviceDeleter> device(
      static_cast<GeometryResults *>(memory));
  // Readback transfers every position below the queue length, including ones
  // no kernel has written yet: never transfer uninitialized bytes.
  if (error == cudaSuccess)
    error = cudaMemsetAsync(device.get(), 0, bytes, stream);
  if (error == cudaSuccess && m_device)
    error = cudaMemcpyAsync(device.get(),
        m_device.get(),
        m_capacity * sizeof(GeometryResults),
        cudaMemcpyDeviceToDevice,
        stream);
  // Earlier kernels may still write the old device storage, and the copy
  // reads it: free neither buffer before the stream completes. The first
  // allocation needs no wait; later kernels on `stream` follow its memset.
  // Every earlier readback completed in its own finish, and the next one
  // transfers the whole queued range, so the old host storage is free to go.
  if (device && (m_device || error != cudaSuccess))
    error = completeStream(error, stream);
  if (error != cudaSuccess) {
    cudaGetLastError(); // A failed growth must not poison a retry.
    return error;
  }
  m_device = std::move(device);
  m_host = std::move(host);
  m_capacity = capacity;
  return cudaSuccess;
}

GeometryResults *PreparationResults::device(size_t position)
{
  assert(position < m_capacity);
  m_unread = true;
  return m_device.get() + position;
}

cudaError_t PreparationResults::readback(size_t count, cudaStream_t stream)
{
  // Positions past the capacity were never written.
  count = std::min(count, m_capacity);
  if (!m_unread || count == 0)
    return cudaSuccess;
  m_unread = false;
  return cudaMemcpyAsync(m_host.get(),
      m_device.get(),
      count * sizeof(GeometryResults),
      cudaMemcpyDeviceToHost,
      stream);
}

} // namespace visrtx
