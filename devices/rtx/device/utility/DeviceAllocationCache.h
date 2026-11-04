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
#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <mutex>

namespace visrtx {

// Device-local cache for backend-owned authored-array uploads only. Returning
// storage NEVER makes it reusable immediately. A frozen pending cohort becomes
// reusable only after deferred CPU submissions and all GPU consumers complete.
// Call allocate/reclaim in the owning device's CUDA context; scene updates and
// reclaim calls are serialized. No mutex is held across CUDA calls or
// callbacks.
struct DeviceAllocationCache
{
  static constexpr size_t MAX_BLOCKS = 256;
  static constexpr size_t MAX_BYTES = 64 * 1024 * 1024;
  explicit DeviceAllocationCache(
      size_t byteLimit = MAX_BYTES, size_t blockLimit = MAX_BLOCKS);
  ~DeviceAllocationCache();
  DeviceAllocationCache(const DeviceAllocationCache &) = delete;
  DeviceAllocationCache &operator=(const DeviceAllocationCache &) = delete;

  cudaError_t allocate(size_t bytes, void **pointer);
  void release(void *pointer, size_t bytes);
  template <typename CompleteDeferred>
  cudaError_t reclaim(CompleteDeferred completeDeferred);
  void close();

 private:
  struct Block
  {
    void *pointer{nullptr};
    size_t bytes{0};
  };
  bool trimReady();
  void finishCohort(
      const std::array<Block, MAX_BLOCKS> &cohort, size_t count, bool reusable);
  const size_t m_byteLimit;
  const size_t m_blockLimit;
  std::mutex m_mutex;
  bool m_enabled{true};
  // Counts include a frozen cohort while reclaim() is outside the lock.
  size_t m_cachedBytes{0};
  size_t m_cachedBlocks{0};
  size_t m_pendingCount{0};
  size_t m_readyCount{0};
  std::array<Block, MAX_BLOCKS> m_pending;
  std::array<Block, MAX_BLOCKS> m_ready;
};

inline DeviceAllocationCache::DeviceAllocationCache(
    size_t byteLimit, size_t blockLimit)
    : m_byteLimit(byteLimit), m_blockLimit(std::min(blockLimit, MAX_BLOCKS))
{}

inline DeviceAllocationCache::~DeviceAllocationCache()
{
  close();
}

inline cudaError_t DeviceAllocationCache::allocate(size_t bytes, void **pointer)
{
  *pointer = nullptr;
  if (!bytes)
    return cudaSuccess;
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (size_t i = 0; i < m_readyCount; ++i) {
      if (m_ready[i].bytes == bytes) {
        *pointer = m_ready[i].pointer;
        m_ready[i] = m_ready[--m_readyCount];
        m_cachedBytes -= bytes;
        --m_cachedBlocks;
        return cudaSuccess;
      }
    }
  }
  auto error = cudaMalloc(pointer, bytes);
  if (error == cudaErrorMemoryAllocation && trimReady()) {
    // Release only certified idle blocks. Pending blocks are not safe to hand
    // back without a completion boundary, and are never borrowed on failure.
    cudaGetLastError();
    *pointer = nullptr;
    error = cudaMalloc(pointer, bytes);
  }
  return error;
}

inline void DeviceAllocationCache::release(void *pointer, size_t bytes)
{
  if (!pointer)
    return;
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_enabled && m_cachedBlocks < m_blockLimit
        && bytes <= m_byteLimit - m_cachedBytes) {
      m_pending[m_pendingCount++] = {pointer, bytes};
      m_cachedBytes += bytes;
      ++m_cachedBlocks;
      return;
    }
  }
  // Overflow and late returns preserve ordinary cudaFree retirement semantics.
  cudaFree(pointer);
}

template <typename CompleteDeferred>
inline cudaError_t DeviceAllocationCache::reclaim(
    CompleteDeferred completeDeferred)
{
  std::array<Block, MAX_BLOCKS> cohort;
  size_t count = 0;
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    count = m_pendingCount;
    if (!count)
      return cudaSuccess;
    std::copy_n(m_pending.begin(), count, cohort.begin());
    m_pendingCount = 0;
  }
  bool reusable = false;
  // Restores the frozen cohort to pending on host exceptions too. This guard
  // borrows the cache; its deleter returns blocks rather than deleting this.
  auto restore = [&](DeviceAllocationCache *) {
    finishCohort(cohort, count, reusable);
  };
  std::unique_ptr<DeviceAllocationCache, decltype(restore)> guard(
      this, restore);
  auto error = completeDeferred();
  // Legacy consumers and application-visible array pointers need the same
  // conservative device-wide retirement cudaFree previously provided per block.
  // One boundary covers the cohort; later returns remain pending for next time.
  if (error == cudaSuccess)
    error = cudaDeviceSynchronize();
  reusable = error == cudaSuccess;
  return error;
}

inline void DeviceAllocationCache::finishCohort(
    const std::array<Block, MAX_BLOCKS> &cohort, size_t count, bool reusable)
{
  bool freeCohort = false;
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    freeCohort = !m_enabled;
    for (size_t i = 0; i < count; ++i) {
      const auto block = cohort[i];
      if (freeCohort) {
        m_cachedBytes -= block.bytes;
        --m_cachedBlocks;
      } else if (reusable)
        m_ready[m_readyCount++] = block;
      else
        m_pending[m_pendingCount++] = block;
    }
  }
  if (freeCohort)
    for (size_t i = 0; i < count; ++i)
      cudaFree(cohort[i].pointer);
}

inline bool DeviceAllocationCache::trimReady()
{
  std::array<Block, MAX_BLOCKS> blocks;
  size_t count = 0;
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    count = m_readyCount;
    std::copy_n(m_ready.begin(), count, blocks.begin());
    m_readyCount = 0;
    m_cachedBlocks -= count;
    for (size_t i = 0; i < count; ++i)
      m_cachedBytes -= blocks[i].bytes;
  }
  for (size_t i = 0; i < count; ++i)
    cudaFree(blocks[i].pointer);
  return count != 0;
}

inline void DeviceAllocationCache::close()
{
  std::array<Block, MAX_BLOCKS> blocks;
  size_t count = 0;
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_enabled = false;
    for (size_t i = 0; i < m_pendingCount; ++i)
      blocks[count++] = m_pending[i];
    for (size_t i = 0; i < m_readyCount; ++i)
      blocks[count++] = m_ready[i];
    m_pendingCount = m_readyCount = 0;
    m_cachedBlocks -= count;
    for (size_t i = 0; i < count; ++i)
      m_cachedBytes -= blocks[i].bytes;
  }
  for (size_t i = 0; i < count; ++i)
    cudaFree(blocks[i].pointer);
}

} // namespace visrtx
