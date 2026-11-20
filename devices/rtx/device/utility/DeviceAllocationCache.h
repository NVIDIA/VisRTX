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
#include <mutex>
#include <nonstd/scope.hpp>

namespace visrtx {

// Device-local cache for backend-owned authored-array uploads only. Returning
// storage NEVER makes it reusable immediately: reclaim() freezes the blocks
// returned so far, and they become reusable only after deferred CPU
// submissions and all GPU consumers complete. Call allocate/reclaim in the
// owning device's CUDA context; scene updates and reclaim calls are
// serialized. No mutex is held across CUDA calls or callbacks.
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
  void *remove(size_t index);
  bool freeBlocks(bool readyOnly);
  const size_t m_byteLimit;
  const size_t m_blockLimit;
  std::mutex m_mutex;
  bool m_enabled{true};
  size_t m_cachedBytes{0};
  // m_blocks holds three runs: [0, m_readyEnd) is reusable, [m_readyEnd,
  // m_frozenEnd) is the cohort a reclaim() in progress froze, and
  // [m_frozenEnd, m_count) was returned since and waits for the next one.
  size_t m_readyEnd{0};
  size_t m_frozenEnd{0};
  size_t m_count{0};
  std::array<Block, MAX_BLOCKS> m_blocks;
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
    for (size_t i = 0; i < m_readyEnd; ++i) {
      if (m_blocks[i].bytes == bytes) {
        *pointer = remove(i);
        return cudaSuccess;
      }
    }
  }
  auto error = cudaMalloc(pointer, bytes);
  if (error == cudaErrorMemoryAllocation && freeBlocks(true)) {
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
    if (m_enabled && m_count < m_blockLimit
        && bytes <= m_byteLimit - m_cachedBytes) {
      m_blocks[m_count++] = {pointer, bytes};
      m_cachedBytes += bytes;
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
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_frozenEnd == m_count)
      return cudaSuccess;
    m_frozenEnd = m_count;
  }
  bool reusable = false;
  // Runs on host exceptions too: a cohort that was not certified idle goes
  // back to waiting.
  auto finish = nonstd::make_scope_exit([&] {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (reusable)
      m_readyEnd = m_frozenEnd;
    else
      m_frozenEnd = m_readyEnd;
  });
  auto error = completeDeferred();
  // These blocks used to be retired by cudaFree, which waits for the whole
  // context. One wait of the same reach covers the cohort without a list of
  // the streams that may have read it; later returns wait for the next
  // reclaim().
  if (error == cudaSuccess)
    error = cudaDeviceSynchronize();
  reusable = error == cudaSuccess;
  return error;
}

// Call with the mutex held. The hole is filled from the end of the block's own
// run, then from the end of each later run, so the three stay contiguous.
inline void *DeviceAllocationCache::remove(size_t index)
{
  const auto block = m_blocks[index];
  if (index < m_readyEnd) {
    m_blocks[index] = m_blocks[--m_readyEnd];
    index = m_readyEnd;
  }
  if (index < m_frozenEnd) {
    m_blocks[index] = m_blocks[--m_frozenEnd];
    index = m_frozenEnd;
  }
  m_blocks[index] = m_blocks[--m_count];
  m_cachedBytes -= block.bytes;
  return block.pointer;
}

// Frees the reusable blocks, or every block. Returns whether any was freed.
inline bool DeviceAllocationCache::freeBlocks(bool readyOnly)
{
  bool freed = false;
  for (;;) {
    void *pointer = nullptr;
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      if ((readyOnly ? m_readyEnd : m_count) == 0)
        return freed;
      pointer = remove(0);
    }
    cudaFree(pointer);
    freed = true;
  }
}

inline void DeviceAllocationCache::close()
{
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_enabled = false;
  }
  freeBlocks(false);
}

} // namespace visrtx
