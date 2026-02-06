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

#include <stdint.h>

#include <array>
#include <memory>
#include <atomic>
#include <mutex>

#ifdef _WIN32
#include <intrin.h>
#endif

// This is an array of exponentially growing chunks
// Only addition of elements needs to be protected
// by a lock while elements remain accessible even
// during a resizing operation

template <typename T>
class ConcurrentArray
{
  static const uint64_t offset = 8u;

  std::atomic<uint64_t> elements{0};
  std::array<std::unique_ptr<T[]>, 64> blocks{};
  mutable std::mutex mutex;

  static uint64_t msb(uint64_t x)
  {
#ifdef _WIN64
    unsigned long index;
    _BitScanReverse64(&index, x);
    return index;
#elif defined(_WIN32)
    unsigned long index;
    if (x >> 32) {
      _BitScanReverse(&index, x >> 32);
      return index + 32;
    } else {
      _BitScanReverse(&index, x);
      return index;
    }
#else
    return UINT64_C(63) - __builtin_clzll(x);
#endif
  }
  static uint64_t block(uint64_t i)
  {
    return msb((i >> (offset - 1u)) | 1u);
  }
  static uint64_t index(uint64_t i, uint64_t b)
  {
    uint64_t saturated = b ? b - 1 : 0;
    uint64_t block_size = (1u << (saturated + offset));
    uint64_t mask = block_size - 1u;
    return i & mask;
  }

 public:
  uint64_t size() const
  {
    return elements;
  }
  T &operator[](uint64_t i)
  {
    uint64_t b = block(i);
    uint64_t idx = index(i, b);
    return blocks[b][idx];
  }
  const T &operator[](uint64_t i) const
  {
    uint64_t b = block(i);
    uint64_t idx = index(i, b);
    return blocks[b][idx];
  }
  uint64_t add()
  {
    uint64_t index = elements++;
    uint64_t b = block(index);

    std::lock_guard<std::mutex> guard(mutex);
    if (blocks[b] == nullptr) {
      uint64_t saturated = b ? b - 1 : 0;
      uint64_t block_size = (1u << (saturated + offset));
      blocks[b].reset(new T[block_size]);
    }
    return index;
  }
};
