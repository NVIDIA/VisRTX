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
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace visrtx {

struct PreparationResult
{
  float value;
  uint32_t invalid;
};

// A device-local high-water cache of pinned pages. Each lease owns two result
// slots and retains its page independently of the pool's lifetime. Returning a
// lease is legal only after its GPU writers have completed; geometry collection
// provides that ordering. Acquisition/release may run on different host
// threads.
struct GeometryReadbackPool
{
 private:
  struct Page;

 public:
  static constexpr size_t SLOTS_PER_PAGE = 256;

  struct Lease
  {
    Lease() = default;
    ~Lease();
    Lease(const Lease &) = delete;
    Lease &operator=(const Lease &) = delete;
    Lease(Lease &&other) noexcept;
    Lease &operator=(Lease &&other) noexcept;

    PreparationResult *data() const;
    void reset();

   private:
    friend struct GeometryReadbackPool;
    std::shared_ptr<Page> m_page;
    size_t m_index{0};
  };

  GeometryReadbackPool() = default;
  GeometryReadbackPool(const GeometryReadbackPool &) = delete;
  GeometryReadbackPool &operator=(const GeometryReadbackPool &) = delete;

  cudaError_t acquire(Lease &lease);

 private:
  std::mutex m_mutex;
  std::vector<std::shared_ptr<Page>> m_pages;
};

struct GeometryReadbackPool::Page
{
  Page();
  ~Page();
  bool take(size_t &index);
  void release(size_t index);

  PreparationResult *memory{nullptr};
  std::mutex mutex;
  std::array<size_t, SLOTS_PER_PAGE> freeSlots;
  size_t freeCount{SLOTS_PER_PAGE};
};

// Inlined definitions ////////////////////////////////////////////////////////

inline GeometryReadbackPool::Page::Page()
{
  for (size_t i = 0; i < SLOTS_PER_PAGE; ++i)
    freeSlots[i] = SLOTS_PER_PAGE - i - 1;
}

inline GeometryReadbackPool::Page::~Page()
{
  if (memory)
    cudaFreeHost(memory);
}

inline bool GeometryReadbackPool::Page::take(size_t &index)
{
  std::lock_guard<std::mutex> lock(mutex);
  if (freeCount == 0)
    return false;
  index = freeSlots[--freeCount];
  return true;
}

inline void GeometryReadbackPool::Page::release(size_t index)
{
  std::lock_guard<std::mutex> lock(mutex);
  assert(index < SLOTS_PER_PAGE && freeCount < SLOTS_PER_PAGE);
  freeSlots[freeCount++] = index;
}

inline GeometryReadbackPool::Lease::~Lease()
{
  reset();
}

inline GeometryReadbackPool::Lease::Lease(Lease &&other) noexcept
    : m_page(std::move(other.m_page)), m_index(other.m_index)
{}

inline GeometryReadbackPool::Lease &GeometryReadbackPool::Lease::operator=(
    Lease &&other) noexcept
{
  if (this != &other) {
    reset();
    m_page = std::move(other.m_page);
    m_index = other.m_index;
  }
  return *this;
}

inline PreparationResult *GeometryReadbackPool::Lease::data() const
{
  return m_page ? m_page->memory + 2 * m_index : nullptr;
}

inline void GeometryReadbackPool::Lease::reset()
{
  if (m_page) {
    m_page->release(m_index);
    m_page.reset();
  }
}

inline cudaError_t GeometryReadbackPool::acquire(Lease &lease)
{
  if (lease.data())
    return cudaSuccess;
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto &page : m_pages) {
      if (page->take(lease.m_index)) {
        lease.m_page = page;
        return cudaSuccess;
      }
    }
  }

  // CUDA allocation can wait for other work. Do not hold a pool/page mutex
  // across it; simultaneous growth may allocate an extra page, but is safe.
  auto page = std::make_shared<Page>();
  void *memory = nullptr;
  const auto error =
      cudaMallocHost(&memory, 2 * SLOTS_PER_PAGE * sizeof(PreparationResult));
  if (error != cudaSuccess) {
    cudaGetLastError(); // A failed page allocation must not poison a retry.
    return error;
  }
  page->memory = static_cast<PreparationResult *>(memory);
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_pages.push_back(page);
    const bool acquired = page->take(lease.m_index);
    assert(acquired);
    (void)acquired;
    lease.m_page = std::move(page);
  }
  return cudaSuccess;
}

} // namespace visrtx
