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
#include <atomic>
#include <cstdint>

namespace visrtx {

// Scene submission is serialized. Invalidate BEFORE submitting work on the
// bound stream. A successful retirement covers only the generation captured
// before its wait, never work submitted by a callback during that wait.
// Failure cleanup must still use an unconditional drain, not this cache.
struct StreamRetirement
{
  bool isRetired() const;
  void initialize(cudaStream_t stream);
  void invalidate();
  cudaError_t retire();

 private:
  cudaStream_t m_stream{nullptr};
  std::atomic<uint64_t> m_submitted{1};
  std::atomic<uint64_t> m_retired{0};
};

inline bool StreamRetirement::isRetired() const
{
  const auto generation = m_submitted.load(std::memory_order_acquire);
  return m_retired.load(std::memory_order_acquire) >= generation;
}

inline void StreamRetirement::initialize(cudaStream_t stream)
{
  m_stream = stream;
  m_submitted = 1;
  m_retired = 0;
}

inline void StreamRetirement::invalidate()
{
  m_submitted.fetch_add(1, std::memory_order_release);
}

inline cudaError_t StreamRetirement::retire()
{
  const auto generation = m_submitted.load(std::memory_order_acquire);
  if (m_retired.load(std::memory_order_acquire) >= generation)
    return cudaSuccess;
  const auto error = cudaStreamSynchronize(m_stream);
  if (error == cudaSuccess) {
    auto retired = m_retired.load(std::memory_order_relaxed);
    while (retired < generation
        && !m_retired.compare_exchange_weak(retired,
            generation,
            std::memory_order_release,
            std::memory_order_relaxed)) {
    }
  }
  return error;
}

} // namespace visrtx
