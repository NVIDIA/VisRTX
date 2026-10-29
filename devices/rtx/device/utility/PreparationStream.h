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

namespace visrtx {

// Serialized by the device's scene-update path. GPU dependencies do not block
// the host; finish() is reserved for host readback/publication and scratch
// reuse.
struct PreparationStream
{
  PreparationStream() = default;
  ~PreparationStream();
  PreparationStream(const PreparationStream &) = delete;
  PreparationStream &operator=(const PreparationStream &) = delete;

  cudaError_t initialize();
  void reset();
  cudaStream_t stream() const;
  cudaError_t waitFor(cudaStream_t producer);
  cudaError_t finish();

  // All sources must remain stable until this returns. Drain a partial batch
  // on failure; callers acknowledge dirty state only after success.
  template <typename... Arrays>
  cudaError_t upload(cudaStream_t previousConsumers, Arrays &...arrays);

 private:
  cudaStream_t m_stream{nullptr};
  cudaEvent_t m_dependency{nullptr};
};

// Inlined definitions ////////////////////////////////////////////////////////

inline PreparationStream::~PreparationStream()
{
  reset();
}

inline cudaError_t PreparationStream::initialize()
{
  if (m_stream)
    return cudaSuccess;
  auto error = cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking);
  if (error == cudaSuccess)
    error = cudaEventCreateWithFlags(&m_dependency, cudaEventDisableTiming);
  if (error != cudaSuccess)
    reset();
  return error;
}

inline void PreparationStream::reset()
{
  if (m_stream)
    cudaStreamSynchronize(m_stream);
  if (m_dependency)
    cudaEventDestroy(m_dependency);
  if (m_stream)
    cudaStreamDestroy(m_stream);
  m_dependency = nullptr;
  m_stream = nullptr;
}

inline cudaStream_t PreparationStream::stream() const
{
  return m_stream;
}

inline cudaError_t PreparationStream::waitFor(cudaStream_t producer)
{
  auto error = cudaEventRecord(m_dependency, producer);
  if (error == cudaSuccess)
    error = cudaStreamWaitEvent(m_stream, m_dependency, 0);
  return error;
}

inline cudaError_t PreparationStream::finish()
{
  return cudaStreamSynchronize(m_stream);
}

template <typename... Arrays>
inline cudaError_t PreparationStream::upload(
    cudaStream_t previousConsumers, Arrays &...arrays)
{
  auto error = waitFor(previousConsumers);
  auto enqueue = [&](auto &array) {
    if (error == cudaSuccess)
      error = array.enqueueUpload(m_stream);
  };
  (enqueue(arrays), ...);
  const auto completion = finish();
  if (error != cudaSuccess || completion != cudaSuccess)
    cudaGetLastError(); // Keep a recoverable API error out of the next batch.
  return error == cudaSuccess ? completion : error;
}

} // namespace visrtx
