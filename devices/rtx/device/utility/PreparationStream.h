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
// std
#include <cstddef>
#include <memory>

namespace visrtx {

/*
 * The device's scene-update stream: a nonblocking CUDA stream beside the
 * render stream. Geometry preparation, BVH builds and scene-table uploads
 * submit to it, ordered after the work they depend on by event rather than by
 * blocking the host. Calls are serialized by the device's scene-update path;
 * nothing here locks.
 */
struct PreparationStream
{
  PreparationStream() = default;
  ~PreparationStream();
  PreparationStream(const PreparationStream &) = delete;
  PreparationStream &operator=(const PreparationStream &) = delete;

  // Device lifetime. `renderStream` is the stream frames render on.
  cudaError_t initialize(cudaStream_t renderStream);
  void reset();
  cudaStream_t stream() const;
  cudaStream_t renderStream() const;

  // GPU-side ordering; neither blocks the host. The first asks CUDA whether
  // the render stream is idle, so no submission site has to announce itself,
  // and is skipped when it is: an event recorded on an idle stream is already
  // complete and orders nothing.
  cudaError_t orderAfterRender();
  cudaError_t orderAfter(cudaStream_t producer);
  // Waits for this stream's submitted work.
  cudaError_t sync();

  // Orders after rendering, copies, waits. All sources must remain stable
  // until this returns. A partial batch is drained on failure; callers
  // acknowledge dirty state only after success.
  template <typename... Arrays>
  cudaError_t upload(Arrays &...arrays);

  // Pinned host memory of at least `bytes` for a readback that completes
  // before its caller returns (the BVH build's), so one block serves every
  // caller. Allocated by the first call. Null with `error` set on failure.
  void *readback(size_t bytes, cudaError_t &error);

 private:
  cudaStream_t m_stream{nullptr};
  cudaStream_t m_renderStream{nullptr};
  struct PinnedDeleter
  {
    void operator()(void *memory) const
    {
      cudaFreeHost(memory);
    }
  };
  cudaEvent_t m_dependency{nullptr};
  std::unique_ptr<void, PinnedDeleter> m_readback;
  size_t m_readbackBytes{0};
};

// Inlined definitions ////////////////////////////////////////////////////////

inline PreparationStream::~PreparationStream()
{
  reset();
}

inline cudaError_t PreparationStream::initialize(cudaStream_t renderStream)
{
  m_renderStream = renderStream;
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

inline cudaStream_t PreparationStream::renderStream() const
{
  return m_renderStream;
}

inline cudaError_t PreparationStream::orderAfterRender()
{
  return cudaStreamQuery(m_renderStream) == cudaSuccess
      ? cudaSuccess
      : orderAfter(m_renderStream);
}

inline cudaError_t PreparationStream::orderAfter(cudaStream_t producer)
{
  auto error = cudaEventRecord(m_dependency, producer);
  if (error == cudaSuccess)
    error = cudaStreamWaitEvent(m_stream, m_dependency, 0);
  return error;
}

inline cudaError_t PreparationStream::sync()
{
  return cudaStreamSynchronize(m_stream);
}

template <typename... Arrays>
inline cudaError_t PreparationStream::upload(Arrays &...arrays)
{
  // Unlike orderAfterRender(), a registry or world-table upload records and
  // waits on the event even when the render stream is idle. Kept as it was:
  // dropping the wait there has not been decided.
  auto error = orderAfter(m_renderStream);
  auto enqueue = [&](auto &array) {
    if (error == cudaSuccess)
      error = array.enqueueUpload(m_stream);
  };
  (enqueue(arrays), ...);
  const auto completion = sync();
  if (error != cudaSuccess || completion != cudaSuccess)
    cudaGetLastError(); // Keep a recoverable API error out of the next batch.
  return error == cudaSuccess ? completion : error;
}

inline void *PreparationStream::readback(size_t bytes, cudaError_t &error)
{
  error = cudaSuccess;
  if (bytes > m_readbackBytes) {
    m_readback.reset();
    m_readbackBytes = 0;
    void *memory = nullptr;
    error = cudaMallocHost(&memory, bytes);
    if (error == cudaSuccess) {
      m_readback.reset(memory);
      m_readbackBytes = bytes;
    }
  }
  return m_readback.get();
}

} // namespace visrtx
