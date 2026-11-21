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

#include "GeometryPreparation.h"

namespace visrtx {

GeometryPreparation::~GeometryPreparation() = default;

cudaError_t GeometryPreparation::initialize(cudaStream_t renderStream)
{
  m_renderStream = renderStream;
  return m_stream.initialize();
}

void GeometryPreparation::shutdown()
{
  // Disable cached returns and free idle/pending blocks before destroying
  // streams. Any allocation outliving the device falls back to ordinary free.
  m_inputAllocations->close();
  m_stream.reset();
}

std::shared_ptr<DeviceAllocationCache> GeometryPreparation::inputAllocations()
    const
{
  return m_inputAllocations;
}

cudaStream_t GeometryPreparation::stream() const
{
  return m_stream.stream();
}

cudaError_t GeometryPreparation::orderAfterRender()
{
  return cudaStreamQuery(m_renderStream) == cudaSuccess
      ? cudaSuccess
      : m_stream.waitFor(m_renderStream);
}

cudaError_t GeometryPreparation::orderAfter(cudaStream_t producer)
{
  return m_stream.waitFor(producer);
}

cudaError_t GeometryPreparation::sync()
{
  return m_stream.finish();
}

GeometryPreparation::BvhReadback *GeometryPreparation::bvhReadback(
    cudaError_t &error)
{
  error = cudaSuccess;
  if (!m_bvhReadback) {
    void *memory = nullptr;
    error = cudaMallocHost(&memory, sizeof(BvhReadback));
    if (error == cudaSuccess)
      m_bvhReadback.reset(static_cast<BvhReadback *>(memory));
  }
  return m_bvhReadback.get();
}

cudaError_t GeometryPreparation::retireRender()
{
  // Anything but success, a failed query included, takes the wait: only a
  // completed stream permits releasing what its work reads.
  return cudaStreamQuery(m_renderStream) == cudaSuccess
      ? cudaSuccess
      : cudaStreamSynchronize(m_renderStream);
}

cudaError_t GeometryPreparation::orderAfterRenderAlways()
{
  return m_stream.waitFor(m_renderStream);
}

} // namespace visrtx
