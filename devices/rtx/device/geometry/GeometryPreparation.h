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

#include "Preprocessing.h"
#include "utility/DeviceAllocationCache.h"
#include "utility/PreparationStream.h"
// std
#include <memory>

namespace visrtx {

struct DeviceGlobalState;

/*
 * Owns what geometry preparation runs on: the preparation stream, the
 * small-geometry batch, the scalar results and the input allocation cache.
 */
struct GeometryPreparation
{
  struct BvhReadback
  {
    box3 bounds;
    uint64_t compactedSize;
  };

  GeometryPreparation() = default;
  ~GeometryPreparation();
  GeometryPreparation(const GeometryPreparation &) = delete;
  GeometryPreparation &operator=(const GeometryPreparation &) = delete;

  // Device lifetime. shutdown() closes the allocation cache and destroys the
  // stream.
  cudaError_t initialize(cudaStream_t renderStream);
  void shutdown();

  // Backs device copies of authored host arrays.
  std::shared_ptr<DeviceAllocationCache> inputAllocations() const;

  // Other work on the preparation stream (BVH builds, scene tables).
  cudaStream_t stream() const;
  // GPU-side ordering; neither blocks the host. The first asks CUDA whether
  // the render stream is idle, so no submission site has to announce itself,
  // and is skipped when it is.
  cudaError_t orderAfterRender();
  cudaError_t orderAfter(cudaStream_t producer);
  // Waits for the stream only; queued geometry results are left alone.
  cudaError_t sync();
  // Orders after rendering, copies, waits. Sources stay stable until return.
  template <typename... Arrays>
  cudaError_t upload(Arrays &...arrays);
  // Pinned; allocated by the first build. Builds are serialized and complete
  // before returning, so one serves them all. Null with `error` set on
  // failure.
  BvhReadback *bvhReadback(cudaError_t &error);

  // Unconditional forms: host-blocking retire, for callers about to release
  // memory that render work may still read, and the event wait that
  // Geometry::beginPreparation() takes even on an idle render stream.
  cudaError_t retireRender();
  cudaError_t orderAfterRenderAlways();

 private:
  friend struct DeviceGlobalState;
  friend struct Preprocessing;

  cudaStream_t m_renderStream{nullptr};
  PreparationStream m_stream;
  std::shared_ptr<DeviceAllocationCache> m_inputAllocations{
      std::make_shared<DeviceAllocationCache>()};
  // Scalar results of queued geometry preparations, by queue position.
  PreparationResults m_results;
  SmallGeometryBatch m_smallBatch;
  std::unique_ptr<BvhReadback, PinnedDeleter> m_bvhReadback;
};

// Inlined definitions ////////////////////////////////////////////////////////

template <typename... Arrays>
inline cudaError_t GeometryPreparation::upload(Arrays &...arrays)
{
  return m_stream.upload(m_renderStream, arrays...);
}

} // namespace visrtx
