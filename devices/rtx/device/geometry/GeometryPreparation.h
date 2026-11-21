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
#include "utility/PreparationQueue.h"
#include "utility/PreparationStream.h"
// helium
#include <helium/utility/IntrusivePtr.h>
#include <helium/utility/TimeStamp.h>
// std
#include <memory>
#include <unordered_map>

namespace visrtx {

struct Array;
struct Geometry;

/*
 * Owns everything that prepares geometry on the GPU: the preparation stream,
 * the two collection queues, the small-geometry batch, the scalar results and
 * the input allocation cache.
 *
 * One scene update is one transaction:
 *
 *   commit     Geometry::commitParameters() captures its dirty host arrays and
 *              queues itself (stageInputs).
 *   upload     The first finalizer uploads every queued geometry's inputs and
 *              waits once (uploadInputs). A finalizer whose arrays no commit
 *              captured queues and uploads them the same way.
 *   submit     Each finalizer queues bounds/area work; nothing is published
 *              (defer). Small geometries are held for one batched launch.
 *   collect    collect() launches the batch, reads all results back in one
 *              transfer, waits once, then publishes each geometry.
 *
 * A CUDA failure fails every queued geometry; a numeric failure fails its own.
 * Calls are serialized by the device's scene-update path; nothing here locks.
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
  // stream; call collect() first, while objects may still be released.
  cudaError_t initialize(cudaStream_t renderStream);
  void shutdown();

  // True while input uploads or results are being collected. Reentrant
  // property queries and rendering are refused meanwhile.
  bool busy() const;

  // Completes the transaction: publishes every queued geometry. Returns
  // false, doing nothing, when reentered from a collection.
  bool collect();

  // For a caller about to hand memory back to the application. The first
  // waits for a host-to-device input copy in flight; the second for rendering
  // and queued preparation that read an application-owned device array.
  cudaError_t waitForHostReaders();
  cudaError_t waitForDeviceReaders();

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

  // For geometry code. fail() marks the geometry failed, drains both streams
  // so no queued work still targets its storage, and reports. check() returns
  // true on cudaSuccess, otherwise fail()s. finish() waits for the geometry's
  // submitted work, failing it on error.
  void fail(Geometry *geometry, cudaError_t error, const char *operation);
  bool check(Geometry *geometry, cudaError_t error, const char *operation);
  bool finish(Geometry *geometry);

 private:
  friend struct Geometry;
  friend struct Preprocessing;

  // Geometry protocol, in transaction order.
  void stageInputs(Geometry *geometry);
  bool uploadInputs();
  cudaError_t stageArray(Array *array, helium::TimeStamp modified);
  // Queues once; returns the position that indexes m_results.
  size_t defer(Geometry *geometry);
  // Launches the small batch, reads back, waits. Fails all queued on error.
  cudaError_t flush();
  // The other render dependencies: a host-blocking retire (skipped when the
  // render stream is idle), for callers about to release memory that render
  // work may still read, and the event wait that Geometry::beginPreparation()
  // takes even on an idle render stream.
  cudaError_t retireRender();
  cudaError_t orderAfterRenderAlways();

  bool uploadingInputs() const;
  bool pending() const;
  void failInputUploads();

  cudaStream_t m_renderStream{nullptr};
  PreparationStream m_stream;
  std::shared_ptr<DeviceAllocationCache> m_inputAllocations{
      std::make_shared<DeviceAllocationCache>()};
  // Scalar results of queued geometry preparations, by queue position.
  PreparationResults m_results;
  SmallGeometryBatch m_smallBatch;
  std::unique_ptr<BvhReadback, PinnedDeleter> m_bvhReadback;
  std::unordered_map<const Array *, helium::TimeStamp> m_inputVersions;
  // Last, so queued geometries are released before the storage above.
  PreparationQueue<helium::IntrusivePtr<Geometry>> m_inputUploads;
  PreparationQueue<helium::IntrusivePtr<Geometry>> m_preparations;
};

// Inlined definitions ////////////////////////////////////////////////////////

template <typename... Arrays>
inline cudaError_t GeometryPreparation::upload(Arrays &...arrays)
{
  return m_stream.upload(m_renderStream, arrays...);
}

} // namespace visrtx
