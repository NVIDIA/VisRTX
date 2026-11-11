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

#include <vector>
#include "ReadbackPool.h"
#include "gpu/gpu_objects.h"
#include "utility/DeviceBuffer.h"

namespace visrtx {
struct Geometry;

// GPU views use logical array beginnings, never application allocation bases.
// Geometry::beginPreparation() completes authored host-array uploads before
// these views are consumed; kernels need no legacy default-stream dependency.
struct PrimitiveInput
{
  GeometryType type{GeometryType::TRIANGLE};
  size_t count{0};
  const vec3 *vertices{nullptr};
  const void *indices{nullptr};
  const float *radii{nullptr};
  const uint8_t *caps{nullptr};
  float radius{0.f};
  uint8_t capFlags{0};
  const SDFPrimitive *sdf{nullptr};
  float blendFactor{1.f};
  float blendLerpFactor{0.5f};
  float noiseFactor{0.f};
};

// One block handles one small geometry. Bounds and area inputs are separate:
// a lazily requested sphere area distribution need not carry vertex positions.
struct SmallGeometryJob
{
  PrimitiveInput boundsInput;
  PrimitiveInput areaInput;
  box3 *bounds{nullptr};
  float *cdf{nullptr};
  PreparationResult *results{nullptr};
};

struct Preprocessing;

// Device-owned, single-flight batch. The geometry collection queue retains
// owners until finish() drains every queued descriptor, kernel and readback.
struct SmallGeometryBatch
{
  static constexpr size_t MAX_PRIMITIVES = 256;
  SmallGeometryBatch() = default;
  ~SmallGeometryBatch();
  SmallGeometryBatch(const SmallGeometryBatch &) = delete;
  SmallGeometryBatch &operator=(const SmallGeometryBatch &) = delete;
  void enqueue(Preprocessing *preprocessing);
  cudaError_t finish(cudaStream_t stream);

 private:
  std::vector<Preprocessing *> m_pending;
  std::vector<SmallGeometryJob> m_jobs;
  DeviceBuffer m_descriptors;
  DeviceBuffer m_results;
  PreparationResult *m_readback{nullptr};
  size_t m_readbackCapacity{0};
  cudaError_t reserveResults(size_t count);
};

// Per-geometry reusable workspace and the owner of its scalar results.
// Submission retains pinned readback storage; complete() publishes the epsilon
// scale, total area and area CDF after the device's collection barrier.
struct Preprocessing
{
  enum class Status
  {
    Ready, // Every submitted result is published.
    Pending, // Submitted results await complete().
    Failed, // The last collection failed; what it was to publish stays zero.
  };

  Preprocessing() = default;
  ~Preprocessing();
  Preprocessing(const Preprocessing &) = delete;
  Preprocessing &operator=(const Preprocessing &) = delete;
  Preprocessing(Preprocessing &&) = delete;
  Preprocessing &operator=(Preprocessing &&) = delete;

  Status status() const;
  // Published results. The CDF is null when the total area is zero.
  float epsilonScale() const;
  float totalArea() const;
  const float *areaCdf() const;

  // Success means queued, not published. Small jobs defer GPU submission to
  // the next preparation completion boundary. Submission zeroes the result it
  // replaces until complete() publishes the new one.
  bool areas(Geometry *geometry, const PrimitiveInput &input);
  bool bounds(
      Geometry *geometry, const PrimitiveInput &input, DeviceBuffer &aabbs);
  // Drops the published area distribution and its storage, and cancels the
  // publication of an area submission still awaiting complete().
  void releaseAreas();
  bool complete(Geometry *geometry);

 private:
  friend struct SmallGeometryBatch;
  bool reserveReadback(Geometry *geometry);
  void enqueueSmall(Geometry *geometry);
  void submit(Geometry *geometry);
  SmallGeometryJob m_small;
  bool m_smallQueued{false};
  GeometryReadbackPool::Lease m_readbackLease;
  PreparationResult *m_readback{nullptr};
  Status m_status{Status::Ready};
  bool m_boundsSubmitted{false};
  bool m_areasSubmitted{false};
  // SDF bounds are padded per primitive and carry no epsilon scale.
  bool m_boundsHaveEpsilonScale{false};
  float m_epsilonScale{0.f};
  float m_totalArea{0.f};
  DeviceBuffer m_areaCdf;
  DeviceBuffer m_prefix;
  DeviceBuffer m_scratch;
  DeviceBuffer m_result;
};

// Marks the geometry's preparation failed, drains the preparation and render
// streams so no queued work still targets its storage, and reports the error.
void failPreparation(
    Geometry *geometry, cudaError_t error, const char *operation);
// Returns true on cudaSuccess; otherwise calls failPreparation().
bool failPreparationOnError(
    Geometry *geometry, cudaError_t error, const char *operation);
bool reservePreparation(Geometry *geometry, DeviceBuffer &buffer, size_t bytes);
bool finishPreparation(Geometry *geometry);
bool generateQuadIndices(Geometry *geometry,
    const uvec4 *indices,
    size_t count,
    DeviceBuffer &output);
bool generateCurveIndices(
    Geometry *geometry, size_t count, DeviceBuffer &output);
bool generateCurveRadii(
    Geometry *geometry, size_t count, float radius, DeviceBuffer &output);

} // namespace visrtx
