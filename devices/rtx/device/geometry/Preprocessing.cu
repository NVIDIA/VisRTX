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

#include "Geometry.h"
#include "Preprocessing.h"
#include "gpu/intersectPrimitives.h"

#include <algorithm>
#include <cmath>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_scan.cuh>
#include <limits>
#include <utility>

namespace visrtx {
namespace {
using Result = PreparationResult;

VISRTX_DEVICE double primitiveArea(PrimitiveInput in, size_t i)
{
  double area;
  if (in.type == GeometryType::SPHERE) {
    const auto *indices = static_cast<const uint32_t *>(in.indices);
    const uint32_t v = indices ? indices[i] : uint32_t(i);
    const float r = glm::abs(in.radii ? in.radii[v] : in.radius);
    area = 4.0 * double(kPi) * double(r) * double(r);
  } else if (in.type == GeometryType::CONE
      || in.type == GeometryType::CYLINDER) {
    const auto *indices = static_cast<const uvec2 *>(in.indices);
    const uvec2 c = indices ? indices[i] : uvec2(2 * i, 2 * i + 1);
    const bool cone = in.type == GeometryType::CONE;
    const double r0 =
        double(glm::abs(in.radii ? in.radii[cone ? c.x : i] : in.radius));
    const double r1 = cone ? double(glm::abs(in.radii[c.y])) : r0;
    const double len = double(glm::length(in.vertices[c.y] - in.vertices[c.x]));
    const bool cap0 = in.caps ? in.caps[c.x] != 0 : in.capFlags & CAP_FIRST;
    const bool cap1 = in.caps ? in.caps[c.y] != 0 : in.capFlags & CAP_SECOND;
    area = 0.0;
    if (len > 0.0 && (cone || r0 > 0.0)) {
      area = cone
          ? double(kPi) * (r0 + r1) * sqrt(len * len + (r0 - r1) * (r0 - r1))
          : 2.0 * double(kPi) * r0 * len;
      if (cap0)
        area += double(kPi) * r0 * r0;
      if (cap1)
        area += double(kPi) * r1 * r1;
    }
  } else {
    const auto *indices = static_cast<const uvec3 *>(in.indices);
    const uvec3 t = indices ? indices[i] : uvec3(0, 1, 2) + uint32_t(3 * i);
    area = 0.5
        * glm::length(glm::cross(in.vertices[t.y] - in.vertices[t.x],
            in.vertices[t.z] - in.vertices[t.x]));
  }
  return area;
}

VISRTX_GLOBAL void primitiveAreas(
    PrimitiveInput in, double *areas, Result *result)
{
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= in.count)
    return;
  const double area = primitiveArea(in, i);
  if (!isfinite(area) || area < 0.0)
    atomicOr(&result->invalid, 1u);
  areas[i] = area;
}

VISRTX_GLOBAL void normalizeAreas(
    const double *prefix, size_t n, float *cdf, Result *result)
{
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const double total = prefix[n - 1];
  const float area = float(total);
  if (!isfinite(prefix[i]) || prefix[i] < 0.0 || !isfinite(area)
      || (total > 0.0 && area == 0.f))
    atomicOr(&result->invalid, 1u);
  cdf[i] = total > 0.0 ? (i + 1 == n ? 1.f : float(prefix[i] / total)) : 0.f;
  if (i == 0)
    result->value = area;
}
// Encode numeric failure in the scalar readback; cone/cylinder bounds have
// have a separate status readback for this generation.
VISRTX_GLOBAL void publishArea(Result *result)
{
  if (result->invalid)
    result->value = __int_as_float(0x7fffffff);
}

VISRTX_DEVICE box3 computeSDFPrimitiveAABB(const SDFPrimitive &g)
{
  const float disp = g.userParams.x; // displacement amplitude padding
  const float r0 = glm::max(0.f, g.r0);
  const float r1 = glm::max(0.f, g.r1);

  switch (static_cast<SDFType>(g.type)) {
  case SDFType::SPHERE:
  case SDFType::CUT_SPHERE:
    return box3(g.p0 - (r0 + disp), g.p0 + (r0 + disp));

  case SDFType::TORUS:
    return box3(g.p0 - (r0 + r1 + disp), g.p0 + (r0 + r1 + disp));

  case SDFType::ELLIPSOID:
    // p1 stores radii vector for the ellipsoid
    return box3(g.p0 - (g.p1 + disp), g.p0 + (g.p1 + disp));

  case SDFType::PILL:
  case SDFType::CONE_PILL:
  case SDFType::CONE_PILL_SIGMOID:
  case SDFType::CONE:
  case SDFType::VESICA:
  default: {
    const float maxR = glm::max(r0, r1) + disp;
    return box3(glm::min(g.p0, g.p1) - maxR, glm::max(g.p0, g.p1) + maxR);
  }
  }
}

VISRTX_DEVICE Result primitiveBound(PrimitiveInput in, size_t i, box3 &bound)
{
  Result result{};
  if (in.type == GeometryType::SDF) {
    const SDFPrimitive &g = in.sdf[i];
    box3 b = computeSDFPrimitiveAABB(g);
    float padding = in.noiseFactor * glm::max(0.f, g.r0) * 0.1f;
    if (g.numNeighbours > 0) {
      const float r0 = glm::max(0.f, g.r0);
      const float r1 = glm::max(0.f, g.r1 >= 0.f ? g.r1 : g.r0);
      padding +=
          glm::mix(glm::min(r0, r1), glm::max(r0, r1), in.blendLerpFactor)
          * in.blendFactor;
    }
    b.lower -= padding;
    b.upper += padding;
    bound = b;
    for (int axis = 0; axis < 3; ++axis) {
      if (!isfinite(b.lower[axis]) || !isfinite(b.upper[axis]))
        result.invalid = 1u;
    }
    return result;
  }
  const bool sphere = in.type == GeometryType::SPHERE;
  const auto *indices = static_cast<const uvec2 *>(in.indices);
  const auto *sphereIndices = static_cast<const uint32_t *>(in.indices);
  const size_t center = sphere ? (sphereIndices ? sphereIndices[i] : i) : 0;
  const uvec2 c = sphere ? uvec2(center)
      : indices          ? indices[i]
                         : uvec2(2 * i, 2 * i + 1);
  const bool cone = in.type == GeometryType::CONE;
  const float r = cone
      ? glm::max(glm::abs(in.radii[c.x]), glm::abs(in.radii[c.y]))
      : glm::abs(in.radii ? in.radii[sphere ? center : i] : in.radius);
  const vec3 v0 = in.vertices[c.x], v1 = in.vertices[c.y];
  const box3 b(glm::min(v0, v1) - r, glm::max(v0, v1) + r);
  bound = b;
  const vec3 m = glm::max(glm::abs(b.lower), glm::abs(b.upper));
  const float scale = glm::max(m.x, glm::max(m.y, m.z));
  if (!isfinite(r) || !isfinite(v0.x) || !isfinite(v0.y) || !isfinite(v0.z)
      || !isfinite(v1.x) || !isfinite(v1.y) || !isfinite(v1.z)
      || !isfinite(scale))
    result.invalid = 1u;
  else
    result.value = scale;
  return result;
}

VISRTX_GLOBAL void primitiveBounds(
    PrimitiveInput in, box3 *bounds, Result *result)
{
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= in.count)
    return;
  const auto summary = primitiveBound(in, i, bounds[i]);
  if (summary.invalid)
    atomicOr(&result->invalid, 1u);
  else
    atomicMax(reinterpret_cast<unsigned *>(&result->value),
        __float_as_uint(summary.value));
}

// A batch is one launch, with independent block-local reductions/scans. The
// general multi-block/CUB path remains available above MAX_PRIMITIVES.
VISRTX_GLOBAL void smallGeometry(const SmallGeometryJob *jobs)
{
  constexpr int threads = SmallGeometryBatch::MAX_PRIMITIVES;
  using FloatReduce = cub::BlockReduce<float, threads>;
  using StatusReduce = cub::BlockReduce<unsigned, threads>;
  using Scan = cub::BlockScan<double, threads>;
  __shared__ typename FloatReduce::TempStorage scaleStorage;
  __shared__ typename StatusReduce::TempStorage statusStorage;
  __shared__ typename Scan::TempStorage scanStorage;
  const auto &job = jobs[blockIdx.x];
  const unsigned i = threadIdx.x;
  // Write only the requested results: an earlier launch in this collection
  // may have written the other one.
  if (job.bounds) {
    Result local{};
    if (i < job.boundsInput.count)
      local = primitiveBound(job.boundsInput, i, job.bounds[i]);
    const float scale =
        FloatReduce(scaleStorage)
            .Reduce(local.value,
                [] __device__(float a, float b) { return glm::max(a, b); });
    const unsigned invalid = StatusReduce(statusStorage).Sum(local.invalid);
    if (i == 0)
      job.results->bounds = {scale, invalid};
    __syncthreads(); // statusStorage is reused for the area validation.
  }
  if (job.cdf) {
    const auto n = job.areaInput.count;
    const double area = i < n ? primitiveArea(job.areaInput, i) : 0.0;
    double prefix, total;
    Scan(scanStorage).InclusiveSum(area, prefix, total);
    const float value = float(total);
    const unsigned localInvalid = !isfinite(area) || area < 0.0
        || !isfinite(prefix) || prefix < 0.0 || !isfinite(value)
        || (total > 0.0 && value == 0.f);
    if (i < n)
      job.cdf[i] =
          total > 0.0 ? (i + 1 == n ? 1.f : float(prefix / total)) : 0.f;
    const unsigned invalid = StatusReduce(statusStorage).Sum(localInvalid);
    if (i == 0)
      job.results->area = {
          invalid ? __int_as_float(0x7fffffff) : value, invalid};
  }
}

VISRTX_GLOBAL void quadIndices(const uvec4 *indices, size_t n, uvec3 *out)
{
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const uvec4 q = indices ? indices[i] : uvec4(4 * i) + uvec4(0, 1, 2, 3);
  out[2 * i] = uvec3(q.x, q.y, q.w);
  out[2 * i + 1] = uvec3(q.z, q.w, q.y);
}

VISRTX_GLOBAL void curveIndices(size_t n, uint32_t *out)
{
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = uint32_t(i);
}

VISRTX_GLOBAL void curveRadii(size_t n, float radius, float *out)
{
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = radius;
}
} // namespace

bool generateQuadIndices(
    Geometry *g, const uvec4 *indices, size_t n, DeviceBuffer &out)
{
  if (!reservePreparation(g, out, 2 * n * sizeof(uvec3)))
    return false;
  if (n)
    quadIndices<<<(n + 255) / 256,
        256,
        0,
        g->deviceState()->preparation.stream()>>>(
        indices, n, out.ptrAs<uvec3>());
  return finishPreparation(g);
}

bool generateCurveIndices(Geometry *g, size_t n, DeviceBuffer &out)
{
  if (!reservePreparation(g, out, n * sizeof(uint32_t)))
    return false;
  if (n)
    curveIndices<<<(n + 255) / 256,
        256,
        0,
        g->deviceState()->preparation.stream()>>>(n, out.ptrAs<uint32_t>());
  return failPreparationOnError(g, cudaGetLastError(), "curve indices");
}

bool generateCurveRadii(Geometry *g, size_t n, float radius, DeviceBuffer &out)
{
  if (!reservePreparation(g, out, n * sizeof(float)))
    return false;
  if (n)
    curveRadii<<<(n + 255) / 256,
        256,
        0,
        g->deviceState()->preparation.stream()>>>(
        n, radius, out.ptrAs<float>());
  return failPreparationOnError(g, cudaGetLastError(), "curve radii");
}

void failPreparation(Geometry *g, cudaError_t error, const char *operation)
{
  g->markPreparationFailed();
  // A failed dispatch/readback can follow successful queued kernels. Drain
  // preparation before retrying or freeing scratch. Keep the render-stream
  // drain on this error path too: this helper also handles input retirement
  // and dependency failures, which may precede a successful producer edge.
  g->deviceState()->finishGeometryPreparation();
  cudaStreamSynchronize(g->deviceState()->stream);
  g->reportMessage(ANARI_SEVERITY_ERROR,
      "geometry preprocessing failed during %s: %s",
      operation,
      cudaGetErrorString(error));
}

bool failPreparationOnError(
    Geometry *g, cudaError_t error, const char *operation)
{
  if (error == cudaSuccess)
    return true;
  failPreparation(g, error, operation);
  return false;
}

bool reservePreparation(Geometry *g, DeviceBuffer &buffer, size_t bytes)
{
  return failPreparationOnError(
      g, buffer.tryReserve(bytes), "allocating device storage");
}

bool finishPreparation(Geometry *g)
{
  // Also establishes an explicit boundary for legacy default-stream consumers.
  return failPreparationOnError(g, cudaGetLastError(), "launch")
      && failPreparationOnError(g,
          g->deviceState()->finishGeometryPreparation(),
          "execution completion");
}

void SmallGeometryBatch::enqueue(Preprocessing *preprocessing)
{
  m_pending.push_back(preprocessing);
}

cudaError_t SmallGeometryBatch::launch(
    PreparationResults &results, cudaStream_t stream)
{
  cudaError_t error = cudaSuccess;
  if (!m_pending.empty()) {
    size_t positions = 0;
    for (auto *p : m_pending)
      positions = std::max(positions, p->m_position + 1);
    // No host allocations after submission: descriptors remain stable even
    // when an error requires a subsequent fallback drain.
    m_jobs.clear();
    m_jobs.reserve(m_pending.size());
    if (m_pending.size() > size_t(std::numeric_limits<int>::max()))
      error = cudaErrorInvalidValue;
    else
      error = results.reserve(positions, stream);
    if (error == cudaSuccess) {
      for (auto *p : m_pending) {
        m_jobs.push_back(p->m_small);
        m_jobs.back().results = results.device(p->m_position);
      }
      error = m_descriptors.enqueueUpload(m_jobs.data(), m_jobs.size(), stream);
    }
    if (error == cudaSuccess) {
      smallGeometry<<<unsigned(m_jobs.size()), MAX_PRIMITIVES, 0, stream>>>(
          m_descriptors.ptrAs<SmallGeometryJob>());
      error = cudaGetLastError();
    }
  }
  for (auto *p : m_pending) {
    p->m_smallQueued = false;
    p->m_small = {};
  }
  m_pending.clear();
  return error;
}

void Preprocessing::enqueueSmall(Geometry *g)
{
  if (!m_smallQueued) {
    g->deviceState()->smallGeometryBatch.enqueue(this);
    m_smallQueued = true;
  }
}

Preprocessing::~Preprocessing() = default;

void Preprocessing::submit(Geometry *g)
{
  m_position = g->deferPreparation();
  m_status = Status::Pending;
}

GeometryResults *Preprocessing::reserveResults(Geometry *g)
{
  auto *state = g->deviceState();
  if (!failPreparationOnError(g,
          state->preparationResults.reserve(
              m_position + 1, state->preparation.stream()),
          "allocating preparation results"))
    return nullptr;
  return state->preparationResults.device(m_position);
}

Preprocessing::Status Preprocessing::status() const
{
  return m_status;
}

float Preprocessing::epsilonScale() const
{
  return m_epsilonScale;
}

float Preprocessing::totalArea() const
{
  return m_totalArea;
}

const float *Preprocessing::areaCdf() const
{
  return m_totalArea > 0.f ? m_areaCdf.ptrAs<float>() : nullptr;
}

void Preprocessing::releaseAreas()
{
  m_areasSubmitted = false;
  m_totalArea = 0.f;
  m_areaCdf.reset();
}

bool Preprocessing::complete(Geometry *g)
{
  const bool bounds = std::exchange(m_boundsSubmitted, false);
  const bool areas = std::exchange(m_areasSubmitted, false);
  m_status = Status::Failed;
  if (g->preparationFailed())
    return false;
  if (!bounds && !areas) {
    m_status = Status::Ready;
    return true;
  }
  const auto &results = g->deviceState()->preparationResults.host(m_position);
  if (bounds && results.bounds.invalid)
    return failPreparationOnError(
        g, cudaErrorInvalidValue, "non-finite bounds");
  if (areas && !std::isfinite(results.area.value))
    return failPreparationOnError(
        g, cudaErrorInvalidValue, "non-publishable primitive area");
  if (bounds)
    m_epsilonScale = m_boundsHaveEpsilonScale ? results.bounds.value : 0.f;
  if (areas) {
    m_totalArea = results.area.value;
    if (m_totalArea == 0.f)
      m_areaCdf.reset();
  }
  m_status = Status::Ready;
  return true;
}

bool Preprocessing::bounds(
    Geometry *g, const PrimitiveInput &in, DeviceBuffer &aabbs)
{
  const auto stream = g->deviceState()->preparation.stream();
  m_epsilonScale = 0.f;
  submit(g);
  m_boundsSubmitted = true;
  m_boundsHaveEpsilonScale = in.type != GeometryType::SDF;
  if (in.count > 0 && in.count <= SmallGeometryBatch::MAX_PRIMITIVES) {
    if (!reservePreparation(g, aabbs, in.count * sizeof(box3)))
      return false;
    m_small.boundsInput = in;
    m_small.bounds = aabbs.ptrAs<box3>();
    enqueueSmall(g);
    return true;
  }
  if (!reservePreparation(g, aabbs, in.count * sizeof(box3)))
    return false;
  auto *results = reserveResults(g);
  auto *result = results ? &results->bounds : nullptr;
  if (!result
      || !failPreparationOnError(g,
          cudaMemsetAsync(result, 0, sizeof(Result), stream),
          "clearing bounds status"))
    return false;
  if (in.count)
    primitiveBounds<<<(in.count + 255) / 256, 256, 0, stream>>>(
        in, aabbs.ptrAs<box3>(), result);
  return failPreparationOnError(g, cudaGetLastError(), "bounds evaluation");
}

bool Preprocessing::areas(Geometry *g, const PrimitiveInput &in)
{
  m_totalArea = 0.f;
  if (in.count == 0) {
    m_areaCdf.reset();
    return true;
  }
  if (in.count > size_t(std::numeric_limits<int>::max()))
    return failPreparationOnError(g, cudaErrorInvalidValue, "primitive count");
  submit(g);
  m_areasSubmitted = true;
  if (in.count <= SmallGeometryBatch::MAX_PRIMITIVES) {
    if (!reservePreparation(g, m_areaCdf, in.count * sizeof(float)))
      return false;
    m_small.areaInput = in;
    m_small.cdf = m_areaCdf.ptrAs<float>();
    enqueueSmall(g);
    return true;
  }
  const auto stream = g->deviceState()->preparation.stream();
  size_t scratchBytes = 0;
  if (!reservePreparation(g, m_prefix, in.count * sizeof(double))
      || !reservePreparation(g, m_areaCdf, in.count * sizeof(float)))
    return false;
  auto *results = reserveResults(g);
  auto *result = results ? &results->area : nullptr;
  if (!result
      || !failPreparationOnError(g,
          cudaMemsetAsync(result, 0, sizeof(Result), stream),
          "clearing area status"))
    return false;
  auto *prefix = m_prefix.ptrAs<double>();
  primitiveAreas<<<(in.count + 255) / 256, 256, 0, stream>>>(
      in, prefix, result);
  if (!failPreparationOnError(g, cudaGetLastError(), "area evaluation")
      || !failPreparationOnError(g,
          cub::DeviceScan::InclusiveSum(
              nullptr, scratchBytes, prefix, prefix, int(in.count), stream),
          "scan workspace query")
      || !reservePreparation(g, m_scratch, scratchBytes)
      || !failPreparationOnError(g,
          cub::DeviceScan::InclusiveSum(m_scratch.ptr(),
              scratchBytes,
              prefix,
              prefix,
              int(in.count),
              stream),
          "area scan"))
    return false;
  normalizeAreas<<<(in.count + 255) / 256, 256, 0, stream>>>(
      prefix, in.count, m_areaCdf.ptrAs<float>(), result);
  if (!failPreparationOnError(g, cudaGetLastError(), "CDF normalization"))
    return false;
  publishArea<<<1, 1, 0, stream>>>(result);
  return failPreparationOnError(g, cudaGetLastError(), "area publication");
}
} // namespace visrtx
