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

#include <cmath>
#include <cub/device/device_scan.cuh>
#include <limits>

namespace visrtx {
namespace {
struct Result
{
  float value;
  uint32_t invalid;
};

VISRTX_GLOBAL void primitiveAreas(
    PrimitiveInput in, double *areas, Result *result)
{
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= in.count)
    return;
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
// already returned their status word for this generation.
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

VISRTX_GLOBAL void primitiveBounds(
    PrimitiveInput in, box3 *bounds, Result *result)
{
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= in.count)
    return;
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
    bounds[i] = b;
    for (int axis = 0; axis < 3; ++axis) {
      if (!isfinite(b.lower[axis]) || !isfinite(b.upper[axis]))
        atomicOr(&result->invalid, 1u);
    }
    return;
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
  bounds[i] = b;
  const vec3 m = glm::max(glm::abs(b.lower), glm::abs(b.upper));
  const float scale = glm::max(m.x, glm::max(m.y, m.z));
  if (!isfinite(r) || !isfinite(v0.x) || !isfinite(v0.y) || !isfinite(v0.z)
      || !isfinite(v1.x) || !isfinite(v1.y) || !isfinite(v1.z)
      || !isfinite(scale))
    atomicOr(&result->invalid, 1u);
  else
    atomicMax(
        reinterpret_cast<unsigned *>(&result->value), __float_as_uint(scale));
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
  return checkPreparation(g, cudaGetLastError(), "curve indices");
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
  return checkPreparation(g, cudaGetLastError(), "curve radii");
}

bool checkPreparation(Geometry *g, cudaError_t error, const char *operation)
{
  if (error == cudaSuccess)
    return true;
  g->setPreparationFailed(true);
  // A failed dispatch/readback can follow successful queued kernels. Drain
  // preparation before retrying or freeing scratch. Keep the render-stream
  // drain on this error path too: this helper also handles input retirement
  // and dependency failures, which may precede a successful producer edge.
  g->deviceState()->preparation.finish();
  cudaStreamSynchronize(g->deviceState()->stream);
  g->reportMessage(ANARI_SEVERITY_ERROR,
      "geometry preprocessing failed during %s: %s",
      operation,
      cudaGetErrorString(error));
  return false;
}

bool reservePreparation(Geometry *g, DeviceBuffer &buffer, size_t bytes)
{
  // Failed allocations can retain their requested capacity without storage.
  // Clear it so a later preparation can retry the same-sized allocation.
  if (!buffer)
    buffer.reset();
  buffer.reserve(bytes);
  // Consume the runtime error even when missing storage identifies allocation
  // failure. Otherwise the next successful allocation inherits this error.
  const auto error = cudaGetLastError();
  return checkPreparation(g,
      error != cudaSuccess ? error
          : bytes && (!buffer || buffer.bytes() < bytes)
          ? cudaErrorMemoryAllocation
          : cudaSuccess,
      "allocating device storage");
}

bool finishPreparation(Geometry *g)
{
  // Also establishes an explicit boundary for legacy default-stream consumers.
  return checkPreparation(g, cudaGetLastError(), "launch")
      && checkPreparation(
          g, g->deviceState()->preparation.finish(), "execution completion");
}

bool Preprocessing::bounds(
    Geometry *g, const PrimitiveInput &in, DeviceBuffer &aabbs, float &scale)
{
  const auto stream = g->deviceState()->preparation.stream();
  scale = 0.f;
  if (!reservePreparation(g, aabbs, in.count * sizeof(box3))
      || !reservePreparation(g, m_result, sizeof(Result))
      || !checkPreparation(g,
          cudaMemsetAsync(m_result.ptr(), 0, sizeof(Result), stream),
          "clearing bounds status"))
    return false;
  if (in.count)
    primitiveBounds<<<(in.count + 255) / 256, 256, 0, stream>>>(
        in, aabbs.ptrAs<box3>(), m_result.ptrAs<Result>());
  Result host{};
  const bool sdf = in.type == GeometryType::SDF;
  if (!checkPreparation(g, cudaGetLastError(), "bounds evaluation")
      || !checkPreparation(g,
          cudaMemcpyAsync(sdf ? static_cast<void *>(&host.invalid) : &host,
              sdf ? static_cast<void *>(&m_result.ptrAs<Result>()->invalid)
                  : m_result.ptr(),
              sdf ? sizeof(host.invalid) : sizeof(host),
              cudaMemcpyDeviceToHost,
              stream),
          "bounds summary")
      || !finishPreparation(g))
    return false;
  if (host.invalid)
    return checkPreparation(g, cudaErrorInvalidValue, "non-finite bounds");
  scale = host.value;
  return true;
}

bool Preprocessing::areas(
    Geometry *g, const PrimitiveInput &in, DeviceBuffer &cdf, float &total)
{
  total = 0.f;
  if (in.count == 0) {
    cdf.reset();
    return true;
  }
  if (in.count > size_t(std::numeric_limits<int>::max()))
    return checkPreparation(g, cudaErrorInvalidValue, "primitive count");
  const auto stream = g->deviceState()->preparation.stream();
  size_t scratchBytes = 0;
  if (!reservePreparation(g, m_prefix, in.count * sizeof(double))
      || !reservePreparation(g, cdf, in.count * sizeof(float))
      || !reservePreparation(g, m_result, sizeof(Result))
      || !checkPreparation(g,
          cudaMemsetAsync(m_result.ptr(), 0, sizeof(Result), stream),
          "clearing area status"))
    return false;
  auto *prefix = m_prefix.ptrAs<double>();
  auto *result = m_result.ptrAs<Result>();
  primitiveAreas<<<(in.count + 255) / 256, 256, 0, stream>>>(
      in, prefix, result);
  if (!checkPreparation(g, cudaGetLastError(), "area evaluation")
      || !checkPreparation(g,
          cub::DeviceScan::InclusiveSum(
              nullptr, scratchBytes, prefix, prefix, int(in.count), stream),
          "scan workspace query")
      || !reservePreparation(g, m_scratch, scratchBytes)
      || !checkPreparation(g,
          cub::DeviceScan::InclusiveSum(m_scratch.ptr(),
              scratchBytes,
              prefix,
              prefix,
              int(in.count),
              stream),
          "area scan"))
    return false;
  normalizeAreas<<<(in.count + 255) / 256, 256, 0, stream>>>(
      prefix, in.count, cdf.ptrAs<float>(), result);
  if (!checkPreparation(g, cudaGetLastError(), "CDF normalization"))
    return false;
  publishArea<<<1, 1, 0, stream>>>(result);
  Result host{};
  if (!checkPreparation(g, cudaGetLastError(), "area publication")
      || !checkPreparation(g,
          cudaMemcpyAsync(&host.value,
              &result->value,
              sizeof(host.value),
              cudaMemcpyDeviceToHost,
              stream),
          "area readback")
      || !finishPreparation(g))
    return false;
  if (!std::isfinite(host.value))
    return checkPreparation(
        g, cudaErrorInvalidValue, "non-publishable primitive area");
  total = host.value;
  if (total == 0.f)
    cdf.reset();
  return true;
}
} // namespace visrtx
