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

#include "geometry/IsosurfaceLinear.h"
#include "geometry/IsosurfaceSample.h"
#include "gpu/gpu_math.h"
#include "gpu/gridTraversal.h"
#include "gpu/intersectPrimitives.h"
#include "gpu/shading_api.h"
#include "spatial_field/CustomFieldSamplerInline.h"

// glm
#include <glm/gtx/norm.hpp>
#ifdef VISRTX_USE_NEURAL
#include <optix_types.h>

#include <cuda_fp16.h>
#endif

namespace visrtx {

// Helper functions ///////////////////////////////////////////////////////////

// Analytic-primitive hit kinds: facing in bit 0 so front/back survives into
// AH/CH (and optixIsFrontFaceHit's custom-primitive LSB convention matches).
// The CH side re-derives facing from the outward normal regardless
// (populateHit.h), so nothing correctness-critical rides on the OptiX LSB
// behavior.
constexpr uint32_t HIT_KIND_FRONT = 0u;
constexpr uint32_t HIT_KIND_BACK = 1u;

VISRTX_DEVICE void reportIntersection(
    float t, const vec3 &normal, float u, uint32_t hitKind = HIT_KIND_FRONT)
{
  optixReportIntersection(t,
      hitKind,
      bit_cast<uint32_t>(u),
      bit_cast<uint32_t>(normal.x),
      bit_cast<uint32_t>(normal.y),
      bit_cast<uint32_t>(normal.z));
}

// Report every boundary crossing of an analytic solid — entry (front-facing)
// and exit (back-facing) alike — and let OptiX keep the nearest in
// [tmin, tmax]. Back-facing crossings make interiors visible (camera inside a
// primitive, cut planes) and give transmission rays their exit event. The
// normal stays outward; the CH side orients it toward the ray and records
// facing (populateHit.h), mirroring the triangle convention. A secondary ray
// grazing its own origin surface can reach the surface's own exit crossing —
// the hit.epsilon origin offset + tmin guards against it, exactly as for
// triangles. rd is the (object-space) ray direction the facing is tested
// against.
VISRTX_DEVICE void reportCrossing(const PrimHit &h, const vec3 &rd)
{
  const bool isFront = dot(h.Ng, rd) < 0.f;
  reportIntersection(h.t, h.Ng, h.u, isFront ? HIT_KIND_FRONT : HIT_KIND_BACK);
}

VISRTX_DEVICE void reportIntersection(float t)
{
  optixReportIntersection(t, 0, bit_cast<uint32_t>(0.f));
}

VISRTX_DEVICE void reportIntersectionVolume(const box1 &t)
{
  const auto rd = optixGetObjectRayDirection();
  optixReportIntersection(t.lower,
      0,
      bit_cast<uint32_t>(t.upper),
      bit_cast<uint32_t>(rd.x),
      bit_cast<uint32_t>(rd.y),
      bit_cast<uint32_t>(rd.z));
}

VISRTX_DEVICE void reportIntersectionIsosurface(
    float t, const vec3 &normal, uint32_t isovalueIndex)
{
  optixReportIntersection(t,
      isovalueIndex, // hitKind: read in CH via optixGetHitKind()
      bit_cast<uint32_t>(0.f), // u (unused for isosurface)
      bit_cast<uint32_t>(normal.x),
      bit_cast<uint32_t>(normal.y),
      bit_cast<uint32_t>(normal.z));
}

// Primitive intersectors /////////////////////////////////////////////////////

// Per-endpoint cap enablement: vertex.cap array (element!=0 enables that
// endpoint's cap) overrides the geometry-wide default when present.
VISRTX_DEVICE uint8_t resolveCapBits(
    uint8_t defaultCapFlags, const uint8_t *vertexCaps, const uvec2 &pidx)
{
  if (!vertexCaps)
    return defaultCapFlags;
  uint8_t caps = 0;
  if (vertexCaps[pidx.x])
    caps |= CAP_FIRST;
  if (vertexCaps[pidx.y])
    caps |= CAP_SECOND;
  return caps;
}

VISRTX_DEVICE void intersectSphere(const GeometryGPUData &geometryData)
{
  const auto &sphereData = geometryData.sphere;

  const auto primID =
      sphereData.indices ? sphereData.indices[ray::primID()] : ray::primID();

  const auto center = sphereData.centers[primID];
  const auto radius =
      sphereData.radii ? sphereData.radii[primID] : sphereData.radius;

  const vec3 rd = ray::localDirection();
  forEachSphereCrossing(
      ray::localOrigin(), rd, center, radius, [&](const PrimHit &h) {
        reportCrossing(h, rd);
      });
}

VISRTX_DEVICE void intersectCylinder(const GeometryGPUData &geometryData)
{
  const auto &cylinderData = geometryData.cylinder;

  const uvec2 pidx = cylinderData.indices ? cylinderData.indices[ray::primID()]
                                          : (2 * ray::primID() + uvec2(0, 1));

  const auto p0 = cylinderData.vertices[pidx.x];
  const auto p1 = cylinderData.vertices[pidx.y];

  const float radius = cylinderData.radii ? cylinderData.radii[ray::primID()]
                                          : cylinderData.radius;
  const uint8_t caps = resolveCapBits(
      cylinderData.defaultCapFlags, cylinderData.vertexCaps, pidx);

  const vec3 rd = ray::localDirection();
  forEachCylinderCrossing(
      ray::localOrigin(), rd, p0, p1, radius, caps, [&](const PrimHit &h) {
        reportCrossing(h, rd);
      });
}

VISRTX_DEVICE void intersectCone(const GeometryGPUData &geometryData)
{
  const auto &coneData = geometryData.cone;

  const uvec2 pidx = coneData.indices ? coneData.indices[ray::primID()]
                                      : (2 * ray::primID() + uvec2(0, 1));

  const auto p0 = coneData.vertices[pidx.x];
  const auto p1 = coneData.vertices[pidx.y];

  const float r0 = coneData.radii[pidx.x];
  const float r1 = coneData.radii[pidx.y];

  const uint8_t caps =
      resolveCapBits(coneData.defaultCapFlags, coneData.vertexCaps, pidx);

  const vec3 rd = ray::localDirection();
  forEachConeCrossing(
      ray::localOrigin(), rd, p0, p1, r0, r1, caps, [&](const PrimHit &h) {
        reportCrossing(h, rd);
      });
}

VISRTX_DEVICE bool rayBoxIntersection(
    const vec3 &ro, const vec3 &rd, const box3 &bounds, float &t0, float &t1)
{
  // Per-axis slab test that handles axis-aligned rays (rd[a] == 0) explicitly,
  // instead of (lower - ro) * (1/0). The latter yields 0*inf = NaN when the ray
  // origin sits exactly on a finite box face, which min/max propagate so the
  // test wrongly fails — the source of the axis-aligned tile seams.
  constexpr float inf = std::numeric_limits<float>::infinity();
  t0 = -inf;
  t1 = inf;
  for (int a = 0; a < 3; ++a) {
    if (rd[a] != 0.f) {
      const float inv = 1.f / rd[a];
      const float ta = (bounds.lower[a] - ro[a]) * inv;
      const float tb = (bounds.upper[a] - ro[a]) * inv;
      t0 = max(t0, min(ta, tb));
      t1 = min(t1, max(ta, tb));
    } else if (ro[a] < bounds.lower[a] || ro[a] > bounds.upper[a]) {
      return false;
    }
  }
  return t0 < t1;
}

VISRTX_DEVICE void intersectVolume()
{
  const auto &ss = ray::screenSample();
  const auto &frameData = *ss.frameData;
  const auto &volumeData = ray::volumeData(frameData);

  if (box1 t; rayBoxIntersection(ray::localOrigin(),
          ray::localDirection(),
          volumeData.bounds,
          t.lower,
          t.upper)) {
    const auto &fieldData = ray::fieldData(frameData, volumeData);

    if (box1 tRoi; rayBoxIntersection(ray::localOrigin(),
            ray::localDirection(),
            fieldData.roi,
            tRoi.lower,
            tRoi.upper)) {
      t.lower = std::max(t.lower, tRoi.lower);
      t.upper = std::min(t.upper, tRoi.upper);

      t.lower = std::max(t.lower, ray::tmin());
      if (t.lower < t.upper) {
        reportIntersectionVolume(t);
      }
    }
  }
}

#ifdef VISRTX_USE_NEURAL

VISRTX_DEVICE __half relu(__half x)
{
  return __hmax(__float2half(0.0f), x);
}

VISRTX_DEVICE float __optix_enabled__forwardSDF(
    const NeuralGeometryData &data, const vec3 &p)
{
  // Convert input to half precision
  __half input[3] = {__float2half(p.x), __float2half(p.y), __float2half(p.z)};

  // Create OptixCoopVec for hidden layers
  using OCV = OptixCoopVec<__half, NEURAL_LAYER_SIZE>;
  using OCV_OUT = OptixCoopVec<__half, 1>; // For output layer
  OCV h1, h2;

  // First layer computation
  uint32_t layer = 0;
  for (uint32_t i = 0; i < NEURAL_LAYER_SIZE; ++i) {
    __half acc = data.biases[layer][i];
    for (uint32_t j = 0; j < 3; ++j) {
      acc = __hadd(acc, __hmul(data.weights[layer][i * 3 + j], input[j]));
    }
    h1[i] = relu(acc);
  }

  // Hidden layers computation using optixCoopVecMatMul
  OCV *hb = &h1;
  OCV *ha = &h2;

  for (uint32_t layer = 1; layer < data.nb_layers - 1; ++layer) {
    // Use optixCoopVecMatMul for matrix multiplication
    *ha = optixCoopVecMatMul<OCV, // VecTOut
        OCV, // VecTIn
        OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16, // inputInterpretation
        OPTIX_COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR, // matrixLayout
        false, // transpose
        NEURAL_LAYER_SIZE, // N
        NEURAL_LAYER_SIZE, // K
        OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16, // matrixElementType
        OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16>(*hb, // biasElementType
        (CUdeviceptr)data.weights[layer],
        0,
        (CUdeviceptr)data.biases[layer],
        0,
        NEURAL_LAYER_SIZE * sizeof(__half));

    // Apply ReLU activation to the output buffer
    for (uint32_t i = 0; i < NEURAL_LAYER_SIZE; ++i) {
      (*ha)[i] = relu((*ha)[i]);
    }

    // Swap buffers
    OCV *tmp = hb;
    hb = ha;
    ha = tmp;
  }

  // Output layer computation using optixCoopVecMatMul
  layer = data.nb_layers - 1;
  OCV_OUT output_vec = optixCoopVecMatMul<OCV_OUT, // VecTOut
      OCV, // VecTIn
      OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16, // inputInterpretation
      OPTIX_COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR, // matrixLayout
      false, // transpose
      1, // N
      NEURAL_LAYER_SIZE, // K
      OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16, // matrixElementType
      OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16>(*hb, // biasElementType
      (CUdeviceptr)data.weights[layer],
      0,
      (CUdeviceptr)data.biases[layer],
      0,
      NEURAL_LAYER_SIZE * sizeof(__half));

  return __half2float(output_vec[0]);
}

VISRTX_DEVICE void intersectNeural(const GeometryGPUData &geometryData)
{
  const auto &neuralData = geometryData.neural;
  const box3 bounds = {neuralData.boundMin, neuralData.boundMax};
  const vec3 &ro = ray::localOrigin();
  const vec3 &rd = ray::localDirection();
  float t0, t1;
  const bool intersection = rayBoxIntersection(ro, rd, bounds, t0, t1);
  const float threshold = neuralData.threshold;
  if (t0 > 0.f && t1 > 0.f && intersection) {
    float t = t0;
    while (t < t1) {
      const vec3 p = ro + t * rd;
      const float d = __optix_enabled__forwardSDF(neuralData, p);
      if (glm::abs(d) < threshold) {
        // Compute gradient in a single pass
        const float dxp = __optix_enabled__forwardSDF(
            neuralData, p + vec3(threshold, 0.f, 0.f));
        const float dyp = __optix_enabled__forwardSDF(
            neuralData, p + vec3(0.f, threshold, 0.f));
        const float dzp = __optix_enabled__forwardSDF(
            neuralData, p + vec3(0.f, 0.f, threshold));
        const vec3 normal = glm::normalize(vec3(dxp - d, dyp - d, dzp - d));
        reportIntersection(t, normal, 0.f);
        break;
      }
      t += d;
    }
  }
}
#endif

#include "gpu/gpu_sdf.h"

// Marches the [t0, t1] segment of one active macrocell on the global sample
// lattice (defined by `phase`) and reports the nearest isovalue crossing in it,
// returning true on a hit. Value-only march + 6-step bisection; the gradient
// (6 extra texture fetches for built-in fields) is taken once, at the refined
// hit. Specialized on the concrete sampler state — sampleValue/sampleNormal
// resolve by ADL — so the hot path stays monomorphic.
template <typename SamplerState>
VISRTX_DEVICE bool marchIsosurfaceSegment(const IsosurfaceGeometryData &iso,
    const SpatialFieldGPUData &field,
    const SamplerState &st,
    const vec3 &ro,
    const vec3 &rd,
    float t0,
    float t1,
    float phase,
    ScreenSample &ss)
{
  // iso.stepSize is an object-space distance; rd (the object ray direction) is
  // non-unit under instance scaling, so divide to get the step in ray-t units.
  const float step = iso.stepSize / length(rd);
  const float *const isovals = iso.isovalues;
  const uint32_t numIsovalues = iso.numIsovalues;

  // Snap onto the global lattice at/just below the segment entry so adjacent
  // cells share samples (seam-free) and the entry face is covered.
  float tStart = floorf((t0 - phase) / step) * step + phase;
  tStart = fmaxf(tStart, ray::tmin());

  const auto valueAt = [&](float t) {
    const vec3 p = ro + t * rd;
    return sampleValue(st, field, p);
  };

  float tPrev = tStart;
  float vPrev = valueAt(tStart);

  // Integer step index (not t += step) so float error can't drop or double the
  // final step near t1 over a long segment.
  const int nSteps = step > 0.f ? int(floorf((t1 - tStart) / step)) : 0;
  for (int i = 1; i <= nSteps; ++i) {
    const float t = tStart + float(i) * step;
    const float v = valueAt(t);

    // Find the nearest isovalue crossing within (tPrev, t]. The field is
    // monotone across one sub-voxel step, so the crossings order by their
    // linear-interpolated t -- pick the nearest from those estimates (no extra
    // samples) and refine only that one by bisection below, instead of
    // bisecting every bracketing isovalue (N root-finds -> 1).
    float bestEst = t1 + 1.f;
    float bestIv = 0.f;
    uint32_t bestIdx = 0;
    bool found = false;
    for (uint32_t i = 0; i < numIsovalues; ++i) {
      const float iv = isovals[i];
      const float fa = vPrev - iv;
      const float fb = v - iv;
      float est;
      if (fa == 0.f)
        est = tPrev; // exact crossing at the previous sample
      else if ((fa < 0.f) != (fb < 0.f))
        est = tPrev + (t - tPrev) * (fa / (fa - fb)); // sign change: fa != fb
      else
        continue;
      if (est < bestEst) {
        bestEst = est;
        bestIv = iv;
        bestIdx = i;
        found = true;
      }
    }

    if (found) {
      // Refine the nearest crossing's root in [tPrev, t] by value-only
      // bisection (skipped when it sits exactly on the previous sample).
      float bestT = bestEst;
      if (bestT > tPrev) {
        float la = tPrev, lb = t, va = vPrev - bestIv, tm = bestT;
        for (int k = 0; k < kIsosurfaceBisectionIters; ++k) {
          tm = 0.5f * (la + lb);
          const float fm = valueAt(tm) - bestIv;
          if ((va < 0.f) != (fm < 0.f)) {
            lb = tm;
          } else {
            la = tm;
            va = fm;
          }
        }
        bestT = tm;
      }

      // The field gradient points toward increasing value, which for most data
      // (e.g. denser tissue) points into the surface. A level set has no
      // inherent front/back, so orient the shading normal toward the incoming
      // ray; otherwise the camera-facing side shades dark. (Object-space flip;
      // correct for non-mirrored instance transforms.)
      const vec3 pHit = ro + bestT * rd;
      vec3 grad = sampleNormal(st, field, pHit);
      // Guard a (near-)zero gradient (field extremum/saddle): normalize(0) is
      // NaN. Fall back to facing the incoming ray.
      vec3 nrm;
      if (dot(grad, grad) < 1e-12f) {
        nrm = normalize(-rd);
      } else {
        if (dot(grad, rd) > 0.f)
          grad = -grad;
        nrm = normalize(grad);
      }
      reportIntersectionIsosurface(bestT, nrm, bestIdx);
      return true; // front-to-back: first bracketing step holds the nearest hit
    }

    tPrev = t;
    vPrev = v;
  }
  return false;
}

// Walks the field's macrocell grid front-to-back (3D-DDA), marching only cells
// whose value range brackets an isovalue, and reports the nearest crossing.
// Empty-space skipping happens here — inactive cells are stepped over with no
// sampling — so the geometry is a single field-bounds AABB rather than one
// primitive per active macrocell. This fixed-step march is now the fallback for
// fields without analytic voxel boundaries (custom fields); built-in grids use
// the per-voxel DDA (marchVoxelsMacrocellSkip) for both filters.
template <typename SamplerState>
VISRTX_DEVICE void marchIsosurface(const IsosurfaceGeometryData &iso,
    const SpatialFieldGPUData &field,
    const SamplerState &st,
    const Ray &objRay,
    ScreenSample &ss)
{
  // Object-space distance -> ray-t units (objRay.dir is non-unit under instance
  // scaling); phase and the exit overrun below both inherit these t units.
  const float step = iso.stepSize / length(objRay.dir);
  const float *const isovals = iso.isovalues;
  const uint32_t numIsovalues = iso.numIsovalues;

  // One sampling phase per (pixel, frame), shared by every cell the ray
  // crosses: a single continuous lattice (no per-cell restart -> no seams)
  // whose phase varies per frame so accumulation averages residual aliasing
  // into noise (the moiré fix).
  const auto &fb = ss.frameData->fb;
  const uint64_t pixelLinear =
      uint64_t(ss.pixel.y) * fb.size.x + uint64_t(ss.pixel.x);
  const uint64_t phaseHash = detail::pcg_mix64(
      (uint64_t(fb.frameID) << 32u) ^ pixelLinear ^ 0x9E3779B97F4A7C15ULL);
  const float phase = float(phaseHash >> 40) * (1.0f / 16777216.0f) * step;

  GridTraversal trav(objRay, field.grid.dims, field.grid.objectBounds);
  while (trav.valid()) {
    const box1 vr = field.grid.valueRanges[trav.cellIndex];
    bool active = vr.lower <= vr.upper;
    if (active) {
      active = false;
      for (uint32_t i = 0; i < numIsovalues; ++i) {
        if (isovals[i] >= vr.lower && isovals[i] <= vr.upper) {
          active = true;
          break;
        }
      }
    }
    if (active) {
      // Extend one step past the cell exit so a crossing near the high face is
      // caught even when the next cell is skipped (value ranges carry a one-
      // voxel margin). Cap at the true ray tmax, NOT objRay.t.upper: the latter
      // is clipped to this brick's AABB, so capping there zeroes the overrun at
      // every brick-exit face and drops a crossing in the last partial step —
      // a depth seam on the brick-boundary plane. Adjacent active cells/bricks
      // re-check the overlap harmlessly; front-to-back keeps the nearest hit.
      const float t1 = fminf(trav.tExit + step, ray::tmax());
      if (marchIsosurfaceSegment(iso,
              field,
              st,
              objRay.org,
              objRay.dir,
              trav.tEntry,
              t1,
              phase,
              ss))
        return;
    }
    trav.next();
  }
}

// Per-voxel crossing test shared by all four grid voxelDDASegment variants
// below — the only per-filter logic, factored out so the four overloads differ
// only in their traversal mechanics. st.filter is a per-field constant, so the
// branch is warp-uniform. Nearest = piecewise-constant face test (one midpoint
// sample; report the entry-face normal at tEntry). Linear = the
// trilinear-along- ray cubic (linearCrossing). Returns true and reports the hit
// on a crossing; otherwise carries the predecessor sample `vPrev` (voxel
// midpoint for nearest, exit-face value for linear) and returns false.
// crossedAxis is the face the ray entered this voxel through. Callers seed
// vPrev per-filter before the first voxel (nearest a quarter-voxel back; linear
// the exact entry-face value, since linearCrossing consumes it as the cubic's
// node-0).
template <typename SamplerState>
VISRTX_DEVICE bool crossingTest(const SamplerState &st,
    const SpatialFieldGPUData &field,
    const vec3 &ro,
    const vec3 &rd,
    float tEntry,
    float tExit,
    int crossedAxis,
    const float *isovals,
    uint32_t numIsovalues,
    float &vPrev,
    ScreenSample &ss)
{
  if (st.filter == SpatialFieldFilter::Nearest) {
    const float tMid = 0.5f * (tEntry + tExit);
    const float vCur = sampleValue(st, field, ro + tMid * rd);
    // Scan all isovalues (no divergent in-loop early-out; the uniform trip
    // count lets ptxas unroll). Every bracket shares this face's t, so report
    // the first.
    int hitIdx = -1;
    for (uint32_t i = 0; i < numIsovalues; ++i) {
      const float iv = isovals[i];
      if (hitIdx < 0 && ((vPrev < iv) != (vCur < iv)))
        hitIdx = int(i);
    }
    if (hitIdx >= 0) {
      vec3 n(0.f); // face the ray entered this voxel through
      n[crossedAxis] = 1.f;
      if (dot(n, rd) > 0.f)
        n = -n;
      reportIntersectionIsosurface(tEntry, n, uint32_t(hitIdx));
      return true; // front-to-back: first crossing is the nearest hit
    }
    vPrev = vCur;
  } else {
    // Exit-face value sampled only on the linear path (nearest keeps its single
    // midpoint sample). Carried as the next voxel's entry value -> one
    // fetch/voxel.
    const float vExit = sampleValue(st, field, ro + tExit * rd);
    float hitT;
    vec3 hitN;
    uint32_t hitIdx;
    if (linearCrossing(st,
            field,
            ro,
            rd,
            tEntry,
            tExit,
            vPrev,
            vExit,
            isovals,
            numIsovalues,
            hitT,
            hitN,
            hitIdx)) {
      reportIntersectionIsosurface(hitT, hitN, hitIdx);
      return true;
    }
    vPrev = vExit;
  }
  return false;
}

// Per-grid voxel-DDA over a [t0,t1] sub-segment, returning true on the nearest
// crossing within it. Each seeds the predecessor value just before t0 so a
// crossing on the segment's entry face is caught (callers pass a one-voxel
// margin past the segment so the shared face with the next macrocell is covered
// too). These are the narrow phase of the two-level traversal below.

// Structured-regular: piecewise-constant under nearest filtering, so the
// isosurface is axis-aligned voxel faces. Walk the voxel grid in sample-
// coordinate space (value boundaries are the integer planes); the DDA t is
// shared with the object ray, so a sign change across a face gives the crossing
// t AND the face axis directly.
VISRTX_DEVICE bool voxelDDASegment(const IsosurfaceGeometryData &iso,
    const SpatialFieldGPUData &field,
    const StructuredRegularSamplerState &st,
    const vec3 &ro,
    const vec3 &rd,
    float t0,
    float t1,
    ScreenSample &ss)
{
  const float *const isovals = iso.isovalues;
  const uint32_t numIsovalues = iso.numIsovalues;

  Ray cRay; // object ray in sample-coordinate space (shared t parameterization)
  cRay.org = (ro - st.origin) * st.invSpacing + st.offset;
  cRay.dir = rd * st.invSpacing;
  cRay.t.lower = t0;
  cRay.t.upper = t1;

  GridTraversal trav(cRay, st.dims, box3(vec3(0.f), vec3(st.dims)));
  if (!trav.valid())
    return false;

  const float firstSpan = trav.tExit - trav.tEntry;
  // Nearest seeds a quarter-voxel before entry (an entry-side value for the
  // face sign-test); linear needs the EXACT entry-face value — linearCrossing
  // consumes vPrev as vEntry == the cubic's node-0 (f0), so a pre-entry sample
  // would corrupt the first voxel's root in every active segment.
  float vPrev = st.filter == SpatialFieldFilter::Nearest
      ? sampleValue(st, field, ro + (trav.tEntry - 0.25f * firstSpan) * rd)
      : sampleValue(st, field, ro + trav.tEntry * rd);

  while (trav.valid()) {
    if (crossingTest(st,
            field,
            ro,
            rd,
            trav.tEntry,
            trav.tExit,
            trav.crossedAxis,
            isovals,
            numIsovalues,
            vPrev,
            ss))
      return true;
    trav.next();
  }
  return false;
}

// NanoVDB regular: affine world->index map, so the index-space ray is straight
// and shares the object t. Nearest fetch rounds the index (value boundaries at
// half-integer index planes), so shift the DDA grid bounds by -0.5. (Object
// normal uses the index axis directly — exact for axis-aligned grids; a rotated
// map would need the map's inverse-transpose, a follow-up.)
template <typename ValueType>
VISRTX_DEVICE bool voxelDDASegment(const IsosurfaceGeometryData &iso,
    const SpatialFieldGPUData &field,
    const NvdbRegularSamplerState<ValueType> &st,
    const vec3 &ro,
    const vec3 &rd,
    float t0,
    float t1,
    ScreenSample &ss)
{
  const float *const isovals = iso.isovalues;
  const uint32_t numIsovalues = iso.numIsovalues;

  const auto i0 = nvdbIndexPos(st, ro);
  const auto i1 = nvdbIndexPos(st, ro + rd);
  Ray idxRay;
  idxRay.org = vec3(i0[0], i0[1], i0[2]);
  idxRay.dir = vec3(i1[0] - i0[0], i1[1] - i0[1], i1[2] - i0[2]);
  idxRay.t.lower = t0;
  idxRay.t.upper = t1;

  const vec3 lo(
      st.indexMin[0] - 0.5f, st.indexMin[1] - 0.5f, st.indexMin[2] - 0.5f);
  const vec3 hi(
      st.indexMax[0] + 0.5f, st.indexMax[1] + 0.5f, st.indexMax[2] + 0.5f);
  const ivec3 dims(int(st.indexMax[0] - st.indexMin[0]) + 1,
      int(st.indexMax[1] - st.indexMin[1]) + 1,
      int(st.indexMax[2] - st.indexMin[2]) + 1);

  GridTraversal trav(idxRay, dims, box3(lo, hi));
  if (!trav.valid())
    return false;

  const float firstSpan = trav.tExit - trav.tEntry;
  // Nearest seeds a quarter-voxel before entry (an entry-side value for the
  // face sign-test); linear needs the EXACT entry-face value — linearCrossing
  // consumes vPrev as vEntry == the cubic's node-0 (f0), so a pre-entry sample
  // would corrupt the first voxel's root in every active segment.
  float vPrev = st.filter == SpatialFieldFilter::Nearest
      ? sampleValue(st, field, ro + (trav.tEntry - 0.25f * firstSpan) * rd)
      : sampleValue(st, field, ro + trav.tEntry * rd);

  while (trav.valid()) {
    if (crossingTest(st,
            field,
            ro,
            rd,
            trav.tEntry,
            trav.tExit,
            trav.crossedAxis,
            isovals,
            numIsovalues,
            vPrev,
            ss))
      return true;
    trav.next();
  }
  return false;
}

// Structured rectilinear: uniform voxel grid in sample (texcoord) space, warped
// per-axis into object space. Non-uniform DDA whose integer-texcoord boundary
// positions come straight from the inverse LUT (index->object). Value
// boundaries are at integer texcoord (texel-floor nearest); the per-axis warp
// keeps face normals axis-aligned in object space.
VISRTX_DEVICE bool voxelDDASegment(const IsosurfaceGeometryData &iso,
    const SpatialFieldGPUData &field,
    const StructuredRectilinearSamplerState &st,
    const vec3 &ro,
    const vec3 &rd,
    float t0,
    float t1,
    ScreenSample &ss)
{
  const float *const isovals = iso.isovalues;
  const uint32_t numIsovalues = iso.numIsovalues;
  const vec3 ext = st.axisBoundsMax - st.axisBoundsMin;

  const auto boundaryPos = [&](int a, int m) {
    const float ni = (float(m) - st.offset[a]) / st.dims[a];
    return st.axisBoundsMin[a] + tex1D<float>(st.invAxisLUT[a], ni) * ext[a];
  };

  float t = fmaxf(t0, 0.f);
  const vec3 c = structuredRectilinearCoord(st, ro + t * rd);
  ivec3 vi(int(floorf(c.x)), int(floorf(c.y)), int(floorf(c.z)));
  const ivec3 stepv(
      rd.x > 0.f ? 1 : -1, rd.y > 0.f ? 1 : -1, rd.z > 0.f ? 1 : -1);

  float vPrev; // predecessor sample, seeded voxel-relative on the first step
  bool seeded = false;
  // First-voxel entry face: the axis whose entry boundary the ray crossed most
  // recently (largest entry-t <= t). GridTraversal derives this from its slab
  // test for the uniform grids; the hand-rolled DDA must compute it, else the
  // first voxel always reports an x-face normal.
  int crossedAxis = 0;
  {
    float tEnter = -1e30f;
    const float epsE = 1e-6f * fmaxf(1.f, fabsf(t));
    for (int a = 0; a < 3; ++a) {
      if (rd[a] == 0.f)
        continue;
      const int eb = rd[a] > 0.f ? vi[a] : vi[a] + 1; // boundary behind the ray
      const float te = (boundaryPos(a, eb) - ro[a]) / rd[a];
      if (te <= t + epsE && te > tEnter) {
        tEnter = te;
        crossedAxis = a;
      }
    }
  }

  constexpr int kMaxVoxelSteps = 4096; // hard cap: never hang the intersector
  for (int s = 0; s < kMaxVoxelSteps && t < t1; ++s) {
    float tExit = t1;
    int exitAxis =
        -1; // set below to the exit-boundary axis; -1 => none before t1
    // Scale-relative advance guard: a fixed 1e-6 falls below ULP(t) once t
    // exceeds ~10 (e.g. coords in the tens), collapsing the test to te > t and
    // risking a re-picked boundary / stalled segment in larger scenes.
    const float tEps = 1e-6f * fmaxf(1.f, fabsf(t));
    for (int a = 0; a < 3; ++a) {
      if (rd[a] == 0.f)
        continue;
      const int nb = rd[a] > 0.f ? vi[a] + 1 : vi[a];
      const float te = (boundaryPos(a, nb) - ro[a]) / rd[a];
      if (te > t + tEps && te < tExit) {
        tExit = te;
        exitAxis = a;
      }
    }
    const float tExitSeg = fminf(tExit, t1);
    if (!seeded) {
      // Seed the predecessor a quarter of the entry voxel back (voxel-relative,
      // as the uniform grids do) so an entry-face crossing is caught. A
      // segment-relative backstep collapses to t under fp rounding on long
      // segments, missing it.
      // See the uniform-grid seed note: linear needs the exact entry-face value
      // (cubic node-0), nearest the quarter-voxel-back entry-side value.
      vPrev = st.filter == SpatialFieldFilter::Nearest
          ? sampleValue(st, field, ro + (t - 0.25f * (tExitSeg - t)) * rd)
          : sampleValue(st, field, ro + t * rd);
      seeded = true;
    }
    if (crossingTest(st,
            field,
            ro,
            rd,
            t,
            tExitSeg,
            crossedAxis,
            isovals,
            numIsovalues,
            vPrev,
            ss))
      return true;
    if (exitAxis < 0) // no forward boundary before t1: the segment ends here
      break;
    crossedAxis = exitAxis;
    t = tExit;
    vi[exitAxis] += stepv[exitAxis];
  }
  return false;
}

// NanoVDB rectilinear: as structured rectilinear, but boundary world positions
// are inverse-LUT (rect index -> uniform index) composed with the precomputed
// uniform-index -> world map. Value boundaries at rect-index half-integers.
template <typename ValueType>
VISRTX_DEVICE bool voxelDDASegment(const IsosurfaceGeometryData &iso,
    const SpatialFieldGPUData &field,
    const NvdbRectilinearSamplerState<ValueType> &st,
    const vec3 &ro,
    const vec3 &rd,
    float t0,
    float t1,
    ScreenSample &ss)
{
  const float *const isovals = iso.isovalues;
  const uint32_t numIsovalues = iso.numIsovalues;

  const auto boundaryPos = [&](int a, float rb) {
    const float nr = (rb - st.offsetUp[a]) / st.scaleUp[a];
    const float nu = tex1D<float>(st.invAxisLUT[a], nr);
    const float ip = nu / st.scaleDown[a] + st.offsetDown[a]; // uniform index
    return st.worldOrigin[a] + ip * st.worldVoxelStep[a];
  };

  float t = fmaxf(t0, 0.f);
  const vec3 pEntry = ro + t * rd;
  const auto idxEntry = worldToIndexRectilinear(st, &pEntry);
  ivec3 vi(int(roundf(idxEntry[0])),
      int(roundf(idxEntry[1])),
      int(roundf(idxEntry[2])));
  const ivec3 stepv(
      rd.x > 0.f ? 1 : -1, rd.y > 0.f ? 1 : -1, rd.z > 0.f ? 1 : -1);

  float vPrev; // predecessor sample, seeded voxel-relative on the first step
  bool seeded = false;
  // First-voxel entry face: axis whose entry boundary the ray crossed most
  // recently (largest entry-t <= t), as the uniform GridTraversal would derive
  // it — otherwise the first voxel always reports an x-face normal.
  int crossedAxis = 0;
  {
    float tEnter = -1e30f;
    const float epsE = 1e-6f * fmaxf(1.f, fabsf(t));
    for (int a = 0; a < 3; ++a) {
      if (rd[a] == 0.f)
        continue;
      const float rb = rd[a] > 0.f ? float(vi[a]) - 0.5f : float(vi[a]) + 0.5f;
      const float te = (boundaryPos(a, rb) - ro[a]) / rd[a];
      if (te <= t + epsE && te > tEnter) {
        tEnter = te;
        crossedAxis = a;
      }
    }
  }

  constexpr int kMaxVoxelSteps = 4096; // hard cap: never hang the intersector
  for (int s = 0; s < kMaxVoxelSteps && t < t1; ++s) {
    float tExit = t1;
    int exitAxis =
        -1; // set below to the exit-boundary axis; -1 => none before t1
    // Scale-relative advance guard: a fixed 1e-6 falls below ULP(t) once t
    // exceeds ~10 (e.g. coords in the tens), collapsing the test to te > t and
    // risking a re-picked boundary / stalled segment in larger scenes.
    const float tEps = 1e-6f * fmaxf(1.f, fabsf(t));
    for (int a = 0; a < 3; ++a) {
      if (rd[a] == 0.f)
        continue;
      const float rb = rd[a] > 0.f ? float(vi[a]) + 0.5f : float(vi[a]) - 0.5f;
      const float te = (boundaryPos(a, rb) - ro[a]) / rd[a];
      if (te > t + tEps && te < tExit) {
        tExit = te;
        exitAxis = a;
      }
    }
    const float tExitSeg = fminf(tExit, t1);
    if (!seeded) {
      // Seed the predecessor a quarter of the entry voxel back (voxel-relative,
      // as the uniform grids do) so an entry-face crossing is caught. A
      // segment-relative backstep collapses to t under fp rounding on long
      // segments, missing it.
      // See the uniform-grid seed note: linear needs the exact entry-face value
      // (cubic node-0), nearest the quarter-voxel-back entry-side value.
      vPrev = st.filter == SpatialFieldFilter::Nearest
          ? sampleValue(st, field, ro + (t - 0.25f * (tExitSeg - t)) * rd)
          : sampleValue(st, field, ro + t * rd);
      seeded = true;
    }
    if (crossingTest(st,
            field,
            ro,
            rd,
            t,
            tExitSeg,
            crossedAxis,
            isovals,
            numIsovalues,
            vPrev,
            ss))
      return true;
    if (exitAxis < 0) // no forward boundary before t1: the segment ends here
      break;
    crossedAxis = exitAxis;
    t = tExit;
    vi[exitAxis] += stepv[exitAxis];
  }
  return false;
}

// Two-level traversal: walk the coarse macrocell grid (empty-space skipping via
// its value ranges) and only run the per-voxel DDA inside macrocells whose
// range brackets an isovalue. Each active segment is extended one voxel past
// the cell exit so a crossing on the shared face with a skipped neighbour is
// still found; front-to-back keeps the nearest hit. (`voxelDDASegment` resolves
// per grid.)
template <typename SamplerState>
VISRTX_DEVICE void marchVoxelsMacrocellSkip(const IsosurfaceGeometryData &iso,
    const SpatialFieldGPUData &field,
    const SamplerState &st,
    const Ray &objRay,
    ScreenSample &ss)
{
  const float *const isovals = iso.isovalues;
  const uint32_t numIsovalues = iso.numIsovalues;
  // ~one voxel, to cover the exit face. iso.stepSize is object-space;
  // objRay.dir is non-unit under instance scaling, so convert to ray-t units.
  const float margin = 2.f * iso.stepSize / length(objRay.dir);

  GridTraversal mc(objRay, field.grid.dims, field.grid.objectBounds);
  while (mc.valid()) {
    const box1 vr = field.grid.valueRanges[mc.cellIndex];
    bool active = vr.lower <= vr.upper;
    if (active) {
      active = false;
      for (uint32_t i = 0; i < numIsovalues; ++i) {
        if (isovals[i] >= vr.lower && isovals[i] <= vr.upper) {
          active = true;
          break;
        }
      }
    }
    if (active) {
      // Cap at the true ray tmax, NOT objRay.t.upper (which is clipped to this
      // brick's AABB): capping at the brick edge zeroes the margin at every
      // brick-exit face and drops a crossing on the face shared with the next
      // brick -- a depth seam on the brick-boundary plane. The per-grid DDA
      // stays in bounds via its own grid clamp; adjacent bricks re-check the
      // overlap harmlessly and front-to-back keeps the nearest hit. (Mirrors
      // the linear ray-march path above.)
      const float t1 = fminf(mc.tExit + margin, ray::tmax());
      if (voxelDDASegment(
              iso, field, st, objRay.org, objRay.dir, mc.tEntry, t1, ss))
        return;
    }
    mc.next();
  }
}

// Voxel-DDA isosurface intersection for built-in grids. Concrete grid samplers
// run the two-level macrocell-skip + per-voxel DDA for BOTH filters: nearest
// does the piecewise-constant face test, linear solves the per-voxel trilinear
// cubic (linearCrossing). The fixed-step ray march below is no longer reached
// by built-in fields. This generic form is the ray-march fallback for fields
// without analytic voxel boundaries (e.g. custom fields); built-in fields are
// gated to require a space-skipping grid in Isosurface::finalize.
template <typename SamplerState>
VISRTX_DEVICE void marchIsosurfaceVoxels(const IsosurfaceGeometryData &iso,
    const SpatialFieldGPUData &field,
    const SamplerState &st,
    const Ray &objRay,
    ScreenSample &ss)
{
  marchIsosurface(iso, field, st, objRay, ss);
}

VISRTX_DEVICE void marchIsosurfaceVoxels(const IsosurfaceGeometryData &iso,
    const SpatialFieldGPUData &field,
    const StructuredRegularSamplerState &st,
    const Ray &objRay,
    ScreenSample &ss)
{
  // Both filters traverse the voxel-DDA now: nearest does the face test, linear
  // solves the per-voxel cubic (the fixed-step march is no longer reached).
  marchVoxelsMacrocellSkip(iso, field, st, objRay, ss);
}

template <typename ValueType>
VISRTX_DEVICE void marchIsosurfaceVoxels(const IsosurfaceGeometryData &iso,
    const SpatialFieldGPUData &field,
    const NvdbRegularSamplerState<ValueType> &st,
    const Ray &objRay,
    ScreenSample &ss)
{
  // Both filters traverse the voxel-DDA now (linear solves the per-voxel
  // cubic).
  marchVoxelsMacrocellSkip(iso, field, st, objRay, ss);
}

VISRTX_DEVICE void marchIsosurfaceVoxels(const IsosurfaceGeometryData &iso,
    const SpatialFieldGPUData &field,
    const StructuredRectilinearSamplerState &st,
    const Ray &objRay,
    ScreenSample &ss)
{
  // Both filters traverse the voxel-DDA now (linear solves the per-voxel
  // cubic).
  marchVoxelsMacrocellSkip(iso, field, st, objRay, ss);
}

template <typename ValueType>
VISRTX_DEVICE void marchIsosurfaceVoxels(const IsosurfaceGeometryData &iso,
    const SpatialFieldGPUData &field,
    const NvdbRectilinearSamplerState<ValueType> &st,
    const Ray &objRay,
    ScreenSample &ss)
{
  // Both filters traverse the voxel-DDA now (linear solves the per-voxel
  // cubic).
  marchVoxelsMacrocellSkip(iso, field, st, objRay, ss);
}

VISRTX_DEVICE void intersectIsosurface(const GeometryGPUData &geometryData)
{
  const auto &iso = geometryData.isosurface;
  auto &ss = ray::screenSample();
  const auto &field = ss.frameData->registry.fields[iso.field];

  Ray objRay;
  objRay.org = ray::localOrigin();
  objRay.dir = ray::localDirection();
  objRay.t.lower = ray::tmin();
  objRay.t.upper = ray::tmax();

  // Limit the march to this brick (one BVH primitive == one active-region
  // brick) so its DDA walks only its own macrocells; the BVH has already culled
  // bricks the ray misses.
  box1 tBrick;
  if (!rayBoxIntersection(objRay.org,
          objRay.dir,
          iso.brickBounds[ray::primID()],
          tBrick.lower,
          tBrick.upper))
    return;
  objRay.t.lower = fmaxf(objRay.t.lower, tBrick.lower);
  objRay.t.upper = fminf(objRay.t.upper, tBrick.upper);

  // Clip to the field's region of interest, matching the volume path.
  box1 tRoi;
  if (!rayBoxIntersection(
          objRay.org, objRay.dir, field.roi, tRoi.lower, tRoi.upper))
    return;
  objRay.t.lower = fmaxf(objRay.t.lower, tRoi.lower);
  objRay.t.upper = fminf(objRay.t.upper, tRoi.upper);
  if (!(objRay.t.lower < objRay.t.upper))
    return;

  // Dispatch once on the concrete field sampler; the marcher calls the shared
  // sampleValue/sampleNormal, resolved by ADL on the state type.
  VolumeSamplingState st;
  switch (field.samplerCallableIndex) {
#define X(ENTRY, MEMBER, INIT_FN)                                              \
  case SbtCallableEntryPoints::ENTRY:                                          \
    INIT_FN(st.MEMBER, &field);                                                \
    marchIsosurfaceVoxels(iso, field, st.MEMBER, objRay, ss);                  \
    return;
    VISRTX_ISOSURFACE_FIELD_VARIANTS
#undef X
  case SbtCallableEntryPoints::SpatialFieldSamplerCustom:
    // Custom fields have no inline sampler; sample via the SBT callables
    // (initSamplerState/sampleValue/sampleNormal route through
    // optixDirectCall). The generic marchIsosurfaceVoxels falls back to the
    // fixed-step ray march.
    initSamplerState(st, field);
    marchIsosurfaceVoxels(iso, field, st, objRay, ss);
    return;
  default:
    return; // unsupported field type
  }
}

// Generic geometry dispatch //////////////////////////////////////////////////

VISRTX_DEVICE void intersectGeometry()
{
  const auto &ss = ray::screenSample();
  const auto &frameData = *ss.frameData;
  const auto &surfaceData = ray::surfaceData(frameData);
  const auto &geometryData = getGeometryData(frameData, surfaceData.geometry);

  switch (geometryData.type) {
  case GeometryType::SPHERE:
    intersectSphere(geometryData);
    break;
  case GeometryType::CYLINDER:
    intersectCylinder(geometryData);
    break;
  case GeometryType::CONE:
    intersectCone(geometryData);
    break;
  case GeometryType::SDF:
    intersectSDF(geometryData);
    break;
  case GeometryType::ISOSURFACE:
    intersectIsosurface(geometryData);
    break;
#ifdef VISRTX_USE_NEURAL
  case GeometryType::NEURAL:
    intersectNeural(geometryData);
    break;
#endif
  }
}

// Main intersection dispatch /////////////////////////////////////////////////

VISRTX_GLOBAL void __intersection__()
{
  if (ray::isIntersectingSurfaces())
    intersectGeometry();
  else
    intersectVolume();
}

} // namespace visrtx
