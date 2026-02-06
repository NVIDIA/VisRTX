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

#include "gpu/gpu_math.h"
#include "gpu/shading_api.h"

// glm
#include <glm/gtx/norm.hpp>
#ifdef VISRTX_USE_NEURAL
#include <optix_types.h>

#include <cuda_fp16.h>
#endif

namespace visrtx {

// Helper functions ///////////////////////////////////////////////////////////

VISRTX_DEVICE void reportIntersection(float t, const vec3 &normal, float u)
{
  optixReportIntersection(t,
      0,
      bit_cast<uint32_t>(u),
      bit_cast<uint32_t>(normal.x),
      bit_cast<uint32_t>(normal.y),
      bit_cast<uint32_t>(normal.z));
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

// Primitive intersectors /////////////////////////////////////////////////////

VISRTX_DEVICE void intersectSphere(const GeometryGPUData &geometryData)
{
  const auto &sphereData = geometryData.sphere;

  const auto primID =
      sphereData.indices ? sphereData.indices[ray::primID()] : ray::primID();

  const auto center = sphereData.centers[primID];
  const auto radius =
      sphereData.radii ? sphereData.radii[primID] : sphereData.radius;

  const vec3 d = ray::localDirection();
  const float rd2 = 1.f / dot(d, d);
  const vec3 CO = center - ray::localOrigin();
  const float projCO = dot(CO, d) * rd2;
  const vec3 perp = CO - projCO * d;
  const float l2 = glm::dot(perp, perp);
  const float r2 = radius * radius;
  if (l2 > r2)
    return;
  const float td = glm::sqrt((r2 - l2) * rd2);
  const float t = projCO - td;
  const vec3 h = ray::localOrigin() + t * ray::localDirection();
  const vec3 n = h - center;
  reportIntersection(t, n, 0.f);
}

VISRTX_DEVICE void intersectCylinder(const GeometryGPUData &geometryData)
{
  const auto &cylinderData = geometryData.cylinder;

  const uvec2 pidx = cylinderData.indices ? cylinderData.indices[ray::primID()]
                                          : (2 * ray::primID() + uvec2(0, 1));

  const auto p0 = cylinderData.vertices[pidx.x];
  const auto p1 = cylinderData.vertices[pidx.y];

  const float radius =
      glm::abs(cylinderData.radii ? cylinderData.radii[ray::primID()]
                                  : cylinderData.radius);
  const bool caps = cylinderData.caps;

  const vec3 ro = ray::localOrigin();
  const vec3 rd = ray::localDirection();

  const vec3 s = p1 - p0; // axis
  const vec3 sxd = glm::cross(s, rd);
  const float a = glm::dot(sxd, sxd); // (s x d)^2

  if (a == 0.f)
    return;

  const vec3 f = p0 - ro;
  const vec3 sxf = glm::cross(s, f);
  const float ra = 1.0f / a;
  const float ts =
      glm::dot(sxd, sxf) * ra; // (s x d)(s x f) / (s x d)^2, in ray-space
  const vec3 fp = f - ts * rd; // f' = p0 - closest point to axis

  const float s2 = glm::dot(s, s); // s^2
  const vec3 perp = glm::cross(s, fp); // s x f'
  const float c =
      (radius * radius) * s2 - glm::dot(perp, perp); //  r^2 s^2 - (s x f')^2

  if (c < 0.f)
    return;

  float td = glm::sqrt(c * ra);
  const float tin = ts - td;
  const float tout = ts + td;

  // clip to cylinder caps
  const float sf = glm::dot(s, f);
  const float sd = glm::dot(s, rd);

  const float u_in = (tin * sd - sf) * (1.f / s2);
  const vec3 N_in = -td * rd - fp - u_in * s;
  const float u_out = (tout * sd - sf) * (1.f / s2);
  const vec3 N_out = td * rd - fp - u_out * s;

  if (sd != 0.0f) { // Note: sd will become zero if cylinder is oriented
                    // perpendicular to ray direction.
    const float rsd = 1.f / sd;
    const float tA = sf * rsd;
    const float tB = tA + s2 * rsd;

    const float cap_tin = glm::min(tA, tB);
    const float cap_tout = glm::max(tA, tB);

    if (tin > cap_tin && tin < cap_tout)
      reportIntersection(tin, N_in, u_in);
    if (tout > cap_tin && tout < cap_tout)
      reportIntersection(tout, N_out, u_out);
  } else if (sf <= 0.0f && sf >= -s2) {
    reportIntersection(tin, N_in, u_in);
    reportIntersection(tout, N_out, u_out);
  }

  if (caps) {
    const float a = s2 - sd * sd;
    const float b = s2 * glm::dot(-f, rd) + (sf * sd);
    const float c = s2 * glm::dot(-f, -f) - (sf * sf) - (radius * radius * s2);
    float h = b * b - a * c;

    if (h < 0.f)
      return;

    h = glm::sqrt(h);
    const float y = ((-b - h) / a) * sd - sf;
    const float d = ((y < 0.f ? 0.f : s2) + sf) / sd;

    if (glm::abs(b + a * d) < h) {
      auto n = s * glm::sign(y) / s2;
      reportIntersection(d, n, y < 0.f ? 0.f : 1.f);
    }
  }
}

VISRTX_DEVICE void intersectCone(const GeometryGPUData &geometryData)
{
  const auto &coneData = geometryData.cone;

  const uvec2 pidx = coneData.indices ? coneData.indices[ray::primID()]
                                      : (2 * ray::primID() + uvec2(0, 1));

  const auto p0 = coneData.vertices[pidx.x];
  const auto p1 = coneData.vertices[pidx.y];

  const float ra = coneData.radii[pidx.x];
  const float rb = coneData.radii[pidx.y];

  const vec3 ro = ray::localOrigin();
  const vec3 rd = ray::localDirection();

  const vec3 ba = p1 - p0;
  const vec3 oa = ro - p0;
  const vec3 ob = ro - p1;

  const float m0 = glm::dot(ba, ba);
  const float m1 = glm::dot(oa, ba);
  const float m2 = glm::dot(ob, ba);
  const float m3 = glm::dot(rd, ba);

  if (m1 < 0.0f) {
    if (glm::length2(oa * m3 - rd * m1) < (ra * ra * m3 * m3))
      reportIntersection(-m1 / m3, -ba * glm::inversesqrt(m0), 0.f);
  } else if (m2 > 0.0f) {
    if (glm::length2(ob * m3 - rd * m2) < (rb * rb * m3 * m3))
      reportIntersection(-m2 / m3, ba * glm::inversesqrt(m0), 1.f);
  }

  const float m4 = glm::dot(rd, oa);
  const float m5 = glm::dot(oa, oa);
  const float rr = ra - rb;
  const float hy = m0 + rr * rr;

  float k2 = m0 * m0 - m3 * m3 * hy;
  float k1 = m0 * m0 * m4 - m1 * m3 * hy + m0 * ra * (rr * m3 * 1.0f);
  float k0 = m0 * m0 * m5 - m1 * m1 * hy + m0 * ra * (rr * m1 * 2.0f - m0 * ra);

  const float h = k1 * k1 - k2 * k0;
  if (h < 0.0f)
    return;

  const float t = (-k1 - glm::sqrt(h)) / k2;

  const float y = m1 + t * m3;
  if (y > 0.0f && y < m0) {
    reportIntersection(t,
        glm::normalize(m0 * (m0 * (oa + t * rd) + rr * ba * ra) - ba * hy * y),
        position(y, box1(0.f, m0)));
  }
}

VISRTX_DEVICE bool rayBoxIntersection(
    const vec3 &ro, const vec3 &rd, const box3 &bounds, float &t0, float &t1)
{
  const vec3 mins = (bounds.lower - ro) * (1.f / rd);
  const vec3 maxs = (bounds.upper - ro) * (1.f / rd);
  const vec3 nears = min(mins, maxs);
  const vec3 fars = max(mins, maxs);
  t0 = max(nears.x, max(nears.y, nears.z));
  t1 = min(fars.x, min(fars.y, fars.z));

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
