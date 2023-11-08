/*
 * Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace visrtx {

// Helper functions ///////////////////////////////////////////////////////////

RT_FUNCTION void reportIntersection(float t, const vec3 &normal, float u)
{
  optixReportIntersection(t,
      0,
      bit_cast<uint32_t>(u),
      bit_cast<uint32_t>(normal.x),
      bit_cast<uint32_t>(normal.y),
      bit_cast<uint32_t>(normal.z));
}

RT_FUNCTION void reportIntersection(float t)
{
  optixReportIntersection(t, 0, bit_cast<uint32_t>(0.f));
}

RT_FUNCTION void reportIntersectionVolume(const box1 &t)
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

RT_FUNCTION void intersectSphere(const GeometryGPUData &geometryData)
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
  reportIntersection(projCO - td);
}

RT_FUNCTION void intersectCylinder(const GeometryGPUData &geometryData)
{
  const auto &cylinderData = geometryData.cylinder;

  const uvec2 pidx = cylinderData.indices ? cylinderData.indices[ray::primID()]
                                          : (2 * ray::primID() + uvec2(0, 1));

  const auto p0 = cylinderData.vertices[pidx.x];
  const auto p1 = cylinderData.vertices[pidx.y];

  const float radius =
      glm::abs(cylinderData.radii ? cylinderData.radii[ray::primID()]
                                  : cylinderData.radius);

  const vec3 ro = ray::localOrigin();
  const vec3 rd = ray::localDirection();

  vec3 ca = p1 - p0;
  vec3 oc = ro - p0;

  float caca = glm::dot(ca, ca);
  float card = glm::dot(ca, rd);
  float caoc = glm::dot(ca, oc);

  float a = caca - card * card;
  float b = caca * glm::dot(oc, rd) - caoc * card;
  float c = caca * glm::dot(oc, oc) - caoc * caoc - radius * radius * caca;
  float h = b * b - a * c;

  if (h < 0.f)
    return;

  h = glm::sqrt(h);
  float d = (-b - h) / a;

  float y = caoc + d * card;
  if (y > 0.f && y < caca) {
    auto n = (oc + d * rd - ca * y / caca) / radius;
    reportIntersection(d, n, position(y, box1(0.f, caca)));
  }

  d = ((y < 0.f ? 0.f : caca) - caoc) / card;

  if (glm::abs(b + a * d) < h) {
    auto n = ca * glm::sign(y) / caca;
    reportIntersection(d, n, y < 0.f ? 0.f : 1.f);
  }
}

RT_FUNCTION void intersectCone(const GeometryGPUData &geometryData)
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

RT_FUNCTION void intersectVolume()
{
  auto &hit = ray::rayData<VolumeHit>();
  if (hit.volID == ray::primID() && hit.instID == ray::instID())
    return;

  const auto &ss = ray::screenSample();
  const auto &frameData = *ss.frameData;
  const auto &volumeData = ray::volumeData(frameData);
  const auto &bounds = volumeData.bounds;
  const vec3 mins =
      (bounds.lower - ray::localOrigin()) * (1.f / ray::localDirection());
  const vec3 maxs =
      (bounds.upper - ray::localOrigin()) * (1.f / ray::localDirection());
  const vec3 nears = glm::min(mins, maxs);
  const vec3 fars = glm::max(mins, maxs);

  box1 t(glm::compMax(nears), glm::compMin(fars));

  if (t.lower < t.upper) {
    const box1 rayt{ray::tmin(), ray::tmax()};
    t.lower = clamp(t.lower, rayt);
    t.upper = clamp(t.upper, rayt);
    reportIntersectionVolume(t);
  }
}

// Generic geometry dispatch //////////////////////////////////////////////////

RT_FUNCTION void intersectGeometry()
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
  }
}

// Main intersection dispatch /////////////////////////////////////////////////

RT_PROGRAM void __intersection__()
{
  if (ray::isIntersectingSurfaces())
    intersectGeometry();
  else
    intersectVolume();
}

} // namespace visrtx
