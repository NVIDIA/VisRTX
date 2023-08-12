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

#include "gpu/shading_api.h"

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

  const uvec2 pidx = cylinderData.indices
      ? cylinderData.indices[ray::primID()]
      : uvec2(2 * ray::primID(), 2 * ray::primID() + 1);

  const auto p0 = cylinderData.vertices[pidx.x];
  const auto p1 = cylinderData.vertices[pidx.y];

  const float radius = cylinderData.radii ? cylinderData.radii[ray::primID()]
                                          : cylinderData.radius;

  const vec3 ro = ray::localOrigin();
  const vec3 rd = ray::localDirection();

  const vec3 cZ = p1 - p0;
  const vec3 q = ro - p0;

  const float z2 = glm::dot(cZ, cZ);
  const float d = glm::dot(cZ, rd);
  const float c = glm::dot(cZ, q);

  const float A = z2 - (d * d);
  const float B = z2 * glm::dot(q, rd) - c * d;
  const float C = z2 * glm::dot(q, q) - (c * c) - (radius * radius) * z2;

  float radical = B * B - A * C;
  if (radical < 0.f)
    return;

  radical = glm::sqrt(radical);

  // First hit //

  const float tin = (-B - radical) / A;
  const float yin = c + tin * d;
  if (yin > 0.f && yin < z2) {
    const vec3 normal = (q + tin * rd - cZ * yin * (1.f / z2)) * (1.f / radius);
    reportIntersection(tin, normal, yin * (1.f / z2));
  } else if (cylinderData.caps) {
    const float tcapin = (((yin < 0.f) ? 0.f : z2) - c) / d;
    if (abs(B + A * tcapin) < radical) {
      const float us = yin < 0.f ? -1.f : 1.f;
      const vec3 normal = cZ * us / z2;
      reportIntersection(tin, normal, (yin < 0.f) ? 0.f : 1.f);
    }
  }

  // Second hit //

  const float tout = (-B + radical) / A;
  const float yout = c + tout * d;
  if (yout > 0.f && yout < z2) {
    const vec3 normal =
        (q + tout * rd - cZ * yout * (1.f / z2)) * (1.f / radius);
    reportIntersection(tout, normal, yout * (1.f / z2));
  } else if (cylinderData.caps) {
    const float tcapout = (((yout < 0.f) ? 0.f : z2) - c) / d;
    if (abs(B + A * tcapout) < radical) {
      const float us = yout < 0.f ? -1.f : 1.f;
      const vec3 normal = cZ * us / z2;
      reportIntersection(tout, normal, (yout < 0.f) ? 0.f : 1.f);
    }
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
