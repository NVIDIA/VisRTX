/*
 * Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gpu/curveHelpers.h"
#include "gpu/gpu_util.h"

namespace visrtx {

RT_FUNCTION const GeometryGPUData &getGeometryData(
    const FrameGPUData &frameData, DeviceObjectIndex idx)
{
  return frameData.registry.geometries[idx];
}

RT_FUNCTION const MaterialGPUData &getMaterialData(
    const FrameGPUData &frameData, DeviceObjectIndex idx)
{
  return frameData.registry.materials[idx];
}

namespace ray {

RT_FUNCTION vec3 origin()
{
  return make_vec3(optixGetWorldRayOrigin());
}

RT_FUNCTION vec3 localOrigin()
{
  return make_vec3(optixGetObjectRayOrigin());
}

RT_FUNCTION vec3 direction()
{
  return make_vec3(optixGetWorldRayDirection());
}

RT_FUNCTION vec3 localDirection()
{
  return make_vec3(optixGetObjectRayDirection());
}

RT_FUNCTION float tmin()
{
  return optixGetRayTmin();
}

RT_FUNCTION float tmax()
{
  return optixGetRayTmax();
}

RT_FUNCTION float t()
{
  return tmax();
}

RT_FUNCTION float volume_out_t()
{
  return bit_cast<float>(optixGetAttribute_0());
}

RT_FUNCTION vec3 volume_local_direction()
{
  return vec3(bit_cast<float>(optixGetAttribute_1()),
      bit_cast<float>(optixGetAttribute_2()),
      bit_cast<float>(optixGetAttribute_3()));
}

RT_FUNCTION vec3 hitpoint()
{
  return origin() + (t() * direction());
}

RT_FUNCTION vec2 uv(GeometryType type)
{
  switch (type) {
  case GeometryType::TRIANGLE:
  case GeometryType::QUAD:
  case GeometryType::CONE: {
    const ::float2 values = optixGetTriangleBarycentrics();
    return vec2(values.x, values.y);
  }
  case GeometryType::CURVE: {
    const float u = optixGetCurveParameter();
    return vec2(u, 1.f - u);
  }
  case GeometryType::SPHERE:
  case GeometryType::CYLINDER:
  default: {
    const float u = bit_cast<float>(optixGetAttribute_0());
    return vec2(u, 1.f - u);
  }
  }
}

RT_FUNCTION vec3 uvw(GeometryType type)
{
  const auto values = uv(type);
  return vec3(1.f - values.x - values.y, values.x, values.y);
}

RT_FUNCTION uint32_t primID()
{
  return optixGetPrimitiveIndex();
}

RT_FUNCTION uint32_t objID()
{
  return optixGetSbtGASIndex();
}

RT_FUNCTION uint32_t instID()
{
  return optixGetInstanceIndex();
}

RT_FUNCTION ScreenSample &screenSample()
{
  return *detail::getPRD<ScreenSample>(detail::PRDSelector::SCREEN_SAMPLE);
}

template <typename T>
RT_FUNCTION T &rayData()
{
  return *detail::getPRD<T>(detail::PRDSelector::RAY_DATA);
}

RT_FUNCTION bool isIntersectingSurfaces()
{
  return optixGetPayload_4();
}

RT_FUNCTION bool isIntersectingVolumes()
{
  return !isIntersectingSurfaces();
}

RT_FUNCTION const SurfaceGPUData &surfaceData(const FrameGPUData &frameData)
{
  auto &inst = frameData.world.surfaceInstances[ray::instID()];
  auto idx = inst.surfaces[ray::objID()];
  return frameData.registry.surfaces[idx];
}

RT_FUNCTION const VolumeGPUData &volumeData(const FrameGPUData &frameData)
{
  auto &inst = frameData.world.volumeInstances[ray::instID()];
  auto idx = inst.volumes[ray::objID()];
  return frameData.registry.volumes[idx];
}

RT_FUNCTION void computeNormal(
    const GeometryGPUData &ggd, uint32_t primID, SurfaceHit &hit)
{
  const vec3 b = ray::uvw(ggd.type);

  switch (ggd.type) {
  case GeometryType::TRIANGLE: {
    const auto *indices = ggd.tri.indices;
    const uvec3 idx =
        indices ? ggd.tri.indices[primID] : uvec3(0, 1, 2) + primID * 3;

    const vec3 v0 = ggd.tri.vertices[idx.x];
    const vec3 v1 = ggd.tri.vertices[idx.y];
    const vec3 v2 = ggd.tri.vertices[idx.z];
    hit.Ng = cross(v1 - v0, v2 - v0);
    if (!optixIsFrontFaceHit())
      hit.Ng = -hit.Ng;

    if (ggd.tri.vertexNormals != nullptr) {
      const vec3 n0 = ggd.tri.vertexNormals[idx.x];
      const vec3 n1 = ggd.tri.vertexNormals[idx.y];
      const vec3 n2 = ggd.tri.vertexNormals[idx.z];
      hit.Ns = b.x * n0 + b.y * n1 + b.z * n2;
      if (dot(hit.Ng, hit.Ns) < 0.f)
        hit.Ns = -hit.Ns;
    } else
      hit.Ns = hit.Ng;

    break;
  }
  case GeometryType::QUAD: {
    const auto *indices = ggd.quad.indices;
    const uvec3 idx =
        indices ? ggd.quad.indices[primID] : uvec3(0, 1, 2) + primID * 3;
    const vec3 v0 = ggd.quad.vertices[idx.x];
    const vec3 v1 = ggd.quad.vertices[idx.y];
    const vec3 v2 = ggd.quad.vertices[idx.z];
    hit.Ng = cross(v1 - v0, v2 - v0);

    if (!optixIsFrontFaceHit())
      hit.Ng = -hit.Ng;
    hit.Ns = hit.Ng;
    break;
  }
  case GeometryType::SPHERE: {
    if (ggd.sphere.indices) {
      hit.Ng = hit.Ns =
          hitpoint() - ggd.sphere.centers[ggd.sphere.indices[primID]];
    } else
      hit.Ng = hit.Ns = hitpoint() - ggd.sphere.centers[primID];
    break;
  }
  case GeometryType::CYLINDER: {
    hit.Ng = hit.Ns = vec3(bit_cast<float>(optixGetAttribute_1()),
        bit_cast<float>(optixGetAttribute_2()),
        bit_cast<float>(optixGetAttribute_3()));
    break;
  }
  case GeometryType::CONE: {
    const auto *indices = ggd.cone.indices;
    const uvec3 idx =
        indices ? ggd.cone.indices[primID] : uvec3(0, 1, 2) + primID * 3;
    const vec3 v0 = ggd.cone.vertices[idx.x];
    const vec3 v1 = ggd.cone.vertices[idx.y];
    const vec3 v2 = ggd.cone.vertices[idx.z];
    hit.Ng = cross(v1 - v0, v2 - v0);

    if (!optixIsFrontFaceHit())
      hit.Ng = -hit.Ng;
    hit.Ns = hit.Ng;
    break;
  }
  case GeometryType::CURVE: {
    const uint32_t idx = ggd.curve.indices[primID];
    const vec3 v0 = ggd.curve.vertices[idx + 0];
    const vec3 v1 = ggd.curve.vertices[idx + 1];
    const float r0 = ggd.curve.radii[idx + 0];
    const float r1 = ggd.curve.radii[idx + 1];
    vec4 controlPoints[2] = {{v0.x, v0.y, v0.z, r0}, {v1.x, v1.y, v1.z, r1}};

    LinearBSplineSegment interpolator(controlPoints);
    auto hp =
        optixTransformPointFromWorldToObjectSpace((::float3 &)hit.hitpoint);
    auto u = optixGetCurveParameter();
    hit.Ng = hit.Ns =
        curveSurfaceNormal(interpolator, u, vec3(hp.x, hp.y, hp.z));
    break;
  }
  default:
    break;
  }

  hit.Ng = normalize(make_vec3(
      optixTransformNormalFromObjectToWorldSpace((::float3 &)hit.Ng)));
  hit.Ns = normalize(make_vec3(
      optixTransformNormalFromObjectToWorldSpace((::float3 &)hit.Ns)));
}

RT_FUNCTION void populateSurfaceHit(SurfaceHit &hit)
{
  auto &ss = ray::screenSample();
  auto &fd = *ss.frameData;
  auto &sd = ray::surfaceData(fd);

  auto &gd = getGeometryData(fd, sd.geometry);
  auto &md = getMaterialData(fd, sd.material);

  hit.foundHit = true;
  hit.geometry = &gd;
  hit.material = &md;
  hit.t = ray::t();
  hit.hitpoint = ray::hitpoint();
  hit.uvw = ray::uvw(gd.type);
  hit.primID = ray::primID();
  hit.epsilon = epsilonFrom(ray::hitpoint(), ray::direction(), ray::t());
  ray::computeNormal(gd, ray::primID(), hit);
}

RT_FUNCTION void populateVolumeHit(VolumeHit &hit)
{
  auto &ss = ray::screenSample();
  auto &fd = *ss.frameData;

  hit.foundHit = true;
  hit.volumeData = &ray::volumeData(fd);

  const auto ro = optixGetWorldRayOrigin();
  hit.localRay.org = make_vec3(optixTransformPointFromWorldToObjectSpace(ro));
  hit.localRay.dir = ray::volume_local_direction();
  hit.localRay.t.lower = ray::t();
  hit.localRay.t.upper = ray::volume_out_t();
}

RT_FUNCTION void populateHit()
{
  if (ray::isIntersectingSurfaces()) {
    auto &hit = ray::rayData<SurfaceHit>();
    ray::populateSurfaceHit(hit);
  } else {
    auto &hit = ray::rayData<VolumeHit>();
    ray::populateVolumeHit(hit);
  }
}

} // namespace ray
} // namespace visrtx
