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

#include <glm/ext/quaternion_trigonometric.hpp>
#include <glm/fwd.hpp>
#include <glm/geometric.hpp>
#include <glm/gtx/quaternion.hpp>
#include "gpu/curveHelpers.h"
#include "gpu/gpu_objects.h"
#include "gpu/gpu_util.h"

namespace visrtx {

VISRTX_DEVICE const InstanceSurfaceGPUData &getSurfaceInstanceData(
    const FrameGPUData &frameData, DeviceObjectIndex idx)
{
  return frameData.world.surfaceInstances[idx];
}

VISRTX_DEVICE const InstanceVolumeGPUData &getVolumeInstanceData(
    const FrameGPUData &frameData, DeviceObjectIndex idx)
{
  return frameData.world.volumeInstances[idx];
}

VISRTX_DEVICE const GeometryGPUData &getGeometryData(
    const FrameGPUData &frameData, DeviceObjectIndex idx)
{
  return frameData.registry.geometries[idx];
}

VISRTX_DEVICE const MaterialGPUData &getMaterialData(
    const FrameGPUData &frameData, DeviceObjectIndex idx)
{
  return frameData.registry.materials[idx];
}

namespace ray {

VISRTX_DEVICE vec3 origin()
{
  return make_vec3(optixGetWorldRayOrigin());
}

VISRTX_DEVICE vec3 localOrigin()
{
  return make_vec3(optixGetObjectRayOrigin());
}

VISRTX_DEVICE vec3 direction()
{
  return make_vec3(optixGetWorldRayDirection());
}

VISRTX_DEVICE vec3 localDirection()
{
  return make_vec3(optixGetObjectRayDirection());
}

VISRTX_DEVICE float tmin()
{
  return optixGetRayTmin();
}

VISRTX_DEVICE float tmax()
{
  return optixGetRayTmax();
}

VISRTX_DEVICE float t()
{
  return tmax();
}

VISRTX_DEVICE float volume_out_t()
{
  return bit_cast<float>(optixGetAttribute_0());
}

VISRTX_DEVICE vec3 volume_local_direction()
{
  return vec3(bit_cast<float>(optixGetAttribute_1()),
      bit_cast<float>(optixGetAttribute_2()),
      bit_cast<float>(optixGetAttribute_3()));
}

VISRTX_DEVICE vec3 hitpoint()
{
  return origin() + (t() * direction());
}

VISRTX_DEVICE vec2 uv(GeometryType type)
{
  switch (type) {
  case GeometryType::TRIANGLE:
  case GeometryType::QUAD: {
    const ::float2 values = optixGetTriangleBarycentrics();
    return vec2(values.x, values.y);
  }
  case GeometryType::CURVE: {
    const float u = optixGetCurveParameter();
    return vec2(u, 1.f - u);
  }
  case GeometryType::SPHERE:
  case GeometryType::CYLINDER:
  case GeometryType::CONE:
  default: {
    const float u = bit_cast<float>(optixGetAttribute_0());
    return vec2(u, 1.f - u);
  }
  }
}

VISRTX_DEVICE vec3 uvw(GeometryType type)
{
  const auto values = uv(type);
  return vec3(1.f - values.x - values.y, values.x, values.y);
}

VISRTX_DEVICE uint32_t primID()
{
  return optixGetPrimitiveIndex();
}

VISRTX_DEVICE uint32_t objID()
{
  return optixGetSbtGASIndex();
}

VISRTX_DEVICE uint32_t instID()
{
  return optixGetInstanceIndex();
}

VISRTX_DEVICE ScreenSample &screenSample()
{
  return *detail::getPRD<ScreenSample>(detail::PRDSelector::SCREEN_SAMPLE);
}

template <typename T>
VISRTX_DEVICE T &rayData()
{
  return *detail::getPRD<T>(detail::PRDSelector::RAY_DATA);
}

VISRTX_DEVICE bool isIntersectingSurfaces()
{
  return optixGetPayload_4();
}

VISRTX_DEVICE bool isIntersectingVolumes()
{
  return !isIntersectingSurfaces();
}

VISRTX_DEVICE const SurfaceGPUData &surfaceData(const FrameGPUData &frameData)
{
  auto &inst = frameData.world.surfaceInstances[ray::instID()];
  auto idx = inst.surfaces[ray::objID()];
  return frameData.registry.surfaces[idx];
}

VISRTX_DEVICE const VolumeGPUData &volumeData(const FrameGPUData &frameData)
{
  auto &inst = frameData.world.volumeInstances[ray::instID()];
  auto idx = inst.volumes[ray::objID()];
  return frameData.registry.volumes[idx];
}

VISRTX_DEVICE const SpatialFieldGPUData &fieldData(
    const FrameGPUData &frameData, const VolumeGPUData &volumeData)
{
  // Currently only TF1D volume type is supported, so assume this is what we
  // have
  return frameData.registry.fields[volumeData.data.tf1d.field];
}

VISRTX_DEVICE void computeTangentSpace(
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

    hit.Ng = normalize(cross(v1 - v0, v2 - v0));

    if (!optixIsFrontFaceHit())
      hit.Ng = -hit.Ng;

    vec3 n0, n1, n2;
    bool hasVertexNormals = true;
    if (ggd.tri.vertexNormalsFV != nullptr) {
      const uvec3 nidx = uvec3(0, 1, 2) + (hit.primID * 3);
      n0 = ggd.tri.vertexNormalsFV[nidx.x];
      n1 = ggd.tri.vertexNormalsFV[nidx.y];
      n2 = ggd.tri.vertexNormalsFV[nidx.z];
    } else if (ggd.tri.vertexNormals != nullptr) {
      n0 = ggd.tri.vertexNormals[idx.x];
      n1 = ggd.tri.vertexNormals[idx.y];
      n2 = ggd.tri.vertexNormals[idx.z];
    } else {
      hasVertexNormals = false;
    }

    if (hasVertexNormals)
      hit.Ns = b.x * n0 + b.y * n1 + b.z * n2;
    else
      hit.Ns = hit.Ng;

    hit.Ns = normalize(hit.Ns);

    const bool hasTangentsFV = ggd.tri.vertexTangentsFV != nullptr;
    const bool hasTangentsV = ggd.tri.vertexTangents != nullptr;
    if (hasTangentsFV || hasTangentsV) {
      const uvec3 tIdx =
          hasTangentsFV ? uvec3(0, 1, 2) + (hit.primID * 3) : idx;
      const vec4 *tArr =
          hasTangentsFV ? ggd.tri.vertexTangentsFV : ggd.tri.vertexTangents;
      const vec4 t0 = tArr[tIdx.x];
      const vec4 t1 = tArr[tIdx.y];
      const vec4 t2 = tArr[tIdx.z];

      // At UV mirror seams the sign flips between adjacent vertices;
      // barycentric-summing the signs and applying a single sign at the
      // hit point would carve seam edges into the tangent frame.
      // Build each vertex's bitangent with its own sign and normal,
      // then blend B and T independently — same convention
      // as glTF Sample Renderer, PBRT, Filament.
      const vec3 N0 = hasVertexNormals ? n0 : hit.Ng;
      const vec3 N1 = hasVertexNormals ? n1 : hit.Ng;
      const vec3 N2 = hasVertexNormals ? n2 : hit.Ng;
      const vec3 B0 = t0.w * cross(N0, vec3(t0));
      const vec3 B1 = t1.w * cross(N1, vec3(t1));
      const vec3 B2 = t2.w * cross(N2, vec3(t2));

      hit.tU = normalize(b.x * vec3(t0) + b.y * vec3(t1) + b.z * vec3(t2));
      hit.tV = normalize(b.x * B0 + b.y * B1 + b.z * B2);
    } else {
      auto tangentSpace = computeOrthonormalBasis(hit.Ng);
      hit.tU = tangentSpace[0];
      hit.tV = tangentSpace[1];
    }

    if (dot(hit.Ng, hit.Ns) < 0.f) {
      hit.Ns = -hit.Ns;
      hit.tU = -hit.tU;
      hit.tV = -hit.tV;
    }

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

    auto tangentSpace = computeOrthonormalBasis(hit.Ng);
    hit.tU = tangentSpace[0];
    hit.tV = tangentSpace[1];
    break;
  }
  case GeometryType::SPHERE:
  case GeometryType::ISOSURFACE:
  case GeometryType::CONE:
  case GeometryType::NEURAL:
  case GeometryType::CYLINDER:
  case GeometryType::SDF: {
    vec3 n = vec3(bit_cast<float>(optixGetAttribute_1()),
        bit_cast<float>(optixGetAttribute_2()),
        bit_cast<float>(optixGetAttribute_3()));
    // Analytic intersectors report entry AND exit crossings with the OUTWARD
    // (object-space) normal; orient it toward the ray and record facing.
    // Do the facing test in WORLD space — transform the normal to world and
    // compare to the world ray direction — rather than reconstructing the
    // object-space ray direction as worldToObject * worldDir. That
    // reconstruction does not round-trip optixGetObjectRayDirection() for
    // rotated/sheared instances, so the flip boundary landed in the wrong place
    // and half the surface got an inward normal, rendering black (issue #336).
    // Isosurface/SDF/neural normals already face the ray, so this leaves them
    // on isFrontFace == true unchanged.
    const vec3 worldN = make_vec3(
        optixTransformNormalFromObjectToWorldSpace((::float3 &)n));
    hit.isFrontFace = dot(worldN, ray::direction()) < 0.f;
    if (!hit.isFrontFace)
      n = -n;
    hit.Ng = hit.Ns = n;
    auto tangentSpace = computeOrthonormalBasis(hit.Ng);
    hit.tU = tangentSpace[0];
    hit.tV = tangentSpace[1];
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
    auto tangentSpace = computeOrthonormalBasis(hit.Ng);
    hit.tU = tangentSpace[0];
    hit.tV = tangentSpace[1];
    break;
  }
  default:
    break;
  }

  hit.Ng = normalize(make_vec3(
      optixTransformNormalFromObjectToWorldSpace((::float3 &)hit.Ng)));
  hit.Ns = normalize(make_vec3(
      optixTransformNormalFromObjectToWorldSpace((::float3 &)hit.Ns)));

  hit.tU = normalize(make_vec3(
      optixTransformVectorFromObjectToWorldSpace((::float3 &)hit.tU)));
  hit.tV = normalize(make_vec3(
      optixTransformVectorFromObjectToWorldSpace((::float3 &)hit.tV)));
}

VISRTX_DEVICE void cullbackFaces()
{
  if (optixIsFrontFaceHit())
    return;
  auto &ss = ray::screenSample();
  auto &fd = *ss.frameData;
  auto &sd = ray::surfaceData(fd);
  auto &gd = getGeometryData(fd, sd.geometry);
  const bool cull = (gd.type == GeometryType::TRIANGLE && gd.tri.cullBackfaces)
      || (gd.type == GeometryType::QUAD && gd.quad.cullBackfaces);
  if (cull)
    optixIgnoreIntersection();
}

VISRTX_DEVICE void cullCutPlane()
{
  auto &ss = screenSample();
  auto &fd = *ss.frameData;
  const auto &cp = fd.renderer.cutPlane;
  if (cp == vec4(0.f))
    return;
  const vec3 N(cp.x, cp.y, cp.z);
  const vec3 hitPos = hitpoint();
  if (glm::dot(N, hitPos) + cp.w < 0.f)
    optixIgnoreIntersection();
}

// hit.epsilon floor for analytic primitives, in units of the geometry's
// object-space coordinate scale (GeometryGPUData::epsilonScale). The
// intersectors' quadratics run at that scale, so their fp noise band is
// ~tens of ulps OF THAT SCALE; a secondary ray must clear it or phantom
// self-hits shadow the surface (acne rings on large ground spheres).
// epsilonFrom alone under-lifts wherever the hitpoint's own coordinates are
// small (e.g. near the origin on a giant sphere centered at -r*Y). ~64 fp32
// ulps.
constexpr float kAnalyticEpsilonScale = 0x1.p-17f;

VISRTX_DEVICE void populateSurfaceHit(SurfaceHit &hit)
{
  const auto &ss = ray::screenSample();
  const auto &fd = *ss.frameData;
  const auto &sd = ray::surfaceData(fd);

  const auto &gd = getGeometryData(fd, sd.geometry);
  const auto &md = getMaterialData(fd, sd.material);
  const auto &isd = getSurfaceInstanceData(fd, ray::instID());

  hit.foundHit = true;
  hit.isFrontFace = optixIsFrontFaceHit();
  hit.instance = &isd;
  hit.geometry = &gd;
  hit.material = &md;
  hit.t = ray::t();
  hit.hitpoint = ray::hitpoint();
  hit.uvw = ray::uvw(gd.type);
  hit.primID = ray::primID();
  if (gd.type == GeometryType::ISOSURFACE)
    hit.primID = optixGetHitKind();
  hit.objID = sd.id;
  hit.instID = isd.id;
  hit.epsilon = epsilonFrom(ray::hitpoint(), ray::direction(), ray::t());
  if (gd.epsilonScale > 0.f) {
    // Object-space scale -> world units via the object->world distance ratio
    // along the ray (same construction as the isosurface branch below).
    const vec3 wdir = ray::direction();
    const vec3 odir = mat3(hit.instance->worldToObject) * wdir;
    hit.epsilon = fmaxf(hit.epsilon,
        gd.epsilonScale * kAnalyticEpsilonScale * length(wdir) / length(odir));
  }
  if (gd.type == GeometryType::ISOSURFACE) {
    // Isosurface hits lie on a voxel-resolution surface — exact voxel faces for
    // the nearest voxel-DDA, bisection-localized for the marched linear/custom
    // path. Either way the surface has voxel-scale relief, so a secondary ray
    // offset by less than a voxel grazes into adjacent voxels and self-occludes
    // (AO/shadow acne; on blocky nearest surfaces, a per-voxel waffle).
    // stepSize is half the smallest voxel, so 2*stepSize lifts the ray clear by
    // one. stepSize is object-space but hit.epsilon is world-space, so under a
    // scaled instance scale the lift by the object->world distance ratio along
    // the ray: |worldDir|/|objDir|. optixGetObjectRayDirection is illegal in
    // closest-hit, so derive objDir from the stored worldToObject linear map
    // instead.
    const vec3 wdir = ray::direction();
    const vec3 odir = mat3(hit.instance->worldToObject) * wdir;
    hit.epsilon = fmaxf(hit.epsilon,
        2.f * gd.isosurface.stepSize * length(wdir) / length(odir));
  }
  ray::computeTangentSpace(gd, ray::primID(), hit);
}

VISRTX_DEVICE void populateVolumeHit(VolumeHit &hit)
{
  auto &ss = ray::screenSample();
  auto &fd = *ss.frameData;

  auto &ivd = getVolumeInstanceData(fd, ray::instID());

  hit.foundHit = true;
  hit.volume = &ray::volumeData(fd);
  hit.instance = &ivd;

  const auto ro = optixGetWorldRayOrigin();
  hit.localRay.org = make_vec3(optixTransformPointFromWorldToObjectSpace(ro));
  hit.localRay.dir = ray::volume_local_direction();
  hit.localRay.t.lower = ray::t();
  hit.localRay.t.upper = ray::volume_out_t();
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

VISRTX_DEVICE void populateHit()
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
