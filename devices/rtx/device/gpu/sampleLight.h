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

#include "gpu/evalEmission.h" // evaluateSurfaceEmission (Stage 2 sampled emission)
#include "gpu/gpu_math.h"
#include "gpu/gpu_objects.h"
#include "gpu/gpu_util.h"
#include "gpu/intersectPrimitives.h" // CapBit (cylinder/cone cap enablement)

// glm
#include <glm/ext/matrix_float3x3.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/color_space.hpp>

// cuda
#include <device_atomic_functions.h>

// cccl
#include <cub/thread/thread_search.cuh>

// std
#include <algorithm>
#include <cmath>
#include <limits>

// Windows.h is draw in by thread_search.cuh, so we need to undef OPAQUE
#ifdef OPAQUE
#undef OPAQUE
#endif

namespace visrtx {

// Light sampling result containing direction, distance, radiance and PDF
struct LightSample
{
  vec3 radiance; // Emitted radiance in direction of hit point (W⋅sr⁻¹⋅m⁻²)
  vec3 dir; // Unit direction vector from hit point to light sample
  float dist; // Distance from hit point to light sample
  float pdf; // Probability density function value for this sample
};

// Shadow rays toward a Geometry Light's sampled point must stop just short of
// it: the point lies on real, opaque emissive geometry that would otherwise
// self-occlude the shadow ray (~15% energy loss). Relative so it scales with
// distance; analytic lights have no geometry there and are unaffected.
constexpr float GEOMETRY_LIGHT_SHADOW_EPSILON = 1.0e-3f;

// Cone taper |r0-r1| below which the frustum is treated as a cylinder and the
// axial fraction is taken as the uniform sample directly — the (rt-r0)/(r1-r0)
// back-solve would otherwise divide by ~0. Absolute (not scale-relative) so a
// degenerate r0==r1==0 cone can't reach a divide-by-zero.
constexpr float CONE_TAPER_EPSILON = 1.0e-8f;

namespace detail {

VISRTX_DEVICE LightSample sampleDirectionalLight(
    const LightGPUData &ld, const mat4 &xfm)
{
  LightSample ls;
  // Transform light direction to world space and negate to get direction TO
  // light (ld.distant.direction points FROM the light source)
  ls.dir = xfmVec(xfm, -ld.distant.direction);
  ls.dist = std::numeric_limits<float>::infinity();
  // For directional lights, irradiance is the amount of light per unit area
  // arriving at the surface (W/m²)
  ls.radiance = ld.color * ld.distant.irradiance;
  // Delta function: directional light has no spatial extent, so PDF = 1
  ls.pdf = 1.f;

  return ls;
}

VISRTX_DEVICE LightSample samplePointLight(
    const LightGPUData &ld, const mat4 &xfm, const vec3 &origin)
{
  LightSample ls;
  // Calculate vector from hit point to light position
  ls.dir = xfmPoint(xfm, ld.point.position) - origin;
  ls.dist = length(ls.dir);
  ls.dir /= ls.dist;
  // Apply inverse square law: intensity falls off as 1/r²
  // This converts intensity (W/sr) to radiance at the hit point
  ls.radiance = ld.color * ld.point.intensity / pow2(ls.dist);
  // Delta function: point light has no spatial extent, so PDF = 1
  ls.pdf = 1.f;

  return ls;
}

VISRTX_DEVICE LightSample sampleSphereLight(
    const LightGPUData &ld, const mat4 &xfm, const vec3 &origin, RandState &rs)
{
  LightSample ls;
  auto u1 = pcg_uniform(&rs);
  auto u2 = pcg_uniform(&rs);

  // Uniform sampling on unit sphere using Marsaglia's method
  // u1 maps to z-coordinate: z ∈ [-1, 1]
  auto z = 1.f - 2.f * u1;
  // r is the radius in the xy-plane for this z-level
  auto r = sqrtf(std::max(0.f, 1.f - z * z));
  // u2 maps to azimuthal angle: φ ∈ [0, 2π]
  auto phi = kTwoPi * u2;
  auto x = r * cosf(phi);
  auto y = r * sinf(phi);

  // Scale by sphere radius to get point on sphere surface
  auto p = vec3(x, y, z) * ld.sphere.radius;
  auto worldSamplePos = xfmPoint(xfm, ld.sphere.position + p);
  ls.dir = worldSamplePos - origin;
  ls.dist = length(ls.dir);
  ls.dir /= ls.dist;

  // Sphere emits uniformly in all directions (Lambertian)
  ls.radiance = ld.color * ld.sphere.intensity;

  // Convert area PDF to solid angle PDF for proper Monte Carlo integration
  // Area PDF = 1 / (4πr²), but we need solid angle PDF
  // Conversion: pdf_solid_angle = pdf_area * distance² / |cos θ|
  // For sphere: cos θ = dot(surface_normal, -light_direction)
  // Surface normal at sampled point: direction from sphere center to sample
  // point
  auto worldSphereCenter = xfmPoint(xfm, ld.sphere.position);
  auto surfaceNormal = normalize(worldSamplePos - worldSphereCenter);
  auto cosTheta = dot(surfaceNormal, -ls.dir);

  if (cosTheta > 0.0f) {
    // Note: For non-uniform scaling transforms, the area calculation would need
    // to account for the transform's effect on surface area (determinant of
    // jacobian) Currently assumes uniform scaling or no scaling of the light
    // geometry
    float areaPdf = 1.f / (4.f * kPi * ld.sphere.radius * ld.sphere.radius);
    ls.pdf = areaPdf * pow2(ls.dist) / cosTheta;
  } else {
    // Back-facing surface element contributes no light
    ls.radiance = vec3(0.0f);
    ls.pdf = 0.0f;
  }

  return ls;
}

VISRTX_DEVICE LightSample sampleRectLight(
    const LightGPUData &ld, const mat4 &xfm, const vec3 &origin, RandState &rs)
{
  LightSample ls;
  auto uv = vec2(pcg_uniform(&rs), pcg_uniform(&rs));

  // Uniform sampling on rectangle: uv ∈ [0,1]² maps to rectangle
  auto rectangleSample = ld.rect.edge1 * uv.x + ld.rect.edge2 * uv.y;
  auto worldPos = xfmPoint(xfm, ld.rect.position + rectangleSample);
  ls.dir = worldPos - origin;
  ls.dist = length(ls.dir);
  ls.dir /= ls.dist;

  // Calculate rectangle normal and area from cross product
  auto normal = cross(ld.rect.edge1, ld.rect.edge2);
  auto area = length(normal);
  normal = normalize(xfmVec(xfm, normal));

  // Apply Lambert's cosine law: radiance ∝ cos(θ) where θ is angle to normal
  auto cosTheta = dot(normal, -ls.dir);

  // Handle front/back face emission based on light configuration
  if (ld.rect.side.back) {
    if (ld.rect.side.front)
      cosTheta = fabsf(cosTheta); // Both sides: always positive
    else
      cosTheta = -cosTheta; // Back only: flip to back face
  }
  // Front only: use cosTheta as-is (positive for front face)

  if (cosTheta > 0.0f) {
    // Lambertian radiance. cosTheta is handled through pdf below.
    ls.radiance = ld.color * ld.rect.intensity;

    // Convert area PDF to solid angle PDF for proper Monte Carlo integration
    // Area PDF = 1 / area, Solid angle PDF = area_pdf * distance² / |cos θ|
    float areaPdf = 1.0f / area;
    ls.pdf = areaPdf * pow2(ls.dist) / cosTheta;
  } else {
    // No emission toward surfaces facing away from the light
    ls.radiance = vec3(0.0f);
    ls.pdf = 0.0f;
  }

  return ls;
}

VISRTX_DEVICE LightSample sampleRingLight(
    const LightGPUData &ld, const mat4 &xfm, const vec3 &origin, RandState &rs)
{
  LightSample ls;
  auto u1 = pcg_uniform(&rs);
  auto u2 = pcg_uniform(&rs);

  // Sample angle uniformly around the ring: φ ∈ [0, 2π]
  auto phi = kTwoPi * u1;

  // Sample radial position uniformly by area between inner and outer radius
  // For uniform area sampling: r² = u₂(R² - r²) + r² where R=outer, r=inner
  auto outerRadius = ld.ring.radius;
  auto innerRadius = ld.ring.innerRadius;
  auto r = sqrtf(u2 * (outerRadius * outerRadius - innerRadius * innerRadius)
      + innerRadius * innerRadius);

  // Create orthonormal basis with ring direction as normal
  auto direction = normalize(ld.ring.direction);
  auto basis = computeOrthonormalBasis(direction);

  // Convert polar coordinates (r, φ) to Cartesian in ring's local frame
  auto localX = r * cosf(phi);
  auto localY = r * sinf(phi);
  auto samplePos = basis[0] * localX + basis[1] * localY;

  // Calculate direction and distance to light sample point
  ls.dir = xfmPoint(xfm, ld.ring.position + samplePos) - origin;
  ls.dist = length(ls.dir);
  ls.dir /= ls.dist;

  auto worldDirection = xfmVec(xfm, direction);

  // Calculate spotlight-like cone attenuation
  float spot;
  auto cosTheta = dot(worldDirection, -ls.dir);
  if (cosTheta < ld.ring.cosOuterAngle) {
    // Outside cone: no illumination
    spot = 0.0f;
  } else if (cosTheta > ld.ring.cosInnerAngle) {
    // Inside inner cone: full illumination
    spot = 1.0f;
  } else {
    // Falloff region: smooth interpolation using smoothstep function
    // smoothstep(t) = 3t² - 2t³ provides C¹ continuity
    spot = (cosTheta - ld.ring.cosOuterAngle)
        / (ld.ring.cosInnerAngle - ld.ring.cosOuterAngle);
    spot = spot * spot * (3.0f - 2.0f * spot);
  }

  if (spot > 0.0f) {
    if (cosTheta > 0.0f) {
      // Lambertian radiance. cosTheta is handled through pdf below.
      ls.radiance = ld.color * ld.ring.intensity * spot;

      // Convert area PDF to solid angle PDF for proper Monte Carlo integration
      // Ring area = π(R² - r²), so area PDF = 1 / ring_area
      // Solid angle PDF = area_pdf * distance² / |cos θ|
      float areaPdf = ld.ring.oneOverArea; // This is 1 / ring_area
      ls.pdf = areaPdf * pow2(ls.dist) / cosTheta;
    } else {
      ls.radiance = vec3(0.0f);
      ls.pdf = 0.0f;
    }
  } else {
    ls.radiance = vec3(0.0f);
    ls.pdf = 0.0f;
  }

  return ls;
}

VISRTX_DEVICE LightSample sampleSpotLight(
    const LightGPUData &ld, const mat4 &xfm, const vec3 &origin)
{
  LightSample ls;
  // Calculate direction from light to hit point
  ls.dir = xfmPoint(xfm, ld.spot.position) - origin;
  ls.dist = length(ls.dir);
  ls.dir /= ls.dist;

  // Transform spot light direction to world space
  auto worldDirection = normalize(xfmVec(xfm, ld.spot.direction));

  // Calculate angle between light direction and direction to hit point
  // spot = cos(angle_between_directions)
  float spot = dot(worldDirection, -ls.dir);

  // Apply spotlight cone attenuation with smooth falloff
  if (spot < ld.spot.cosOuterAngle)
    spot = 0.f; // Outside cone: no illumination
  else if (spot > ld.spot.cosInnerAngle)
    spot = 1.f; // Inside inner cone: full illumination
  else {
    // Falloff region: smooth interpolation using smoothstep
    spot = (spot - ld.spot.cosOuterAngle)
        / (ld.spot.cosInnerAngle - ld.spot.cosOuterAngle);
    spot = spot * spot * (3.f - 2.f * spot); // smoothstep function
  }

  // Apply inverse square law with spotlight attenuation
  ls.radiance = ld.color * ld.spot.intensity * spot / pow2(ls.dist);
  // Delta function for point light source
  ls.pdf = spot > 0.0f ? 1.f : 0.0f;
  return ls;
}

// Binary search for the first index i such that cdf[i] >= u (cub::LowerBound):
// inverse transform sampling of a discrete cumulative distribution. Templated
// on the CDF element type so the light-pick CDF (double, to preserve dim-light
// masses below float epsilon) and the float HDRI/primitive CDFs share one path.
template <typename T>
VISRTX_DEVICE int inverseSampleCDF(const T *cdf, int size, float u)
{
  return cub::LowerBound(cdf, size, T(u));
}

// The three object-space vertex indices of triangle primID, indexed or soup.
VISRTX_DEVICE uvec3 triangleIndices(
    const TriangleGeometryData &tri, uint32_t primID)
{
  return tri.indices ? tri.indices[primID] : uvec3(0, 1, 2) + primID * 3;
}

// Solid-angle pdf of uniform-in-object-area sampling of a Geometry Light, at a
// point on a triangle whose object/world twice-areas and total object area are
// given. Uniform-in-object-area maps to world density
// (1/A_obj_total)·(A_obj_tri /A_world_tri), then to solid angle by
// dist²/|cosθ|. Exact under any affine instance transform. Shared by the
// sampler and the hit-side MIS pdf so the two can never drift — MIS
// unbiasedness depends on them being identical.
VISRTX_DEVICE float geometryLightSolidAnglePdf(float objTwiceArea,
    float worldTwiceArea,
    float totalObjArea,
    float dist,
    float absCosTheta)
{
  return (objTwiceArea / worldTwiceArea) / totalObjArea * pow2(dist)
      / absCosTheta;
}

// Radiance emitted by a Geometry Light at the sampled surface point, toward the
// shading point. A constant emitter uses the baked average (fast path); a
// sampler/attribute emitter is evaluated through the material's own emission
// entry point at a SYNTHETIC hit, so next-event radiance matches the path-hit
// deposit exactly (MIS stays unbiased). `uvw` is the geometry's parametric
// coordinate at the sample (see the per-geometry samplers). The synthetic hit
// points at the REAL surface instance (surfaceInstanceIndex) so
// instance-uniform attribute emission resolves identically to the deposit; a
// transform-only fallback covers the (unexpected) -1 case.
VISRTX_DEVICE vec3 evalGeometryLightEmission(ScreenSample &ss,
    const LightGPUData &ld,
    const mat4 &xfm,
    uint32_t primID,
    const vec3 &uvw,
    const vec3 &worldPoint,
    const vec3 &nsWorld,
    const vec3 &outgoingDir,
    DeviceObjectIndex surfaceInstanceIndex)
{
  const auto &registry = ss.frameData->registry;
  const auto &md = registry.materials[ld.geometry.materialIndex];
  if (md.emissionIsConstant)
    return ld.geometry.radiance;

  SurfaceHit hit{};
  hit.geometry = &registry.geometries[ld.geometry.geometryIndex];
  hit.material = &md;
  hit.primID = primID;
  hit.uvw = uvw;
  hit.hitpoint = worldPoint;
  // A material EDF (MDL) is single-sided, but triangle Geometry Lights are
  // double-sided: the deposit orients the shading normal toward the incoming
  // ray and the triangle sampler pdf uses |cos|. Orient the synthetic normal
  // toward the receiver so the EDF evaluates on the sampled side. A no-op for
  // the analytic samplers (outward normal, far side culled) and for
  // orientation-independent native-PBR emission.
  // The geometric normal reused as shading normal, a synthesized tangent basis,
  // and object id 0 make this hit faithful only for orientation-/tangent-/
  // object-id-independent emission — the diffuse case kFaithfulSet admits
  // (material/EmissionPolicy.h). Enriching this hit is what grows that set.
  const vec3 ns = dot(nsWorld, outgoingDir) < 0.0f ? -nsWorld : nsWorld;
  hit.Ng = hit.Ns = ns;
  const mat3 basis = computeOrthonormalBasis(ns);
  hit.tU = basis[0];
  hit.tV = basis[1];
  hit.objID = 0;
  hit.instID = 0;
  hit.t = 0.0f;
  hit.epsilon = 0.0f;
  hit.isFrontFace = true;

  // Every geometry light carries a valid surface-instance index (World fills it
  // from the same layout the surface instances are built with), so the in-range
  // path is the live one.
  const auto &world = ss.frameData->world;
  if (surfaceInstanceIndex >= 0
      && static_cast<size_t>(surfaceInstanceIndex)
          < world.numSurfaceInstances) {
    hit.instance = &world.surfaceInstances[surfaceInstanceIndex];
    return evaluateSurfaceEmission(*ss.frameData, md, hit, outgoingDir);
  }

  // Defensive fallback, unreachable for real geometry lights: a transform-only
  // instance. An out-of-range index means the host/device surface-instance
  // layout drifted (the World-side assert is stripped under NDEBUG), so
  // bounding it here keeps the drift from dereferencing out of bounds on
  // device.
  InstanceSurfaceGPUData fallback{};
  fallback.objectToWorld = glm::transpose(mat4x3(xfm));
  fallback.worldToObject = glm::transpose(mat4x3(glm::affineInverse(xfm)));
  hit.instance = &fallback;
  return evaluateSurfaceEmission(*ss.frameData, md, hit, outgoingDir);
}

// Sample a point on a triangle Geometry Light. Picks a primitive by its
// object-space area, samples it uniformly, and reports the EXACT solid-angle
// pdf via geometryLightSolidAnglePdf, so the estimator stays unbiased under any
// affine instance transform (non-uniform scale included). Double-sided,
// matching the view-independent constant emission the hit deposit uses.
VISRTX_DEVICE LightSample sampleTriangleGeometryLight(const LightGPUData &ld,
    const TriangleGeometryData &tri,
    const mat4 &xfm,
    const vec3 &origin,
    ScreenSample &ss,
    DeviceObjectIndex surfaceInstanceIndex)
{
  LightSample ls{};
  if (tri.numPrimitives == 0 || ld.geometry.area <= 0.0f)
    return ls;

  const uint32_t primID = glm::min(
      uint32_t(inverseSampleCDF(
          tri.primAreaCdf, int(tri.numPrimitives), pcg_uniform(&ss.rs))),
      tri.numPrimitives - 1);
  const uvec3 idx = triangleIndices(tri, primID);
  const vec3 v0 = tri.vertices[idx.x];
  const vec3 e1o = tri.vertices[idx.y] - v0;
  const vec3 e2o = tri.vertices[idx.z] - v0;

  // Uniform barycentric sample of the triangle; b0/b1/b2 weight v0/v1/v2 and
  // are the hit's uvw for attribute/texcoord interpolation at the sampled
  // point.
  const float su = sqrtf(pcg_uniform(&ss.rs));
  const float u2 = pcg_uniform(&ss.rs);
  const float b1 = su * (1.0f - u2);
  const float b2 = su * u2;
  const vec3 pObj = v0 + e1o * b1 + e2o * b2;

  vec3 nWorld = cross(xfmVec(xfm, e1o), xfmVec(xfm, e2o));
  const float worldTriTwiceArea = length(nWorld);

  const vec3 worldPoint = xfmPoint(xfm, pObj);
  ls.dir = worldPoint - origin;
  ls.dist = length(ls.dir);
  ls.dir /= ls.dist;

  if (worldTriTwiceArea <= 0.0f)
    return ls;
  nWorld /= worldTriTwiceArea;
  const float cosTheta = fabsf(dot(nWorld, -ls.dir));
  if (cosTheta <= 0.0f)
    return ls; // radiance/pdf left zero

  ls.radiance = evalGeometryLightEmission(ss,
      ld,
      xfm,
      primID,
      vec3(1.0f - b1 - b2, b1, b2),
      worldPoint,
      nWorld,
      -ls.dir,
      surfaceInstanceIndex);
  ls.pdf = geometryLightSolidAnglePdf(length(cross(e1o, e2o)),
      worldTriTwiceArea,
      ld.geometry.area,
      ls.dist,
      cosTheta);

  return ls;
}

// Finish a SINGLE-sided (outward) Geometry Light area sample from an
// object-space surface point and its OUTWARD object normal: world
// direction/distance, the emitted radiance at the point, and the EXACT affine
// solid-angle pdf. The world area element and normal come from two transformed
// orthonormal object tangents
// (|cross(M t1, M t2)|), so it is exact under any affine instance transform —
// the same Jacobian the triangle path uses. Shared by the sphere/cylinder/cone
// samplers; the pdf must match geometryLightHitPdf on the deposit side for MIS
// to partition to 1. Radiance/pdf are left zero when the sample faces away.
VISRTX_DEVICE LightSample finishAreaLightSample(ScreenSample &ss,
    const LightGPUData &ld,
    const mat4 &xfm,
    const vec3 &origin,
    const vec3 &pObj,
    const vec3 &nObjOut,
    uint32_t primID,
    const vec3 &uvw,
    DeviceObjectIndex surfaceInstanceIndex)
{
  LightSample ls{};
  const vec3 worldPoint = xfmPoint(xfm, pObj);
  ls.dir = worldPoint - origin;
  ls.dist = length(ls.dir);
  ls.dir /= ls.dist;

  const mat3 basis = computeOrthonormalBasis(nObjOut);
  vec3 nWorld = cross(xfmVec(xfm, basis[0]), xfmVec(xfm, basis[1]));
  const float worldAreaScale =
      length(nWorld); // world-area per unit object-area
  if (worldAreaScale <= 0.0f)
    return ls;
  nWorld /= worldAreaScale;
  // Orient to the outward hemisphere: (M⁻ᵀnObj)·(M nObj) = |nObj|² > 0, so
  // xfmVec(xfm, nObjOut) is a valid outward sign reference under any affine M.
  if (dot(nWorld, xfmVec(xfm, nObjOut)) < 0.0f)
    nWorld = -nWorld;

  const float cosTheta = dot(nWorld, -ls.dir);
  if (cosTheta <= 0.0f)
    return ls; // far side; self-occluded, contributes nothing

  ls.radiance = evalGeometryLightEmission(ss,
      ld,
      xfm,
      primID,
      uvw,
      worldPoint,
      nWorld,
      -ls.dir,
      surfaceInstanceIndex);
  // Object-area element is 1 (orthonormal tangents); world element is
  // worldAreaScale. geometryLightSolidAnglePdf folds pick(1/totalArea) and the
  // area→solid-angle Jacobian identically to the triangle path.
  ls.pdf = geometryLightSolidAnglePdf(
      1.0f, worldAreaScale, ld.geometry.area, ls.dist, cosTheta);
  return ls;
}

// Sample a point on a sphere-set Geometry Light. Picks a sphere by its
// object-space area (4πr²), samples that sphere's surface uniformly
// (Marsaglia). SINGLE-sided (outward): a closed sphere self-occludes its far
// hemisphere.
VISRTX_DEVICE LightSample sampleSphereGeometryLight(const LightGPUData &ld,
    const SphereGeometryData &sph,
    const mat4 &xfm,
    const vec3 &origin,
    ScreenSample &ss,
    DeviceObjectIndex surfaceInstanceIndex)
{
  if (sph.numPrimitives == 0 || ld.geometry.area <= 0.0f)
    return {};

  const uint32_t primID = glm::min(
      uint32_t(inverseSampleCDF(
          sph.primAreaCdf, int(sph.numPrimitives), pcg_uniform(&ss.rs))),
      sph.numPrimitives - 1);
  const uint32_t vi = sph.indices ? sph.indices[primID] : primID;
  const vec3 c = sph.centers[vi];
  const float r = fabsf(sph.radii ? sph.radii[vi] : sph.radius);

  // Uniform point on the object-space unit sphere (Marsaglia); nObj is the unit
  // outward object normal there.
  const float z = 1.0f - 2.0f * pcg_uniform(&ss.rs);
  const float rho = sqrtf(fmaxf(0.0f, 1.0f - z * z));
  const float phi = kTwoPi * pcg_uniform(&ss.rs);
  const vec3 nObj = vec3(rho * cosf(phi), rho * sinf(phi), z);

  // Sphere attributes are per-primitive (no interpolation); uvw = (0,0,1)
  // matches the intersector's constant sphere parameter.
  return finishAreaLightSample(ss,
      ld,
      xfm,
      origin,
      c + nObj * r,
      nObj,
      primID,
      vec3(0.0f, 0.0f, 1.0f),
      surfaceInstanceIndex);
}

// Per-endpoint cap enablement, matching the intersector's resolveCapBits: a
// non-null vertex.cap array (element != 0) overrides the geometry-wide default.
VISRTX_DEVICE void resolveEndpointCaps(const uint8_t *vertexCaps,
    uint8_t defaultCapFlags,
    const uvec2 &idx,
    bool &cap0,
    bool &cap1)
{
  cap0 =
      vertexCaps ? (vertexCaps[idx.x] != 0) : bool(defaultCapFlags & CAP_FIRST);
  cap1 = vertexCaps ? (vertexCaps[idx.y] != 0)
                    : bool(defaultCapFlags & CAP_SECOND);
}

// Uniform point on a disk of radius `rad` in the plane spanned by (e0,e1)
// centered at `c`; used for cylinder/cone caps.
VISRTX_DEVICE vec3 sampleDisk(
    const vec3 &c, const vec3 &e0, const vec3 &e1, float rad, ScreenSample &ss)
{
  const float rr = rad * sqrtf(pcg_uniform(&ss.rs));
  const float phi = kTwoPi * pcg_uniform(&ss.rs);
  return c + rr * (cosf(phi) * e0 + sinf(phi) * e1);
}

// Sample a point on a cylinder-set Geometry Light: pick a cylinder by object
// area (lateral 2πrL + enabled caps πr²), then pick lateral vs a cap by
// sub-area and sample it uniformly. Outward object normal is radial on the
// wall, ±axis on a cap. Single-sided (outward).
VISRTX_DEVICE LightSample sampleCylinderGeometryLight(const LightGPUData &ld,
    const CylinderGeometryData &cyl,
    const mat4 &xfm,
    const vec3 &origin,
    ScreenSample &ss,
    DeviceObjectIndex surfaceInstanceIndex)
{
  if (cyl.numPrimitives == 0 || ld.geometry.area <= 0.0f)
    return {};

  const uint32_t primID = glm::min(
      uint32_t(inverseSampleCDF(
          cyl.primAreaCdf, int(cyl.numPrimitives), pcg_uniform(&ss.rs))),
      cyl.numPrimitives - 1);
  const uvec2 idx =
      cyl.indices ? cyl.indices[primID] : uvec2(0, 1) + primID * 2;
  const vec3 p0 = cyl.vertices[idx.x];
  const vec3 p1 = cyl.vertices[idx.y];
  const float r = fabsf(cyl.radii ? cyl.radii[primID] : cyl.radius);
  const vec3 axis = p1 - p0;
  const float len = length(axis);
  if (len <= 0.0f || r <= 0.0f)
    return {};
  const vec3 axisN = axis / len;
  const mat3 basis =
      computeOrthonormalBasis(axisN); // basis[0],basis[1] ⊥ axisN

  bool cap0, cap1;
  resolveEndpointCaps(cyl.vertexCaps, cyl.defaultCapFlags, idx, cap0, cap1);
  const float latArea = kTwoPi * r * len;
  const float capArea = kPi * r * r;

  float pick = pcg_uniform(&ss.rs)
      * (latArea + (cap0 ? capArea : 0.0f) + (cap1 ? capArea : 0.0f));
  vec3 pObj, nObj;
  float axialU; // axial fraction along p0→p1; the hit's uvw parameter
  if (pick < latArea) {
    axialU = pcg_uniform(&ss.rs);
    const float phi = kTwoPi * pcg_uniform(&ss.rs);
    const vec3 rho = cosf(phi) * basis[0] + sinf(phi) * basis[1];
    pObj = p0 + axialU * axis + r * rho;
    nObj = rho;
  } else if (cap0 && pick < latArea + capArea) {
    pObj = sampleDisk(p0, basis[0], basis[1], r, ss);
    nObj = -axisN;
    axialU = 0.0f;
  } else {
    pObj = sampleDisk(p1, basis[0], basis[1], r, ss);
    nObj = axisN;
    axialU = 1.0f;
  }
  // uvw = (0, u, 1-u): u weights endpoint p1, (1-u) weights p0 (see the
  // intersector and readAttributeValue's cylinder branch).
  return finishAreaLightSample(ss,
      ld,
      xfm,
      origin,
      pObj,
      nObj,
      primID,
      vec3(0.0f, axialU, 1.0f - axialU),
      surfaceInstanceIndex);
}

// Sample a point on a cone-set Geometry Light: pick a cone by object area
// (frustum lateral π(r0+r1)·slant + enabled caps πr²), then pick lateral vs a
// cap. Lateral uses the radius-weighted axial CDF r(t)=√((1-u)r0²+u·r1²); the
// outward object normal is the tilted slant normal on the wall, ±axis on a cap.
// Single-sided.
VISRTX_DEVICE LightSample sampleConeGeometryLight(const LightGPUData &ld,
    const ConeGeometryData &cone,
    const mat4 &xfm,
    const vec3 &origin,
    ScreenSample &ss,
    DeviceObjectIndex surfaceInstanceIndex)
{
  if (cone.numPrimitives == 0 || ld.geometry.area <= 0.0f)
    return {};

  const uint32_t primID = glm::min(
      uint32_t(inverseSampleCDF(
          cone.primAreaCdf, int(cone.numPrimitives), pcg_uniform(&ss.rs))),
      cone.numPrimitives - 1);
  const uvec2 idx =
      cone.indices ? cone.indices[primID] : uvec2(0, 1) + primID * 2;
  const vec3 p0 = cone.vertices[idx.x];
  const vec3 p1 = cone.vertices[idx.y];
  // cone.radii is mandatory (Cone::isValid requires vertex.radius), so it is
  // never null when a cone Geometry Light exists — no global-radius fallback.
  const float r0 = fabsf(cone.radii[idx.x]);
  const float r1 = fabsf(cone.radii[idx.y]);
  const vec3 axis = p1 - p0;
  const float len = length(axis);
  if (len <= 0.0f)
    return {};
  const vec3 axisN = axis / len;
  const mat3 basis = computeOrthonormalBasis(axisN);

  bool cap0, cap1;
  resolveEndpointCaps(cone.vertexCaps, cone.defaultCapFlags, idx, cap0, cap1);
  const float slant = sqrtf(len * len + (r0 - r1) * (r0 - r1));
  const float latArea = kPi * (r0 + r1) * slant;
  const float cap0Area = cap0 ? kPi * r0 * r0 : 0.0f;
  const float cap1Area = cap1 ? kPi * r1 * r1 : 0.0f;

  float pick = pcg_uniform(&ss.rs) * (latArea + cap0Area + cap1Area);
  vec3 pObj, nObj;
  float axialT; // axial fraction along p0→p1; the hit's uvw parameter
  if (pick < latArea) {
    const float u = pcg_uniform(&ss.rs);
    const float rt = sqrtf((1.0f - u) * r0 * r0 + u * r1 * r1); // radius at t
    axialT = (fabsf(r1 - r0) > CONE_TAPER_EPSILON) ? (rt - r0) / (r1 - r0) : u;
    const float phi = kTwoPi * pcg_uniform(&ss.rs);
    const vec3 rho = cosf(phi) * basis[0] + sinf(phi) * basis[1];
    pObj = p0 + axialT * axis + rt * rho;
    // Outward slant normal: radial tilted toward the axis by the taper slope.
    nObj = normalize(rho + ((r0 - r1) / len) * axisN);
  } else if (cap0Area > 0.0f && pick < latArea + cap0Area) {
    pObj = sampleDisk(p0, basis[0], basis[1], r0, ss);
    nObj = -axisN;
    axialT = 0.0f;
  } else {
    pObj = sampleDisk(p1, basis[0], basis[1], r1, ss);
    nObj = axisN;
    axialT = 1.0f;
  }
  return finishAreaLightSample(ss,
      ld,
      xfm,
      origin,
      pObj,
      nObj,
      primID,
      vec3(0.0f, axialT, 1.0f - axialT),
      surfaceInstanceIndex);
}

// Dispatch a Geometry Light sample by the backing geometry's type. The geometry
// carries its own type, so GeometryLightGPUData needs no discriminator.
VISRTX_DEVICE LightSample sampleGeometryLight(const LightGPUData &ld,
    const mat4 &xfm,
    const vec3 &origin,
    ScreenSample &ss,
    DeviceObjectIndex surfaceInstanceIndex)
{
  const auto &geom =
      ss.frameData->registry.geometries[ld.geometry.geometryIndex];
  switch (geom.type) {
  case GeometryType::TRIANGLE:
    return sampleTriangleGeometryLight(
        ld, geom.tri, xfm, origin, ss, surfaceInstanceIndex);
  case GeometryType::SPHERE:
    return sampleSphereGeometryLight(
        ld, geom.sphere, xfm, origin, ss, surfaceInstanceIndex);
  case GeometryType::CYLINDER:
    return sampleCylinderGeometryLight(
        ld, geom.cylinder, xfm, origin, ss, surfaceInstanceIndex);
  case GeometryType::CONE:
    return sampleConeGeometryLight(
        ld, geom.cone, xfm, origin, ss, surfaceInstanceIndex);
  default:
    return {};
  }
}

VISRTX_DEVICE LightSample sampleHDRILight(
    const LightGPUData &ld, const mat4 &xfm, const vec3 &dir)
{
  // Convert direction to spherical coordinates for environment map lookup
  auto thetaPhi = sphericalCoordsFromDirection(ld.hdri.xfm * dir);
  // Map spherical coordinates to UV texture coordinates
  // θ ∈ [0,π] → v ∈ [0,1], φ ∈ [0,2π] → u ∈ [0,1]
  auto uv = glm::vec2(thetaPhi.y, thetaPhi.x) / glm::vec2(kTwoPi, kPi);

  auto radiance = sampleHDRI(ld, uv);
  // pdf_ω = (L/totalL) · pdfWeight; the equirectangular sinθ jacobian is
  // already folded into the CDF (computeWeightedLuminance) and into
  // pdfWeight's 2π²/(W·H) factor, so do not re-multiply by sinθ here.
  auto pdf = dot(radiance, {0.2126f, 0.7152f, 0.0722f}) * ld.hdri.pdfWeight;

  LightSample ls;
  ls.dir = xfmVec(xfm, dir);
  ls.dist =
      std::numeric_limits<float>::infinity(); // Environment is at infinity
  ls.radiance = radiance * ld.hdri.scale * ld.color;
  ls.pdf = pdf;

  return ls;
}

VISRTX_DEVICE LightSample sampleHDRILight(
    const LightGPUData &ld, const mat4 &xfm, RandState &rs)
{
  // Importance sampling using hierarchical (marginal/conditional) CDF approach
  // First sample row (y) using marginal CDF, then column (x) using conditional
  // CDF
  auto y =
      inverseSampleCDF(ld.hdri.marginalCDF, ld.hdri.size.y, pcg_uniform(&rs));
  auto x = inverseSampleCDF(ld.hdri.conditionalCDF + y * ld.hdri.size.x,
      ld.hdri.size.x,
      pcg_uniform(&rs));

  auto xy = glm::uvec2(x, y);

#ifdef VISRTX_ENABLE_HDRI_SAMPLING_DEBUG
  if (ld.hdri.samples) {
    atomicInc(ld.hdri.samples + y * ld.hdri.size.x + x, ~0u);
  }
#endif
  // Add sub-pixel jitter to avoid aliasing
  auto jitter = glm::vec2(pcg_uniform(&rs), pcg_uniform(&rs));
  auto uv =
      glm::clamp((glm::vec2(xy) + jitter) / glm::vec2(ld.hdri.size), 0.f, 1.f);

  // Convert UV coordinates to spherical coordinates
  // uv.y ∈ [0,1] → θ ∈ [0,π], uv.x ∈ [0,1] → φ ∈ [0,2π]
  auto thetaPhi = kPi * glm::vec2(uv.y, 2.0f * (uv.x));

  // pdf_ω = (L/totalL) · pdfWeight; the equirectangular sinθ jacobian is
  // already folded into the CDF and pdfWeight, so do not re-multiply here.
  auto radiance = sampleHDRI(ld, uv);
  auto pdf = dot(radiance, {0.2126f, 0.7152f, 0.0722f}) * ld.hdri.pdfWeight;

  LightSample ls;
  // Transform spherical direction to world space
  // ld.hdri.xfm is orthogonal, so we can use right-hand multiplication
  // instead of explicitly transposing/inverting the matrix
  ls.dir = xfmVec(xfm, sphericalCoordsToDirection(thetaPhi) * ld.hdri.xfm);
  ls.dist = 1e20f; // Environment is effectively at infinity
  ls.radiance = radiance * ld.hdri.scale * ld.color;
  ls.pdf = pdf;

  return ls;
}

} // namespace detail

// surfaceInstanceIndex is used only by Geometry Lights (index into
// world.surfaceInstances for instance-attribute emission); -1 for all others.
VISRTX_DEVICE LightSample sampleLight(ScreenSample &ss,
    const vec3 &origin,
    DeviceObjectIndex idx,
    const mat4 &xfm,
    DeviceObjectIndex surfaceInstanceIndex)
{
  auto &ld = ss.frameData->registry.lights[idx];

  switch (ld.type) {
  case LightType::DIRECTIONAL:
    return detail::sampleDirectionalLight(ld, xfm);
  case LightType::POINT:
    return detail::samplePointLight(ld, xfm, origin);
  case LightType::SPHERE:
    return detail::sampleSphereLight(ld, xfm, origin, ss.rs);
  case LightType::RECT:
    return detail::sampleRectLight(ld, xfm, origin, ss.rs);
  case LightType::SPOT:
    return detail::sampleSpotLight(ld, xfm, origin);
  case LightType::RING:
    return detail::sampleRingLight(ld, xfm, origin, ss.rs);
  case LightType::HDRI:
    return detail::sampleHDRILight(ld, xfm, ss.rs);
  case LightType::GEOMETRY:
    return detail::sampleGeometryLight(
        ld, xfm, origin, ss, surfaceInstanceIndex);
  default:
    break;
  }

  return {};
}

} // namespace visrtx
