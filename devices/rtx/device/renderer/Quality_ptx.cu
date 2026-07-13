/*
 * Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <optix_device.h>
#include "gpu/createScreenSample.h"
#include "gpu/evalShading.h"
#include "gpu/gpu_debug.h"
#include "gpu/gpu_decl.h"
#include "gpu/gpu_math.h"
#include "gpu/gpu_util.h"
#include "gpu/intersectRay.h"
#include "gpu/lightPickPower.h"
#include "gpu/populateHit.h"
#include "gpu/renderer/common.h"
#include "gpu/renderer/shadowTransmittance.h"
#include "gpu/sampleLight.h"
#include "gpu/shadingState.h"
#include "gpu/volumeIntegration.h"

#include <limits>

namespace visrtx {

constexpr float PATH_CONTRIBUTION_EPSILON = 1.0e-8f;
constexpr float ATTENUATION_EPSILON = std::numeric_limits<float>::epsilon();
constexpr float INV_4PI = 1.0f / (4.0f * kPi);
// Pre-shadow skip: a contribution below SHADOW_SKIP_EPSILON can't survive RGB
// quantisation even unattenuated, so skipping the trace entirely costs nothing.
constexpr float SHADOW_SKIP_EPSILON = 1.0e-5f;
// RR start depth. Volume scatter engages earlier — throughput shrinks by
// medium albedo each scatter, so dense regions need RR sooner. Surface
// bounces don't shrink throughput so reliably; keep their conservative
// threshold.
constexpr int RUSSIAN_ROULETTE_START_DEPTH = 3;
constexpr int RUSSIAN_ROULETTE_START_DEPTH_VOLUME = 1;
constexpr float VOLUME_SCATTER_EPSILON = 1.0e-4f;

DECLARE_FRAME_DATA(frameData)

struct VolumeDistanceSample
{
  bool didScatter;
  vec3 albedo;
  float depth;
  vec3 normal;
  float extinction;
  uint32_t objID;
  uint32_t instID;
};

struct SampleDetails
{
  vec3 color;
  float opacity;
  vec3 albedo;
  float depth;
  vec3 normal;
};

VISRTX_DEVICE void accumPixelSample(
    const FrameGPUData &frame, const uvec2 &pixel, const SampleDetails &sample)
{
  accumPixelSample(frame,
      pixel,
      vec4(sample.color, sample.opacity),
      sample.albedo,
      sample.normal);
}

// Per-channel surface shadow blocking (vec3): opacity scaled by how much light
// the tinted transmission lets through. Component-wise 0 = transparent,
// 1 = fully blocking.
VISRTX_DEVICE vec3 shadowBlocking(const MaterialShadingState &shadingState)
{
  return materialEvaluateOpacity(shadingState)
      * (1.0f - materialEvaluateTransmission(shadingState));
}

VISRTX_DEVICE bool shouldTerminatePath(ScreenSample &ss,
    int depth,
    vec3 &contribution,
    bool useRussianRoulette,
    int rrStartDepth = RUSSIAN_ROULETTE_START_DEPTH,
    float maxSurvivalProb = 0.95f)
{
  if (glm::all(glm::lessThan(contribution, vec3(PATH_CONTRIBUTION_EPSILON))))
    return true;

  if (!useRussianRoulette || depth < rrStartDepth)
    return false;

  // Survival cap. Surface default 0.95 keeps contributing paths alive
  // (bouncing is cheap). White-smoke / dense-cloud volumes keep
  // max(contribution) near 1 forever; the lower 0.5 cap forces probabilistic
  // termination so the warp doesn't pin on the longest lane.
  const float maxContribution =
      glm::max(contribution.x, glm::max(contribution.y, contribution.z));
  const float survivalProb = glm::min(maxSurvivalProb, maxContribution);
  if (pcg_uniform(&ss.rs) > survivalProb)
    return true;

  contribution /= survivalProb;
  return false;
}

// A NEE light sample plus which BSDF-reachable kind it is. The environment (env
// MIS) and Geometry Lights can both also be hit by a BSDF escape/continuation,
// so both get a balance-heuristic weight; all other light types cannot, so they
// take w_nee = 1.
struct SurfaceLightSample
{
  LightSample ls;
  bool isEnv;
  bool isGeometry;
};

// Power (relative flux) of the ambient term, treated as an infinite hemisphere
// light so it competes in the same Pick Power currency as the light instances
// (irradiance × scene cross-section, matching lightPickPower's infinite lights).
VISRTX_DEVICE float ambientPickPower(const FrameGPUData &frameData)
{
  const auto &r = frameData.renderer;
  if (r.ambientIntensity <= 0.0f)
    return 0.0f;
  const float radius = frameData.world.sceneRadius;
  // Irradiance from a constant-radiance ambient hemisphere is pi * L; times the
  // scene cross-section pi * R^2 for the flux the pick weights compare.
  return luminance(r.ambientColor) * r.ambientIntensity * kPi * kPi * radius
      * radius;
}

// Pick a light instance ∝ Pick Power via the world's cumulative CDF. Clamped so
// float rounding at u≈1 cannot index past the last instance.
VISRTX_DEVICE size_t pickLightInstance(const WorldGPUData &world, float u)
{
  const size_t n = world.numLightInstances;
  const size_t idx =
      size_t(detail::inverseSampleCDF(world.lightPickCdf, int(n), u));
  return glm::min(idx, n - 1);
}

// Discrete probability that pickLightInstance selected `idx`, folded with the
// ambient stratum so P(pick) sums to 1 across every pick candidate. The CDF is
// normalized by totalLightPower, so its per-slot delta is power_i/totalLightPower.
VISRTX_DEVICE float instancePickProbability(
    const WorldGPUData &world, size_t idx, float totalPower)
{
  const float lo = idx > 0 ? world.lightPickCdf[idx - 1] : 0.0f;
  const float conditional = world.lightPickCdf[idx] - lo;
  return conditional * world.totalLightPower / totalPower;
}

// Aggregate probability that the Light Pick lands on the HDRI environment. Both
// env-MIS sides fold this into the env light density. Falls back to the uniform
// stratum fraction when no light carries Pick Power (matching sampleLights).
VISRTX_DEVICE float envPickProbability(const FrameGPUData &frameData)
{
  const auto &world = frameData.world;
  const float ambientPower = ambientPickPower(frameData);
  const float totalPower = world.totalLightPower + ambientPower;
  if (totalPower > 0.0f)
    return world.hdriPower / totalPower;
  // All-dark fallback: mirror sampleLights' uniform stratum count exactly
  // (ambient counted iff it carries Pick Power) so both MIS sides agree.
  const size_t numStrata = world.numLightInstances + (ambientPower > 0.0f ? 1 : 0);
  return numStrata > 0
      ? float(world.numHdriLightInstances) / float(numStrata)
      : 0.0f;
}

// NEE density a Geometry Light would report for a BSDF ray that hit it, for the
// hit-side MIS weight. Recomputed from the hit — the same exact per-triangle
// Jacobian the sampler uses and the same isotropic pick probability the CDF
// used — so wNee and wBsdf evaluate one pdf function and partition to 1.
// Returns 0 when the hit surface is not a Geometry Light (deposit stays weight
// 1). Pick Power uses the same emission the CDF was built from so the two sides
// agree: the material's emissionAverage for a non-constant emitter (a mean for
// native textured emission; for MDL the dynamic-recipe live mean, or the unit
// proxy when no recipe resolves), and the hit's own (constant) emission
// otherwise. `emission` is the surface's evaluated radiance.
VISRTX_DEVICE float geometryLightHitPdf(const FrameGPUData &frameData,
    const SurfaceHit &hit,
    const vec3 &rayDir,
    const vec3 &emission)
{
  // Only a sampleable-emissive area-samplable surface is a Geometry Light. The
  // type guard is load-bearing: GeometryGPUData is a union, so reading `.tri`/
  // `.sphere` on the wrong type (never NEE-sampled) would misread it and wrongly
  // down-weight the deposit.
  if (!hit.material->emissionIsSampleable)
    return 0.0f;
  // objectToWorld is a row-stored mat3x4 (glm column i = OptiX row i of M), so
  // mat3(objectToWorld) is Mᵀ; transpose it back to M — the linear map the NEE
  // sampler uses via xfmVec — or the area Jacobian is wrong under a non-symmetric
  // instance transform (rotation + non-uniform scale).
  const mat3 o2w = transpose(mat3(hit.instance->objectToWorld));
  const float cosTheta = fabsf(dot(hit.Ng, rayDir));
  if (cosTheta <= 0.0f)
    return 0.0f;

  // Per-type: the exact solid-angle pdf the NEE sampler would report for this
  // hit, plus the object-space total area feeding the pick probability. Kept
  // identical to the samplers in sampleLight.h so wNee and wBsdf partition to 1.
  float solidAnglePdf = 0.0f;
  float totalArea = 0.0f;

  if (hit.geometry->type == GeometryType::TRIANGLE) {
    const auto &tri = hit.geometry->tri;
    if (tri.totalArea <= 0.0f || tri.numPrimitives == 0)
      return 0.0f;
    const uvec3 idx = detail::triangleIndices(tri, hit.primID);
    const vec3 v0 = tri.vertices[idx.x];
    const vec3 e1o = tri.vertices[idx.y] - v0;
    const vec3 e2o = tri.vertices[idx.z] - v0;
    const float worldTwice = length(cross(o2w * e1o, o2w * e2o));
    if (worldTwice <= 0.0f)
      return 0.0f;
    solidAnglePdf = detail::geometryLightSolidAnglePdf(
        length(cross(e1o, e2o)), worldTwice, tri.totalArea, hit.t, cosTheta);
    totalArea = tri.totalArea;
  } else {
    // Sphere/cylinder/cone samplers are SINGLE-sided (outward): finishAreaLightSample
    // culls the far hemisphere (cosTheta <= 0). hit.Ng is ray-oriented so fabsf above
    // cannot recover facing — an interior (back-face) hit is never NEE-sampled, so its
    // NEE pdf must be 0. Otherwise the deposit sees pNee > 0 and down-weights via MIS
    // while NEE contributes nothing, losing the interior fraction (dark shell inside).
    if (!hit.isFrontFace)
      return 0.0f;
    // The unit-tangent samplers depend only on the OUTWARD object normal at the point
    // (worldAreaScale = |cross(M t1,M t2)|, invariant to the tangent basis). Recover it
    // generically from the world normal — o2wᵀ·Ng ∝ nObj since Ng = normalize(M⁻ᵀ·nObj)
    // — so no per-surface (lateral vs cap, slant) math is needed and it matches
    // finishAreaLightSample exactly.
    uint32_t numPrimitives = 0;
    if (hit.geometry->type == GeometryType::SPHERE) {
      totalArea = hit.geometry->sphere.totalArea;
      numPrimitives = hit.geometry->sphere.numPrimitives;
    } else if (hit.geometry->type == GeometryType::CYLINDER) {
      totalArea = hit.geometry->cylinder.totalArea;
      numPrimitives = hit.geometry->cylinder.numPrimitives;
    } else if (hit.geometry->type == GeometryType::CONE) {
      totalArea = hit.geometry->cone.totalArea;
      numPrimitives = hit.geometry->cone.numPrimitives;
    } else {
      return 0.0f;
    }
    if (totalArea <= 0.0f || numPrimitives == 0)
      return 0.0f;
    const vec3 nObj = normalize(transpose(o2w) * hit.Ng);
    const mat3 basis = computeOrthonormalBasis(nObj);
    const float worldAreaScale = length(cross(o2w * basis[0], o2w * basis[1]));
    if (worldAreaScale <= 0.0f)
      return 0.0f;
    solidAnglePdf = detail::geometryLightSolidAnglePdf(
        1.0f, worldAreaScale, totalArea, hit.t, cosTheta);
  }

  const float totalPower =
      frameData.world.totalLightPower + ambientPickPower(frameData);
  if (totalPower <= 0.0f)
    return 0.0f;

  LightGPUData ld{};
  ld.type = LightType::GEOMETRY;
  ld.geometry.geometryIndex = -1; // unused by lightPickPower
  // A constant emitter's CDF weight is its own radiance (== emissionAverage for
  // the native path); a textured emitter's varies per point, so its CDF was
  // built from the mean — use that here so pick probabilities match.
  ld.geometry.radiance = hit.material->emissionIsConstant
      ? emission
      : hit.material->emissionAverage;
  ld.geometry.area = totalArea;
  const float pickProb =
      lightPickPower(ld, mat4(o2w), frameData.world.sceneRadius) / totalPower;

  return solidAnglePdf * pickProb;
}

// One Light Pick: which candidate (a light instance or the ambient term) was
// drawn, and its discrete pick probability folded into the returned pdf. Shared
// by the surface and volume samplers, which differ only in how they turn the
// ambient stratum into a direction.
struct PickedCandidate
{
  bool valid; // false when there are no candidates at all
  bool isAmbient;
  size_t instance; // index into world.lightInstances (when !isAmbient)
  float pickPdf; // probability of this pick, ∝ Pick Power
};

VISRTX_DEVICE PickedCandidate pickCandidate(
    ScreenSample &ss, const FrameGPUData &frameData)
{
  const auto &world = frameData.world;
  const float ambientPower = ambientPickPower(frameData);
  const bool hasAmbient = ambientPower > 0.0f;

  if (world.numLightInstances == 0 && !hasAmbient)
    return {false, false, 0, 0.0f};

  const float totalPower = world.totalLightPower + ambientPower;

  // Fallback when no candidate carries Pick Power (all dark): uniform pick keeps
  // the estimator unbiased and avoids a divide-by-zero.
  if (!(totalPower > 0.0f)) {
    const size_t numStrata = world.numLightInstances + (hasAmbient ? 1 : 0);
    const size_t selected =
        glm::min(size_t(pcg_uniform(&ss.rs) * float(numStrata)), numStrata - 1);
    const float pickPdf = 1.0f / float(numStrata);
    if (hasAmbient && selected == world.numLightInstances)
      return {true, true, 0, pickPdf};
    return {true, false, selected, pickPdf};
  }

  if (hasAmbient && pcg_uniform(&ss.rs) * totalPower < ambientPower)
    return {true, true, 0, ambientPower / totalPower};

  const size_t selected = pickLightInstance(world, pcg_uniform(&ss.rs));
  return {true,
      false,
      selected,
      instancePickProbability(world, selected, totalPower)};
}

VISRTX_DEVICE SurfaceLightSample sampleLights(ScreenSample &ss,
    const FrameGPUData &frameData,
    const vec3 &origin,
    const vec3 &normal)
{
  const PickedCandidate pick = pickCandidate(ss, frameData);
  if (!pick.valid)
    return {};

  if (pick.isAmbient) {
    // Cosine-weighted hemisphere sample; pdf cos(theta)/pi folded with the pick
    // probability so MIS weights see the full joint pdf.
    const auto &rp = frameData.renderer;
    const vec3 dir = sampleHemisphere(ss.rs, normal);
    const float cosNs = fmaxf(0.f, dot(dir, normal));
    return {LightSample{rp.ambientColor * rp.ambientIntensity,
                dir,
                std::numeric_limits<float>::max(),
                pick.pickPdf * cosNs * kInvPi},
        false,
        false};
  }

  const auto &li = frameData.world.lightInstances[pick.instance];
  auto ls =
      sampleLight(ss, origin, li.lightIndex, li.xfm, li.surfaceInstanceIndex);
  ls.pdf *= pick.pickPdf;
  const LightType type = frameData.registry.lights[li.lightIndex].type;
  return {ls, type == LightType::HDRI, type == LightType::GEOMETRY};
}

VISRTX_DEVICE LightSample sampleLightsVolume(
    ScreenSample &ss, const FrameGPUData &frameData, const vec3 &origin)
{
  const PickedCandidate pick = pickCandidate(ss, frameData);
  if (!pick.valid)
    return {};

  if (pick.isAmbient) {
    // Uniform-sphere sample (pdf 1/(4π)) to match the isotropic phase function.
    const auto &rp = frameData.renderer;
    const vec3 dir = randomDir(ss.rs);
    return LightSample{rp.ambientColor * rp.ambientIntensity,
        dir,
        std::numeric_limits<float>::max(),
        pick.pickPdf * INV_4PI};
  }

  const auto &li = frameData.world.lightInstances[pick.instance];
  auto ls =
      sampleLight(ss, origin, li.lightIndex, li.xfm, li.surfaceInstanceIndex);
  ls.pdf *= pick.pickPdf;
  return ls;
}

VISRTX_DEVICE
VolumeDistanceSample sampleVolumeDistance(
    ScreenSample &ss, Ray ray, bool needNormal)
{
  VolumeDistanceSample volumeHit = {
      false, vec3(0.0f), ray.t.upper, vec3(0.0f), 0.0f, ~0u, ~0u};

  // Skip the gradient-based normal computation on non-primary bounces.
  volumeHit.depth = sampleDistanceAllVolumes(ss,
      ray,
      RayType::PRIMARY,
      ray.t.upper,
      volumeHit.albedo,
      volumeHit.extinction,
      volumeHit.didScatter,
      volumeHit.objID,
      volumeHit.instID,
      needNormal ? &volumeHit.normal : nullptr);
  return volumeHit;
}

VISRTX_GLOBAL void __closesthit__shading()
{
  ray::populateHit();
}

VISRTX_GLOBAL void __anyhit__shading()
{
  ray::cullCutPlane();
}

VISRTX_GLOBAL void __miss__shading()
{
  if (ray::isIntersectingSurfaces()) {
    auto &hit = ray::rayData<SurfaceHit>();
    hit.foundHit = false;
  } else {
    auto &hit = ray::rayData<VolumeHit>();
    hit.foundHit = false;
  }
}

VISRTX_GLOBAL void __closesthit__shadow() {}

VISRTX_GLOBAL void __anyhit__shadow()
{
  auto &attenuation = ray::rayData<vec3>();

  if (ray::isIntersectingSurfaces()) {
    ray::cullCutPlane();
    SurfaceHit hit;
    ray::populateSurfaceHit(hit);

    // Fully opaque material: skip the init / opacity / transmission callable
    // dispatch chain and just block the ray.
    if (hit.material->isFullyOpaque) {
      attenuation = vec3(0.0f);
      optixTerminateRay();
      return;
    }

    MaterialShadingState shadingState;
    materialInitShading(&shadingState, frameData, *hit.material, hit);
    auto blocking = shadowBlocking(shadingState);

    attenuation *= (1.0f - blocking);

    if (glm::all(glm::lessThanEqual(attenuation, vec3(ATTENUATION_EPSILON))))
      optixTerminateRay();
    else
      optixIgnoreIntersection();
  } else {
    VolumeHit hit;
    ray::populateVolumeHit(hit);

    // Unbiased ratio-tracking transmittance over this volume segment.
    // Scalar σ_t (TF is monochrome) broadcast to vec3 in the callee.
    ratioTrackTransmittanceVolume(ray::screenSample(), hit, attenuation);

    if (glm::all(glm::lessThanEqual(attenuation, vec3(ATTENUATION_EPSILON))))
      optixTerminateRay();
    else
      optixIgnoreIntersection();
  }
}

VISRTX_GLOBAL void __miss__shadow() {}

VISRTX_GLOBAL void __raygen__()
{
  auto ss = createScreenSample(frameData);
  if (pixelOutOfFrame(ss.pixel, frameData.fb))
    return;

  const auto &rendererParams = frameData.renderer;
  const auto &qualityParams = rendererParams.params.quality;

  for (int i = 0; i < rendererParams.numIterations; ++i) {
    bool isVeryFirstRay = i == 0 && ss.frameData->fb.frameID == 0;
    // Halton sample index: per-pixel ordinal across the whole frame's
    // sample budget. Using `frameID * numIterations + i` keeps the
    // per-pixel Halton indices contiguous [0, totalSpp), which is the
    // optimal QMC stratification.
    const uint32_t sampleIdx = uint32_t(ss.frameData->fb.frameID)
            * uint32_t(rendererParams.numIterations)
        + uint32_t(i);
    auto ray = makePrimaryRay(ss, sampleIdx, isVeryFirstRay);

    applyCuttingPlane(rendererParams.cutPlane, ray);

    SampleDetails sample = {
        vec3(0.0f), 0.0f, vec3(0.0f), ray.t.upper, vec3(0.0f)};

    auto sampleContribution = vec3(1.0f);

    // The environment (visible HDRI lights) is sampled both by NEE at every
    // scatter vertex (HDRIs are in the light list) and by a BSDF ray that
    // escapes to it. Balance-heuristic MIS combines the two: `bsdfPdf` carries
    // the solid-angle pdf of the bounce that produced the current ray, so the
    // miss can weight the escape estimator by bsdfPdf/(bsdfPdf + pLight). The
    // primary ray is a delta event (the directly visible backdrop), so it
    // starts at +inf => w_bsdf = 1.
    float bsdfPdf = INFINITY;

    // Probability the power-proportional Light Pick lands on the environment,
    // matching sampleLights. Folded into the env light density on both MIS
    // sides so wNee and wBsdf use identical pdf functions.
    const float envPickProb = envPickProbability(frameData);

    // Coverage pass-throughs are not light-transport events, so they track a
    // separate, generous budget instead of spending bounceDepth — a deep stack
    // of alpha cutouts must not starve the indirect-bounce budget.
    int bounceDepth = 0;
    int transparencyDepth = 0;
    while (bounceDepth < qualityParams.maxRayDepth) {
      const bool isFirstBounce = bounceDepth == 0 && transparencyDepth == 0;

      SurfaceHit surfaceHit = {};
      intersectSurface(ss,
          ray,
          RayType::PRIMARY,
          &surfaceHit,
          primaryRayOptiXFlags(rendererParams));

      float volumeUpperBound = surfaceHit.foundHit ? surfaceHit.t : ray.t.upper;
      auto volumeRay = Ray{ray.org, ray.dir, {ray.t.lower, volumeUpperBound}};

      auto volumeSample = sampleVolumeDistance(ss, volumeRay, isFirstBounce);

      if (volumeSample.didScatter) {
        const vec3 scatterPos = ray.org + ray.dir * volumeSample.depth;

        {
          LightSample lightSample =
              sampleLightsVolume(ss, frameData, scatterPos);
          // Gate on a positive pdf, NOT a fixed epsilon: a dim light's pick
          // probability can make the joint pdf legitimately tiny, and dividing
          // by it stays unbiased. An epsilon floor would drop those samples and
          // render the dim light black — the very bright+dim case the power pick
          // targets.
          if (lightSample.pdf > 0.0f && lightSample.dist > 0.0f) {
            const vec3 directLight = volumeSample.albedo * lightSample.radiance
                * INV_4PI / lightSample.pdf;
            const vec3 contribUpper = sampleContribution * directLight;
            const float maxContrib = glm::max(
                contribUpper.x, glm::max(contribUpper.y, contribUpper.z));
            if (maxContrib >= SHADOW_SKIP_EPSILON) {
              const float eps = VOLUME_SCATTER_EPSILON;
              const Ray shadowRay = {
                  scatterPos + lightSample.dir * eps,
                  lightSample.dir,
                  {eps, lightSample.dist},
              };
              // Adaptive RR knob: w in (0, 1] = maxContrib / 0.5. Dim rays
              // raise the in-trace RR threshold so ratio-tracking kills them
              // sooner. RR estimator stays unbiased; cap inside RR bounds
              // amplification.
              ss.shadowContribWeight = glm::min(1.0f, maxContrib * 2.0f);
              const auto attenuation = surfaceShadowTransmittance(ss, shadowRay)
                  * volumeShadowTransmittance(ss, shadowRay);
              ss.shadowContribWeight = 1.0f;
              sample.color += contribUpper * attenuation;
            }
          }
        }

        accumulateValue(sample.opacity, 1.0f, sample.opacity);

        // Record first-hit AOVs (object/instance id, depth, albedo, normal)
        // BEFORE the contribution-based path termination below. A fully opaque
        // but zero-albedo (black) volume scatters here, then drives
        // sampleContribution to 0 via the albedo multiply — shouldTerminatePath
        // would break out before these AOVs were written, leaving the object-id
        // / depth / normal buffers unset for opaque-black regions. The AOVs are
        // first-hit metadata, independent of the path's radiance contribution.
        if (isFirstBounce) {
          setPixelIds(frameData.fb,
              ss.pixel,
              volumeSample.depth,
              volumeSample.objID,
              volumeSample.objID,
              volumeSample.instID);
          sample.depth = volumeSample.depth;
          sample.albedo = volumeSample.albedo;
          const vec3 volumeNormal = glm::length(volumeSample.normal) > 0.01f
              ? volumeSample.normal
              : -ray.dir;
          sample.normal = volumeNormal;
        }

        sampleContribution *= volumeSample.albedo;
        if (shouldTerminatePath(ss,
                bounceDepth,
                sampleContribution,
                true,
                RUSSIAN_ROULETTE_START_DEPTH_VOLUME,
                /*maxSurvivalProb=*/0.5f))
          break;

        const vec3 scatterDir = randomDir(ss.rs);
        ray = Ray{scatterPos + scatterDir * VOLUME_SCATTER_EPSILON, scatterDir};
        // The volume NEE above already sampled the environment at this scatter
        // point, so the continuation ray must not re-deposit it on a miss
        // (bsdfPdf = 0 => w_bsdf = 0). Env MIS for volumes is left as-is.
        bsdfPdf = 0.0f;
        ++bounceDepth;
        continue;
      }

      if (surfaceHit.foundHit) {
        MaterialShadingState shadingState;
        materialInitShading(
            &shadingState, frameData, *surfaceHit.material, surfaceHit);

        const vec3 materialEmission =
            materialEvaluateEmission(shadingState, -ray.dir);
        const vec3 materialTint = materialEvaluateTint(shadingState);
        const float opacity = materialEvaluateOpacity(shadingState);

        if (isFirstBounce) {
          setPixelIds(frameData.fb,
              ss.pixel,
              surfaceHit.t,
              surfaceHit.primID,
              surfaceHit.objID,
              surfaceHit.instID);
          sample.depth = surfaceHit.t;
          sample.normal = materialEvaluateNormal(shadingState);
          sample.albedo = materialTint;
        }

        // Emission, direct lighting are scaled by opacity analytically rather
        // than gated stochastically below. A Geometry Light reached by a finite-
        // pdf bounce is also sampled by NEE, so MIS-weight the deposit against
        // that; a delta/primary bounce (bsdfPdf == +inf) keeps weight 1 since
        // NEE cannot reach it, as do non-sampled emissive surfaces (pNee == 0).
        float wEmission = 1.0f;
        if (!isinf(bsdfPdf)) {
          const float pNee = geometryLightHitPdf(
              frameData, surfaceHit, ray.dir, materialEmission);
          if (pNee > 0.0f)
            wEmission = bsdfPdf / (bsdfPdf + pNee);
        }
        sample.color += wEmission * sampleContribution * opacity * materialEmission;
        // Sample around the shading normal so the cosine-weighted hemisphere's
        // pdf matches the BRDF's NdotL (which uses Ns). Sampling around Ng
        // would bias the Lambertian estimator by cos_Ns/cos_Ng on smooth or
        // bump-mapped surfaces.
        const vec3 shadowOrigin =
            shadingHitpoint(surfaceHit) + surfaceHit.Ng * surfaceHit.epsilon;
        const SurfaceLightSample lightPick =
            sampleLights(ss, frameData, shadowOrigin, surfaceHit.Ns);
        const LightSample &lightSample = lightPick.ls;
        // Positive-pdf gate (not an epsilon): a dim light's tiny pick
        // probability keeps NEE unbiased; an epsilon floor would drop it and
        // render it black in bright+dim scenes.
        if (lightSample.pdf > 0.0f && lightSample.dist > 0.0f) {
          // Gate on the shading normal so the terminator follows the smooth
          // surface; gating on Ng would carve the per-triangle facet shape
          // into the lit/unlit boundary at grazing light angles.
          const float lightDotNs = dot(lightSample.dir, surfaceHit.Ns);
          if (lightDotNs > 0.0f) {
            const vec3 directLight = materialShadeSurface(
                shadingState, surfaceHit, lightSample, -ray.dir);
            // Env MIS: only the HDRI environment can also be reached by the
            // BSDF escape, so only it gets a balance-heuristic weight. The
            // light density uses envPdf on BOTH sides (here and at the miss),
            // not lightSample.pdf, so wNee and wBsdf use identical pdf functions
            // and partition to 1 exactly — unbiased regardless of how closely
            // envPdf tracks the NEE importance pdf (the NEE estimator still
            // divides by its true lightSample.pdf, which carries the same
            // envPickProb, inside materialShadeSurface).
            // Other light types: p_bsdf = 0 => w_nee = 1 (behaviour unchanged).
            float wNee = 1.0f;
            if (lightPick.isEnv) {
              const float pBsdf =
                  materialEvalPdf(shadingState, -ray.dir, lightSample.dir);
              const float pLight =
                  envPdf(frameData, lightSample.dir) * envPickProb;
              wNee = pLight / (pLight + pBsdf);
            } else if (lightPick.isGeometry) {
              // lightSample.pdf is the exact NEE density (solid-angle × pick
              // probability); the BSDF continuation can also hit this Geometry
              // Light,
              // so weight against it. Mirrors geometryLightHitPdf on the deposit.
              const float pBsdf =
                  materialEvalPdf(shadingState, -ray.dir, lightSample.dir);
              wNee = lightSample.pdf / (lightSample.pdf + pBsdf);
            }
            const vec3 contribUpper =
                wNee * sampleContribution * opacity * directLight;
            const float maxContrib = glm::max(
                contribUpper.x, glm::max(contribUpper.y, contribUpper.z));
            if (maxContrib >= SHADOW_SKIP_EPSILON) {
              // A Geometry Light's sampled point lies on real, opaque geometry,
              // so stop the shadow ray just short of it or it self-occludes on
              // the emissive surface itself (~15% energy loss). Analytic lights
              // have no geometry there and keep the exact distance.
              const float shadowDist = lightPick.isGeometry
                  ? lightSample.dist * (1.0f - GEOMETRY_LIGHT_SHADOW_EPSILON)
                  : lightSample.dist;
              const Ray shadowRay = {
                  shadowOrigin,
                  lightSample.dir,
                  {surfaceHit.epsilon, shadowDist},
              };
              ss.shadowContribWeight = glm::min(1.0f, maxContrib * 2.0f);
              const auto attenuation = surfaceShadowTransmittance(ss, shadowRay)
                  * volumeShadowTransmittance(ss, shadowRay);
              ss.shadowContribWeight = 1.0f;
              sample.color += contribUpper * attenuation;
            }
          }
        }

        // Resolve geometric alpha stochastically for the continuation
        if (pcg_uniform(&ss.rs) > opacity) {
          if (++transparencyDepth > qualityParams.maxTransparencyDepth)
            break;
          ray = Ray{surfaceHit.hitpoint - surfaceHit.Ng * surfaceHit.epsilon,
              ray.dir};
          continue;
        }

        auto nextRay = materialNextRay(shadingState, ray, ss.rs);
        sampleContribution *= nextRay.contributionWeight;

        // Carry the bounce's solid-angle pdf for the env-MIS weight at a miss.
        // Reflection/diffuse lobes report a finite pdf (MIS-combined with NEE);
        // a transmission lobe reports +inf (NEE can't reach the env behind the
        // surface, so the escape owns it => w_bsdf = 1).
        bsdfPdf = nextRay.pdf;

        if (!continuesThroughSurface(nextRay))
          accumulateValue(sample.opacity, 1.0f, sample.opacity);

        if (shouldTerminatePath(ss, bounceDepth, sampleContribution, true))
          break;

        const float side = continuesThroughSurface(nextRay) ? -1.0f : 1.0f;
        ray =
            Ray{surfaceHit.hitpoint + surfaceHit.Ng * surfaceHit.epsilon * side,
                normalize(vec3(nextRay.direction))};
      }

      if (!surfaceHit.foundHit && !volumeSample.didScatter) {
        // Deposit the environment, MIS-weighted against NEE. pLight mirrors the
        // NEE env density: the HDRI importance pdf (envPdf) folded with the same
        // power-proportional env pick probability sampleLights applied. bsdfPdf
        // == +inf (delta / transmission / primary ray) => w_bsdf = 1.
        if (vec3 hdri; getBackgroundLight(frameData, ray.dir, hdri)) {
          const float pLight = envPdf(frameData, ray.dir) * envPickProb;
          const float wBsdf =
              isinf(bsdfPdf) ? 1.0f : bsdfPdf / (bsdfPdf + pLight);
          sample.color += wBsdf * sampleContribution * hdri;
          accumulateValue(sample.opacity, 1.f, sample.opacity);
        }

        if (isFirstBounce)
          setPixelIds(frameData.fb, ss.pixel, ray.t.upper, ~0u, ~0u, ~0u);

        break;
      }

      // Only a surface bounce reaches here (volume scatter and coverage
      // pass-through continue earlier, an environment miss breaks).
      ++bounceDepth;
    }

    accumPixelSample(frameData, ss.pixel, sample);
  }
}

} // namespace visrtx
