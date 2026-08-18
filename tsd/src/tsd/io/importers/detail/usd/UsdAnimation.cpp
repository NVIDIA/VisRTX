// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/importers/detail/usd/UsdAnimation.h"
#include "tsd/animation/AnimationManager.hpp"
#include "tsd/io/animation/UsdGeometryFileBinding.hpp"
#include "tsd/io/animation/UsdInstancerFileBinding.hpp"
// usd
#include <pxr/usd/usdGeom/pointBased.h>
#include <pxr/usd/usdGeom/pointInstancer.h>
#include <pxr/usd/usdGeom/xformable.h>
// std
#include <algorithm>
#include <cmath>
#include <vector>

namespace tsd::io::usd {

using namespace tsd::core;

namespace {

// Beyond this a rotation between two keys is subdivided; a quarter turn keeps
// spherical interpolation faithful without densifying content that does not
// need it.
constexpr float MAX_ROTATION_STEP = float(M_PI) * 0.5f;
constexpr int MAX_DENSIFY_DEPTH = 6;

// Largest angle any rotation basis vector turns through between two frames.
float rotationDelta(const tsd::math::mat4 &a, const tsd::math::mat4 &b)
{
  float retval = 0.f;
  for (int axis = 0; axis < 3; ++axis) {
    const auto u = tsd::math::normalize(tsd::math::to_float3(a[axis]));
    const auto v = tsd::math::normalize(tsd::math::to_float3(b[axis]));
    retval = std::max(
        retval, std::acos(std::clamp(tsd::math::dot(u, v), -1.f, 1.f)));
  }
  return retval;
}

struct TransformSampler
{
  pxr::UsdGeomXformable xformable;

  tsd::math::mat4 at(double time) const;
};

tsd::math::mat4 TransformSampler::at(double time) const
{
  pxr::GfMatrix4d local(1.0);
  bool resetsXformStack = false;
  xformable.GetLocalTransformation(
      &local, &resetsXformStack, pxr::UsdTimeCode(time));
  return toTsdMat4(local);
}

// A full turn authored with only two keys is degenerate under spherical
// interpolation: the endpoints coincide and the motion collapses. Recursively
// insert the midpoint wherever the path from a key to the interval's midpoint
// turns further than a quarter turn.
void densifyInterval(const TransformSampler &sampler,
    double t0,
    const tsd::math::mat4 &m0,
    double t1,
    const tsd::math::mat4 &m1,
    int depth,
    std::vector<double> &outTimes,
    std::vector<tsd::math::mat4> &outFrames)
{
  const double tm = 0.5 * (t0 + t1);
  const auto mm = sampler.at(tm);

  const bool needsSplit = depth < MAX_DENSIFY_DEPTH
      && (rotationDelta(m0, mm) > MAX_ROTATION_STEP
          || rotationDelta(mm, m1) > MAX_ROTATION_STEP);

  if (!needsSplit)
    return;

  densifyInterval(sampler, t0, m0, tm, mm, depth + 1, outTimes, outFrames);
  outTimes.push_back(tm);
  outFrames.push_back(mm);
  densifyInterval(sampler, tm, mm, t1, m1, depth + 1, outTimes, outFrames);
}

} // namespace

std::vector<float> normalizeSampleTimes(
    const pxr::UsdStageRefPtr &stage, const std::vector<double> &times)
{
  if (times.empty())
    return {};

  double start = stage->GetStartTimeCode();
  double end = stage->GetEndTimeCode();
  if (!(end > start)) {
    start = times.front();
    end = times.back();
  }
  const double span = end - start;

  std::vector<float> retval;
  retval.reserve(times.size());
  for (double t : times)
    retval.push_back(span > 0.0 ? float((t - start) / span) : 0.f);
  return retval;
}

void addTransformAnimation(
    ImportContext &ctx, const pxr::SdfPath &primPath, LayerNodeRef node)
{
  auto prim = ctx.stage->GetPrimAtPath(primPath);
  if (!prim)
    return;

  pxr::UsdGeomXformable xformable(prim);
  if (!xformable)
    return;

  std::vector<double> authoredTimes;
  xformable.GetTimeSamples(&authoredTimes);
  if (authoredTimes.size() < 2)
    return;

  TransformSampler sampler{xformable};

  std::vector<double> times;
  std::vector<tsd::math::mat4> frames;
  times.push_back(authoredTimes.front());
  frames.push_back(sampler.at(authoredTimes.front()));
  for (size_t i = 1; i < authoredTimes.size(); ++i) {
    const auto next = sampler.at(authoredTimes[i]);
    densifyInterval(sampler,
        authoredTimes[i - 1],
        frames.back(),
        authoredTimes[i],
        next,
        0,
        times,
        frames);
    times.push_back(authoredTimes[i]);
    frames.push_back(next);
  }

  addTransformStepBinding(ctx.animation(),
      node,
      frames,
      normalizeSampleTimes(ctx.stage, times));
  ctx.reportSampleCount(authoredTimes.size());
}

size_t pointInstancerSampleCount(
    ImportContext &ctx, const pxr::SdfPath &primPath)
{
  pxr::UsdGeomPointInstancer instancer(ctx.stage->GetPrimAtPath(primPath));
  if (!instancer)
    return 0;

  // Every attribute that can move a placement, including the two velocity
  // attributes Hydra folds into the instance transforms it computes.
  const pxr::UsdAttribute attributes[] = {instancer.GetPositionsAttr(),
      instancer.GetOrientationsAttr(),
      instancer.GetScalesAttr(),
      instancer.GetVelocitiesAttr(),
      instancer.GetAngularVelocitiesAttr(),
      instancer.GetProtoIndicesAttr(),
      instancer.GetInvisibleIdsAttr()};

  size_t retval = 0;
  for (const auto &attribute : attributes) {
    if (attribute)
      retval = std::max(retval, size_t(attribute.GetNumTimeSamples()));
  }
  return retval;
}

void addInstancerAnimation(ImportContext &ctx,
    const pxr::SdfPath &primPath,
    size_t prototypeIndex,
    LayerNodeRef arrayNode,
    ArrayRef transforms,
    size_t sampleCount)
{
  ctx.animation().emplaceFileBinding<UsdInstancerFileBinding>(&ctx.scene,
      ctx.session,
      arrayNode,
      transforms,
      ctx.filePath,
      primPath.GetString(),
      prototypeIndex);
  ctx.reportSampleCount(sampleCount);
}

void addDeformingGeometryAnimation(
    ImportContext &ctx, const pxr::SdfPath &primPath, GeometryRef geometry)
{
  auto prim = ctx.stage->GetPrimAtPath(primPath);
  if (!prim || !geometry)
    return;

  pxr::UsdGeomPointBased pointBased(prim);
  if (!pointBased)
    return;

  std::vector<double> sampleTimes;
  pointBased.GetPointsAttr().GetTimeSamples(&sampleTimes);
  if (sampleTimes.size() < 2)
    return;

  // One eager frame is already in the Scene; the rest is pulled from the
  // shared Stage Session on demand (ADR 0018).
  ctx.animation().emplaceFileBinding<UsdGeometryFileBinding>(&ctx.scene,
      geometry.data(),
      ctx.session,
      ctx.filePath,
      primPath.GetString());
  ctx.reportSampleCount(sampleTimes.size());
}

} // namespace tsd::io::usd
