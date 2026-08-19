// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/importers/detail/usd/UsdInstancing.h"
#include "tsd/io/importers/detail/usd/UsdAnimation.h"
#include "tsd/io/importers/detail/usd/UsdGeometry.h"
// usd
#include <pxr/base/gf/quatf.h>
#include <pxr/base/gf/quath.h>
#include <pxr/imaging/hd/instancerTopologySchema.h>
#include <pxr/imaging/hd/primOriginSchema.h>
#include <pxr/imaging/hd/primvarsSchema.h>
#include <pxr/imaging/hd/sceneIndexPrimView.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hd/xformSchema.h>
#include <pxr/usd/usdGeom/pointInstancer.h>
#include <pxr/usd/usdGeom/xformable.h>
// std
#include <algorithm>
#include <utility>
#include <vector>

namespace tsd::io::usd {

using namespace tsd::core;

namespace {

tsd::math::mat4 flattenedXformOf(
    const pxr::HdSceneIndexBaseRefPtr &sceneIndex, const pxr::SdfPath &primPath)
{
  auto prim = sceneIndex->GetPrim(primPath);
  auto xform = pxr::HdXformSchema::GetFromParent(prim.dataSource);
  if (!xform)
    return tsd::math::IDENTITY_MAT4;
  auto matrix = xform.GetMatrix();
  return matrix ? toTsdMat4(matrix->GetTypedValue(0))
                : tsd::math::IDENTITY_MAT4;
}

// The Stage path a resolved prototype prim came from, which is where its
// authored animation lives.
pxr::SdfPath originOf(const pxr::HdSceneIndexPrim &prim)
{
  auto origin = pxr::HdPrimOriginSchema::GetFromParent(prim.dataSource);
  if (!origin)
    return {};
  return origin.GetOriginPath(pxr::HdPrimOriginSchemaTokens->scenePath);
}

bool subtreeHasAnimatedTransforms(ImportContext &ctx,
    const pxr::HdSceneIndexBaseRefPtr &sceneIndex,
    const pxr::SdfPath &root)
{
  for (const pxr::SdfPath &path : pxr::HdSceneIndexPrimView(sceneIndex, root)) {
    const auto origin = originOf(sceneIndex->GetPrim(path));
    if (origin.IsEmpty())
      continue;
    auto prim = ctx.stage->GetPrimAtPath(origin);
    if (!prim)
      continue;
    pxr::UsdGeomXformable xformable(prim);
    if (!xformable)
      continue;
    std::vector<double> times;
    xformable.GetTimeSamples(&times);
    if (times.size() > 1)
      return true;
  }
  return false;
}

std::shared_ptr<PrototypeContent> convertPrototype(ImportContext &ctx,
    const pxr::HdSceneIndexBaseRefPtr &sceneIndex,
    const pxr::SdfPath &prototypeRoot,
    InstancerRegistry &registry)
{
  const auto key = prototypeRoot.GetString();
  if (auto found = registry.prototypes.find(key);
      found != registry.prototypes.end())
    return found->second;

  auto content = std::make_shared<PrototypeContent>();
  content->internalTransformsAnimated =
      subtreeHasAnimatedTransforms(ctx, sceneIndex, prototypeRoot);

  // Either way the Prototype's gprims convert exactly once and are shared by
  // every placement. Baking is what an animated Prototype gives up, not
  // sharing: its gprims keep their own transforms and are expanded per
  // placement as Layer nodes referencing these same objects.
  const auto rootXform = flattenedXformOf(sceneIndex, prototypeRoot);
  const auto inverseRoot = tsd::math::inverse(rootXform);
  for (const pxr::SdfPath &path :
      pxr::HdSceneIndexPrimView(sceneIndex, prototypeRoot)) {
    if (ctx.isClaimed(path))
      continue;
    auto prim = sceneIndex->GetPrim(path);
    if (!isGeometryPrimType(prim.primType))
      continue;
    const auto bake = content->internalTransformsAnimated
        ? tsd::math::IDENTITY_MAT4
        : tsd::math::mul(inverseRoot, flattenedXformOf(sceneIndex, path));
    for (auto &surface :
        convertGeometry(ctx, sceneIndex, path, prim, bake).surfaces)
      content->surfaces.push_back(surface);
    if (content->internalTransformsAnimated)
      content->gprimPaths.push_back(path);
  }
  ctx.report->convertedPrims += content->surfaces.size();

  registry.prototypes[key] = content;
  return content;
}

// Expanded fallback for Prototypes whose internal transforms are animated and
// so cannot be baked: mirror the Prototype subtree beneath each placement,
// still referencing the objects converted once by convertPrototype().
void expandPrototype(ImportContext &ctx,
    const pxr::HdSceneIndexBaseRefPtr &sceneIndex,
    const pxr::SdfPath &prototypeRoot,
    const PrototypeContent &content,
    LayerNodeRef parent)
{
  const auto inverseRoot =
      tsd::math::inverse(flattenedXformOf(sceneIndex, prototypeRoot));

  for (size_t i = 0;
       i < content.surfaces.size() && i < content.gprimPaths.size();
       ++i) {
    const auto &path = content.gprimPaths[i];
    const auto local =
        tsd::math::mul(inverseRoot, flattenedXformOf(sceneIndex, path));
    auto node = ctx.scene->insertChildTransformNode(
        parent, local, path.GetName().c_str());
    ctx.scene->insertChildObjectNode(
        node, content.surfaces[i], content.surfaces[i]->name().c_str());
  }
}

// The per-instance transforms an instancer carries, either directly as
// matrices or composed from translate/rotate/scale primvars.
std::vector<tsd::math::mat4> readInstanceTransforms(
    const pxr::HdSceneIndexPrim &prim, size_t instanceCount)
{
  auto primvars = pxr::HdPrimvarsSchema::GetFromParent(prim.dataSource);

  auto valueOf = [&](const pxr::TfToken &name) {
    auto primvar = primvars.GetPrimvar(name);
    if (!primvar)
      return pxr::VtValue();
    auto source = primvar.GetPrimvarValue();
    return source ? source->GetValue(0) : pxr::VtValue();
  };

  std::vector<tsd::math::mat4> retval(instanceCount, tsd::math::IDENTITY_MAT4);

  const auto transforms = valueOf(pxr::HdInstancerTokens->instanceTransforms);
  if (transforms.IsHolding<pxr::VtMatrix4dArray>()) {
    const auto &m = transforms.UncheckedGet<pxr::VtMatrix4dArray>();
    for (size_t i = 0; i < retval.size() && i < m.size(); ++i)
      retval[i] = toTsdMat4(m[i]);
    return retval;
  }

  const auto translations =
      valueOf(pxr::HdInstancerTokens->instanceTranslations);
  const auto rotations = valueOf(pxr::HdInstancerTokens->instanceRotations);
  const auto scales = valueOf(pxr::HdInstancerTokens->instanceScales);

  for (size_t i = 0; i < retval.size(); ++i) {
    auto transform = tsd::math::IDENTITY_MAT4;

    if (scales.IsHolding<pxr::VtVec3fArray>()) {
      const auto &s = scales.UncheckedGet<pxr::VtVec3fArray>();
      if (i < s.size()) {
        transform = tsd::math::mul(
            tsd::math::scaling_matrix(float3(s[i][0], s[i][1], s[i][2])),
            transform);
      }
    }

    auto applyRotation = [&](float x, float y, float z, float w) {
      transform = tsd::math::mul(
          tsd::math::rotation_matrix(tsd::math::float4(x, y, z, w)), transform);
    };
    if (rotations.IsHolding<pxr::VtQuathArray>()) {
      const auto &r = rotations.UncheckedGet<pxr::VtQuathArray>();
      if (i < r.size()) {
        const auto imaginary = r[i].GetImaginary();
        applyRotation(float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
            float(r[i].GetReal()));
      }
    } else if (rotations.IsHolding<pxr::VtQuatfArray>()) {
      const auto &r = rotations.UncheckedGet<pxr::VtQuatfArray>();
      if (i < r.size()) {
        const auto imaginary = r[i].GetImaginary();
        applyRotation(imaginary[0], imaginary[1], imaginary[2], r[i].GetReal());
      }
    }

    if (translations.IsHolding<pxr::VtVec3fArray>()) {
      const auto &t = translations.UncheckedGet<pxr::VtVec3fArray>();
      if (i < t.size()) {
        transform = tsd::math::mul(
            tsd::math::translation_matrix(float3(t[i][0], t[i][1], t[i][2])),
            transform);
      }
    }

    retval[i] = transform;
  }

  return retval;
}

// Every resolved instancer prim UsdImaging synthesised for native instancing.
// Both passes over that subtree -- discovering placement paths before the
// traversal, attaching Prototypes after it -- have to agree on this set, so
// they share the walk that finds it rather than each filtering their own.
std::vector<pxr::SdfPath> nativeInstancerPaths(
    const pxr::HdSceneIndexBaseRefPtr &sceneIndex)
{
  std::vector<pxr::SdfPath> retval;

  const pxr::SdfPath root(NATIVE_INSTANCING_ROOT);
  if (sceneIndex->GetPrim(root).dataSource == nullptr
      && sceneIndex->GetChildPrimPaths(root).empty())
    return retval;

  for (const pxr::SdfPath &path : pxr::HdSceneIndexPrimView(sceneIndex, root)) {
    if (sceneIndex->GetPrim(path).primType == pxr::HdPrimTypeTokens->instancer)
      retval.push_back(path);
  }

  return retval;
}

} // namespace

bool InstancerPlacements::isVisible(int instanceId) const
{
  if (mask.empty())
    return true;
  return size_t(instanceId) >= mask.size() || mask[size_t(instanceId)];
}

std::vector<tsd::math::mat4> InstancerPlacements::forPrototype(
    size_t prototypeIndex) const
{
  std::vector<tsd::math::mat4> retval;
  if (prototypeIndex >= instanceIndices.size())
    return retval;

  for (int index : instanceIndices[prototypeIndex]) {
    if (!isVisible(index))
      continue;
    if (size_t(index) < transforms.size())
      retval.push_back(transforms[size_t(index)]);
  }
  return retval;
}

InstancerPlacements readInstancerPlacements(const pxr::HdSceneIndexPrim &prim)
{
  InstancerPlacements retval;
  auto schema = pxr::HdInstancerTopologySchema::GetFromParent(prim.dataSource);
  if (!schema)
    return retval;

  if (auto prototypes = schema.GetPrototypes())
    retval.prototypes = prototypes->GetTypedValue(0);
  if (auto locations = schema.GetInstanceLocations())
    retval.instanceLocations = locations->GetTypedValue(0);
  if (auto mask = schema.GetMask())
    retval.mask = mask->GetTypedValue(0);

  auto indices = schema.GetInstanceIndices();
  for (size_t i = 0; i < indices.GetNumElements(); ++i) {
    auto element = indices.GetElement(i);
    retval.instanceIndices.push_back(
        element ? element->GetTypedValue(0) : pxr::VtIntArray());
  }

  // Native instancing attaches its Prototypes at each USD Instance's own node,
  // so it never asks forPrototype() for transforms; reading them would be the
  // most expensive part of this call and all of it wasted.
  if (!retval.instanceLocations.empty())
    return retval;

  size_t instanceCount = 0;
  for (const auto &element : retval.instanceIndices) {
    for (int i : element)
      instanceCount = std::max(instanceCount, size_t(i) + 1);
  }
  retval.transforms = readInstanceTransforms(prim, instanceCount);

  return retval;
}

std::vector<double> pointInstancerSampleTimes(const pxr::UsdPrim &prim)
{
  pxr::UsdGeomPointInstancer instancer(prim);
  if (!instancer)
    return {};

  // Every attribute that can move a placement, including the two velocity
  // attributes Hydra folds into the instance transforms it computes.
  const pxr::UsdAttribute attributes[] = {instancer.GetPositionsAttr(),
      instancer.GetOrientationsAttr(),
      instancer.GetScalesAttr(),
      instancer.GetVelocitiesAttr(),
      instancer.GetAngularVelocitiesAttr(),
      instancer.GetProtoIndicesAttr(),
      instancer.GetInvisibleIdsAttr()};

  std::vector<double> retval;
  for (const auto &attribute : attributes) {
    if (!attribute)
      continue;
    std::vector<double> times;
    attribute.GetTimeSamples(&times);
    retval.insert(retval.end(), times.begin(), times.end());
  }

  std::sort(retval.begin(), retval.end());
  retval.erase(std::unique(retval.begin(), retval.end()), retval.end());
  return retval;
}

InstancerRegistry::InstancerRegistry(
    const pxr::HdSceneIndexBaseRefPtr &sceneIndex)
{
  // Reading the placements the instancers name, rather than every path the
  // traversal will see, is what keeps recordNode() to a handful of nodes on a
  // large Stage -- and to none at all on the Stages with no native instancing,
  // where there is nothing here to walk either.
  for (const pxr::SdfPath &path : nativeInstancerPaths(sceneIndex)) {
    const auto schema = pxr::HdInstancerTopologySchema::GetFromParent(
        sceneIndex->GetPrim(path).dataSource);
    if (!schema)
      continue;
    const auto locations = schema.GetInstanceLocations();
    if (!locations)
      continue;
    for (const auto &location : locations->GetTypedValue(0))
      m_placementPaths.insert(location.GetString());
  }
}

LayerNodeRef InstancerRegistry::nodeFor(
    const pxr::SdfPath &primPath, LayerNodeRef fallback) const
{
  auto found = m_nodeForPrimPath.find(primPath.GetString());
  return found != m_nodeForPrimPath.end() ? found->second : fallback;
}

void InstancerRegistry::recordNode(
    const pxr::SdfPath &primPath, LayerNodeRef node)
{
  auto key = primPath.GetString();
  if (m_placementPaths.count(key))
    m_nodeForPrimPath[std::move(key)] = node;
}

void convertInstancer(ImportContext &ctx,
    const pxr::HdSceneIndexBaseRefPtr &sceneIndex,
    const pxr::SdfPath &primPath,
    const pxr::HdSceneIndexPrim &prim,
    LayerNodeRef node,
    InstancerRegistry &registry)
{
  const auto placementsOfPrim = readInstancerPlacements(prim);
  if (placementsOfPrim.prototypes.empty())
    return;

  // Native instancing is resolved against each USD Instance's own node after
  // the hierarchy has been mirrored.
  if (!placementsOfPrim.instanceLocations.empty())
    return;

  const auto animatedSamples = pointInstancerSampleCount(ctx, primPath);
  bool boundAnyPrototype = false;

  for (size_t protoIndex = 0; protoIndex < placementsOfPrim.prototypes.size();
       ++protoIndex) {
    auto content = convertPrototype(
        ctx, sceneIndex, placementsOfPrim.prototypes[protoIndex], registry);

    const auto placements = placementsOfPrim.forPrototype(protoIndex);
    if (placements.empty())
      continue;

    if (content->internalTransformsAnimated) {
      for (size_t i = 0; i < placements.size(); ++i) {
        auto placementNode = ctx.scene->insertChildTransformNode(node,
            placements[i],
            (primPath.GetName() + "_" + std::to_string(i)).c_str());
        expandPrototype(ctx,
            sceneIndex,
            placementsOfPrim.prototypes[protoIndex],
            *content,
            placementNode);
      }
      continue;
    }

    // One transform-array node so hardware instancing is used rather than
    // thousands of individual nodes. Its children must be object nodes: the
    // render index does not push a transform-array node's matrices onto the
    // transform stack, which is why Prototype geometry is baked (ADR 0016).
    auto transformArray =
        ctx.scene->createArray(ANARI_FLOAT32_MAT4, placements.size());
    transformArray->setData(placements.data(), placements.size());
    transformArray->setName((primPath.GetString() + "_transforms").c_str());

    auto arrayNode = ctx.scene->insertChildTransformArrayNode(
        node, transformArray.data(), primPath.GetName().c_str());
    for (auto &surface : content->surfaces)
      ctx.scene->insertChildObjectNode(
          arrayNode, surface, surface->name().c_str());

    // The Array this Prototype's placements just went into is the Array the
    // binding re-fills; handing it over here is what keeps a scrub from having
    // to find it again by name.
    if (animatedSamples > 1) {
      addInstancerAnimation(
          ctx, primPath, protoIndex, arrayNode, transformArray);
      boundAnyPrototype = true;
    }
  }

  // One animated prim, however many Prototypes it scatters.
  if (boundAnyPrototype)
    ctx.reportAnimatedPrim(animatedSamples);
}

void attachNativeInstances(ImportContext &ctx,
    const pxr::HdSceneIndexBaseRefPtr &sceneIndex,
    InstancerRegistry &registry,
    LayerNodeRef importRoot)
{
  for (const pxr::SdfPath &path : nativeInstancerPaths(sceneIndex)) {
    if (ctx.isClaimed(path))
      continue;

    const auto placements = readInstancerPlacements(sceneIndex->GetPrim(path));
    if (placements.prototypes.empty() || placements.instanceLocations.empty())
      continue;

    auto content =
        convertPrototype(ctx, sceneIndex, placements.prototypes[0], registry);

    const auto &indices = placements.instanceIndices.empty()
        ? pxr::VtIntArray()
        : placements.instanceIndices[0];

    for (size_t i = 0; i < placements.instanceLocations.size(); ++i) {
      const int index = size_t(i) < indices.size() ? indices[i] : int(i);
      if (!placements.isVisible(index))
        continue;

      // Each USD Instance becomes one node referencing the same shared
      // objects, so editing the Prototype's material affects every placement
      // as it does in USD.
      auto placementNode =
          registry.nodeFor(placements.instanceLocations[i], importRoot);

      if (content->internalTransformsAnimated) {
        expandPrototype(
            ctx, sceneIndex, placements.prototypes[0], *content, placementNode);
      } else {
        for (auto &surface : content->surfaces) {
          ctx.scene->insertChildObjectNode(
              placementNode, surface, surface->name().c_str());
        }
      }
    }
  }
}

} // namespace tsd::io::usd
