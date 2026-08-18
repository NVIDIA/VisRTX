// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/importers/detail/usd/UsdGeometry.h"
#include "tsd/io/importers/detail/usd/UsdMaterials.h"
#include "tsd/io/importers/detail/usd/UsdSubdivision.h"
// usd
#include <pxr/imaging/hd/materialBindingsSchema.h>
#include <pxr/imaging/hd/tokens.h>
// std
#include <map>
#include <string>
#include <vector>

namespace tsd::io::usd {

using namespace tsd::core;

namespace {

// The material a resolved prim binds, or an empty path when it binds none.
pxr::SdfPath boundMaterialPathOf(const pxr::HdSceneIndexPrim &prim)
{
  auto bindings = pxr::HdMaterialBindingsSchema::GetFromParent(prim.dataSource);
  if (auto binding = bindings.GetMaterialBinding()) {
    if (auto path = binding.GetPath())
      return path->GetTypedValue(0);
  }
  return {};
}

// A prim with no bound material takes its colour from the display-colour and
// display-opacity primvars, so unmaterialed content looks as it does in a
// reference viewer instead of taking TSD's default.
MaterialRef displayColorMaterial(ImportContext &ctx,
    const pxr::SdfPath &primPath,
    const pxr::HdSceneIndexPrim &prim)
{
  auto retval = ctx.scene.createObject<Material>(tokens::material::matte);
  retval->setName((primPath.GetString() + "_displayColor").c_str());

  const auto display = readDisplayColor(prim);
  if (display.hasColor)
    retval->setParameter("color", display.color);
  if (display.hasOpacity)
    retval->setParameter("opacity", display.opacity);
  return retval;
}

// The subsets a mesh divides itself into, in the order the resolver visits
// them, so Part names line up with the materials resolved for them.
std::vector<pxr::SdfPath> subsetPathsOf(
    const pxr::HdSceneIndexBaseRefPtr &sceneIndex, const pxr::SdfPath &primPath)
{
  std::vector<pxr::SdfPath> retval;
  for (const auto &childPath : sceneIndex->GetChildPrimPaths(primPath)) {
    if (sceneIndex->GetPrim(childPath).primType
        == pxr::HdPrimTypeTokens->geomSubset)
      retval.push_back(childPath);
  }
  return retval;
}

// Resolve the materials this gprim needs, and with them the attribute-slot
// assignment the resolver must use. This is the half of conversion that does
// not change over time, which is why it happens once and its answer is
// replayed rather than recomputed.
struct MaterialPlan
{
  std::map<std::string, MaterialRef> byPart;
  std::map<std::string, std::string> uvNamesByPart;
  MaterialRef fallback;
};

MaterialPlan planMaterials(ImportContext &ctx,
    const pxr::HdSceneIndexBaseRefPtr &sceneIndex,
    const pxr::SdfPath &primPath,
    const pxr::HdSceneIndexPrim &prim)
{
  MaterialPlan retval;

  const auto resolved =
      resolveMaterial(ctx, sceneIndex, boundMaterialPathOf(prim));
  retval.fallback = resolved.material
      ? resolved.material
      : displayColorMaterial(ctx, primPath, prim);

  // A material names the primvar its texture reader wants. A subset without a
  // material of its own falls back to this one, and this one to the
  // conventional name.
  const std::string meshUvName =
      resolved.uvPrimvarName.empty() ? "st" : resolved.uvPrimvarName;

  const auto meshName = primPath.GetString();
  retval.byPart[meshName] = retval.fallback;
  retval.uvNamesByPart[meshName] = meshUvName;

  if (prim.primType != pxr::HdPrimTypeTokens->mesh)
    return retval;

  for (const auto &subsetPath : subsetPathsOf(sceneIndex, primPath)) {
    auto subsetPrim = sceneIndex->GetPrim(subsetPath);
    const auto subsetResolved =
        resolveMaterial(ctx, sceneIndex, boundMaterialPathOf(subsetPrim));

    const auto subsetName = subsetPath.GetString();
    retval.byPart[subsetName] = subsetResolved.material ? subsetResolved.material
                                                        : retval.fallback;
    retval.uvNamesByPart[subsetName] = subsetResolved.uvPrimvarName.empty()
        ? meshUvName
        : subsetResolved.uvPrimvarName;
  }

  return retval;
}

// Turn resolved Parts into TSD objects. Attributes carrying a shared key are
// built once and pointed at by every Part that names them, which is how a
// subdivided mesh's Surfaces end up sharing one position Array.
void buildParts(ImportContext &ctx,
    const ResolvedGeometry &resolved,
    const MaterialPlan &materials,
    ConvertedGeometry &out)
{
  std::map<std::string, ArrayRef> sharedArrays;

  for (const auto &part : resolved.parts) {
    auto geometry = ctx.scene.createObject<Geometry>(part.subtype);
    geometry->setName(part.name.c_str());

    for (const auto &attribute : part.attributes) {
      if (!attribute.valid())
        continue;

      ArrayRef array;
      if (!attribute.sharedKey.empty()) {
        auto &shared = sharedArrays[attribute.sharedKey];
        if (!shared) {
          shared = ctx.scene.createArray(attribute.type, attribute.count());
          shared->setData(attribute.data());
        }
        array = shared;
      } else {
        array = ctx.scene.createArray(attribute.type, attribute.count());
        array->setData(attribute.data());
      }
      geometry->setParameterObject(attribute.parameter, *array);
    }

    for (const auto &[name, value] : part.scalars)
      geometry->setParameter(name, value);

    auto found = materials.byPart.find(part.name);
    auto material =
        found != materials.byPart.end() ? found->second : materials.fallback;

    out.geometryByPart.emplace_back(part.name, geometry);
    out.surfaces.push_back(
        ctx.scene.createSurface(part.name.c_str(), geometry, material));
  }
}

} // namespace

ConvertedGeometry convertGeometry(ImportContext &ctx,
    const pxr::HdSceneIndexBaseRefPtr &sceneIndex,
    const pxr::SdfPath &primPath,
    const pxr::HdSceneIndexPrim &prim,
    const tsd::math::mat4 &bakeXform)
{
  ConvertedGeometry retval;

  // Materials first: which primvar a Part's material reads as texture
  // coordinates decides which slot every other primvar falls into.
  const auto materials = planMaterials(ctx, sceneIndex, primPath, prim);

  retval.resolveOptions.bakeXform = bakeXform;
  retval.resolveOptions.refinementLevel = ctx.options.refinementLevel;
  retval.resolveOptions.refine =
      meshWantsRefinement(ctx.stage, ctx.options.refinementLevel, primPath);
  retval.resolveOptions.uvNamesByPart = materials.uvNamesByPart;

  const auto resolved =
      resolveGeometry(sceneIndex, primPath, prim, retval.resolveOptions);
  if (!resolved.valid())
    return retval;

  buildParts(ctx, resolved, materials, retval);
  return retval;
}

} // namespace tsd::io::usd
