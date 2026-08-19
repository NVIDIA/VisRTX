// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/importers/detail/usd/UsdGeometry.h"
#include "tsd/io/importers/detail/usd/UsdMaterials.h"
#include "tsd/io/importers/detail/usd/UsdSubdivision.h"
// usd
#include <pxr/imaging/hd/materialBindingsSchema.h>
#include <pxr/imaging/hd/tokens.h>
// std
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
  auto retval = ctx.scene->createObject<Material>(tokens::material::matte);
  retval->setName((primPath.GetString() + "_displayColor").c_str());

  const auto display = readDisplayColor(prim);
  if (display.color)
    retval->setParameter("color", *display.color);
  if (display.opacity)
    retval->setParameter("opacity", *display.opacity);
  return retval;
}

// The materials this gprim's Parts need, and with them the texture-coordinate
// primvar each Part's attribute assignment must be built around. This is the
// half of conversion that does not change over time, which is why it happens
// once and its answer is replayed rather than recomputed.
struct MaterialPlan
{
  FlatMap<std::string, MaterialRef> byPart;
  FlatMap<std::string, std::string> uvNamesByPart;
  MaterialRef fallback;

  // True when some material asked for a texture-coordinate primvar other than
  // the conventional one, which is the only case where knowing the materials
  // changes what a resolve produces.
  bool anyNonDefaultUv{false};
};

// Deliberately driven by the Parts a resolve actually produced: resolving a
// material creates scene objects, and a subset that draws no triangles must
// not leave a Material and its textures behind that nothing references.
MaterialPlan planMaterials(ImportContext &ctx,
    const pxr::HdSceneIndexBaseRefPtr &sceneIndex,
    const pxr::SdfPath &primPath,
    const pxr::HdSceneIndexPrim &prim,
    const ResolvedGeometry &resolved)
{
  MaterialPlan retval;

  const auto meshResolved =
      resolveMaterial(ctx, sceneIndex, boundMaterialPathOf(prim));
  retval.fallback = meshResolved.material
      ? meshResolved.material
      : displayColorMaterial(ctx, primPath, prim);

  // A material names the primvar its texture reader wants. A subset without a
  // material of its own falls back to this one, and this one to the
  // conventional name.
  const std::string meshUvName =
      meshResolved.uvPrimvarName.empty() ? "st" : meshResolved.uvPrimvarName;
  retval.anyNonDefaultUv = meshUvName != "st";

  const auto meshName = primPath.GetString();
  for (const auto &part : resolved.parts) {
    if (part.name == meshName) {
      retval.byPart.set(part.name, retval.fallback);
      retval.uvNamesByPart.set(part.name, meshUvName);
      continue;
    }

    auto subsetPrim = sceneIndex->GetPrim(pxr::SdfPath(part.name));
    const auto subsetResolved =
        resolveMaterial(ctx, sceneIndex, boundMaterialPathOf(subsetPrim));
    retval.byPart.set(part.name,
        subsetResolved.material ? subsetResolved.material : retval.fallback);

    const auto subsetUvName = subsetResolved.uvPrimvarName.empty()
        ? meshUvName
        : subsetResolved.uvPrimvarName;
    retval.uvNamesByPart.set(part.name, subsetUvName);
    retval.anyNonDefaultUv =
        retval.anyNonDefaultUv || subsetUvName != "st";
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
  FlatMap<std::string, ArrayRef> sharedArrays;

  for (const auto &part : resolved.parts) {
    auto geometry = ctx.scene->createObject<Geometry>(part.subtype);
    geometry->setName(part.name.c_str());

    for (const auto &attribute : part.attributes) {
      if (!attribute.valid())
        continue;

      ArrayRef array;
      if (auto *shared = sharedArrays.at(attribute.sharedKey))
        array = *shared;
      if (!array) {
        array = ctx.scene->createArray(attribute.type, attribute.count());
        array->setData(attribute.data());
        if (!attribute.sharedKey.empty())
          sharedArrays.set(attribute.sharedKey, array);
      }
      geometry->setParameterObject(attribute.parameter, *array);
    }

    for (const auto &[name, value] : part.scalars)
      geometry->setParameter(name, value);

    auto *found = materials.byPart.at(part.name);
    auto material = found ? *found : materials.fallback;

    out.geometryByPart.emplace_back(part.name, geometry);
    out.surfaces.push_back(
        ctx.scene->createSurface(part.name.c_str(), geometry, material));
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

  // Nothing is built for a gprim that resolves to nothing -- not even the
  // material it binds, which would otherwise outlive the geometry that was
  // supposed to reference it.
  if (!geometryWillResolve(prim))
    return retval;

  retval.resolveOptions.bakeXform = bakeXform;
  retval.resolveOptions.refinementLevel = ctx.options->refinementLevel;
  retval.resolveOptions.refine =
      meshWantsRefinement(ctx.stage, ctx.options->refinementLevel, primPath);

  // Resolve first, so materials are resolved only for the Parts that exist.
  auto resolved =
      resolveGeometry(sceneIndex, primPath, prim, retval.resolveOptions);
  if (!resolved.valid())
    return retval;

  const auto materials =
      planMaterials(ctx, sceneIndex, primPath, prim, resolved);
  retval.resolveOptions.uvNamesByPart = materials.uvNamesByPart;

  // Which primvar a material reads as texture coordinates decides which slot
  // every other primvar falls into, so a material naming something other than
  // the conventional `st` means the first resolve assigned them wrongly. That
  // costs one extra resolve, at import only: a binding is handed the answer.
  if (materials.anyNonDefaultUv) {
    resolved = resolveGeometry(sceneIndex, primPath, prim, retval.resolveOptions);
    if (!resolved.valid())
      return retval;
  }

  // Record the assignment the resolve settled on, so a scrub replays it rather
  // than re-deriving it from whatever primvars that frame happens to carry.
  for (const auto &part : resolved.parts)
    retval.resolveOptions.slotPrimvarsByPart.set(part.name, part.slotPrimvars);

  buildParts(ctx, resolved, materials, retval);
  return retval;
}

} // namespace tsd::io::usd
