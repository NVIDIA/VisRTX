// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/importers/detail/usd/UsdGeometry.h"
#include "tsd/io/importers/detail/usd/UsdMaterials.h"
#include "tsd/io/importers/detail/usd/UsdSubdivision.h"
// usd
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/imaging/hd/basisCurvesSchema.h>
#include <pxr/imaging/hd/basisCurvesTopologySchema.h>
#include <pxr/imaging/hd/coneSchema.h>
#include <pxr/imaging/hd/cylinderSchema.h>
#include <pxr/imaging/hd/geomSubsetSchema.h>
#include <pxr/imaging/hd/materialBindingsSchema.h>
#include <pxr/imaging/hd/meshSchema.h>
#include <pxr/imaging/hd/meshTopology.h>
#include <pxr/imaging/hd/meshUtil.h>
#include <pxr/imaging/hd/primvarsSchema.h>
#include <pxr/imaging/hd/sphereSchema.h>
#include <pxr/imaging/hd/tokens.h>
// std
#include <algorithm>
#include <map>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace tsd::io::usd {

using namespace tsd::core;

namespace {

///////////////////////////////////////////////////////////////////////////////
// Primvar plumbing ///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// One primvar, already flattened out of any indexing, with the interpolation
// that decides which TSD attribute slot it lands in.
struct Primvar
{
  pxr::VtValue value;
  pxr::TfToken interpolation;
  pxr::TfToken role;

  bool valid() const;
};

bool Primvar::valid() const
{
  return !value.IsEmpty() && value.IsArrayValued() && value.GetArraySize() > 0;
}

Primvar readPrimvar(
    const pxr::HdPrimvarsSchema &primvars, const pxr::TfToken &name)
{
  Primvar retval;
  auto primvar = primvars.GetPrimvar(name);
  if (!primvar)
    return retval;
  if (auto value = primvar.GetFlattenedPrimvarValue())
    retval.value = value->GetValue(0);
  if (auto interpolation = primvar.GetInterpolation())
    retval.interpolation = interpolation->GetTypedValue(0);
  if (auto role = primvar.GetRole())
    retval.role = role->GetTypedValue(0);
  return retval;
}

// Which TSD parameter prefix an interpolation maps onto. Constant primvars
// have no per-element ANARI slot and are handled by the caller where they
// carry meaning (display colour), otherwise dropped.
const char *prefixForInterpolation(const pxr::TfToken &interpolation)
{
  if (interpolation == pxr::HdPrimvarSchemaTokens->uniform)
    return "primitive.";
  if (interpolation == pxr::HdPrimvarSchemaTokens->faceVarying)
    return "faceVarying.";
  if (interpolation == pxr::HdPrimvarSchemaTokens->vertex
      || interpolation == pxr::HdPrimvarSchemaTokens->varying)
    return "vertex.";
  return nullptr;
}

anari::DataType anariTypeOfPrimvar(const pxr::VtValue &value)
{
  if (value.IsHolding<pxr::VtFloatArray>())
    return ANARI_FLOAT32;
  if (value.IsHolding<pxr::VtVec2fArray>())
    return ANARI_FLOAT32_VEC2;
  if (value.IsHolding<pxr::VtVec3fArray>())
    return ANARI_FLOAT32_VEC3;
  if (value.IsHolding<pxr::VtVec4fArray>())
    return ANARI_FLOAT32_VEC4;
  return ANARI_UNKNOWN;
}

///////////////////////////////////////////////////////////////////////////////
// Shared conversion helpers //////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// Prototype-internal transforms are baked into vertex data (ADR 0016);
// everything else passes through untouched.
std::vector<float3> bakedPositions(
    const pxr::VtVec3fArray &source, const tsd::math::mat4 &bakeXform)
{
  const bool bake = bakeXform != tsd::math::IDENTITY_MAT4;
  std::vector<float3> retval;
  retval.reserve(source.size());
  for (const auto &p : source) {
    float3 v(p[0], p[1], p[2]);
    if (bake) {
      const auto t = tsd::math::mul(bakeXform, float4(v.x, v.y, v.z, 1.f));
      v = float3(t.x, t.y, t.z);
    }
    retval.push_back(v);
  }
  return retval;
}

// USD authors widths; TSD geometry takes radii.
void bindRadiiFromWidths(ImportContext &ctx,
    GeometryRef &geometry,
    const Primvar &widths,
    size_t vertexCount)
{
  if (!widths.valid() || !widths.value.IsHolding<pxr::VtFloatArray>())
    return;
  const auto &w = widths.value.UncheckedGet<pxr::VtFloatArray>();
  if (w.empty())
    return;

  std::vector<float> radii;
  radii.reserve(vertexCount);
  for (size_t i = 0; i < vertexCount; ++i)
    radii.push_back(0.5f * w[std::min(i, w.size() - 1)]);

  auto array = ctx.scene.createArray(ANARI_FLOAT32, radii.size());
  array->setData(radii.data(), radii.size());
  geometry->setParameterObject("vertex.radius", *array);
}

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
    const Primvar &displayColor,
    const Primvar &displayOpacity)
{
  auto retval = ctx.scene.createObject<Material>(tokens::material::matte);
  retval->setName((primPath.GetString() + "_displayColor").c_str());
  if (displayColor.valid()
      && displayColor.value.IsHolding<pxr::VtVec3fArray>()) {
    const auto &c = displayColor.value.UncheckedGet<pxr::VtVec3fArray>();
    retval->setParameter("color", float3(c[0][0], c[0][1], c[0][2]));
  }
  if (displayOpacity.valid()
      && displayOpacity.value.IsHolding<pxr::VtFloatArray>()) {
    const auto &o = displayOpacity.value.UncheckedGet<pxr::VtFloatArray>();
    retval->setParameter("opacity", o[0]);
  }
  return retval;
}

///////////////////////////////////////////////////////////////////////////////
// Mesh conversion ////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// Apply a transform to the float-typed array a primvar holds, whatever its
// component count. Anything else has no ANARI attribute slot and so yields an
// empty value.
template <typename Fn>
pxr::VtValue transformFloatArray(const pxr::VtValue &value, Fn &&fn)
{
  if (value.IsHolding<pxr::VtFloatArray>())
    return pxr::VtValue(fn(value.UncheckedGet<pxr::VtFloatArray>()));
  if (value.IsHolding<pxr::VtVec2fArray>())
    return pxr::VtValue(fn(value.UncheckedGet<pxr::VtVec2fArray>()));
  if (value.IsHolding<pxr::VtVec3fArray>())
    return pxr::VtValue(fn(value.UncheckedGet<pxr::VtVec3fArray>()));
  if (value.IsHolding<pxr::VtVec4fArray>())
    return pxr::VtValue(fn(value.UncheckedGet<pxr::VtVec4fArray>()));
  return {};
}

// Expand a uniform (per-face) primvar to per-triangle values using the
// triangulation's record of which coarse face each triangle came from.
template <typename T>
pxr::VtArray<T> expandUniform(
    const pxr::VtArray<T> &source, const pxr::VtIntArray &primitiveParams)
{
  pxr::VtArray<T> retval;
  retval.reserve(primitiveParams.size());
  for (int param : primitiveParams) {
    const int face = pxr::HdMeshUtil::DecodeFaceIndexFromCoarseFaceParam(param);
    retval.push_back(source[size_t(face) < source.size() ? size_t(face)
                                                         : source.size() - 1]);
  }
  return retval;
}

pxr::VtValue expandUniformValue(
    const pxr::VtValue &value, const pxr::VtIntArray &primitiveParams)
{
  return transformFloatArray(value, [&](const auto &source) {
    return expandUniform(source, primitiveParams);
  });
}

// Select the values belonging to a chosen set of triangles out of an array
// laid out in triangle order -- one value per triangle for per-primitive data,
// three for per-corner data.
template <typename T>
pxr::VtArray<T> gatherTriangles(const pxr::VtArray<T> &source,
    const std::vector<uint32_t> &triangles,
    size_t valuesPerTriangle)
{
  pxr::VtArray<T> retval;
  retval.reserve(triangles.size() * valuesPerTriangle);
  for (uint32_t triangle : triangles) {
    const size_t base = size_t(triangle) * valuesPerTriangle;
    for (size_t i = 0; i < valuesPerTriangle; ++i)
      retval.push_back(source[base + i]);
  }
  return retval;
}

pxr::VtValue gatherTrianglesValue(const pxr::VtValue &value,
    const std::vector<uint32_t> &triangles,
    size_t valuesPerTriangle)
{
  return transformFloatArray(value, [&](const auto &source) {
    return gatherTriangles(source, triangles, valuesPerTriangle);
  });
}

// Everything one converted mesh needs to hand to the Surface builder.
struct ConvertedMesh
{
  ArrayRef vertexPosition;
  pxr::VtVec3iArray triangleIndices;
  pxr::VtIntArray primitiveParams;
};

// A primvar expanded onto the triangulated topology and ready to bind. Vertex
// data stays as authored and is indexed by the triangle indices, so its Array
// is created once and shared by every Surface built from the mesh; uniform and
// face-varying data are laid out in triangle order and have to be gathered per
// Surface, because a subset draws only some of the triangles.
struct TriangulatedPrimvar
{
  pxr::VtValue value;
  const char *prefix{nullptr};
  size_t valuesPerTriangle{0};
  ArrayRef sharedArray;

  // Vertex data is the only kind every Surface can point at unchanged.
  bool isShared() const;
};

bool TriangulatedPrimvar::isShared() const
{
  return valuesPerTriangle == 0;
}

// Kept sorted by name: the order primvars are visited decides which of them
// takes each spare attribute slot, and that has to be stable across runs. This
// is why the mesh converter reaches for std::map rather than FlatMap.
using TriangulatedPrimvars = std::map<std::string, TriangulatedPrimvar>;

} // namespace

bool isGeometryPrimType(const pxr::TfToken &primType)
{
  return primType == pxr::HdPrimTypeTokens->mesh
      || primType == pxr::HdPrimTypeTokens->points
      || primType == pxr::HdPrimTypeTokens->basisCurves
      || primType == pxr::HdPrimTypeTokens->sphere
      || primType == pxr::HdPrimTypeTokens->cone
      || primType == pxr::HdPrimTypeTokens->cylinder;
}

namespace {

// Expand every primvar that has an attribute slot onto the triangulated
// topology, once, so that each Surface built from the mesh only has to select
// the values for its own triangles. Primvars whose expansion fails or comes up
// short of the triangulation are left out rather than bound partially.
TriangulatedPrimvars triangulatePrimvars(const pxr::HdMeshUtil &meshUtil,
    const ConvertedMesh &mesh,
    const std::map<std::string, Primvar> &primvars)
{
  TriangulatedPrimvars retval;
  for (const auto &[name, primvar] : primvars) {
    if (name == pxr::HdPrimvarsSchemaTokens->points.GetString()
        || name == pxr::HdTokens->displayOpacity.GetString())
      continue;

    TriangulatedPrimvar attribute;
    attribute.prefix = prefixForInterpolation(primvar.interpolation);
    if (!attribute.prefix || anariTypeOfPrimvar(primvar.value) == ANARI_UNKNOWN)
      continue;

    if (primvar.interpolation == pxr::HdPrimvarSchemaTokens->uniform) {
      attribute.value = expandUniformValue(primvar.value, mesh.primitiveParams);
      attribute.valuesPerTriangle = 1;
    } else if (primvar.interpolation
        == pxr::HdPrimvarSchemaTokens->faceVarying) {
      const auto result = meshUtil.ComputeTriangulatedFaceVaryingPrimvar(
          pxr::HdGetValueData(primvar.value),
          int(primvar.value.GetArraySize()),
          pxr::HdGetValueTupleType(primvar.value).type,
          &attribute.value);
      if (result != pxr::HdMeshComputationResult::Success)
        continue;
      attribute.valuesPerTriangle = 3;
    } else {
      attribute.value = primvar.value;
    }

    // Whatever will be gathered has to cover the whole triangulation, since
    // any subset may ask for any triangle. Vertex data is bound as authored
    // and indexed by the triangle indices, so there is nothing to check here.
    if (!attribute.value.IsArrayValued() || attribute.value.GetArraySize() == 0
        || attribute.value.GetArraySize()
            < mesh.triangleIndices.size() * attribute.valuesPerTriangle)
      continue;

    retval.emplace(name, std::move(attribute));
  }
  return retval;
}

// Bind one expanded primvar onto a geometry drawing `triangles`.
void bindTrianglePrimvar(ImportContext &ctx,
    GeometryRef &geometry,
    TriangulatedPrimvar &primvar,
    const std::vector<uint32_t> &triangles,
    const std::string &tsdName)
{
  ArrayRef array;
  if (primvar.isShared()) {
    if (!primvar.sharedArray) {
      primvar.sharedArray = ctx.scene.createArray(
          anariTypeOfPrimvar(primvar.value), primvar.value.GetArraySize());
      primvar.sharedArray->setData(pxr::HdGetValueData(primvar.value));
    }
    array = primvar.sharedArray;
  } else {
    const auto selected = gatherTrianglesValue(
        primvar.value, triangles, primvar.valuesPerTriangle);
    if (!selected.IsArrayValued() || selected.GetArraySize() == 0)
      return;
    array = ctx.scene.createArray(
        anariTypeOfPrimvar(selected), selected.GetArraySize());
    array->setData(pxr::HdGetValueData(selected));
  }

  geometry->setParameterObject(
      Token((primvar.prefix + tsdName).c_str()), *array);
}

// Append the triangles one coarse face produced to a Surface's selection.
void appendTrianglesOfFace(std::vector<uint32_t> &selection,
    const std::vector<std::vector<uint32_t>> &trianglesOfFace,
    size_t face)
{
  const auto &triangles = trianglesOfFace[face];
  selection.insert(selection.end(), triangles.begin(), triangles.end());
}

// One Surface's worth of the mesh: the triangles it draws, with every primvar
// re-indexed to match. `uvName` is whichever primvar this Surface's own
// material reads, which is why the attribute slots cannot be assigned once for
// the whole mesh.
GeometryRef buildTriangleGeometry(ImportContext &ctx,
    const ConvertedMesh &mesh,
    TriangulatedPrimvars &attributes,
    const std::vector<uint32_t> &triangles,
    const std::string &uvName,
    const char *name)
{
  auto geometry = ctx.scene.createObject<Geometry>(tokens::geometry::triangle);
  geometry->setName(name);
  geometry->setParameterObject("vertex.position", *mesh.vertexPosition);

  std::vector<uint3> indices;
  indices.reserve(triangles.size());
  for (uint32_t triangle : triangles) {
    const auto &t = mesh.triangleIndices[triangle];
    indices.push_back(uint3(t[0], t[1], t[2]));
  }
  auto indexArray = ctx.scene.createArray(ANARI_UINT32_VEC3, indices.size());
  indexArray->setData(indices.data(), indices.size());
  geometry->setParameterObject("primitive.index", *indexArray);

  auto bind = [&](const std::string &primvarName, const std::string &tsdName) {
    auto found = attributes.find(primvarName);
    if (found != attributes.end())
      bindTrianglePrimvar(ctx, geometry, found->second, triangles, tsdName);
  };

  // Normals, UVs, display colour, then any remaining primvars in name order so
  // the attribute assignment is deterministic.
  const auto normalsName = pxr::HdPrimvarsSchemaTokens->normals.GetString();
  const auto colorName = pxr::HdTokens->displayColor.GetString();
  bind(normalsName, "normal");
  bind(uvName, "attribute0");
  bind(colorName, "color");

  int nextAttribute = 1;
  for (auto &[primvarName, primvar] : attributes) {
    if (nextAttribute > 3)
      break;
    if (primvarName == normalsName || primvarName == colorName
        || primvarName == uvName)
      continue;
    bindTrianglePrimvar(ctx,
        geometry,
        primvar,
        triangles,
        "attribute" + std::to_string(nextAttribute++));
  }

  return geometry;
}

std::vector<SurfaceRef> convertMesh(ImportContext &ctx,
    const pxr::HdSceneIndexBaseRefPtr &sceneIndex,
    const pxr::SdfPath &primPath,
    const pxr::HdSceneIndexPrim &prim,
    const tsd::math::mat4 &bakeXform)
{
  auto meshSchema = pxr::HdMeshSchema::GetFromParent(prim.dataSource);
  auto topologySchema = meshSchema.GetTopology();
  auto primvarsSchema = pxr::HdPrimvarsSchema::GetFromParent(prim.dataSource);

  // Resolve every primvar up front so that refinement can replace the
  // vertex-interpolated ones in place.
  std::map<std::string, Primvar> primvars;
  for (const auto &name : primvarsSchema.GetPrimvarNames()) {
    auto primvar = readPrimvar(primvarsSchema, name);
    if (primvar.valid())
      primvars.emplace(name.GetString(), std::move(primvar));
  }

  auto lookup = [&](const std::string &name) -> Primvar {
    auto found = primvars.find(name);
    return found == primvars.end() ? Primvar() : found->second;
  };

  auto points = lookup(pxr::HdPrimvarsSchemaTokens->points.GetString());
  if (!points.valid() || !points.value.IsHolding<pxr::VtVec3fArray>())
    return {};

  auto faceVertexCounts = intArrayOf(topologySchema.GetFaceVertexCounts());
  auto faceVertexIndices = intArrayOf(topologySchema.GetFaceVertexIndices());
  auto holeIndices = intArrayOf(topologySchema.GetHoleIndices());
  if (faceVertexCounts.empty() || faceVertexIndices.empty())
    return {};

  auto orientationSource = topologySchema.GetOrientation();
  const auto orientation = orientationSource
      ? orientationSource->GetTypedValue(0)
      : pxr::HdTokens->rightHanded;

  if (meshWantsRefinement(ctx, primPath)) {
    MeshPrimvars sourcePrimvars;
    for (const auto &[name, primvar] : primvars) {
      if (name == pxr::HdPrimvarsSchemaTokens->points.GetString())
        continue;
      if (primvar.interpolation == pxr::HdPrimvarSchemaTokens->vertex
          || primvar.interpolation == pxr::HdPrimvarSchemaTokens->varying)
        sourcePrimvars.vertex.emplace_back(name, primvar.value);
      else if (primvar.interpolation == pxr::HdPrimvarSchemaTokens->faceVarying)
        sourcePrimvars.faceVarying.emplace_back(name, primvar.value);
      else if (primvar.interpolation == pxr::HdPrimvarSchemaTokens->uniform)
        sourcePrimvars.uniform.emplace_back(name, primvar.value);
    }

    auto refined = refineMesh(meshSchema,
        faceVertexCounts,
        faceVertexIndices,
        holeIndices,
        orientation,
        points.value.UncheckedGet<pxr::VtVec3fArray>(),
        sourcePrimvars,
        ctx.options.refinementLevel);

    if (refined.valid) {
      faceVertexCounts = refined.faceVertexCounts;
      faceVertexIndices = refined.faceVertexIndices;
      holeIndices = refined.holeIndices;
      points.value = pxr::VtValue(refined.points);
      auto writeBack = [&](const std::vector<NamedPrimvar> &group) {
        for (const auto &[name, value] : group)
          primvars[name].value = value;
      };
      writeBack(refined.primvars.vertex);
      writeBack(refined.primvars.faceVarying);
      writeBack(refined.primvars.uniform);
    }
  }

  pxr::HdMeshTopology topology(pxr::PxOsdOpenSubdivTokens->none,
      orientation,
      faceVertexCounts,
      faceVertexIndices,
      holeIndices);

  // OpenUSD's own topology-aware triangulation handles non-convex polygons
  // and holes; a hand-rolled fan does not.
  pxr::HdMeshUtil meshUtil(&topology, primPath);
  ConvertedMesh mesh;
  meshUtil.ComputeTriangleIndices(&mesh.triangleIndices, &mesh.primitiveParams);
  if (mesh.triangleIndices.empty())
    return {};

  // Resolve the bound material first: it names the UV primvar to bind.
  const auto resolved =
      resolveMaterial(ctx, sceneIndex, boundMaterialPathOf(prim));

  const auto positions =
      bakedPositions(points.value.UncheckedGet<pxr::VtVec3fArray>(), bakeXform);
  mesh.vertexPosition =
      ctx.scene.createArray(ANARI_FLOAT32_VEC3, positions.size());
  mesh.vertexPosition->setData(positions.data(), positions.size());

  auto attributes = triangulatePrimvars(meshUtil, mesh, primvars);

  // Material //

  auto material = resolved.material;
  if (!material) {
    material = displayColorMaterial(ctx,
        primPath,
        lookup(pxr::HdTokens->displayColor.GetString()),
        lookup(pxr::HdTokens->displayOpacity.GetString()));
  }

  // A material names the primvar its texture reader wants. A subset without a
  // material of its own falls back to this one, and this one to the
  // conventional name.
  const std::string meshUvName =
      resolved.uvPrimvarName.empty() ? "st" : resolved.uvPrimvarName;

  std::vector<uint32_t> allTriangles(mesh.triangleIndices.size());
  std::iota(allTriangles.begin(), allTriangles.end(), 0u);

  // Per-face material subsets each become their own Surface over their own
  // triangles, sharing this mesh's vertex arrays.
  std::vector<SurfaceRef> retval;
  std::vector<pxr::SdfPath> subsetPaths;
  for (const auto &childPath : sceneIndex->GetChildPrimPaths(primPath)) {
    if (sceneIndex->GetPrim(childPath).primType
        == pxr::HdPrimTypeTokens->geomSubset)
      subsetPaths.push_back(childPath);
  }

  if (subsetPaths.empty()) {
    auto geometry = buildTriangleGeometry(
        ctx, mesh, attributes, allTriangles, meshUvName, primPath.GetText());
    retval.push_back(
        ctx.scene.createSurface(primPath.GetText(), geometry, material));
    return retval;
  }

  // Map each coarse face to the triangles it produced, once, so every subset
  // can select its own triangles cheaply.
  std::vector<std::vector<uint32_t>> trianglesOfFace;
  for (size_t i = 0; i < mesh.primitiveParams.size(); ++i) {
    const int face = pxr::HdMeshUtil::DecodeFaceIndexFromCoarseFaceParam(
        mesh.primitiveParams[i]);
    if (face < 0)
      continue;
    if (trianglesOfFace.size() <= size_t(face))
      trianglesOfFace.resize(size_t(face) + 1);
    trianglesOfFace[size_t(face)].push_back(uint32_t(i));
  }

  std::vector<bool> faceIsClaimed(trianglesOfFace.size(), false);

  for (const auto &subsetPath : subsetPaths) {
    auto subsetPrim = sceneIndex->GetPrim(subsetPath);
    auto subsetSchema =
        pxr::HdGeomSubsetSchema::GetFromParent(subsetPrim.dataSource);
    const auto faceIndices = intArrayOf(subsetSchema.GetIndices());
    if (faceIndices.empty())
      continue;

    std::vector<uint32_t> subsetTriangles;
    for (int face : faceIndices) {
      if (face < 0 || size_t(face) >= trianglesOfFace.size())
        continue;
      faceIsClaimed[size_t(face)] = true;
      appendTrianglesOfFace(subsetTriangles, trianglesOfFace, size_t(face));
    }
    if (subsetTriangles.empty())
      continue;

    // A subset resolves its own material, which may read a different UV
    // primvar than the mesh's does, so its attributes are bound to suit it.
    const auto subsetResolved =
        resolveMaterial(ctx, sceneIndex, boundMaterialPathOf(subsetPrim));
    const std::string &subsetUvName = subsetResolved.uvPrimvarName.empty()
        ? meshUvName
        : subsetResolved.uvPrimvarName;

    auto subsetGeometry = buildTriangleGeometry(ctx,
        mesh,
        attributes,
        subsetTriangles,
        subsetUvName,
        subsetPath.GetText());

    retval.push_back(ctx.scene.createSurface(subsetPath.GetText(),
        subsetGeometry,
        subsetResolved.material ? subsetResolved.material : material));
  }

  // Faces no subset claimed keep the mesh's own binding rather than going
  // missing with the geometry that no Surface would have drawn.
  std::vector<uint32_t> unclaimedTriangles;
  for (size_t face = 0; face < trianglesOfFace.size(); ++face) {
    if (!faceIsClaimed[face])
      appendTrianglesOfFace(unclaimedTriangles, trianglesOfFace, face);
  }

  // No subset drew anything -- with nothing to divide the mesh up, draw all of
  // it, including any triangle whose coarse face could not be identified.
  if (retval.empty())
    unclaimedTriangles = allTriangles;

  if (!unclaimedTriangles.empty()) {
    auto geometry = buildTriangleGeometry(ctx,
        mesh,
        attributes,
        unclaimedTriangles,
        meshUvName,
        primPath.GetText());
    retval.push_back(
        ctx.scene.createSurface(primPath.GetText(), geometry, material));
  }

  return retval;
}

///////////////////////////////////////////////////////////////////////////////
// Points and curves //////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// Points, curves, and quadrics resolve their material the same way a mesh
// does, including the display-colour fallback for unmaterialed prims.
MaterialRef materialForPrim(ImportContext &ctx,
    const pxr::HdSceneIndexBaseRefPtr &sceneIndex,
    const pxr::SdfPath &primPath,
    const pxr::HdSceneIndexPrim &prim)
{
  if (auto material =
          resolveMaterial(ctx, sceneIndex, boundMaterialPathOf(prim)).material)
    return material;

  auto primvars = pxr::HdPrimvarsSchema::GetFromParent(prim.dataSource);
  return displayColorMaterial(ctx,
      primPath,
      readPrimvar(primvars, pxr::HdTokens->displayColor),
      readPrimvar(primvars, pxr::HdTokens->displayOpacity));
}

std::vector<SurfaceRef> convertPoints(ImportContext &ctx,
    const pxr::HdSceneIndexBaseRefPtr &sceneIndex,
    const pxr::SdfPath &primPath,
    const pxr::HdSceneIndexPrim &prim,
    const tsd::math::mat4 &bakeXform)
{
  auto primvars = pxr::HdPrimvarsSchema::GetFromParent(prim.dataSource);
  const auto points =
      readPrimvar(primvars, pxr::HdPrimvarsSchemaTokens->points);
  if (!points.valid() || !points.value.IsHolding<pxr::VtVec3fArray>())
    return {};

  auto geometry = ctx.scene.createObject<Geometry>(tokens::geometry::sphere);
  geometry->setName(primPath.GetText());

  const auto positions =
      bakedPositions(points.value.UncheckedGet<pxr::VtVec3fArray>(), bakeXform);
  auto positionArray =
      ctx.scene.createArray(ANARI_FLOAT32_VEC3, positions.size());
  positionArray->setData(positions.data(), positions.size());
  geometry->setParameterObject("vertex.position", *positionArray);

  bindRadiiFromWidths(ctx,
      geometry,
      readPrimvar(primvars, pxr::HdPrimvarsSchemaTokens->widths),
      positions.size());

  return {ctx.scene.createSurface(primPath.GetText(),
      geometry,
      materialForPrim(ctx, sceneIndex, primPath, prim))};
}

std::vector<SurfaceRef> convertCurves(ImportContext &ctx,
    const pxr::HdSceneIndexBaseRefPtr &sceneIndex,
    const pxr::SdfPath &primPath,
    const pxr::HdSceneIndexPrim &prim,
    const tsd::math::mat4 &bakeXform)
{
  auto curvesSchema = pxr::HdBasisCurvesSchema::GetFromParent(prim.dataSource);
  auto topologySchema = curvesSchema.GetTopology();
  auto primvars = pxr::HdPrimvarsSchema::GetFromParent(prim.dataSource);

  const auto points =
      readPrimvar(primvars, pxr::HdPrimvarsSchemaTokens->points);
  if (!points.valid() || !points.value.IsHolding<pxr::VtVec3fArray>())
    return {};

  const auto vertexCounts = intArrayOf(topologySchema.GetCurveVertexCounts());
  if (vertexCounts.empty())
    return {};

  auto geometry = ctx.scene.createObject<Geometry>(tokens::geometry::curve);
  geometry->setName(primPath.GetText());

  const auto positions =
      bakedPositions(points.value.UncheckedGet<pxr::VtVec3fArray>(), bakeXform);
  auto positionArray =
      ctx.scene.createArray(ANARI_FLOAT32_VEC3, positions.size());
  positionArray->setData(positions.data(), positions.size());
  geometry->setParameterObject("vertex.position", *positionArray);

  // A curve segment index per consecutive vertex pair within each curve.
  std::vector<uint32_t> segments;
  uint32_t base = 0;
  for (int count : vertexCounts) {
    for (int i = 0; i + 1 < count; ++i)
      segments.push_back(base + uint32_t(i));
    base += uint32_t(count);
  }
  if (!segments.empty()) {
    auto indexArray = ctx.scene.createArray(ANARI_UINT32, segments.size());
    indexArray->setData(segments.data(), segments.size());
    geometry->setParameterObject("primitive.index", *indexArray);
  }

  bindRadiiFromWidths(ctx,
      geometry,
      readPrimvar(primvars, pxr::HdPrimvarsSchemaTokens->widths),
      positions.size());

  return {ctx.scene.createSurface(primPath.GetText(),
      geometry,
      materialForPrim(ctx, sceneIndex, primPath, prim))};
}

///////////////////////////////////////////////////////////////////////////////
// Analytic quadrics //////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

std::vector<SurfaceRef> convertQuadric(ImportContext &ctx,
    const pxr::HdSceneIndexBaseRefPtr &sceneIndex,
    const pxr::SdfPath &primPath,
    const pxr::HdSceneIndexPrim &prim,
    const tsd::math::mat4 &bakeXform)
{
  auto readDouble = [](const pxr::HdDoubleDataSourceHandle &h, double alt) {
    return h ? h->GetTypedValue(0) : alt;
  };

  GeometryRef geometry;
  const float3 origin = [&] {
    if (bakeXform == tsd::math::IDENTITY_MAT4)
      return float3(0.f);
    const auto t = tsd::math::mul(bakeXform, float4(0.f, 0.f, 0.f, 1.f));
    return float3(t.x, t.y, t.z);
  }();

  if (prim.primType == pxr::HdPrimTypeTokens->sphere) {
    auto schema = pxr::HdSphereSchema::GetFromParent(prim.dataSource);
    const float radius = float(readDouble(schema.GetRadius(), 1.0));
    geometry = ctx.scene.createObject<Geometry>(tokens::geometry::sphere);
    auto positions = ctx.scene.createArray(ANARI_FLOAT32_VEC3, 1);
    positions->setData(&origin, 1);
    geometry->setParameterObject("vertex.position", *positions);
    geometry->setParameter("radius", radius);
  } else if (prim.primType == pxr::HdPrimTypeTokens->cone
      || prim.primType == pxr::HdPrimTypeTokens->cylinder) {
    const bool isCone = prim.primType == pxr::HdPrimTypeTokens->cone;
    double height = 2.0;
    double radius = 1.0;
    pxr::TfToken axis = pxr::HdConeSchemaTokens->Z;
    if (isCone) {
      auto schema = pxr::HdConeSchema::GetFromParent(prim.dataSource);
      height = readDouble(schema.GetHeight(), height);
      radius = readDouble(schema.GetRadius(), radius);
      if (auto a = schema.GetAxis())
        axis = a->GetTypedValue(0);
    } else {
      auto schema = pxr::HdCylinderSchema::GetFromParent(prim.dataSource);
      height = readDouble(schema.GetHeight(), height);
      radius = readDouble(schema.GetRadius(), radius);
      if (auto a = schema.GetAxis())
        axis = a->GetTypedValue(0);
    }

    // Fold the shape's spine axis into the emitted endpoints so that the
    // shape stays analytic rather than becoming a mesh.
    float3 spine(0.f, 0.f, 1.f);
    if (axis == pxr::HdConeSchemaTokens->X)
      spine = float3(1.f, 0.f, 0.f);
    else if (axis == pxr::HdConeSchemaTokens->Y)
      spine = float3(0.f, 1.f, 0.f);

    const float half = float(height) * 0.5f;
    float3 endpoints[2] = {origin - spine * half, origin + spine * half};
    if (bakeXform != tsd::math::IDENTITY_MAT4) {
      for (auto &e : endpoints) {
        const auto t = tsd::math::mul(bakeXform, float4(e.x, e.y, e.z, 1.f));
        e = float3(t.x, t.y, t.z);
      }
    }

    geometry = ctx.scene.createObject<Geometry>(
        isCone ? tokens::geometry::cone : tokens::geometry::cylinder);
    auto positions = ctx.scene.createArray(ANARI_FLOAT32_VEC3, 2);
    positions->setData(endpoints, 2);
    geometry->setParameterObject("vertex.position", *positions);
    if (isCone) {
      const float radii[2] = {float(radius), 0.f};
      auto radiusArray = ctx.scene.createArray(ANARI_FLOAT32, 2);
      radiusArray->setData(radii, 2);
      geometry->setParameterObject("vertex.radius", *radiusArray);
    } else {
      geometry->setParameter("radius", float(radius));
    }
  }

  if (!geometry)
    return {};

  geometry->setName(primPath.GetText());
  return {ctx.scene.createSurface(primPath.GetText(),
      geometry,
      materialForPrim(ctx, sceneIndex, primPath, prim))};
}

} // namespace

std::vector<SurfaceRef> convertGeometry(ImportContext &ctx,
    const pxr::HdSceneIndexBaseRefPtr &sceneIndex,
    const pxr::SdfPath &primPath,
    const pxr::HdSceneIndexPrim &prim,
    const tsd::math::mat4 &bakeXform)
{
  if (prim.primType == pxr::HdPrimTypeTokens->mesh)
    return convertMesh(ctx, sceneIndex, primPath, prim, bakeXform);
  if (prim.primType == pxr::HdPrimTypeTokens->points)
    return convertPoints(ctx, sceneIndex, primPath, prim, bakeXform);
  if (prim.primType == pxr::HdPrimTypeTokens->basisCurves)
    return convertCurves(ctx, sceneIndex, primPath, prim, bakeXform);
  return convertQuadric(ctx, sceneIndex, primPath, prim, bakeXform);
}

} // namespace tsd::io::usd
