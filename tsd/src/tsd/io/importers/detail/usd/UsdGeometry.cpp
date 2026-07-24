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
  if (value.IsHolding<pxr::VtFloatArray>())
    return pxr::VtValue(expandUniform(
        value.UncheckedGet<pxr::VtFloatArray>(), primitiveParams));
  if (value.IsHolding<pxr::VtVec2fArray>())
    return pxr::VtValue(expandUniform(
        value.UncheckedGet<pxr::VtVec2fArray>(), primitiveParams));
  if (value.IsHolding<pxr::VtVec3fArray>())
    return pxr::VtValue(expandUniform(
        value.UncheckedGet<pxr::VtVec3fArray>(), primitiveParams));
  if (value.IsHolding<pxr::VtVec4fArray>())
    return pxr::VtValue(expandUniform(
        value.UncheckedGet<pxr::VtVec4fArray>(), primitiveParams));
  return {};
}

// Everything one converted mesh needs to hand to the Surface builder. The
// vertex arrays are created once and shared by every material subset.
struct ConvertedMesh
{
  ArrayRef vertexPosition;
  std::vector<std::pair<Token, ArrayRef>> sharedParameters;
  pxr::VtVec3iArray triangleIndices;
  pxr::VtIntArray primitiveParams;
};

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

// Bind one primvar onto a geometry, tessellating or expanding it so that it
// matches the triangulated topology.
void bindMeshPrimvar(ImportContext &ctx,
    GeometryRef &geometry,
    const pxr::HdMeshUtil &meshUtil,
    const ConvertedMesh &mesh,
    const Primvar &primvar,
    const std::string &tsdName)
{
  const char *prefix = prefixForInterpolation(primvar.interpolation);
  if (!prefix)
    return;

  pxr::VtValue value = primvar.value;
  if (primvar.interpolation == pxr::HdPrimvarSchemaTokens->uniform)
    value = expandUniformValue(value, mesh.primitiveParams);
  else if (primvar.interpolation == pxr::HdPrimvarSchemaTokens->faceVarying) {
    pxr::VtValue triangulated;
    const auto result = meshUtil.ComputeTriangulatedFaceVaryingPrimvar(
        pxr::HdGetValueData(primvar.value),
        int(primvar.value.GetArraySize()),
        pxr::HdGetValueTupleType(primvar.value).type,
        &triangulated);
    if (result != pxr::HdMeshComputationResult::Success)
      return;
    value = triangulated;
  }

  const auto type = anariTypeOfPrimvar(value);
  if (type == ANARI_UNKNOWN || !value.IsArrayValued()
      || value.GetArraySize() == 0)
    return;

  auto array = ctx.scene.createArray(type, value.GetArraySize());
  array->setData(pxr::HdGetValueData(value));
  geometry->setParameterObject(Token((prefix + tsdName).c_str()), *array);
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

  auto geometry = ctx.scene.createObject<Geometry>(tokens::geometry::triangle);
  geometry->setName(primPath.GetText());

  const auto positions =
      bakedPositions(points.value.UncheckedGet<pxr::VtVec3fArray>(), bakeXform);
  mesh.vertexPosition =
      ctx.scene.createArray(ANARI_FLOAT32_VEC3, positions.size());
  mesh.vertexPosition->setData(positions.data(), positions.size());
  geometry->setParameterObject("vertex.position", *mesh.vertexPosition);

  auto indexArray =
      ctx.scene.createArray(ANARI_UINT32_VEC3, mesh.triangleIndices.size());
  indexArray->setData(
      (const uint3 *)mesh.triangleIndices.data(), mesh.triangleIndices.size());
  geometry->setParameterObject("primitive.index", *indexArray);

  // Normals, UVs, display colour, then any remaining float-typed primvars in
  // name order so the attribute assignment is deterministic.
  const auto normals = lookup(pxr::HdPrimvarsSchemaTokens->normals.GetString());
  if (normals.valid())
    bindMeshPrimvar(ctx, geometry, meshUtil, mesh, normals, "normal");

  const std::string uvName =
      resolved.uvPrimvarName.empty() ? "st" : resolved.uvPrimvarName;
  const auto uvs = lookup(uvName);
  if (uvs.valid())
    bindMeshPrimvar(ctx, geometry, meshUtil, mesh, uvs, "attribute0");

  const auto displayColor = lookup(pxr::HdTokens->displayColor.GetString());
  if (displayColor.valid()
      && displayColor.interpolation != pxr::HdPrimvarSchemaTokens->constant) {
    bindMeshPrimvar(ctx, geometry, meshUtil, mesh, displayColor, "color");
  }

  int nextAttribute = 1;
  for (const auto &[name, primvar] : primvars) {
    if (nextAttribute > 3)
      break;
    if (name == pxr::HdPrimvarsSchemaTokens->points.GetString()
        || name == pxr::HdPrimvarsSchemaTokens->normals.GetString()
        || name == pxr::HdTokens->displayColor.GetString()
        || name == pxr::HdTokens->displayOpacity.GetString() || name == uvName)
      continue;
    if (anariTypeOfPrimvar(primvar.value) == ANARI_UNKNOWN)
      continue;
    if (!prefixForInterpolation(primvar.interpolation))
      continue;
    bindMeshPrimvar(ctx,
        geometry,
        meshUtil,
        mesh,
        primvar,
        "attribute" + std::to_string(nextAttribute++));
  }

  // Material //

  auto material = resolved.material;
  if (!material) {
    material = displayColorMaterial(ctx,
        primPath,
        displayColor,
        lookup(pxr::HdTokens->displayOpacity.GetString()));
  }

  // Per-face material subsets each become their own Surface over their own
  // index array, sharing this mesh's vertex arrays.
  std::vector<SurfaceRef> retval;
  std::vector<pxr::SdfPath> subsetPaths;
  for (const auto &childPath : sceneIndex->GetChildPrimPaths(primPath)) {
    if (sceneIndex->GetPrim(childPath).primType
        == pxr::HdPrimTypeTokens->geomSubset)
      subsetPaths.push_back(childPath);
  }

  if (subsetPaths.empty()) {
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

  for (const auto &subsetPath : subsetPaths) {
    auto subsetPrim = sceneIndex->GetPrim(subsetPath);
    auto subsetSchema =
        pxr::HdGeomSubsetSchema::GetFromParent(subsetPrim.dataSource);
    const auto faceIndices = intArrayOf(subsetSchema.GetIndices());
    if (faceIndices.empty())
      continue;

    std::vector<uint3> subsetTriangles;
    for (int face : faceIndices) {
      if (face < 0 || size_t(face) >= trianglesOfFace.size())
        continue;
      for (uint32_t triangle : trianglesOfFace[size_t(face)]) {
        const auto &t = mesh.triangleIndices[triangle];
        subsetTriangles.push_back(uint3(t[0], t[1], t[2]));
      }
    }
    if (subsetTriangles.empty())
      continue;

    auto subsetGeometry =
        ctx.scene.createObject<Geometry>(tokens::geometry::triangle);
    subsetGeometry->setName(subsetPath.GetText());
    subsetGeometry->setParameterObject("vertex.position", *mesh.vertexPosition);
    auto subsetIndex =
        ctx.scene.createArray(ANARI_UINT32_VEC3, subsetTriangles.size());
    subsetIndex->setData(subsetTriangles.data(), subsetTriangles.size());
    subsetGeometry->setParameterObject("primitive.index", *subsetIndex);

    // Share every vertex-interpolated attribute the parent mesh carries.
    for (const auto &name :
        {"vertex.normal", "vertex.attribute0", "vertex.color"}) {
      if (auto *array = geometry->parameterValueAsObject<Array>(name))
        subsetGeometry->setParameterObject(Token(name), *array);
    }

    auto subsetMaterial =
        resolveMaterial(ctx, sceneIndex, boundMaterialPathOf(subsetPrim))
            .material;

    retval.push_back(ctx.scene.createSurface(subsetPath.GetText(),
        subsetGeometry,
        subsetMaterial ? subsetMaterial : material));
  }

  if (retval.empty()) {
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
