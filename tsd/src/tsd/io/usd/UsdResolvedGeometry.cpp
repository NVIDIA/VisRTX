// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/usd/UsdResolvedGeometry.h"
#include "tsd/io/importers/detail/usd/UsdSubdivision.h"
#include "tsd/scene/objects/Array.hpp"
// usd
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/imaging/hd/basisCurvesSchema.h>
#include <pxr/imaging/hd/basisCurvesTopologySchema.h>
#include <pxr/imaging/hd/coneSchema.h>
#include <pxr/imaging/hd/cylinderSchema.h>
#include <pxr/imaging/hd/geomSubsetSchema.h>
#include <pxr/imaging/hd/meshSchema.h>
#include <pxr/imaging/hd/meshTopology.h>
#include <pxr/imaging/hd/meshUtil.h>
#include <pxr/imaging/hd/primvarsSchema.h>
#include <pxr/imaging/hd/sphereSchema.h>
#include <pxr/imaging/hd/tokens.h>
// std
#include <algorithm>
#include <limits>
#include <numeric>
#include <string_view>
#include <utility>

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
// Shared resolution helpers //////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// Prototype-internal transforms are baked into vertex data (ADR 0016);
// everything else passes through untouched.
pxr::VtVec3fArray bakedPositions(
    const pxr::VtVec3fArray &source, const tsd::math::mat4 &bakeXform)
{
  if (bakeXform == tsd::math::IDENTITY_MAT4)
    return source;

  pxr::VtVec3fArray retval;
  retval.reserve(source.size());
  for (const auto &p : source) {
    const auto t = tsd::math::mul(bakeXform, float4(p[0], p[1], p[2], 1.f));
    retval.push_back(pxr::GfVec3f(t.x, t.y, t.z));
  }
  return retval;
}

// The one way an attribute joins a Part: nothing that would bind as an empty
// or untyped Array gets in.
void addTypedAttribute(ResolvedPart &part,
    Token parameter,
    anari::DataType type,
    pxr::VtValue value,
    std::string sharedKey = {})
{
  if (type == ANARI_UNKNOWN || !value.IsArrayValued()
      || value.GetArraySize() == 0)
    return;
  part.attributes.push_back(
      {parameter, type, std::move(value), std::move(sharedKey)});
}

// The same, for the float-typed primvar data whose ANARI type is inferable
// from what the VtValue holds.
void addAttribute(ResolvedPart &part,
    Token parameter,
    pxr::VtValue value,
    std::string sharedKey = {})
{
  const auto type = anariTypeOfPrimvar(value);
  addTypedAttribute(
      part, parameter, type, std::move(value), std::move(sharedKey));
}

// USD authors widths; TSD geometry takes radii. Prims with no authored
// widths (common for Blender hair exports) would otherwise inherit ANARI's
// default radius of 1 world unit, which dwarfs most scenes -- instead fall
// back to a small radius scaled to the prim's own bounds so strands stay
// hair-like at any scene scale.
void resolveRadii(ResolvedPart &part,
    const Primvar &widths,
    const pxr::VtVec3fArray &positions)
{
  const bool haveWidths = widths.valid()
      && widths.value.IsHolding<pxr::VtFloatArray>()
      && !widths.value.UncheckedGet<pxr::VtFloatArray>().empty();

  if (!haveWidths) {
    float3 lo(std::numeric_limits<float>::max());
    float3 hi(std::numeric_limits<float>::lowest());
    for (const auto &p : positions) {
      const float3 v(p[0], p[1], p[2]);
      lo = tsd::math::min(lo, v);
      hi = tsd::math::max(hi, v);
    }
    const float diagonal = positions.empty() ? 0.f : tsd::math::length(hi - lo);
    part.scalars.emplace_back(
        Token("radius"), diagonal > 0.f ? 1e-3f * diagonal : 1e-3f);
    return;
  }

  const auto &w = widths.value.UncheckedGet<pxr::VtFloatArray>();
  pxr::VtFloatArray radii;
  radii.reserve(positions.size());
  for (size_t i = 0; i < positions.size(); ++i)
    radii.push_back(0.5f * w[std::min(i, w.size() - 1)]);
  addAttribute(part, Token("vertex.radius"), pxr::VtValue(radii));
}

///////////////////////////////////////////////////////////////////////////////
// Mesh resolution ////////////////////////////////////////////////////////////
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

// Reverse texture coordinates. USD authors `st` v-up, while the coordinates
// TSD hands ANARI run down the image. See
// docs/adr/0014-store-images-in-anari-orientation.md.
pxr::VtValue reversedTexCoordV(const pxr::VtValue &value)
{
  if (!value.IsHolding<pxr::VtVec2fArray>())
    return value;
  auto uv = value.UncheckedGet<pxr::VtVec2fArray>();
  for (auto &c : uv)
    c[1] = 1.f - c[1];
  return pxr::VtValue(uv);
}

// The triangulated mesh, before any Part selects from it.
struct TriangulatedMesh
{
  pxr::VtVec3fArray positions;
  pxr::VtVec3iArray triangleIndices;
  pxr::VtIntArray primitiveParams;
};

// A primvar expanded onto the triangulated topology and ready to bind. Vertex
// data stays as authored and is indexed by the triangle indices, so it is
// shared by every Part built from the mesh; uniform and face-varying data are
// laid out in triangle order and have to be gathered per Part, because a
// subset draws only some of the triangles.
struct TriangulatedPrimvar
{
  pxr::VtValue value;
  const char *prefix{nullptr};
  size_t valuesPerTriangle{0};

  // Vertex data is the only kind every Part can point at unchanged.
  bool isShared() const;
};

bool TriangulatedPrimvar::isShared() const
{
  return valuesPerTriangle == 0;
}

// Kept sorted by name: the order primvars are visited decides which of them
// takes each spare attribute slot, and that has to be stable across runs. This
// is why the mesh resolver reaches for std::map rather than FlatMap.
using TriangulatedPrimvars = std::map<std::string, TriangulatedPrimvar>;

// Expand every primvar that has an attribute slot onto the triangulated
// topology, once, so that each Part only has to select the values for its own
// triangles. Primvars whose expansion fails or comes up short of the
// triangulation are left out rather than bound partially.
TriangulatedPrimvars triangulatePrimvars(const pxr::HdMeshUtil &meshUtil,
    const TriangulatedMesh &mesh,
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
      // Unchanged means the mesh is already all triangles: the flattened
      // input is already one value per triangle corner, in triangle order.
      if (result == pxr::HdMeshComputationResult::Unchanged)
        attribute.value = primvar.value;
      else if (result != pxr::HdMeshComputationResult::Success)
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

// Resolve one primvar into one Part's attribute slot. `isUv` marks the one
// primvar this Part's material reads as texture coordinates, which is the only
// binding whose `v` is reversed -- the same primvar in a spare slot elsewhere
// is data TSD knows nothing about, and is resolved as authored.
void resolvePartPrimvar(ResolvedPart &part,
    const std::string &primvarName,
    const TriangulatedPrimvar &primvar,
    const std::vector<uint32_t> &triangles,
    const std::string &tsdName,
    bool isUv)
{
  const Token parameter((primvar.prefix + tsdName).c_str());

  if (primvar.isShared()) {
    auto value = isUv ? reversedTexCoordV(primvar.value) : primvar.value;
    addAttribute(part,
        parameter,
        std::move(value),
        primvarName + (isUv ? "#uv" : "#raw"));
    return;
  }

  auto selected = gatherTrianglesValue(
      primvar.value, triangles, primvar.valuesPerTriangle);
  if (isUv)
    selected = reversedTexCoordV(selected);
  addAttribute(part, parameter, std::move(selected));
}

// One Part's worth of the mesh: the triangles it draws, with every primvar
// re-indexed to match. `uvName` is whichever primvar this Part's own material
// reads, which is why the attribute slots cannot be assigned once for the
// whole mesh.
ResolvedPart resolveTrianglePart(const TriangulatedMesh &mesh,
    const TriangulatedPrimvars &attributes,
    const std::vector<uint32_t> &triangles,
    const std::string &uvName,
    const std::vector<std::string> *replaySlots,
    const std::string &name)
{
  ResolvedPart part;
  part.subtype = tsd::scene::tokens::geometry::triangle;
  part.name = name;

  addAttribute(part,
      Token("vertex.position"),
      pxr::VtValue(mesh.positions),
      "points");

  pxr::VtVec3iArray indices;
  indices.reserve(triangles.size());
  for (uint32_t triangle : triangles)
    indices.push_back(mesh.triangleIndices[triangle]);
  addTypedAttribute(part,
      Token("primitive.index"),
      ANARI_UINT32_VEC3,
      pxr::VtValue(indices));

  auto bind = [&](const std::string &primvarName,
                  const std::string &tsdName,
                  bool isUv = false) {
    auto found = attributes.find(primvarName);
    if (found != attributes.end()) {
      resolvePartPrimvar(
          part, primvarName, found->second, triangles, tsdName, isUv);
    }
  };

  // Normals, UVs, display colour, then any remaining primvars in name order so
  // the attribute assignment is deterministic.
  const auto normalsName = pxr::HdPrimvarsSchemaTokens->normals.GetString();
  const auto colorName = pxr::HdTokens->displayColor.GetString();
  bind(normalsName, "normal");
  bind(uvName, "attribute0", /*isUv=*/true);
  bind(colorName, "color");

  // Replaying a recorded assignment keeps a primvar that appears or disappears
  // mid-sequence from re-slotting the others; a Part being resolved for the
  // first time has nothing to replay and assigns in name order, which is
  // deterministic because the primvar map is sorted.
  if (replaySlots) {
    int slot = 1;
    for (const auto &primvarName : *replaySlots) {
      if (slot > 3)
        break;
      const auto tsdName = "attribute" + std::to_string(slot++);
      part.slotPrimvars.push_back(primvarName);
      auto found = attributes.find(primvarName);
      if (found == attributes.end())
        continue; // the slot stays empty rather than shifting the rest along
      resolvePartPrimvar(
          part, primvarName, found->second, triangles, tsdName, false);
    }
    return part;
  }

  int nextAttribute = 1;
  for (const auto &[primvarName, primvar] : attributes) {
    if (nextAttribute > 3)
      break;
    if (primvarName == normalsName || primvarName == colorName
        || primvarName == uvName)
      continue;
    part.slotPrimvars.push_back(primvarName);
    resolvePartPrimvar(part,
        primvarName,
        primvar,
        triangles,
        "attribute" + std::to_string(nextAttribute++),
        false);
  }

  return part;
}

// Append the triangles one coarse face produced to a Part's selection.
void appendTrianglesOfFace(std::vector<uint32_t> &selection,
    const std::vector<std::vector<uint32_t>> &trianglesOfFace,
    size_t face)
{
  const auto &triangles = trianglesOfFace[face];
  selection.insert(selection.end(), triangles.begin(), triangles.end());
}

std::string uvNameFor(const GeometryResolveOptions &options,
    const std::string &partName,
    const std::string &fallback)
{
  const auto *found = options.uvNamesByPart.at(partName);
  return found ? *found : fallback;
}

const std::vector<std::string> *replaySlotsFor(
    const GeometryResolveOptions &options, const std::string &partName)
{
  return options.slotPrimvarsByPart.at(partName);
}

ResolvedGeometry resolveMesh(const pxr::HdSceneIndexBaseRefPtr &sceneIndex,
    const pxr::SdfPath &primPath,
    const pxr::HdSceneIndexPrim &prim,
    const GeometryResolveOptions &options)
{
  ResolvedGeometry retval;

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

  auto pointsIt = primvars.find(pxr::HdPrimvarsSchemaTokens->points.GetString());
  if (pointsIt == primvars.end()
      || !pointsIt->second.value.IsHolding<pxr::VtVec3fArray>())
    return retval;
  auto points = pointsIt->second;

  auto faceVertexCounts = intArrayOf(topologySchema.GetFaceVertexCounts());
  auto faceVertexIndices = intArrayOf(topologySchema.GetFaceVertexIndices());
  auto holeIndices = intArrayOf(topologySchema.GetHoleIndices());
  if (faceVertexCounts.empty() || faceVertexIndices.empty())
    return retval;

  auto orientationSource = topologySchema.GetOrientation();
  const auto orientation = orientationSource
      ? orientationSource->GetTypedValue(0)
      : pxr::HdTokens->rightHanded;

  if (options.refine) {
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
        options.refinementLevel);

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
  TriangulatedMesh mesh;
  meshUtil.ComputeTriangleIndices(&mesh.triangleIndices, &mesh.primitiveParams);
  if (mesh.triangleIndices.empty())
    return retval;

  mesh.positions = bakedPositions(
      points.value.UncheckedGet<pxr::VtVec3fArray>(), options.bakeXform);

  const auto attributes = triangulatePrimvars(meshUtil, mesh, primvars);

  const std::string meshName = primPath.GetString();
  const std::string meshUvName = uvNameFor(options, meshName, "st");

  std::vector<uint32_t> allTriangles(mesh.triangleIndices.size());
  std::iota(allTriangles.begin(), allTriangles.end(), 0u);

  // Per-face material subsets each become their own Part over their own
  // triangles, sharing this mesh's vertex data.
  std::vector<pxr::SdfPath> subsetPaths;
  for (const auto &childPath : sceneIndex->GetChildPrimPaths(primPath)) {
    if (sceneIndex->GetPrim(childPath).primType
        == pxr::HdPrimTypeTokens->geomSubset)
      subsetPaths.push_back(childPath);
  }

  if (subsetPaths.empty()) {
    retval.parts.push_back(resolveTrianglePart(mesh,
        attributes,
        allTriangles,
        meshUvName,
        replaySlotsFor(options, meshName),
        meshName));
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

    // A subset's material may read a different UV primvar than the mesh's
    // does, so its attributes are resolved to suit it.
    const std::string subsetName = subsetPath.GetString();
    retval.parts.push_back(resolveTrianglePart(mesh,
        attributes,
        subsetTriangles,
        uvNameFor(options, subsetName, meshUvName),
        replaySlotsFor(options, subsetName),
        subsetName));
  }

  // Faces no subset claimed keep the mesh's own binding rather than going
  // missing with the geometry that no Part would have drawn.
  std::vector<uint32_t> unclaimedTriangles;
  for (size_t face = 0; face < trianglesOfFace.size(); ++face) {
    if (!faceIsClaimed[face])
      appendTrianglesOfFace(unclaimedTriangles, trianglesOfFace, face);
  }

  // No subset drew anything -- with nothing to divide the mesh up, draw all of
  // it, including any triangle whose coarse face could not be identified.
  if (retval.parts.empty())
    unclaimedTriangles = allTriangles;

  if (!unclaimedTriangles.empty()) {
    retval.parts.push_back(resolveTrianglePart(mesh,
        attributes,
        unclaimedTriangles,
        meshUvName,
        replaySlotsFor(options, meshName),
        meshName));
  }

  return retval;
}

///////////////////////////////////////////////////////////////////////////////
// Points and curves //////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

ResolvedGeometry resolvePoints(const pxr::SdfPath &primPath,
    const pxr::HdSceneIndexPrim &prim,
    const GeometryResolveOptions &options)
{
  ResolvedGeometry retval;

  auto primvars = pxr::HdPrimvarsSchema::GetFromParent(prim.dataSource);
  const auto points =
      readPrimvar(primvars, pxr::HdPrimvarsSchemaTokens->points);
  if (!points.valid() || !points.value.IsHolding<pxr::VtVec3fArray>())
    return retval;

  ResolvedPart part;
  part.subtype = tsd::scene::tokens::geometry::sphere;
  part.name = primPath.GetString();

  const auto positions = bakedPositions(
      points.value.UncheckedGet<pxr::VtVec3fArray>(), options.bakeXform);
  addAttribute(part, Token("vertex.position"), pxr::VtValue(positions));
  resolveRadii(part,
      readPrimvar(primvars, pxr::HdPrimvarsSchemaTokens->widths),
      positions);

  retval.parts.push_back(std::move(part));
  return retval;
}

ResolvedGeometry resolveCurves(const pxr::SdfPath &primPath,
    const pxr::HdSceneIndexPrim &prim,
    const GeometryResolveOptions &options)
{
  ResolvedGeometry retval;

  auto curvesSchema = pxr::HdBasisCurvesSchema::GetFromParent(prim.dataSource);
  auto topologySchema = curvesSchema.GetTopology();
  auto primvars = pxr::HdPrimvarsSchema::GetFromParent(prim.dataSource);

  const auto points =
      readPrimvar(primvars, pxr::HdPrimvarsSchemaTokens->points);
  if (!points.valid() || !points.value.IsHolding<pxr::VtVec3fArray>())
    return retval;

  const auto vertexCounts = intArrayOf(topologySchema.GetCurveVertexCounts());
  if (vertexCounts.empty())
    return retval;

  ResolvedPart part;
  part.subtype = tsd::scene::tokens::geometry::curve;
  part.name = primPath.GetString();

  const auto positions = bakedPositions(
      points.value.UncheckedGet<pxr::VtVec3fArray>(), options.bakeXform);
  addAttribute(part, Token("vertex.position"), pxr::VtValue(positions));

  // A curve segment index per consecutive vertex pair within each curve.
  pxr::VtUIntArray segments;
  uint32_t base = 0;
  for (int count : vertexCounts) {
    for (int i = 0; i + 1 < count; ++i)
      segments.push_back(base + uint32_t(i));
    base += uint32_t(count);
  }
  if (!segments.empty()) {
    addTypedAttribute(
        part, Token("primitive.index"), ANARI_UINT32, pxr::VtValue(segments));
  }

  resolveRadii(part,
      readPrimvar(primvars, pxr::HdPrimvarsSchemaTokens->widths),
      positions);

  retval.parts.push_back(std::move(part));
  return retval;
}

///////////////////////////////////////////////////////////////////////////////
// Analytic quadrics //////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

ResolvedGeometry resolveQuadric(const pxr::SdfPath &primPath,
    const pxr::HdSceneIndexPrim &prim,
    const GeometryResolveOptions &options)
{
  ResolvedGeometry retval;

  auto readDouble = [](const pxr::HdDoubleDataSourceHandle &h, double alt) {
    return h ? h->GetTypedValue(0) : alt;
  };

  const auto &bakeXform = options.bakeXform;
  const float3 origin = [&] {
    if (bakeXform == tsd::math::IDENTITY_MAT4)
      return float3(0.f);
    const auto t = tsd::math::mul(bakeXform, float4(0.f, 0.f, 0.f, 1.f));
    return float3(t.x, t.y, t.z);
  }();

  ResolvedPart part;
  part.name = primPath.GetString();

  if (prim.primType == pxr::HdPrimTypeTokens->sphere) {
    auto schema = pxr::HdSphereSchema::GetFromParent(prim.dataSource);
    part.subtype = tsd::scene::tokens::geometry::sphere;
    pxr::VtVec3fArray positions{pxr::GfVec3f(origin.x, origin.y, origin.z)};
    addAttribute(part, Token("vertex.position"), pxr::VtValue(positions));
    part.scalars.emplace_back(
        Token("radius"), float(readDouble(schema.GetRadius(), 1.0)));
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

    part.subtype = isCone ? tsd::scene::tokens::geometry::cone
                          : tsd::scene::tokens::geometry::cylinder;
    pxr::VtVec3fArray positions;
    for (const auto &e : endpoints)
      positions.push_back(pxr::GfVec3f(e.x, e.y, e.z));
    addAttribute(part, Token("vertex.position"), pxr::VtValue(positions));

    if (isCone) {
      pxr::VtFloatArray radii{float(radius), 0.f};
      addAttribute(part, Token("vertex.radius"), pxr::VtValue(radii));
    } else {
      part.scalars.emplace_back(Token("radius"), float(radius));
    }
  } else {
    return retval;
  }

  retval.parts.push_back(std::move(part));
  return retval;
}

} // namespace

///////////////////////////////////////////////////////////////////////////////
// Plain-data accessors ///////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

size_t ResolvedAttribute::count() const
{
  return value.IsArrayValued() ? value.GetArraySize() : 0;
}

const void *ResolvedAttribute::data() const
{
  return pxr::HdGetValueData(value);
}

bool ResolvedAttribute::valid() const
{
  return type != ANARI_UNKNOWN && count() > 0 && data() != nullptr;
}

const ResolvedAttribute *ResolvedPart::attribute(Token parameter) const
{
  for (const auto &a : attributes) {
    if (a.parameter == parameter)
      return &a;
  }
  return nullptr;
}

bool ResolvedPart::provides(Token parameter) const
{
  if (attribute(parameter))
    return true;
  for (const auto &[name, value] : scalars) {
    if (name == parameter)
      return true;
  }
  return false;
}

bool ResolvedGeometry::valid() const
{
  return !parts.empty();
}

const ResolvedPart *ResolvedGeometry::part(const std::string &name) const
{
  for (const auto &p : parts) {
    if (p.name == name)
      return &p;
  }
  return nullptr;
}

///////////////////////////////////////////////////////////////////////////////
// Entry points ///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// Cheap enough to ask before anything is built: every check here is a data
// source read, not a conversion.
bool geometryWillResolve(const pxr::HdSceneIndexPrim &prim)
{
  if (!isGeometryPrimType(prim.primType))
    return false;

  if (prim.primType == pxr::HdPrimTypeTokens->sphere
      || prim.primType == pxr::HdPrimTypeTokens->cone
      || prim.primType == pxr::HdPrimTypeTokens->cylinder)
    return true;

  auto primvars = pxr::HdPrimvarsSchema::GetFromParent(prim.dataSource);
  const auto points =
      readPrimvar(primvars, pxr::HdPrimvarsSchemaTokens->points);
  if (!points.valid() || !points.value.IsHolding<pxr::VtVec3fArray>())
    return false;

  if (prim.primType == pxr::HdPrimTypeTokens->mesh) {
    auto topology =
        pxr::HdMeshSchema::GetFromParent(prim.dataSource).GetTopology();
    return !intArrayOf(topology.GetFaceVertexCounts()).empty()
        && !intArrayOf(topology.GetFaceVertexIndices()).empty();
  }

  if (prim.primType == pxr::HdPrimTypeTokens->basisCurves) {
    auto topology =
        pxr::HdBasisCurvesSchema::GetFromParent(prim.dataSource).GetTopology();
    return !intArrayOf(topology.GetCurveVertexCounts()).empty();
  }

  return true;
}

bool isGeometryPrimType(const pxr::TfToken &primType)
{
  return primType == pxr::HdPrimTypeTokens->mesh
      || primType == pxr::HdPrimTypeTokens->points
      || primType == pxr::HdPrimTypeTokens->basisCurves
      || primType == pxr::HdPrimTypeTokens->sphere
      || primType == pxr::HdPrimTypeTokens->cone
      || primType == pxr::HdPrimTypeTokens->cylinder;
}

DisplayColor readDisplayColor(const pxr::HdSceneIndexPrim &prim)
{
  DisplayColor retval;

  auto primvars = pxr::HdPrimvarsSchema::GetFromParent(prim.dataSource);
  const auto color = readPrimvar(primvars, pxr::HdTokens->displayColor);
  const auto opacity = readPrimvar(primvars, pxr::HdTokens->displayOpacity);

  if (color.valid() && color.value.IsHolding<pxr::VtVec3fArray>()) {
    const auto &c = color.value.UncheckedGet<pxr::VtVec3fArray>();
    retval.color = float3(c[0][0], c[0][1], c[0][2]);
  }
  if (opacity.valid() && opacity.value.IsHolding<pxr::VtFloatArray>())
    retval.opacity = opacity.value.UncheckedGet<pxr::VtFloatArray>()[0];
  return retval;
}

ResolvedGeometry resolveGeometry(
    const pxr::HdSceneIndexBaseRefPtr &sceneIndex,
    const pxr::SdfPath &primPath,
    const pxr::HdSceneIndexPrim &prim,
    const GeometryResolveOptions &options)
{
  if (prim.primType == pxr::HdPrimTypeTokens->mesh)
    return resolveMesh(sceneIndex, primPath, prim, options);
  if (prim.primType == pxr::HdPrimTypeTokens->points)
    return resolvePoints(primPath, prim, options);
  if (prim.primType == pxr::HdPrimTypeTokens->basisCurves)
    return resolveCurves(primPath, prim, options);
  return resolveQuadric(primPath, prim, options);
}

namespace {

// Whether a parameter is one this module owns, and so may clear when a resolve
// stops providing it. Anything else on the Geometry was put there by something
// that is not a per-frame resolve, and is left alone.
bool isResolvedParameterName(const char *name)
{
  const std::string_view view(name ? name : "");
  return view.rfind("vertex.", 0) == 0 || view.rfind("primitive.", 0) == 0
      || view.rfind("faceVarying.", 0) == 0 || view == "radius";
}

// Drop what this Part no longer provides, so a primvar that stops resolving
// leaves nothing of the previous frame behind -- including the case where a
// prim swaps a `vertex.radius` array for a scalar `radius` or back.
void clearStaleParameters(
    scene::Geometry &geometry, const ResolvedPart &part)
{
  std::vector<Token> stale;
  for (size_t i = 0; i < geometry.numParameters(); ++i) {
    const auto *name = geometry.parameterNameAt(i);
    if (isResolvedParameterName(name) && !part.provides(Token(name)))
      stale.push_back(Token(name));
  }
  for (auto name : stale)
    geometry.removeParameter(name);
}

} // namespace

bool refillGeometry(scene::Scene &scene,
    scene::Geometry &geometry,
    const ResolvedPart &part,
    RefillCache &cache)
{
  if (geometry.subtype() != part.subtype)
    return false;

  for (const auto &attribute : part.attributes) {
    if (!attribute.valid())
      continue;

    // A buffer shared with another Part of the same gprim is written once and
    // then only re-bound, so a mesh's Surfaces keep pointing at one Array and
    // that Array is not rewritten once per Surface per frame.
    if (!attribute.sharedKey.empty()) {
      if (auto *shared = cache.sharedArrays.at(attribute.sharedKey)) {
        geometry.setParameterObject(attribute.parameter, **shared);
        continue;
      }
    }

    auto *array =
        geometry.parameterValueAsObject<scene::Array>(attribute.parameter);
    if (array && array->size() == attribute.count()
        && array->elementType() == attribute.type) {
      array->setData(attribute.data());
      if (!attribute.sharedKey.empty())
        cache.sharedArrays.set(attribute.sharedKey, array->self());
      continue;
    }

    // A TSD Array's size is fixed at construction, so an element count that
    // moves costs one allocation and one rebind for that parameter. Every
    // parameter of the Part is written in this one pass, so the Geometry is
    // never left half in one frame and half in another.
    auto replacement = scene.createArray(attribute.type, attribute.count());
    replacement->setData(attribute.data());
    if (array)
      replacement->setName(array->name().c_str());
    geometry.setParameterObject(attribute.parameter, *replacement);
    if (!attribute.sharedKey.empty())
      cache.sharedArrays.set(attribute.sharedKey, replacement);
  }

  for (const auto &[name, value] : part.scalars)
    geometry.setParameter(name, value);

  clearStaleParameters(geometry, part);

  return true;
}

} // namespace tsd::io::usd
