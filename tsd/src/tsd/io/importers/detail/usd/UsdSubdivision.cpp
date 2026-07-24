// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/importers/detail/usd/UsdSubdivision.h"
// usd
#include <pxr/imaging/hd/subdivisionTagsSchema.h>
#include <pxr/imaging/pxOsd/meshTopology.h>
#include <pxr/imaging/pxOsd/refinerFactory.h>
#include <pxr/imaging/pxOsd/subdivTags.h>
#include <pxr/imaging/pxOsd/tokens.h>
#include <pxr/usd/usdGeom/mesh.h>
// opensubdiv
#include <opensubdiv/far/primvarRefiner.h>
#include <opensubdiv/far/topologyLevel.h>
// std
#include <cstring>
#include <vector>

namespace tsd::io::usd {

using namespace tsd::core;

namespace {

// Adapter OpenSubdiv's PrimvarRefiner interpolates through. It carries a fixed
// number of floats so one implementation covers every float-typed primvar.
template <int N>
struct FloatTuple
{
  float value[N];

  void Clear();
  void AddWithWeight(const FloatTuple<N> &src, float weight);
};

template <int N>
void FloatTuple<N>::Clear()
{
  for (int i = 0; i < N; ++i)
    value[i] = 0.f;
}

template <int N>
void FloatTuple<N>::AddWithWeight(const FloatTuple<N> &src, float weight)
{
  for (int i = 0; i < N; ++i)
    value[i] += weight * src.value[i];
}

// Refine one buffer of N-component floats through every level of `refiner`,
// returning the values at the last level.
template <int N>
std::vector<FloatTuple<N>> refineBuffer(
    OpenSubdiv::Far::TopologyRefiner &refiner,
    const float *source,
    size_t count)
{
  const int maxLevel = refiner.GetMaxLevel();

  size_t total = 0;
  for (int level = 0; level <= maxLevel; ++level)
    total += size_t(refiner.GetLevel(level).GetNumVertices());

  std::vector<FloatTuple<N>> buffer(total);
  std::memcpy(buffer.data(), source, sizeof(float) * N * count);

  OpenSubdiv::Far::PrimvarRefiner primvarRefiner(refiner);
  FloatTuple<N> *src = buffer.data();
  for (int level = 1; level <= maxLevel; ++level) {
    FloatTuple<N> *dst = src + refiner.GetLevel(level - 1).GetNumVertices();
    primvarRefiner.Interpolate(level, src, dst);
    src = dst;
  }

  const int lastCount = refiner.GetLevel(maxLevel).GetNumVertices();
  return std::vector<FloatTuple<N>>(src, src + lastCount);
}

template <int N, typename VtArrayT>
bool refineTypedPrimvar(OpenSubdiv::Far::TopologyRefiner &refiner,
    const pxr::VtValue &value,
    pxr::VtValue *out)
{
  if (!value.IsHolding<VtArrayT>())
    return false;
  const auto &source = value.UncheckedGet<VtArrayT>();
  if (source.empty())
    return false;

  const auto refined = refineBuffer<N>(
      refiner, reinterpret_cast<const float *>(source.cdata()), source.size());

  VtArrayT result(refined.size());
  std::memcpy(
      result.data(), refined.data(), sizeof(float) * N * refined.size());
  *out = pxr::VtValue(result);
  return true;
}

pxr::VtIntArray intArrayOf(const pxr::HdIntArrayDataSourceHandle &source)
{
  return source ? source->GetTypedValue(0) : pxr::VtIntArray();
}

pxr::VtFloatArray floatArrayOf(const pxr::HdFloatArrayDataSourceHandle &source)
{
  return source ? source->GetTypedValue(0) : pxr::VtFloatArray();
}

pxr::PxOsdSubdivTags readSubdivTags(const pxr::HdMeshSchema &meshSchema)
{
  pxr::PxOsdSubdivTags retval;
  auto tags = meshSchema.GetSubdivisionTags();
  if (!tags)
    return retval;

  if (auto rule = tags.GetInterpolateBoundary())
    retval.SetVertexInterpolationRule(rule->GetTypedValue(0));
  if (auto rule = tags.GetFaceVaryingLinearInterpolation())
    retval.SetFaceVaryingInterpolationRule(rule->GetTypedValue(0));
  if (auto rule = tags.GetTriangleSubdivisionRule())
    retval.SetTriangleSubdivision(rule->GetTypedValue(0));

  retval.SetCreaseIndices(intArrayOf(tags.GetCreaseIndices()));
  retval.SetCreaseLengths(intArrayOf(tags.GetCreaseLengths()));
  retval.SetCreaseWeights(floatArrayOf(tags.GetCreaseSharpnesses()));
  retval.SetCornerIndices(intArrayOf(tags.GetCornerIndices()));
  retval.SetCornerWeights(floatArrayOf(tags.GetCornerSharpnesses()));

  return retval;
}

} // namespace

bool meshWantsRefinement(const ImportContext &ctx, const pxr::SdfPath &primPath)
{
  if (ctx.options.refinementLevel <= 0)
    return false;

  auto prim = ctx.stage->GetPrimAtPath(primPath);
  if (!prim)
    return false;

  pxr::UsdGeomMesh mesh(prim);
  if (!mesh)
    return false;

  auto attribute = mesh.GetSubdivisionSchemeAttr();
  if (!attribute || !attribute.HasAuthoredValue())
    return false;

  pxr::TfToken scheme;
  if (!attribute.Get(&scheme))
    return false;

  return scheme != pxr::PxOsdOpenSubdivTokens->none;
}

RefinedMesh refineMesh(const pxr::HdMeshSchema &meshSchema,
    const pxr::VtIntArray &faceVertexCounts,
    const pxr::VtIntArray &faceVertexIndices,
    const pxr::VtIntArray &holeIndices,
    const pxr::TfToken &orientation,
    const pxr::VtVec3fArray &points,
    const std::vector<std::pair<std::string, pxr::VtValue>> &vertexPrimvars,
    int refinementLevel)
{
  RefinedMesh retval;

  auto schemeSource = meshSchema.GetSubdivisionScheme();
  const auto scheme = schemeSource ? schemeSource->GetTypedValue(0)
                                   : pxr::PxOsdOpenSubdivTokens->catmullClark;

  pxr::PxOsdMeshTopology topology(scheme,
      orientation,
      faceVertexCounts,
      faceVertexIndices,
      holeIndices,
      readSubdivTags(meshSchema));

  auto refiner = pxr::PxOsdRefinerFactory::Create(topology);
  if (!refiner)
    return retval;

  OpenSubdiv::Far::TopologyRefiner::UniformOptions options(refinementLevel);
  options.fullTopologyInLastLevel = true;
  refiner->RefineUniform(options);

  const auto &lastLevel = refiner->GetLevel(refiner->GetMaxLevel());
  if (lastLevel.GetNumFaces() == 0 || lastLevel.GetNumVertices() == 0)
    return retval;

  // Topology of the refined level.
  retval.faceVertexCounts.reserve(lastLevel.GetNumFaces());
  for (int face = 0; face < lastLevel.GetNumFaces(); ++face) {
    const auto vertices = lastLevel.GetFaceVertices(face);
    retval.faceVertexCounts.push_back(vertices.size());
    for (int i = 0; i < vertices.size(); ++i)
      retval.faceVertexIndices.push_back(vertices[i]);
  }

  // Points, and every vertex-interpolated primvar through the same refinement.
  {
    pxr::VtValue refinedPoints;
    if (!refineTypedPrimvar<3, pxr::VtVec3fArray>(
            *refiner, pxr::VtValue(points), &refinedPoints))
      return retval;
    retval.points = refinedPoints.UncheckedGet<pxr::VtVec3fArray>();
  }

  for (const auto &[name, value] : vertexPrimvars) {
    pxr::VtValue refined;
    const bool ok =
        refineTypedPrimvar<1, pxr::VtFloatArray>(*refiner, value, &refined)
        || refineTypedPrimvar<2, pxr::VtVec2fArray>(*refiner, value, &refined)
        || refineTypedPrimvar<3, pxr::VtVec3fArray>(*refiner, value, &refined)
        || refineTypedPrimvar<4, pxr::VtVec4fArray>(*refiner, value, &refined);
    if (ok)
      retval.vertexPrimvars.emplace_back(name, refined);
  }

  retval.valid = true;
  return retval;
}

} // namespace tsd::io::usd
