// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/importers/detail/usd/UsdSubdivision.h"
// tsd_core
#include "tsd/core/TSDMath.hpp"
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
#include <algorithm>
#include <cstring>
#include <type_traits>
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
std::vector<FloatTuple<N>> refineVertexBuffer(
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

// Refine one buffer of N-component floats through the face-varying channel
// `channel`, returning the values at the last level.
template <int N>
std::vector<FloatTuple<N>> refineFaceVaryingBuffer(
    OpenSubdiv::Far::TopologyRefiner &refiner,
    const float *source,
    size_t count,
    int channel)
{
  const int maxLevel = refiner.GetMaxLevel();

  size_t total = 0;
  for (int level = 0; level <= maxLevel; ++level)
    total += size_t(refiner.GetLevel(level).GetNumFVarValues(channel));

  std::vector<FloatTuple<N>> buffer(total);
  std::memcpy(buffer.data(), source, sizeof(float) * N * count);

  OpenSubdiv::Far::PrimvarRefiner primvarRefiner(refiner);
  FloatTuple<N> *src = buffer.data();
  for (int level = 1; level <= maxLevel; ++level) {
    FloatTuple<N> *dst =
        src + refiner.GetLevel(level - 1).GetNumFVarValues(channel);
    primvarRefiner.InterpolateFaceVarying(level, src, dst, channel);
    src = dst;
  }

  const int lastCount = refiner.GetLevel(maxLevel).GetNumFVarValues(channel);
  return std::vector<FloatTuple<N>>(src, src + lastCount);
}

// Refinement strategies, as functors rather than lambdas because they carry a
// member template that a local class may not declare.
struct VertexRefiner
{
  OpenSubdiv::Far::TopologyRefiner *refiner{nullptr};

  template <int N>
  std::vector<FloatTuple<N>> operator()(const float *source, size_t count) const
  {
    return refineVertexBuffer<N>(*refiner, source, count);
  }
};

struct FaceVaryingRefiner
{
  OpenSubdiv::Far::TopologyRefiner *refiner{nullptr};
  int channel{0};

  template <int N>
  std::vector<FloatTuple<N>> operator()(const float *source, size_t count) const
  {
    return refineFaceVaryingBuffer<N>(*refiner, source, count, channel);
  }
};

// Apply `refine` to whichever float-typed array `value` holds, writing the
// result back as the same array type.
template <typename REFINE_FCN>
bool refineFloatArray(
    const pxr::VtValue &value, pxr::VtValue *out, REFINE_FCN &&refine)
{
  auto tryType = [&](auto tag, auto componentCount) {
    using VtArrayT = decltype(tag);
    constexpr int N = decltype(componentCount)::value;
    if (!value.IsHolding<VtArrayT>())
      return false;
    const auto &source = value.UncheckedGet<VtArrayT>();
    if (source.empty())
      return false;

    const auto refined = refine.template operator()<N>(
        reinterpret_cast<const float *>(source.cdata()), source.size());

    VtArrayT result(refined.size());
    std::memcpy(
        result.data(), refined.data(), sizeof(float) * N * refined.size());
    *out = pxr::VtValue(result);
    return true;
  };

  return tryType(pxr::VtFloatArray(), std::integral_constant<int, 1>())
      || tryType(pxr::VtVec2fArray(), std::integral_constant<int, 2>())
      || tryType(pxr::VtVec3fArray(), std::integral_constant<int, 3>())
      || tryType(pxr::VtVec4fArray(), std::integral_constant<int, 4>());
}

// Expand a value indexed by a channel's face-varying indices into face-corner
// order, which is what the triangulator downstream expects.
template <typename VtArrayT>
pxr::VtValue expandToFaceCorners(const VtArrayT &values,
    const OpenSubdiv::Far::TopologyLevel &level,
    int channel)
{
  VtArrayT retval;
  for (int face = 0; face < level.GetNumFaces(); ++face) {
    const auto indices = level.GetFaceFVarValues(face, channel);
    for (int i = 0; i < indices.size(); ++i)
      retval.push_back(values[indices[i]]);
  }
  return pxr::VtValue(retval);
}

pxr::VtValue expandValueToFaceCorners(const pxr::VtValue &value,
    const OpenSubdiv::Far::TopologyLevel &level,
    int channel)
{
  if (value.IsHolding<pxr::VtFloatArray>())
    return expandToFaceCorners(
        value.UncheckedGet<pxr::VtFloatArray>(), level, channel);
  if (value.IsHolding<pxr::VtVec2fArray>())
    return expandToFaceCorners(
        value.UncheckedGet<pxr::VtVec2fArray>(), level, channel);
  if (value.IsHolding<pxr::VtVec3fArray>())
    return expandToFaceCorners(
        value.UncheckedGet<pxr::VtVec3fArray>(), level, channel);
  if (value.IsHolding<pxr::VtVec4fArray>())
    return expandToFaceCorners(
        value.UncheckedGet<pxr::VtVec4fArray>(), level, channel);
  return {};
}

// Replicate a per-face value onto every refined face descended from it.
template <typename VtArrayT>
pxr::VtValue replicateToRefinedFaces(
    const VtArrayT &values, const std::vector<int> &coarseFaceOfRefinedFace)
{
  VtArrayT retval;
  retval.reserve(coarseFaceOfRefinedFace.size());
  for (int coarse : coarseFaceOfRefinedFace) {
    retval.push_back(
        values[size_t(coarse) < values.size() ? size_t(coarse)
                                              : values.size() - 1]);
  }
  return pxr::VtValue(retval);
}

pxr::VtValue replicateValueToRefinedFaces(
    const pxr::VtValue &value, const std::vector<int> &coarseFaceOfRefinedFace)
{
  if (value.IsHolding<pxr::VtFloatArray>())
    return replicateToRefinedFaces(
        value.UncheckedGet<pxr::VtFloatArray>(), coarseFaceOfRefinedFace);
  if (value.IsHolding<pxr::VtVec2fArray>())
    return replicateToRefinedFaces(
        value.UncheckedGet<pxr::VtVec2fArray>(), coarseFaceOfRefinedFace);
  if (value.IsHolding<pxr::VtVec3fArray>())
    return replicateToRefinedFaces(
        value.UncheckedGet<pxr::VtVec3fArray>(), coarseFaceOfRefinedFace);
  if (value.IsHolding<pxr::VtVec4fArray>())
    return replicateToRefinedFaces(
        value.UncheckedGet<pxr::VtVec4fArray>(), coarseFaceOfRefinedFace);
  return {};
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

bool meshWantsRefinement(const pxr::UsdStageRefPtr &stage,
    int refinementLevel,
    const pxr::SdfPath &primPath)
{
  if (refinementLevel <= 0 || !stage)
    return false;

  auto prim = stage->GetPrimAtPath(primPath);
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
    const MeshPrimvars &primvars,
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

  // Face-varying primvars arrive already flattened, one value per face corner,
  // so each gets a channel whose topology is simply the corner order.
  const size_t cornerCount = faceVertexIndices.size();
  pxr::VtIntArray cornerOrder(cornerCount);
  for (size_t i = 0; i < cornerCount; ++i)
    cornerOrder[i] = int(i);
  std::vector<pxr::VtIntArray> faceVaryingTopologies(
      primvars.faceVarying.size(), cornerOrder);

  auto refiner = faceVaryingTopologies.empty()
      ? pxr::PxOsdRefinerFactory::Create(topology)
      : pxr::PxOsdRefinerFactory::Create(topology, faceVaryingTopologies);
  if (!refiner)
    return retval;

  OpenSubdiv::Far::TopologyRefiner::UniformOptions options(refinementLevel);
  options.fullTopologyInLastLevel = true;
  refiner->RefineUniform(options);

  const int maxLevel = refiner->GetMaxLevel();
  const auto &lastLevel = refiner->GetLevel(maxLevel);
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

  // Which coarse face each refined face descends from, which is what carries
  // per-face data and hole tags down to the refined level.
  std::vector<int> coarseFaceOfRefinedFace(lastLevel.GetNumFaces(), 0);
  for (int face = 0; face < lastLevel.GetNumFaces(); ++face) {
    int current = face;
    for (int level = maxLevel; level > 0; --level)
      current = refiner->GetLevel(level).GetFaceParentFace(current);
    coarseFaceOfRefinedFace[size_t(face)] = current;
  }

  for (int face = 0; face < lastLevel.GetNumFaces(); ++face) {
    const int coarse = coarseFaceOfRefinedFace[size_t(face)];
    if (std::find(holeIndices.begin(), holeIndices.end(), coarse)
        != holeIndices.end())
      retval.holeIndices.push_back(face);
  }

  const VertexRefiner vertexRefiner{refiner.get()};

  {
    pxr::VtValue refinedPoints;
    if (!refineFloatArray(pxr::VtValue(points), &refinedPoints, vertexRefiner))
      return retval;
    retval.points = refinedPoints.UncheckedGet<pxr::VtVec3fArray>();
  }

  for (const auto &[name, value] : primvars.vertex) {
    pxr::VtValue refined;
    if (refineFloatArray(value, &refined, vertexRefiner))
      retval.primvars.vertex.emplace_back(name, refined);
  }

  for (size_t channel = 0; channel < primvars.faceVarying.size(); ++channel) {
    const auto &[name, value] = primvars.faceVarying[channel];
    const FaceVaryingRefiner faceVaryingRefiner{refiner.get(), int(channel)};

    pxr::VtValue refined;
    if (!refineFloatArray(value, &refined, faceVaryingRefiner))
      continue;
    auto expanded = expandValueToFaceCorners(refined, lastLevel, int(channel));
    if (!expanded.IsEmpty())
      retval.primvars.faceVarying.emplace_back(name, expanded);
  }

  for (const auto &[name, value] : primvars.uniform) {
    auto replicated =
        replicateValueToRefinedFaces(value, coarseFaceOfRefinedFace);
    if (!replicated.IsEmpty())
      retval.primvars.uniform.emplace_back(name, replicated);
  }

  retval.valid = true;
  return retval;
}

} // namespace tsd::io::usd
