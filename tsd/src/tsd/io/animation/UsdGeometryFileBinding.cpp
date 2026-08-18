// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/animation/UsdGeometryFileBinding.hpp"
// tsd_core
#include "tsd/core/DataTree.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/scene/objects/Array.hpp"
// std
#include <vector>
#if TSD_USE_USD
// tsd_io
#include "tsd/io/usd/UsdStageSession.h"
// usd
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/pointBased.h>
#endif

namespace tsd::io {

using namespace tsd::core;

UsdGeometryFileBinding::UsdGeometryFileBinding(scene::Scene *scene,
    scene::Geometry *geometry,
    std::shared_ptr<usd::UsdStageSession> session,
    std::string stageFile,
    std::string primPath)
    : UsdFileBinding(
          scene, std::move(session), std::move(stageFile), std::move(primPath)),
      m_geometry(geometry)
{}

UsdGeometryFileBinding::~UsdGeometryFileBinding() = default;

std::string UsdGeometryFileBinding::kind() const
{
  return "usdGeometry";
}

const char *UsdGeometryFileBinding::logTag() const
{
  return "UsdGeometryFileBinding";
}

void UsdGeometryFileBinding::toDataNode(core::DataNode &node) const
{
  // The Stage's own clock is enough to re-derive everything a scrub needs, so
  // no cache of authored sample times is written; an older Archive that still
  // carries one is simply not read.
  auto *geometry = m_geometry.get();
  node["targetIndex"] = geometry ? geometry->index() : tsd::core::INVALID_INDEX;
  writePathsToDataNode(node);
}

void UsdGeometryFileBinding::onDefragment(const scene::IndexRemapper &cb)
{
  if (m_geometry) {
    const size_t newIndex = cb(m_geometry->type(), m_geometry->index());
    m_geometry.updateDefragmentedIndex(newIndex);
  }
}

void UsdGeometryFileBinding::addCallbackToAnimation(
    tsd::animation::Animation &anim)
{
  anim.addCallbackBinding([this](float t) { this->update(t); });
}

UsdGeometryFileBinding *UsdGeometryFileBinding::addToAnimation(
    tsd::animation::Animation &anim, scene::Scene &scene, core::DataNode &node)
{
  const auto targetIndex =
      node["targetIndex"].getValueOr<size_t>(tsd::core::INVALID_INDEX);
  auto *geometry = static_cast<scene::Geometry *>(
      scene.getObject(ANARI_GEOMETRY, targetIndex));
  if (!geometry) {
    logWarning(
        "[UsdGeometryFileBinding] geometry index %zu not found; skipping",
        targetIndex);
    return nullptr;
  }

  return &anim.emplaceFileBinding<UsdGeometryFileBinding>(&scene,
      geometry,
      std::shared_ptr<usd::UsdStageSession>{},
      node["stageFile"].getValueOr<std::string>(""),
      node["primPath"].getValueOr<std::string>(""));
}

#if TSD_USE_USD

namespace {

// Whether the prim's topology moves with its points. When it does, points,
// indices and primvars are one consistent set that has to be re-pulled
// together, which is re-running conversion -- exactly what this binding exists
// to avoid.
bool topologyIsTimeSampled(const pxr::UsdPrim &prim)
{
  pxr::UsdGeomMesh mesh(prim);
  if (!mesh)
    return false;
  const auto counts = mesh.GetFaceVertexCountsAttr();
  const auto indices = mesh.GetFaceVertexIndicesAttr();
  return (counts && counts.GetNumTimeSamples() > 1)
      || (indices && indices.GetNumTimeSamples() > 1);
}

// Write `values` into `array`, or -- if the count moved -- allocate a
// right-sized Array and rebind the parameter to it, since a TSD Array's size
// is fixed at construction.
void writeVertexArray(scene::Scene &scene,
    scene::Geometry &geometry,
    core::Token parameter,
    const pxr::VtVec3fArray &values)
{
  auto *array = geometry.parameterValueAsObject<scene::Array>(parameter);
  if (!array)
    return;

  if (array->size() == values.size()) {
    array->setData(values.cdata());
    return;
  }

  auto replacement = scene.createArray(ANARI_FLOAT32_VEC3, values.size());
  replacement->setData(values.cdata());
  replacement->setName(array->name().c_str());
  geometry.setParameterObject(parameter, *replacement);
}

} // namespace

void UsdGeometryFileBinding::update(float t)
{
  if (!scene() || !ensureSession())
    return;

  auto *geometry = m_geometry.get();
  if (!geometry)
    return;

  auto prim = session()->stage()->GetPrimAtPath(pxr::SdfPath(primPath()));
  pxr::UsdGeomPointBased pointBased(prim);
  if (!pointBased)
    return;

  // A Stage that carries samples but authored no time-code range has no range
  // to map onto until its own prims say what they cover.
  if (!m_sampleTimesNoted && !session()->hasAuthoredTimeRange()) {
    m_sampleTimesNoted = true;
    std::vector<double> times;
    pointBased.GetPointsAttr().GetTimeSamples(&times);
    noteAuthoredSampleTimes(times);
  }

  // Points are read from the Stage's own schema rather than through the
  // Session's resolved chain, as they always have been (ADR 0018); the Session
  // is shared so that this does not mean a second open of the file. Its
  // resolved scene is not read here, so its Time Code is left alone.
  const auto time = session()->timeCodeAt(t);

  pxr::VtVec3fArray points;
  if (pointBased.GetPointsAttr().Get(&points, time) && !points.empty()) {
    auto *positions =
        geometry->parameterValueAsObject<scene::Array>("vertex.position");
    const bool countMoved = positions && positions->size() != points.size();
    if (countMoved && topologyIsTimeSampled(prim)) {
      if (!m_countChangeReported) {
        m_countChangeReported = true;
        logWarning("[UsdGeometryFileBinding] '%s': vertex count and topology"
                   " both change over time; the frame is left as imported",
            primPath().c_str());
      }
      return;
    }
    writeVertexArray(*scene(), *geometry, "vertex.position", points);
  }

  pxr::VtVec3fArray normals;
  if (pointBased.GetNormalsAttr().Get(&normals, time) && !normals.empty())
    writeVertexArray(*scene(), *geometry, "vertex.normal", normals);
}

#else

void UsdGeometryFileBinding::update(float)
{
  ensureSession(); // reports that this build has no USD, once
}

#endif

} // namespace tsd::io
