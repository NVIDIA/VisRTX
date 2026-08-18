// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/animation/UsdGeometryFileBinding.hpp"
// tsd_core
#include "tsd/core/DataTree.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/scene/objects/Array.hpp"
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
    : FileBinding(scene),
      m_geometry(geometry),
      m_session(std::move(session)),
      m_stageFile(std::move(stageFile)),
      m_primPath(std::move(primPath))
{}

UsdGeometryFileBinding::~UsdGeometryFileBinding() = default;

std::string UsdGeometryFileBinding::kind() const
{
  return "usdGeometry";
}

void UsdGeometryFileBinding::toDataNode(core::DataNode &node) const
{
  // The Stage's own clock is enough to re-derive everything a scrub needs, so
  // no cache of authored sample times is written; an older Archive that still
  // carries one is simply not read.
  auto *geometry = m_geometry.get();
  node["targetIndex"] = geometry ? geometry->index() : tsd::core::INVALID_INDEX;
  node["stageFile"] = m_stageFile;
  node["primPath"] = m_primPath;
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

bool UsdGeometryFileBinding::ensureSession()
{
  if (m_session)
    return true;
  if (m_sessionFailed)
    return false;

  m_session = usd::acquireUsdSession(m_stageFile);
  if (!m_session) {
    m_sessionFailed = true;
    logWarning("[UsdGeometryFileBinding] failed to open stage '%s'",
        m_stageFile.c_str());
  }
  return bool(m_session);
}

void UsdGeometryFileBinding::update(float t)
{
  if (!scene() || !ensureSession())
    return;

  auto *geometry = m_geometry.get();
  if (!geometry)
    return;

  m_session->setTime(m_session->timeCodeAt(t));

  auto prim = m_session->stage()->GetPrimAtPath(pxr::SdfPath(m_primPath));
  pxr::UsdGeomPointBased pointBased(prim);
  if (!pointBased)
    return;

  const auto time = m_session->currentTime();

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
            m_primPath.c_str());
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

bool UsdGeometryFileBinding::ensureSession()
{
  return false;
}

void UsdGeometryFileBinding::update(float)
{
  logError("[UsdGeometryFileBinding] USD not enabled in TSD build.");
}

#endif

} // namespace tsd::io
