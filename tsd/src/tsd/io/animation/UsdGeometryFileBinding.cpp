// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/animation/UsdGeometryFileBinding.hpp"
// tsd_core
#include "tsd/core/DataTree.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/scene/objects/Array.hpp"
#if TSD_USE_USD
// usd
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/pointBased.h>
#endif
// std
#include <algorithm>
#include <cmath>

namespace tsd::io {

using namespace tsd::core;

#if TSD_USE_USD

struct UsdGeometryFileBinding::StageHolder
{
  pxr::UsdStageRefPtr stage;
};

#else

struct UsdGeometryFileBinding::StageHolder
{
};

#endif

UsdGeometryFileBinding::UsdGeometryFileBinding(scene::Scene *scene,
    scene::Geometry *geometry,
    std::string stageFile,
    std::string primPath,
    std::vector<double> sampleTimes)
    : FileBinding(scene),
      m_geometry(geometry),
      m_stageFile(std::move(stageFile)),
      m_primPath(std::move(primPath)),
      m_sampleTimes(std::move(sampleTimes))
{}

std::string UsdGeometryFileBinding::kind() const
{
  return "usdGeometry";
}

void UsdGeometryFileBinding::toDataNode(core::DataNode &node) const
{
  auto *geometry = m_geometry.get();
  node["targetIndex"] = geometry ? geometry->index() : size_t(-1);
  node["stageFile"] = m_stageFile;
  node["primPath"] = m_primPath;

  auto &timesNode = node["sampleTimes"];
  for (double t : m_sampleTimes)
    timesNode.append() = float(t);
}

void UsdGeometryFileBinding::onDefragment(const scene::IndexRemapper &cb)
{
  if (m_geometry) {
    const size_t newIndex = cb(m_geometry->type(), m_geometry->index());
    m_geometry.updateDefragmentedIndex(newIndex);
  }
}

size_t UsdGeometryFileBinding::frameCount() const
{
  return m_sampleTimes.size();
}

int UsdGeometryFileBinding::currentFrame() const
{
  return m_currentFrame;
}

void UsdGeometryFileBinding::addCallbackToAnimation(
    tsd::animation::Animation &anim)
{
  anim.addCallbackBinding([this](float t) { this->update(t); });
}

UsdGeometryFileBinding *UsdGeometryFileBinding::addToAnimation(
    tsd::animation::Animation &anim, scene::Scene &scene, core::DataNode &node)
{
  const auto targetIndex = node["targetIndex"].getValueOr<size_t>(size_t(-1));
  auto *geometry = static_cast<scene::Geometry *>(
      scene.getObject(ANARI_GEOMETRY, targetIndex));
  if (!geometry) {
    logWarning(
        "[UsdGeometryFileBinding] geometry index %zu not found; skipping",
        targetIndex);
    return nullptr;
  }

  std::vector<double> sampleTimes;
  if (auto *timesNode = node.child("sampleTimes")) {
    timesNode->foreach_child([&](core::DataNode &n) {
      sampleTimes.push_back(n.getValueOr<float>(0.f));
    });
  }

  return &anim.emplaceFileBinding<UsdGeometryFileBinding>(&scene,
      geometry,
      node["stageFile"].getValueOr<std::string>(""),
      node["primPath"].getValueOr<std::string>(""),
      std::move(sampleTimes));
}

#if TSD_USE_USD

void UsdGeometryFileBinding::update(float t)
{
  if (m_sampleTimes.empty() || !scene())
    return;

  const int frameCount = int(m_sampleTimes.size());
  const int frame =
      std::clamp(int(std::round(t * float(frameCount - 1))), 0, frameCount - 1);
  if (frame == m_currentFrame && m_stage)
    return;

  if (!m_stage) {
    m_stage = std::make_shared<StageHolder>();
    m_stage->stage = pxr::UsdStage::Open(m_stageFile);
    if (!m_stage->stage) {
      logWarning("[UsdGeometryFileBinding] failed to open stage '%s'",
          m_stageFile.c_str());
      return;
    }
  }

  auto prim = m_stage->stage->GetPrimAtPath(pxr::SdfPath(m_primPath));
  pxr::UsdGeomPointBased pointBased(prim);
  if (!pointBased)
    return;

  auto *geometry = m_geometry.get();
  if (!geometry)
    return;

  const pxr::UsdTimeCode time(m_sampleTimes[size_t(frame)]);

  pxr::VtVec3fArray points;
  if (pointBased.GetPointsAttr().Get(&points, time) && !points.empty()) {
    if (auto *positions =
            geometry->parameterValueAsObject<scene::Array>("vertex.position")) {
      if (positions->size() == points.size())
        positions->setData(points.cdata());
    }
  }

  pxr::VtVec3fArray normals;
  if (pointBased.GetNormalsAttr().Get(&normals, time) && !normals.empty()) {
    if (auto *normalArray =
            geometry->parameterValueAsObject<scene::Array>("vertex.normal")) {
      if (normalArray->size() == normals.size())
        normalArray->setData(normals.cdata());
    }
  }

  m_currentFrame = frame;
}

#else

void UsdGeometryFileBinding::update(float)
{
  logError("[UsdGeometryFileBinding] USD not enabled in TSD build.");
}

#endif

} // namespace tsd::io
