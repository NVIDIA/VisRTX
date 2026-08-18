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
#include "tsd/io/usd/UsdResolvedGeometry.h"
#include "tsd/io/usd/UsdStageSession.h"
// usd
#include <pxr/usd/usdGeom/pointBased.h>
#endif

namespace tsd::io {

using namespace tsd::core;

namespace {

// The transform is written out flat: sixteen floats in the order the matrix
// stores them, which is what reads it back.
void writeMat4(core::DataNode &node, const tsd::math::mat4 &m)
{
  for (int column = 0; column < 4; ++column) {
    for (int row = 0; row < 4; ++row)
      node.append() = m[column][row];
  }
}

tsd::math::mat4 readMat4(core::DataNode *node)
{
  auto retval = tsd::math::IDENTITY_MAT4;
  if (!node)
    return retval;

  std::vector<float> values;
  node->foreach_child(
      [&](core::DataNode &n) { values.push_back(n.getValueOr<float>(0.f)); });
  if (values.size() != 16)
    return retval;

  for (int column = 0; column < 4; ++column) {
    for (int row = 0; row < 4; ++row)
      retval[column][row] = values[size_t(column * 4 + row)];
  }
  return retval;
}

} // namespace

UsdGeometryFileBinding::UsdGeometryFileBinding(scene::Scene *scene,
    std::shared_ptr<usd::UsdStageSession> session,
    std::string stageFile,
    std::string primPath,
    std::vector<Part> parts,
    usd::GeometryResolveOptions resolveOptions)
    : UsdFileBinding(
          scene, std::move(session), std::move(stageFile), std::move(primPath)),
      m_parts(std::move(parts)),
      m_resolveOptions(std::move(resolveOptions))
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
  writePathsToDataNode(node);

  // `targetIndex` names the first Part's geometry, which is all an Archive
  // written before the converter split carried and all such an Archive is read
  // back as.
  auto *first = m_parts.empty() ? nullptr : m_parts.front().geometry.get();
  node["targetIndex"] = first ? first->index() : tsd::core::INVALID_INDEX;

  auto &partsNode = node["parts"];
  for (const auto &part : m_parts) {
    auto *geometry = part.geometry.get();
    if (!geometry)
      continue;
    auto &partNode = partsNode.append();
    partNode["name"] = part.name;
    partNode["targetIndex"] = geometry->index();
  }

  // The half of the conversion that does not change over time, so a scrub
  // reproduces it rather than resolving materials again.
  auto &replay = node["resolve"];
  replay["refine"] = m_resolveOptions.refine;
  replay["refinementLevel"] = m_resolveOptions.refinementLevel;
  writeMat4(replay["bakeXform"], m_resolveOptions.bakeXform);
  auto &uvNode = replay["uvNames"];
  for (const auto &[part, uvName] : m_resolveOptions.uvNamesByPart) {
    auto &entry = uvNode.append();
    entry["part"] = part;
    entry["uv"] = uvName;
  }

  auto &slotNode = replay["slots"];
  for (const auto &[part, primvars] : m_resolveOptions.slotPrimvarsByPart) {
    auto &entry = slotNode.append();
    entry["part"] = part;
    auto &names = entry["primvars"];
    for (const auto &primvar : primvars)
      names.append() = primvar;
  }
}

void UsdGeometryFileBinding::onDefragment(const scene::IndexRemapper &cb)
{
  for (auto &part : m_parts) {
    if (!part.geometry)
      continue;
    const size_t newIndex =
        cb(part.geometry->type(), part.geometry->index());
    part.geometry.updateDefragmentedIndex(newIndex);
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
  const auto primPath = node["primPath"].getValueOr<std::string>("");

  auto geometryAt = [&](size_t index) -> scene::Geometry * {
    return static_cast<scene::Geometry *>(
        scene.getObject(ANARI_GEOMETRY, index));
  };

  std::vector<Part> parts;
  if (auto *partsNode = node.child("parts")) {
    partsNode->foreach_child([&](core::DataNode &partNode) {
      const auto index =
          partNode["targetIndex"].getValueOr<size_t>(tsd::core::INVALID_INDEX);
      if (auto *geometry = geometryAt(index)) {
        parts.push_back(
            {partNode["name"].getValueOr<std::string>(primPath), geometry});
      }
    });
  } else {
    // An Archive written before the converter split names one geometry and
    // nothing else. That is exactly a single Part covering the whole prim.
    const auto index =
        node["targetIndex"].getValueOr<size_t>(tsd::core::INVALID_INDEX);
    if (auto *geometry = geometryAt(index))
      parts.push_back({primPath, geometry});
  }

  if (parts.empty()) {
    logWarning("[UsdGeometryFileBinding] no target geometry for '%s' survives"
               " in the scene; skipping",
        primPath.c_str());
    return nullptr;
  }

  usd::GeometryResolveOptions resolveOptions;
  if (auto *replay = node.child("resolve")) {
    resolveOptions.refine = (*replay)["refine"].getValueOr<bool>(false);
    resolveOptions.refinementLevel =
        (*replay)["refinementLevel"].getValueOr<int>(2);
    resolveOptions.bakeXform = readMat4(replay->child("bakeXform"));
    if (auto *uvNode = replay->child("uvNames")) {
      uvNode->foreach_child([&](core::DataNode &entry) {
        resolveOptions.uvNamesByPart.set(
            entry["part"].getValueOr<std::string>(""),
            entry["uv"].getValueOr<std::string>("st"));
      });
    }
    if (auto *slotNode = replay->child("slots")) {
      slotNode->foreach_child([&](core::DataNode &entry) {
        std::vector<std::string> primvars;
        if (auto *names = entry.child("primvars")) {
          names->foreach_child([&](core::DataNode &n) {
            primvars.push_back(n.getValueOr<std::string>(""));
          });
        }
        resolveOptions.slotPrimvarsByPart.set(
            entry["part"].getValueOr<std::string>(""), std::move(primvars));
      });
    }
  }

  return &anim.emplaceFileBinding<UsdGeometryFileBinding>(&scene,
      std::shared_ptr<usd::UsdStageSession>{},
      node["stageFile"].getValueOr<std::string>(""),
      primPath,
      std::move(parts),
      std::move(resolveOptions));
}

#if TSD_USE_USD

void UsdGeometryFileBinding::update(float t)
{
  if (!scene() || m_parts.empty() || !ensureSession())
    return;

  const pxr::SdfPath path(primPath());

  // A Stage that carries samples but authored no time-code range has no range
  // to map onto until its own prims say what they cover.
  if (!m_sampleTimesNoted && !session()->hasAuthoredTimeRange()) {
    m_sampleTimesNoted = true;
    pxr::UsdGeomPointBased pointBased(session()->stage()->GetPrimAtPath(path));
    if (pointBased) {
      std::vector<double> times;
      pointBased.GetPointsAttr().GetTimeSamples(&times);
      noteAuthoredSampleTimes(times);
    }
  }

  session()->setTime(session()->timeCodeAt(t));

  auto prim = session()->sceneIndex()->GetPrim(path);
  const auto resolved =
      usd::resolveGeometry(session()->sceneIndex(), path, prim, m_resolveOptions);
  if (!resolved.valid())
    return;

  // Every Part is written in one pass, so points, indices and primvars always
  // describe the same frame. A Part that has gone missing is left as imported
  // rather than half-updated: Parts appear and disappear when the mesh's
  // material subsets change, and that means new Surfaces and Materials, which
  // is conversion rather than animation.
  // One cache across the whole set, so a mesh's Surfaces keep sharing one
  // position Array and that Array is written once rather than once per Surface.
  usd::RefillCache cache;

  size_t applied = 0;
  for (auto &part : m_parts) {
    auto *geometry = part.geometry.get();
    const auto *resolvedPart = resolved.part(part.name);
    if (!geometry || !resolvedPart)
      continue;
    if (usd::refillGeometry(*scene(), *geometry, *resolvedPart, cache))
      applied++;
  }

  if (applied < m_parts.size() && !m_partsChangedReported) {
    m_partsChangedReported = true;
    logWarning("[UsdGeometryFileBinding] '%s': %zu of %zu parts no longer"
               " resolve; their geometry is left as imported",
        primPath().c_str(),
        m_parts.size() - applied,
        m_parts.size());
  }
}

#else

void UsdGeometryFileBinding::update(float)
{
  ensureSession(); // reports that this build has no USD, once
}

#endif

} // namespace tsd::io
