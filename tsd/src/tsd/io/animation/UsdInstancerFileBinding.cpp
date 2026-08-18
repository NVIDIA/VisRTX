// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/animation/UsdInstancerFileBinding.hpp"
// tsd_core
#include "tsd/core/DataTree.hpp"
#include "tsd/core/Logging.hpp"
#if TSD_USE_USD
// tsd_io
#include "tsd/io/importers/detail/usd/UsdInstancing.h"
#include "tsd/io/usd/UsdStageSession.h"
#endif

namespace tsd::io {

using namespace tsd::core;

UsdInstancerFileBinding::UsdInstancerFileBinding(scene::Scene *scene,
    std::shared_ptr<usd::UsdStageSession> session,
    scene::LayerNodeRef arrayNode,
    scene::ArrayRef transforms,
    std::string stageFile,
    std::string primPath,
    size_t prototypeIndex)
    : UsdFileBinding(
          scene, std::move(session), std::move(stageFile), std::move(primPath)),
      m_arrayNode(arrayNode),
      m_transforms(transforms),
      m_prototypeIndex(prototypeIndex)
{}

UsdInstancerFileBinding::~UsdInstancerFileBinding() = default;

std::string UsdInstancerFileBinding::kind() const
{
  return "usdInstancer";
}

const char *UsdInstancerFileBinding::logTag() const
{
  return "UsdInstancerFileBinding";
}

void UsdInstancerFileBinding::toDataNode(core::DataNode &node) const
{
  if (m_arrayNode && scene()) {
    auto *layer = (*m_arrayNode).value().layer();
    node["layerName"] = scene()->getLayerName(layer).str();
    node["nodeIndex"] = m_arrayNode->index();
  }
  writePathsToDataNode(node);
  node["prototypeIndex"] = uint64_t(m_prototypeIndex);
}

std::vector<scene::LayerNodeRef> UsdInstancerFileBinding::layerTargets() const
{
  if (!m_arrayNode)
    return {};
  return {m_arrayNode};
}

void UsdInstancerFileBinding::onDefragment(const scene::IndexRemapper &cb)
{
  if (m_transforms) {
    const size_t newIndex = cb(m_transforms->type(), m_transforms->index());
    m_transforms.updateDefragmentedIndex(newIndex);
  }
}

void UsdInstancerFileBinding::addCallbackToAnimation(
    tsd::animation::Animation &anim)
{
  anim.addCallbackBinding([this](float t) { this->update(t); });
}

UsdInstancerFileBinding *UsdInstancerFileBinding::addToAnimation(
    tsd::animation::Animation &anim, scene::Scene &scene, core::DataNode &node)
{
  scene::LayerNodeRef arrayNode;
  if (auto *layerNameNode = node.child("layerName")) {
    const auto layerName =
        core::Token(layerNameNode->getValueOr<std::string>("").c_str());
    if (auto *layer = scene.layer(layerName))
      arrayNode = layer->at(node["nodeIndex"].getValueOr<size_t>(0));
  }

  if (!arrayNode) {
    logWarning(
        "[UsdInstancerFileBinding] transform-array node not found; skipping");
    return nullptr;
  }

  auto *transforms = (*arrayNode)->getTransformArray();
  if (!transforms) {
    logWarning("[UsdInstancerFileBinding] node '%s' is not a transform-array"
               " node; skipping",
        (*arrayNode)->name().c_str());
    return nullptr;
  }

  return &anim.emplaceFileBinding<UsdInstancerFileBinding>(&scene,
      std::shared_ptr<usd::UsdStageSession>{},
      arrayNode,
      transforms->self(),
      node["stageFile"].getValueOr<std::string>(""),
      node["primPath"].getValueOr<std::string>(""),
      size_t(node["prototypeIndex"].getValueOr<uint64_t>(0)));
}

#if TSD_USE_USD

void UsdInstancerFileBinding::update(float t)
{
  if (!scene() || !m_arrayNode || !ensureSession())
    return;

  const pxr::SdfPath path(primPath());

  // A Stage that carries samples but authored no time-code range has no range
  // to map onto until its own prims say what they cover.
  if (!m_sampleTimesNoted && !session()->hasAuthoredTimeRange()) {
    m_sampleTimesNoted = true;
    noteAuthoredSampleTimes(usd::pointInstancerSampleTimes(
        session()->stage()->GetPrimAtPath(path)));
  }

  session()->setTime(session()->timeCodeAt(t));

  auto prim = session()->sceneIndex()->GetPrim(path);
  const auto placements =
      usd::readInstancerPlacements(prim).forPrototype(m_prototypeIndex);
  if (placements.empty())
    return;

  auto *transforms = m_transforms.get();
  if (transforms && transforms->size() == placements.size()) {
    transforms->setData(placements.data(), placements.size());
    return;
  }

  // A TSD Array's size is fixed at construction, so a placement count that
  // moves mid-sequence costs one allocation and one rebind on that frame.
  auto replacement =
      scene()->createArray(ANARI_FLOAT32_MAT4, placements.size());
  replacement->setData(placements.data(), placements.size());
  if (transforms)
    replacement->setName(transforms->name().c_str());

  (*m_arrayNode)->setAsTransformArray(replacement.data());
  m_transforms = replacement;
  scene()->signalLayerStructureChanged((*m_arrayNode).value().layer());
}

#else

void UsdInstancerFileBinding::update(float)
{
  ensureSession(); // reports that this build has no USD, once
}

#endif

} // namespace tsd::io
