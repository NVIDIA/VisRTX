// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/rendering/index/RenderIndexFlatRegistry.hpp"

namespace tsd::rendering {

RenderIndexFlatRegistry::RenderIndexFlatRegistry(
    Scene &scene, tsd::core::Token deviceName, anari::Device d)
    : RenderIndex(scene, deviceName, d)
{}

RenderIndexFlatRegistry::~RenderIndexFlatRegistry() = default;

bool RenderIndexFlatRegistry::isFlat() const
{
  return true;
}

void RenderIndexFlatRegistry::signalObjectAdded(const Object *obj)
{
  if (!obj)
    return;
  RenderIndex::signalObjectAdded(obj);
  updateWorld();
}

void RenderIndexFlatRegistry::signalObjectParameterUseCountZero(const Object *)
{
  // no-op
}

void RenderIndexFlatRegistry::signalObjectLayerUseCountZero(const Object *)
{
  // no-op
}

void RenderIndexFlatRegistry::updateWorld()
{
  auto d = device();
  auto w = world();

  auto syncAllHandles = [&](const auto &objArray) {
    foreach_item_const(
        objArray, [&](auto *obj) { m_cache.getHandle(obj, true); });
  };

  const auto &db = m_ctx->objectDB();
  syncAllHandles(db.surface);
  syncAllHandles(db.volume);
  syncAllHandles(db.light);

  setIndexedArrayObjectsAsAnariObjectArray(d, w, "surface", m_cache.surface);
  setIndexedArrayObjectsAsAnariObjectArray(d, w, "volume", m_cache.volume);
  setIndexedArrayObjectsAsAnariObjectArray(d, w, "light", m_cache.light);

  anari::commitParameters(d, w);
}

} // namespace tsd::rendering
