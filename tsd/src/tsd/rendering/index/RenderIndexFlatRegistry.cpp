// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/rendering/index/RenderIndexFlatRegistry.hpp"

namespace tsd::rendering {

RenderIndexFlatRegistry::RenderIndexFlatRegistry(Context &ctx, anari::Device d)
    : RenderIndex(ctx, d)
{}

RenderIndexFlatRegistry::~RenderIndexFlatRegistry() = default;

void RenderIndexFlatRegistry::signalObjectAdded(const Object *obj)
{
  if (!obj)
    return;
  RenderIndex::signalObjectAdded(obj);
  updateWorld();
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
