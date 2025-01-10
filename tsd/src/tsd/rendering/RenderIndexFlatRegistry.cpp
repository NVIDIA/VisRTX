// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/rendering/RenderIndexFlatRegistry.hpp"

namespace tsd {

RenderIndexFlatRegistry::RenderIndexFlatRegistry(anari::Device d)
    : RenderIndex(d)
{}

RenderIndexFlatRegistry::~RenderIndexFlatRegistry() = default;

void RenderIndexFlatRegistry::updateWorld()
{
  auto d = device();
  auto w = world();

  setIndexedArrayObjectsAsAnariObjectArray(d, w, "surface", m_cache.surface);
  setIndexedArrayObjectsAsAnariObjectArray(d, w, "volume", m_cache.volume);
  setIndexedArrayObjectsAsAnariObjectArray(d, w, "light", m_cache.light);

  anari::commitParameters(d, w);
}

} // namespace tsd
