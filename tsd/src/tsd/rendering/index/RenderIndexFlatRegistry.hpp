// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/rendering/index/RenderIndex.hpp"

namespace tsd::rendering {

/*
 * RenderIndex that registers every Scene object directly in the ANARI world
 * without any layer hierarchy; intended for flat/unstructured object
 * collections.
 *
 * Example:
 *   auto *idx = scene.updateDelegate().emplace<RenderIndexFlatRegistry>(
 *       scene, deviceToken, anariDevice);
 *   idx->populate();
 */
struct RenderIndexFlatRegistry : public RenderIndex
{
  RenderIndexFlatRegistry(
      Scene &scene, tsd::core::Token deviceName, anari::Device d);
  ~RenderIndexFlatRegistry() override;

  bool isFlat() const override;
  void signalObjectAdded(const Object *o) override;
  void signalObjectParameterUseCountZero(const Object *obj) override;
  void signalObjectLayerUseCountZero(const Object *obj) override;

 private:
  void updateWorld() override;
};

} // namespace tsd::rendering
