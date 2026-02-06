// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/procedural.hpp"

namespace tsd::io {

void generate_default_lights(Scene &scene)
{
  auto lightsRoot = scene.defaultLayer()->root()->insert_first_child({});
  (*lightsRoot)->name() = "defaultLights";

  auto light = scene.createObject<tsd::core::Light>(
      tsd::core::tokens::light::directional);
  light->setName("mainDistantLight");
  light->setParameter("direction", tsd::math::float2(0.f, 240.f));

  lightsRoot->insert_first_child({light});
}

} // namespace tsd::io
