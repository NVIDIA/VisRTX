// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd_core
#include <tsd/core/scene/Scene.hpp>
// tsd_io
#include <tsd/io/procedural.hpp>

int main()
{
  tsd::core::Scene scene;
  tsd::io::generate_randomSpheres(scene);
  auto geom = scene.getObject<tsd::core::Geometry>(0);
  geom->setName("main geom");
  tsd::core::print(*geom);
  return 0;
}
