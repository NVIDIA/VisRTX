// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd_core
#include <tsd/scene/Scene.hpp>
// tsd_io
#include <tsd/io/procedural.hpp>

int main()
{
  tsd::scene::Scene scene;
  tsd::io::generate_randomSpheres(scene);
  auto geom = scene.getObject<tsd::scene::Geometry>(0);
  geom->setName("main geom");
  tsd::scene::print(*geom);
  return 0;
}
