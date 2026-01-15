// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd_core
#include <tsd/core/Logging.hpp>
#include <tsd/core/scene/Scene.hpp>
// tsd_io
#include <tsd/io/procedural.hpp>
#include <tsd/io/serialization.hpp>

int main(int argc, char **argv)
{
  tsd::core::setLogToStdout();
  tsd::core::Scene scene;
  if (argc > 1)
    tsd::io::load_Scene(scene, argv[1]);
  else
    tsd::io::generate_material_orb(scene, scene.defaultLayer()->root());
  tsd::io::export_SceneToUSD(scene, "scene.usda", 30);
  return 0;
}
