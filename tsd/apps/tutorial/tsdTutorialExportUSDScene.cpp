// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd_core
#include <tsd/core/Logging.hpp>
#include <tsd/scene/Scene.hpp>
// tsd_io
#include <tsd/io/archives/SceneArchive.hpp>
#include <tsd/io/exporters.hpp>
#include <tsd/io/procedural.hpp>

int main(int argc, char **argv)
{
  tsd::core::setLogToStdout();
  tsd::scene::Scene scene;
  if (argc > 1) {
    if (!tsd::io::load_SceneArchive(scene, argv[1]))
      return 1;
  } else {
    tsd::io::generate_material_orb(scene, scene.defaultLayer()->root());
  }
  tsd::io::export_SceneToUSD(scene, "scene.usda", 30);
  return 0;
}
