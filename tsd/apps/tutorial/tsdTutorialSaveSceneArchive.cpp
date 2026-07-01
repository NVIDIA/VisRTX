// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd_core
#include <tsd/scene/Scene.hpp>
// tsd_io
#include <tsd/io/archives/SceneArchive.hpp>
#include <tsd/io/procedural.hpp>

int main()
{
  tsd::scene::Scene scene;
  tsd::io::generate_material_orb(scene, scene.defaultLayer()->root());
  return tsd::io::save_SceneArchive(scene, "scene.tsd") ? 0 : 1;
}
