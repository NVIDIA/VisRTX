// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/animation/AnimationManager.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
#include "tsd/scene/Scene.hpp"
// std
#include <filesystem>
#include <fstream>

SCENARIO(
    "Volume transfer functions reject missing control points", "[Importers]")
{
  tsd::scene::Scene scene;
  auto volume = scene.createObject<tsd::scene::Volume>(
      tsd::scene::tokens::volume::transferFunction1D);
  tsd::core::TransferFunction transferFunction;

  WHEN("An empty transfer function is applied")
  {
    tsd::io::applyTransferFunction(scene, volume, transferFunction);

    THEN("The volume keeps its default scalar color")
    {
      REQUIRE(volume->parameterValueAsObject<tsd::scene::Array>("color")
          == nullptr);
    }
  }
}

SCENARIO(
    "Single volume file import uses a default transfer function", "[Importers]")
{
  const auto path =
      std::filesystem::temp_directory_path() / "tsd_test_1x1x1_uint8.raw";
  {
    std::ofstream file(path, std::ios::binary);
    const unsigned char voxel = 255;
    file.write(reinterpret_cast<const char *>(&voxel), sizeof(voxel));
  }

  tsd::scene::Scene scene;
  tsd::animation::AnimationManager animMgr(&scene);

  WHEN("A volume is imported through the single-file dispatcher")
  {
    tsd::io::import_file(
        scene, animMgr, {tsd::io::ImporterType::VOLUME, path.string()});

    THEN("The imported volume has a sampled color array")
    {
      REQUIRE(scene.numberOfObjects(ANARI_VOLUME) == 1);
      auto volume = scene.getObject<tsd::scene::Volume>(0);
      REQUIRE(volume);
      auto *color = volume->parameterValueAsObject<tsd::scene::Array>("color");
      REQUIRE(color != nullptr);
      REQUIRE(color->size() == 256);
    }
  }

  std::filesystem::remove(path);
}
