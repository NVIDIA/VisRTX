// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/app/Context.h"
#include "tsd/io/archives/SceneArchive.hpp"
// std
#include <filesystem>
#include <string>
#include <vector>

SCENARIO("The TSD CLI flag selects a Scene Archive", "[App]")
{
  tsd::app::Context context;
  std::vector<std::string> args{
      "tsdViewer", "-tsd", "scene.tsd", "-obj", "mesh.obj"};

  context.parseCommandLine(args);

  REQUIRE(context.commandLine.sceneArchiveFile == "scene.tsd");
  REQUIRE(context.commandLine.filenames.size() == 1);
  REQUIRE(context.commandLine.filenames.front().first
      == tsd::io::ImporterType::OBJ);
  REQUIRE(context.commandLine.filenames.front().second == "mesh.obj;default");
}

SCENARIO("Scene Archive CLI inputs are loaded by TSD App", "[App]")
{
  const auto archive =
      std::filesystem::temp_directory_path() / "tsd_app_scene_archive.tsd";
  std::filesystem::remove(archive);

  tsd::scene::Scene source;
  source.addLayer("archived");
  REQUIRE(tsd::io::save_SceneArchive(source, archive.string().c_str()));

  tsd::app::Context context;
  std::vector<std::string> args{"tsdViewer", "-tsd", archive.string()};
  context.parseCommandLine(args);
  context.setupSceneFromCommandLine();

  REQUIRE(context.tsd.scene.layer("archived") != nullptr);
  std::filesystem::remove(archive);
}
