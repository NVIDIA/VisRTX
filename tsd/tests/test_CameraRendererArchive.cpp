// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/core/DataTree.hpp"
#include "tsd/core/DataTreeMetadata.hpp"
#include "tsd/io/archives/CameraArchive.hpp"
#include "tsd/io/archives/RendererArchive.hpp"
#include "tsd/io/serialization/serialization_internal.hpp"
#include "tsd/scene/Scene.hpp"
// std
#include <cstdio>
#include <filesystem>

SCENARIO("Camera and Renderer Archives replace only their complete pools",
    "[CameraArchive][RendererArchive]")
{
  tsd::scene::Scene source;
  source.defaultCamera()->setName("camera zero");
  source.createObject<tsd::scene::Camera>("orthographic")
      ->setName("camera one");
  source.createRenderer("device", "pathtracer")->setName("renderer zero");

  tsd::core::DataTree cameraTree;
  tsd::core::DataTree rendererTree;
  REQUIRE(tsd::io::serialize_CameraArchive(source, cameraTree.root()));
  REQUIRE(tsd::io::serialize_RendererArchive(source, rendererTree.root()));

  auto cameraMetadata = tsd::core::readDataTreeMetadata(cameraTree.root());
  auto rendererMetadata = tsd::core::readDataTreeMetadata(rendererTree.root());
  REQUIRE(cameraMetadata.metadata);
  REQUIRE(rendererMetadata.metadata);
  REQUIRE(cameraMetadata.metadata->schema == "tsd.scene.cameras");
  REQUIRE(rendererMetadata.metadata->schema == "tsd.scene.renderers");
  REQUIRE(cameraTree.root()["objectDB"].child("renderer") == nullptr);
  REQUIRE(rendererTree.root()["objectDB"].child("camera") == nullptr);

  tsd::scene::Scene target;
  target.defaultCamera()->setName("stale camera");
  target.createRenderer("stale device", "stale renderer");
  target.createObject<tsd::scene::Geometry>("sphere");
  target.addLayer("preserved layer");

  REQUIRE(tsd::io::deserialize_CameraArchive(target, cameraTree.root()));
  REQUIRE(target.numberOfObjects(ANARI_CAMERA) == 2);
  REQUIRE(target.getObject<tsd::scene::Camera>(0)->name() == "camera zero");
  REQUIRE(target.getObject<tsd::scene::Camera>(1)->name() == "camera one");
  REQUIRE(target.numberOfObjects(ANARI_RENDERER) == 1);

  REQUIRE(tsd::io::deserialize_RendererArchive(target, rendererTree.root()));
  REQUIRE(target.numberOfObjects(ANARI_RENDERER) == 1);
  REQUIRE(target.getObject<tsd::scene::Renderer>(0)->name() == "renderer zero");
  REQUIRE(target.numberOfObjects(ANARI_GEOMETRY) == 1);
  REQUIRE(target.layer("preserved layer") != nullptr);

  tsd::core::DataTree legacyTree;
  tsd::io::save_SceneCamerasAndRenderers(source, legacyTree.root());
  REQUIRE(tsd::io::deserialize_CameraArchive(target, legacyTree.root()));
  REQUIRE(tsd::io::deserialize_RendererArchive(target, legacyTree.root()));

  const auto cameraFile =
      (std::filesystem::temp_directory_path() / "tsd_camera_archive.tsd")
          .string();
  const auto rendererFile =
      (std::filesystem::temp_directory_path() / "tsd_renderer_archive.tsd")
          .string();
  REQUIRE(tsd::io::save_CameraArchive(source, cameraFile.c_str()));
  REQUIRE(tsd::io::save_RendererArchive(source, rendererFile.c_str()));
  tsd::scene::Scene fileTarget;
  REQUIRE(tsd::io::load_CameraArchive(fileTarget, cameraFile.c_str()));
  REQUIRE(tsd::io::load_RendererArchive(fileTarget, rendererFile.c_str()));
  std::remove(cameraFile.c_str());
  std::remove(rendererFile.c_str());

  tsd::core::DataTree invalidCameraTree;
  invalidCameraTree.root() = cameraTree.root();
  (*invalidCameraTree.root()["objectDB"]["camera"].child(0))["self"] =
      tsd::core::Any(ANARI_CAMERA, size_t(8));
  const auto camerasBefore = target.numberOfObjects(ANARI_CAMERA);
  REQUIRE_FALSE(
      tsd::io::deserialize_CameraArchive(target, invalidCameraTree.root()));
  REQUIRE(target.numberOfObjects(ANARI_CAMERA) == camerasBefore);

  tsd::core::DataTree invalidRendererTree;
  invalidRendererTree.root() = rendererTree.root();
  invalidRendererTree.root()["objectDB"]["renderer"].child(0)->remove(
      "rendererDeviceName");
  const auto renderersBefore = target.numberOfObjects(ANARI_RENDERER);
  REQUIRE_FALSE(
      tsd::io::deserialize_RendererArchive(target, invalidRendererTree.root()));
  REQUIRE(target.numberOfObjects(ANARI_RENDERER) == renderersBefore);
}
