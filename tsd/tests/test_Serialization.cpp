// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/core/DataTreeMetadata.hpp"
#include "tsd/core/DataTree.hpp"
#include "tsd/io/serialization.hpp"
#include "tsd/scene/Scene.hpp"

SCENARIO("tsd::io camera and renderer subset serialization", "[Serialization]")
{
  GIVEN("A scene with cameras, renderers, and unrelated scene data")
  {
    tsd::scene::Scene source;

    auto defaultCamera = source.defaultCamera();
    defaultCamera->setName("shot_0_camera");
    defaultCamera->setParameter("fovy", 0.75f);
    defaultCamera->setMetadataValue("exposure", 1.5f);

    auto secondCamera = source.createObject<tsd::scene::Camera>("orthographic");
    secondCamera->setName("shot_1_camera");
    secondCamera->setParameter("height", 12.f);

    auto renderer = source.createRenderer("test_device", "pathtracer");
    renderer->setName("shot_renderer");
    renderer->setParameter("pixelSamples", 8);
    renderer->setMetadataValue("quality", 3);

    source.createObject<tsd::scene::Geometry>("sphere");
    source.addLayer("preserved_source_layer");

    tsd::core::DataTree tree;
    auto &root = tree.root();
    root["layers"]["stale"] = "remove me";
    root["animations"]["stale"] = "remove me";

    WHEN("only cameras and renderers are saved")
    {
      tsd::io::save_SceneCamerasAndRenderers(source, root);

      THEN("the output is tagged as a camera and renderer subset")
      {
        auto metadata = tsd::core::readDataTreeMetadata(root);
        REQUIRE(metadata.status == tsd::core::DataTreeMetadataReadStatus::Found);
        REQUIRE(metadata.metadata);
        REQUIRE(metadata.metadata->schema
            == std::string(tsd::io::schema::SCENE_CAMERAS_AND_RENDERERS));
      }

      THEN("the output contains only the camera and renderer object pools")
      {
        REQUIRE(root.child("layers") == nullptr);
        REQUIRE(root.child("animations") == nullptr);

        auto *objectDB = root.child("objectDB");
        REQUIRE(objectDB != nullptr);
        REQUIRE(objectDB->child("camera") != nullptr);
        REQUIRE(objectDB->child("renderer") != nullptr);
        REQUIRE(objectDB->child("geometry") == nullptr);
        REQUIRE(objectDB->child("material") == nullptr);
      }

      AND_WHEN("the subset is loaded into another populated scene")
      {
        tsd::scene::Scene target;
        target.defaultCamera()->setName("old_default_camera");
        auto oldCamera = target.createObject<tsd::scene::Camera>("perspective");
        oldCamera->setName("old_extra_camera");
        auto oldRenderer = target.createRenderer("old_device", "old_renderer");
        oldRenderer->setName("old_renderer");
        target.createObject<tsd::scene::Geometry>("cylinder");
        target.addLayer("keep_me");

        tsd::io::load_SceneCamerasAndRenderers(target, root);

        THEN("only cameras and renderers are replaced")
        {
          REQUIRE(target.numberOfObjects(ANARI_GEOMETRY) == 1);
          REQUIRE(target.numberOfLayers() == 1);
          REQUIRE(target.layer("keep_me") != nullptr);

          REQUIRE(target.numberOfObjects(ANARI_CAMERA) == 2);
          REQUIRE(target.numberOfObjects(ANARI_RENDERER) == 1);
          REQUIRE(target.getObject<tsd::scene::Camera>(0)->name()
              == "shot_0_camera");
          REQUIRE(target.getObject<tsd::scene::Camera>(1)->name()
              == "shot_1_camera");
          REQUIRE(target.getObject<tsd::scene::Renderer>(0)->name()
              == "shot_renderer");
        }

        THEN("camera and renderer object data round-trips")
        {
          auto camera = target.getObject<tsd::scene::Camera>(0);
          REQUIRE(camera);
          REQUIRE(camera->subtype().str() == "perspective");
          REQUIRE(camera->parameter("fovy")->value().getAs<float>() == 0.75f);
          REQUIRE(camera->getMetadataValue("exposure").getAs<float>() == 1.5f);

          auto second = target.getObject<tsd::scene::Camera>(1);
          REQUIRE(second);
          REQUIRE(second->subtype().str() == "orthographic");
          REQUIRE(second->parameter("height")->value().getAs<float>() == 12.f);

          auto restoredRenderer = target.getObject<tsd::scene::Renderer>(0);
          REQUIRE(restoredRenderer);
          REQUIRE(restoredRenderer->subtype().str() == "pathtracer");
          REQUIRE(restoredRenderer->rendererDeviceName().str() == "test_device");
          REQUIRE(restoredRenderer->parameter("pixelSamples")
                      ->value()
                      .getAs<int>()
              == 8);
          REQUIRE(restoredRenderer->getMetadataValue("quality").getAs<int>()
              == 3);
        }
      }
    }
  }

  GIVEN("An empty camera subset")
  {
    tsd::scene::Scene scene;
    tsd::core::DataTree tree;
    tree.root()["objectDB"];

    WHEN("the subset is loaded")
    {
      tsd::io::load_SceneCamerasAndRenderers(scene, tree.root());

      THEN("the scene still has a default camera")
      {
        REQUIRE(scene.defaultCamera());
        REQUIRE(scene.numberOfObjects(ANARI_CAMERA) == 1);
      }
    }
  }
}

SCENARIO("tsd::io scene payload metadata validation", "[Serialization]")
{
  GIVEN("A serializable scene")
  {
    tsd::scene::Scene source;
    source.defaultCamera()->setName("source_camera");
    auto renderer = source.createRenderer("test_device", "pathtracer");
    renderer->setName("source_renderer");

    WHEN("a full scene is serialized")
    {
      tsd::core::DataTree tree;
      tsd::io::save_Scene(source, tree.root(), false);

      THEN("the output is tagged as a full scene")
      {
        auto metadata = tsd::core::readDataTreeMetadata(tree.root());
        REQUIRE(metadata.status == tsd::core::DataTreeMetadataReadStatus::Found);
        REQUIRE(metadata.metadata);
        REQUIRE(metadata.metadata->schema
            == std::string(tsd::io::schema::SCENE_FULL));
      }

      THEN("the camera and renderer subset loader accepts the full scene")
      {
        auto result =
            tsd::io::validate_SceneCamerasAndRenderersPayload(tree.root());
        REQUIRE(result.accepted());
        REQUIRE(result.status == tsd::io::PayloadValidationStatus::Valid);
      }
    }

    WHEN("a camera and renderer subset is loaded as a full scene")
    {
      tsd::core::DataTree subsetTree;
      tsd::io::save_SceneCamerasAndRenderers(source, subsetTree.root());

      tsd::scene::Scene target;
      target.createObject<tsd::scene::Geometry>("sphere");
      target.addLayer("keep_me");

      THEN("validation rejects it before mutation")
      {
        auto result = tsd::io::validate_ScenePayload(subsetTree.root());
        REQUIRE(!result.accepted());
        REQUIRE(
            result.status == tsd::io::PayloadValidationStatus::IncompatibleSchema);

        tsd::io::load_Scene(target, subsetTree.root());
        REQUIRE(target.numberOfObjects(ANARI_GEOMETRY) == 1);
        REQUIRE(target.numberOfLayers() == 1);
        REQUIRE(target.layer("keep_me") != nullptr);
      }
    }

    WHEN("legacy metadata is missing but objectDB exists")
    {
      tsd::core::DataTree legacyTree;
      legacyTree.root()["objectDB"];

      THEN("validation accepts it as legacy")
      {
        auto result = tsd::io::validate_ScenePayload(legacyTree.root());
        REQUIRE(result.accepted());
        REQUIRE(result.status
            == tsd::io::PayloadValidationStatus::MissingMetadataAccepted);
      }
    }

    WHEN("the payload is missing objectDB")
    {
      tsd::core::DataTree invalidTree;
      tsd::core::writeDataTreeMetadata(
          invalidTree.root(), {1, "scene", "tsd.scene.full", 1});

      tsd::scene::Scene target;
      target.createObject<tsd::scene::Geometry>("sphere");
      target.addLayer("keep_me");

      THEN("validation rejects it before mutation")
      {
        auto result = tsd::io::validate_ScenePayload(invalidTree.root());
        REQUIRE(!result.accepted());
        REQUIRE(
            result.status == tsd::io::PayloadValidationStatus::MissingRequiredNode);

        tsd::io::load_Scene(target, invalidTree.root());
        REQUIRE(target.numberOfObjects(ANARI_GEOMETRY) == 1);
        REQUIRE(target.numberOfLayers() == 1);
        REQUIRE(target.layer("keep_me") != nullptr);
      }
    }
  }
}
