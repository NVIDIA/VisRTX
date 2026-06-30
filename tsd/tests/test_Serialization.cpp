// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/animation/Animation.hpp"
#include "tsd/animation/AnimationManager.hpp"
#include "tsd/core/DataTree.hpp"
#include "tsd/core/DataTreeMetadata.hpp"
#include "tsd/io/serialization.hpp"
#include "tsd/scene/Scene.hpp"
// std
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

std::string testFile(const char *name)
{
  return (std::filesystem::temp_directory_path() / name).string();
}

void removeTestFile(const std::string &filename)
{
  std::remove(filename.c_str());
}

tsd::scene::ArrayRef makeFloatArray(tsd::scene::Scene &scene,
    const char *name,
    const std::vector<float> &values)
{
  auto array = scene.createArray(ANARI_FLOAT32, values.size());
  array->setName(name);
  array->setData(values);
  return array;
}

} // namespace

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
        REQUIRE(
            metadata.status == tsd::core::DataTreeMetadataReadStatus::Found);
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
          REQUIRE(
              restoredRenderer->rendererDeviceName().str() == "test_device");
          REQUIRE(
              restoredRenderer->parameter("pixelSamples")->value().getAs<int>()
              == 8);
          REQUIRE(
              restoredRenderer->getMetadataValue("quality").getAs<int>() == 3);
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
        REQUIRE(
            metadata.status == tsd::core::DataTreeMetadataReadStatus::Found);
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
        REQUIRE(result.status
            == tsd::io::PayloadValidationStatus::IncompatibleSchema);

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
        REQUIRE(result.status
            == tsd::io::PayloadValidationStatus::MissingRequiredNode);

        tsd::io::load_Scene(target, invalidTree.root());
        REQUIRE(target.numberOfObjects(ANARI_GEOMETRY) == 1);
        REQUIRE(target.numberOfLayers() == 1);
        REQUIRE(target.layer("keep_me") != nullptr);
      }
    }
  }
}

SCENARIO("tsd::io surface object serialization", "[Serialization]")
{
  GIVEN("A surface with geometry, material, sampler, array data, and metadata")
  {
    tsd::scene::Scene source;
    source.createSurface("unused_surface");

    auto positions =
        makeFloatArray(source, "positions", {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    positions->setMetadataValue("stride", 12);

    auto texture = makeFloatArray(source, "texture", {0.25f, 0.5f, 0.75f});

    auto sampler = source.createObject<tsd::scene::Sampler>(
        tsd::scene::tokens::sampler::image1D);
    sampler->setName("albedo_sampler");
    sampler->setParameterObject("image", *texture);

    auto geometry = source.createObject<tsd::scene::Geometry>(
        tsd::scene::tokens::geometry::triangle);
    geometry->setName("mesh_geometry");
    auto *positionParam =
        geometry->setParameterObject("vertex.position", *positions);
    positionParam->setDescription("positions").setEnabled(false);
    geometry->setMetadataValue("positionBuffer",
        tsd::core::Any(positions->type(), positions->index()));

    auto material = source.createObject<tsd::scene::Material>(
        tsd::scene::tokens::material::matte);
    material->setName("sampled_material");
    material->removeAllParameters();
    material->setParameterObject("color", *sampler);
    material->setParameter("roughness", 0.35f);
    material->setMetadataValue(
        "samplerRef", tsd::core::Any(sampler->type(), sampler->index()));

    auto surface = source.createSurface("root_surface", geometry, material);
    surface->setMetadataValue("priority", 9);
    surface->setMetadataValue(
        "geometryRef", tsd::core::Any(geometry->type(), geometry->index()));

    const auto filename = testFile("tsd_surface_object_roundtrip.tsd");
    removeTestFile(filename);

    WHEN("the surface is exported and imported into a non-empty scene")
    {
      REQUIRE(tsd::io::export_Object(filename.c_str(), *surface));

      tsd::core::DataTree exportedTree;
      REQUIRE(exportedTree.load(filename.c_str()));

      THEN("the payload is tagged as a surface object with local root index 0")
      {
        auto metadata = tsd::core::readDataTreeMetadata(exportedTree.root());
        REQUIRE(
            metadata.status == tsd::core::DataTreeMetadataReadStatus::Found);
        REQUIRE(metadata.metadata);
        REQUIRE(metadata.metadata->fileType == "object");
        REQUIRE(metadata.metadata->schema
            == std::string(tsd::io::schema::OBJECT_SURFACE));

        auto *rootObject = exportedTree.root().child("rootObject");
        REQUIRE(rootObject);
        REQUIRE(rootObject->getValue().type() == ANARI_SURFACE);
        REQUIRE(rootObject->getValue().getAsObjectIndex() == 0);

        auto *surfaceNode =
            exportedTree.root().child("objectDB")->child("surface")->child(0);
        REQUIRE(surfaceNode);
        REQUIRE(surfaceNode->child("self")->getValue().type() == ANARI_SURFACE);
        REQUIRE(surfaceNode->child("self")->getValue().getAsObjectIndex() == 0);
      }

      tsd::scene::Scene target;
      auto existingGeometry = target.createObject<tsd::scene::Geometry>(
          tsd::scene::tokens::geometry::sphere);
      existingGeometry->setName("preexisting_geometry");
      target.addLayer("keep_me");

      auto imported = tsd::io::import_Surface(target, filename.c_str());

      THEN("the import appends objects without creating layers")
      {
        REQUIRE(imported);
        REQUIRE(target.numberOfLayers() == 1);
        REQUIRE(target.layer("keep_me") != nullptr);
        REQUIRE(target.numberOfObjects(ANARI_SURFACE) == 1);
        REQUIRE(target.numberOfObjects(ANARI_GEOMETRY) == 2);
        REQUIRE(target.numberOfObjects(ANARI_MATERIAL) == 2);
        REQUIRE(target.numberOfObjects(ANARI_SAMPLER) == 1);
        REQUIRE(target.numberOfObjects(ANARI_ARRAY) == 2);
      }

      THEN("surface dependencies, metadata, data, and sharing round-trip")
      {
        REQUIRE(imported->name() == "root_surface");
        REQUIRE(imported->getMetadataValue("priority").getAs<int>() == 9);

        auto *importedGeometry = imported->geometry();
        auto *importedMaterial = imported->material();
        REQUIRE(importedGeometry);
        REQUIRE(importedMaterial);
        REQUIRE(importedGeometry->name() == "mesh_geometry");
        REQUIRE(importedMaterial->name() == "sampled_material");

        auto geometryMetadata = imported->getMetadataValue("geometryRef");
        REQUIRE(geometryMetadata.holdsObject());
        REQUIRE(
            geometryMetadata.getAsObjectIndex() == importedGeometry->index());

        auto *positionParam = importedGeometry->parameter("vertex.position");
        REQUIRE(positionParam);
        REQUIRE(!positionParam->isEnabled());
        REQUIRE(positionParam->description() == "positions");

        auto *importedPositions =
            importedGeometry->parameterValueAsObject<tsd::scene::Array>(
                "vertex.position");
        REQUIRE(importedPositions);
        REQUIRE(importedPositions->name() == "positions");
        REQUIRE(
            importedPositions->getMetadataValue("stride").getAs<int>() == 12);
        REQUIRE(importedPositions->size() == 6);
        const auto *positionData = importedPositions->dataAs<float>();
        REQUIRE(positionData[0] == 1.f);
        REQUIRE(positionData[5] == 6.f);

        auto *importedSampler =
            importedMaterial->parameterValueAsObject<tsd::scene::Sampler>(
                "color");
        REQUIRE(importedSampler);
        auto *importedTexture =
            importedSampler->parameterValueAsObject<tsd::scene::Array>("image");
        REQUIRE(importedTexture);
        REQUIRE(importedTexture->name() == "texture");
        REQUIRE(importedTexture->dataAs<float>()[2] == 0.75f);

        auto samplerMetadata = importedMaterial->getMetadataValue("samplerRef");
        REQUIRE(samplerMetadata.holdsObject());
        REQUIRE(samplerMetadata.getAsObjectIndex() == importedSampler->index());
      }
    }

    removeTestFile(filename);
  }
}

SCENARIO("tsd::io volume object serialization", "[Serialization]")
{
  GIVEN("A volume with a spatial field, transfer function arrays, and metadata")
  {
    tsd::scene::Scene source;

    auto fieldData = makeFloatArray(source, "field_data", {0.f, 1.f, 2.f, 3.f});
    auto colors = makeFloatArray(source, "tf_colors", {1.f, 0.f, 0.f, 1.f});
    auto opacity = makeFloatArray(source, "tf_opacity", {0.f, 1.f});

    auto field = source.createObject<tsd::scene::SpatialField>(
        tsd::scene::tokens::spatial_field::structuredRegular);
    field->setName("density_field");
    field->setParameterObject("data", *fieldData);
    field->setMetadataValue(
        "sourceData", tsd::core::Any(fieldData->type(), fieldData->index()));

    auto sampler = source.createObject<tsd::scene::Sampler>(
        tsd::scene::tokens::sampler::image1D);
    sampler->setName("tf_sampler");
    sampler->setParameterObject("image", *colors);

    auto volume = source.createObject<tsd::scene::Volume>(
        tsd::scene::tokens::volume::transferFunction1D);
    volume->setName("root_volume");
    volume->removeAllParameters();
    volume->setParameterObject("value", *field);
    volume->setParameterObject("color", *sampler);
    volume->setParameterObject("opacity", *opacity);
    volume->setMetadataValue(
        "fieldRef", tsd::core::Any(field->type(), field->index()));

    const auto filename = testFile("tsd_volume_object_roundtrip.tsd");
    removeTestFile(filename);

    WHEN("the volume is exported and imported into a non-empty scene")
    {
      REQUIRE(tsd::io::export_Object(filename.c_str(), *volume));

      tsd::core::DataTree exportedTree;
      REQUIRE(exportedTree.load(filename.c_str()));
      REQUIRE(tsd::io::validate_VolumePayload(exportedTree.root()).accepted());

      tsd::scene::Scene target;
      target.createObject<tsd::scene::SpatialField>(
          tsd::scene::tokens::spatial_field::structuredRegular);

      auto imported = tsd::io::import_Volume(target, filename.c_str());

      THEN("the imported volume preserves field, arrays, metadata, and refs")
      {
        REQUIRE(imported);
        REQUIRE(imported->name() == "root_volume");
        REQUIRE(target.numberOfObjects(ANARI_VOLUME) == 1);
        REQUIRE(target.numberOfObjects(ANARI_SPATIAL_FIELD) == 2);
        REQUIRE(target.numberOfObjects(ANARI_ARRAY) == 3);

        auto *importedField =
            imported->parameterValueAsObject<tsd::scene::SpatialField>("value");
        REQUIRE(importedField);
        REQUIRE(importedField->name() == "density_field");

        auto fieldRef = imported->getMetadataValue("fieldRef");
        REQUIRE(fieldRef.holdsObject());
        REQUIRE(fieldRef.getAsObjectIndex() == importedField->index());

        auto *importedData =
            importedField->parameterValueAsObject<tsd::scene::Array>("data");
        REQUIRE(importedData);
        REQUIRE(importedData->dataAs<float>()[3] == 3.f);

        auto sourceData = importedField->getMetadataValue("sourceData");
        REQUIRE(sourceData.holdsObject());
        REQUIRE(sourceData.getAsObjectIndex() == importedData->index());

        auto *importedSampler =
            imported->parameterValueAsObject<tsd::scene::Sampler>("color");
        REQUIRE(importedSampler);
        auto *importedColors =
            importedSampler->parameterValueAsObject<tsd::scene::Array>("image");
        REQUIRE(importedColors);
        REQUIRE(importedColors->name() == "tf_colors");
        REQUIRE(importedColors->dataAs<float>()[0] == 1.f);

        auto *importedOpacity =
            imported->parameterValueAsObject<tsd::scene::Array>("opacity");
        REQUIRE(importedOpacity);
        REQUIRE(importedOpacity->name() == "tf_opacity");
      }
    }

    removeTestFile(filename);
  }
}

SCENARIO("tsd::io object payload validation failures", "[Serialization]")
{
  GIVEN("A full scene payload")
  {
    tsd::scene::Scene scene;
    tsd::core::DataTree tree;
    tsd::io::save_Scene(scene, tree.root(), false);
    const auto filename =
        testFile("tsd_full_scene_rejected_by_object_import.tsd");
    removeTestFile(filename);
    REQUIRE(tree.save(filename.c_str()));

    THEN("object import validators reject it")
    {
      auto result = tsd::io::validate_ObjectPayload(tree.root());
      REQUIRE(!result.accepted());
      REQUIRE(result.status
          == tsd::io::PayloadValidationStatus::IncompatibleSchema);

      tsd::scene::Scene target;
      const auto before = target.numberOfObjects(ANARI_MATERIAL);
      REQUIRE(tsd::io::import_Object(target, filename.c_str()) == nullptr);
      REQUIRE(target.numberOfObjects(ANARI_MATERIAL) == before);
    }

    removeTestFile(filename);
  }

  GIVEN("A surface object file")
  {
    tsd::scene::Scene source;
    auto geometry = source.createObject<tsd::scene::Geometry>(
        tsd::scene::tokens::geometry::sphere);
    auto material = source.createObject<tsd::scene::Material>(
        tsd::scene::tokens::material::matte);
    auto surface = source.createSurface("surface", geometry, material);

    const auto filename = testFile("tsd_invalid_surface_object.tsd");
    removeTestFile(filename);
    REQUIRE(tsd::io::export_Object(filename.c_str(), *surface));

    tsd::core::DataTree tree;
    REQUIRE(tree.load(filename.c_str()));

    WHEN("an extra unreferenced object is present")
    {
      auto &extra = tree.root()["objectDB"]["geometry"].append();
      tsd::io::objectToNode(*geometry, extra);
      extra["self"] = tsd::core::Any(ANARI_GEOMETRY, size_t(1));

      THEN("validation rejects the payload")
      {
        auto result = tsd::io::validate_SurfacePayload(tree.root());
        REQUIRE(!result.accepted());
        REQUIRE(result.status
            == tsd::io::PayloadValidationStatus::IncompatibleSchema);
      }
    }

    removeTestFile(filename);
  }

  GIVEN("A surface payload with a disallowed volume pool")
  {
    tsd::core::DataTree tree;
    auto &root = tree.root();
    tsd::core::writeDataTreeMetadata(root,
        {tsd::core::DATA_TREE_METADATA_ENVELOPE_VERSION,
            "object",
            std::string(tsd::io::schema::OBJECT_SURFACE),
            1});
    root["rootObject"] = tsd::core::Any(ANARI_SURFACE, size_t(0));
    auto &surfaceNode = root["objectDB"]["surface"].append();
    surfaceNode["name"] = "surface";
    surfaceNode["self"] = tsd::core::Any(ANARI_SURFACE, size_t(0));
    surfaceNode["subtype"] = "";
    auto &volumeNode = root["objectDB"]["volume"].append();
    volumeNode["name"] = "volume";
    volumeNode["self"] = tsd::core::Any(ANARI_VOLUME, size_t(0));
    volumeNode["subtype"] =
        tsd::scene::tokens::volume::transferFunction1D.c_str();

    THEN("validation rejects the disallowed pool")
    {
      auto result = tsd::io::validate_SurfacePayload(root);
      REQUIRE(!result.accepted());
      REQUIRE(result.status
          == tsd::io::PayloadValidationStatus::IncompatibleSchema);
    }
  }
}

SCENARIO("tsd::io object export failures", "[Serialization]")
{
  GIVEN("An unsupported root object type")
  {
    tsd::scene::Scene scene;
    auto geometry = scene.createObject<tsd::scene::Geometry>(
        tsd::scene::tokens::geometry::sphere);

    THEN("export fails")
    {
      REQUIRE_FALSE(tsd::io::export_Object(
          testFile("tsd_unsupported_object.tsd").c_str(), *geometry));
    }
  }

  GIVEN("A surface reaching a proxy array")
  {
    tsd::scene::Scene scene;
    auto proxy = scene.createArrayProxy(ANARI_FLOAT32, 4);
    auto geometry = scene.createObject<tsd::scene::Geometry>(
        tsd::scene::tokens::geometry::sphere);
    geometry->setParameterObject("primitive.radius", *proxy);
    auto material = scene.createObject<tsd::scene::Material>(
        tsd::scene::tokens::material::matte);
    auto surface = scene.createSurface("surface", geometry, material);

    THEN("export fails because object files must be self-contained")
    {
      REQUIRE_FALSE(tsd::io::export_Object(
          testFile("tsd_proxy_array_object.tsd").c_str(), *surface));
    }
  }

  GIVEN("A surface reaching an object-typed array")
  {
    tsd::scene::Scene scene;
    auto objectArray = scene.createArray(ANARI_SURFACE, 1);
    auto geometry = scene.createObject<tsd::scene::Geometry>(
        tsd::scene::tokens::geometry::sphere);
    geometry->setParameterObject("surface.ids", *objectArray);
    auto material = scene.createObject<tsd::scene::Material>(
        tsd::scene::tokens::material::matte);
    auto surface = scene.createSurface("surface", geometry, material);

    THEN("export fails because object-valued array data cannot be remapped")
    {
      REQUIRE_FALSE(tsd::io::export_Object(
          testFile("tsd_object_typed_array_object.tsd").c_str(), *surface));
    }
  }
}

SCENARIO("tsd::io layer subtree serialization", "[Serialization]")
{
  GIVEN(
      "A scene with a layer subtree referencing surfaces, a light, and overrides")
  {
    tsd::scene::Scene source;

    auto positions = makeFloatArray(source, "positions", {0.f, 1.f, 2.f});
    auto geometry = source.createObject<tsd::scene::Geometry>(
        tsd::scene::tokens::geometry::triangle);
    geometry->setName("mesh_geometry");
    geometry->setParameterObject("vertex.position", *positions);

    auto material = source.createObject<tsd::scene::Material>(
        tsd::scene::tokens::material::matte);
    material->setName("mesh_material");

    auto surface = source.createSurface("mesh_surface", geometry, material);

    auto light = source.createObject<tsd::scene::Light>("directional");
    light->setName("key_light");
    light->setParameter("irradiance", 3.f);

    auto overrideMaterial = source.createObject<tsd::scene::Material>(
        tsd::scene::tokens::material::matte);
    overrideMaterial->setName("override_material");

    // Build subtree: transform -> { surface (with instance params), light }
    auto *layer = source.defaultLayer();
    auto transformNode = source.insertChildTransformNode(
        layer->root(), tsd::math::IDENTITY_MAT4, "group");
    (*transformNode)
        .value()
        .setAsTransform(tsd::math::mat3{tsd::math::float3(2.f, 2.f, 2.f),
            tsd::math::float3(10.f, 20.f, 30.f),
            tsd::math::float3(1.f, 2.f, 3.f)});

    auto surfaceNode =
        source.insertChildObjectNode(transformNode, surface, "surface_inst");
    (*surfaceNode)
        .value()
        .setInstanceParameter("opacity", tsd::core::Any(0.5f));
    (*surfaceNode)
        .value()
        .setInstanceParameter("materialOverride",
            tsd::core::Any(
                overrideMaterial->type(), overrideMaterial->index()));

    source.insertChildObjectNode(transformNode, light, "light_inst");

    const auto filename = testFile("tsd_layer_subtree_roundtrip.tsd");
    removeTestFile(filename);

    WHEN("the subtree is exported")
    {
      REQUIRE(tsd::io::export_LayerSubtree(filename.c_str(), transformNode));

      tsd::core::DataTree exportedTree;
      REQUIRE(exportedTree.load(filename.c_str()));

      THEN(
          "the payload is tagged as a layer subtree with an objectDB and subtree")
      {
        auto metadata = tsd::core::readDataTreeMetadata(exportedTree.root());
        REQUIRE(
            metadata.status == tsd::core::DataTreeMetadataReadStatus::Found);
        REQUIRE(metadata.metadata);
        REQUIRE(metadata.metadata->fileType == "layer-subtree");
        REQUIRE(metadata.metadata->schema
            == std::string(tsd::io::schema::LAYER_SUBTREE));
        REQUIRE(exportedTree.root().child("objectDB"));
        REQUIRE(exportedTree.root().child("subtree"));

        auto result =
            tsd::io::validate_LayerSubtreePayload(exportedTree.root());
        REQUIRE(result.accepted());
      }
    }

    WHEN("the subtree is imported under a destination node in another scene")
    {
      REQUIRE(tsd::io::export_LayerSubtree(filename.c_str(), transformNode));

      tsd::scene::Scene target;
      target.createObject<tsd::scene::Geometry>(
          tsd::scene::tokens::geometry::sphere);

      auto *targetLayer = target.defaultLayer();
      auto destination = target.insertChildTransformNode(
          targetLayer->root(), tsd::math::IDENTITY_MAT4, "mount");

      auto splicedRoot =
          tsd::io::import_LayerSubtree(target, filename.c_str(), destination);

      THEN(
          "objects are appended and the subtree is grafted under the destination")
      {
        REQUIRE(splicedRoot);
        REQUIRE(target.numberOfObjects(ANARI_SURFACE) == 1);
        REQUIRE(target.numberOfObjects(ANARI_GEOMETRY) == 2); // sphere + mesh
        REQUIRE(target.numberOfObjects(ANARI_MATERIAL)
            == 3); // default + mesh + override
        REQUIRE(target.numberOfObjects(ANARI_LIGHT) == 1);
        REQUIRE(target.numberOfObjects(ANARI_ARRAY) == 1);

        REQUIRE((*splicedRoot).value().isTransform());
        REQUIRE((*splicedRoot).value().name() == "group");

        // Two children: surface instance + light instance.
        int childCount = 0;
        bool sawSurfaceInstance = false;
        bool sawLight = false;
        tsd::core::Any opacity;
        tsd::core::Any materialOverride;
        targetLayer->traverse(splicedRoot, [&](auto &node, int level) {
          if (level == 1) {
            childCount++;
            auto &d = node.value();
            if (d.name() == "surface_inst") {
              sawSurfaceInstance = true;
              opacity = d.getInstanceParameters().at("opacity")
                  ? *d.getInstanceParameters().at("opacity")
                  : tsd::core::Any();
              materialOverride =
                  d.getInstanceParameters().at("materialOverride")
                  ? *d.getInstanceParameters().at("materialOverride")
                  : tsd::core::Any();
            }
            if (d.name() == "light_inst")
              sawLight = true;
          }
          return true;
        });

        REQUIRE(childCount == 2);
        REQUIRE(sawSurfaceInstance);
        REQUIRE(sawLight);

        // Instance parameters round-trip, with the object-valued one remapped
        // to the freshly created target material (not the source index).
        REQUIRE(opacity.getAs<float>() == 0.5f);
        REQUIRE(materialOverride.holdsObject());
        REQUIRE(materialOverride.type() == ANARI_MATERIAL);
        auto *remapped = target.getObject(materialOverride);
        REQUIRE(remapped);
        REQUIRE(remapped->name() == "override_material");
      }
    }

    WHEN("the subtree is imported with no destination node")
    {
      REQUIRE(tsd::io::export_LayerSubtree(filename.c_str(), transformNode));

      tsd::scene::Scene target;
      auto *targetLayer = target.defaultLayer();
      const size_t nodesBefore = targetLayer->size();

      auto splicedRoot = tsd::io::import_LayerSubtree(target, filename.c_str());

      THEN("objects are appended but no subtree is grafted")
      {
        REQUIRE_FALSE(splicedRoot);
        REQUIRE(target.numberOfObjects(ANARI_SURFACE) == 1);
        REQUIRE(target.numberOfObjects(ANARI_LIGHT) == 1);
        REQUIRE(targetLayer->size() == nodesBefore);
      }
    }

    WHEN("the subtree is imported into a fragmented (non-defragmented) scene")
    {
      REQUIRE(tsd::io::export_LayerSubtree(filename.c_str(), transformNode));

      tsd::scene::Scene target;

      // Create then remove objects to leave holes in the object pools without
      // defragmenting, mimicking a running viewer that has deleted objects.
      auto g0 = target.createObject<tsd::scene::Geometry>(
          tsd::scene::tokens::geometry::sphere);
      auto g1 = target.createObject<tsd::scene::Geometry>(
          tsd::scene::tokens::geometry::sphere);
      auto m0 = target.createObject<tsd::scene::Material>(
          tsd::scene::tokens::material::matte);
      target.removeObject(g0.data());
      target.removeObject(m0.data());

      auto *targetLayer = target.defaultLayer();
      auto destination = target.insertChildTransformNode(
          targetLayer->root(), tsd::math::IDENTITY_MAT4, "mount");

      auto splicedRoot =
          tsd::io::import_LayerSubtree(target, filename.c_str(), destination);

      THEN("the import succeeds and grafts the subtree")
      {
        REQUIRE(splicedRoot);
        REQUIRE(target.numberOfObjects(ANARI_SURFACE) == 1);
        REQUIRE(target.numberOfObjects(ANARI_LIGHT) == 1);
        REQUIRE((*splicedRoot).value().name() == "group");
      }
    }
  }
}

SCENARIO("tsd::io layer subtree animations round trip", "[Serialization]")
{
  tsd::scene::Scene source;
  tsd::animation::AnimationManager sourceAnimations(&source);

  auto geometry = source.createObject<tsd::scene::Geometry>(
      tsd::scene::tokens::geometry::sphere);
  geometry->setName("animated_geometry");
  auto material = source.createObject<tsd::scene::Material>(
      tsd::scene::tokens::material::matte);
  auto alternateMaterial = source.createObject<tsd::scene::Material>(
      tsd::scene::tokens::material::matte);
  alternateMaterial->setName("alternate_material");
  auto surface = source.createSurface("animated_surface", geometry, material);
  auto group = source.insertChildTransformNode(source.defaultLayer()->root(),
      tsd::math::IDENTITY_MAT4,
      "animated_group");
  source.insertChildObjectNode(group, surface, "animated_surface");

  const float times[] = {0.f, 1.f};
  const float radii[] = {0.25f, 2.f};
  auto &animation = sourceAnimations.addAnimation("dataset_animation");
  animation.addObjectParameterBinding(
      geometry.data(), "radius", ANARI_FLOAT32, radii, times, 2);
  tsd::scene::Object *materials[] = {material.data(), alternateMaterial.data()};
  animation.addObjectParameterBinding(geometry.data(),
      "animated.material",
      ANARI_MATERIAL,
      materials,
      times,
      2);
  animation.addTransformBinding(group);

  const auto filename = testFile("tsd_layer_subtree_animation.tsd");
  removeTestFile(filename);
  tsd::io::SubtreeIOOptions exportOptions;
  exportOptions.animationManager = &sourceAnimations;
  REQUIRE(tsd::io::export_Subtree(filename.c_str(),
      group,
      {"layer-subtree", tsd::io::schema::LAYER_SUBTREE, false},
      {},
      exportOptions));
  tsd::core::DataTree exported;
  REQUIRE(exported.load(filename.c_str()));
  auto *serializedAnimation = exported.root()["animations"].child(0);
  REQUIRE(serializedAnimation);
  auto *serializedMaterialBinding =
      (*serializedAnimation)["objectBindings"].child(1);
  REQUIRE(serializedMaterialBinding);
  REQUIRE((*serializedMaterialBinding)["dataType"].getValueAs<int>()
      == ANARI_MATERIAL);
  anari::DataType serializedMaterialType = ANARI_UNKNOWN;
  const void *serializedMaterialData = nullptr;
  const size_t *serializedMaterialIndices = nullptr;
  size_t serializedMaterialCount = 0;
  (*serializedMaterialBinding)["data"].getValueAsArray(&serializedMaterialType,
      &serializedMaterialData,
      &serializedMaterialCount);
  serializedMaterialIndices =
      static_cast<const size_t *>(serializedMaterialData);
  REQUIRE(serializedMaterialType == ANARI_MATERIAL);
  REQUIRE(serializedMaterialCount == 2);
  REQUIRE(exported.root()["objectDB"]["material"].numChildren() == 2);
  auto exportedValidation =
      tsd::io::validate_LayerSubtreePayload(exported.root());
  INFO(exportedValidation.message);
  REQUIRE(exportedValidation.accepted());

  tsd::scene::Scene target;
  tsd::animation::AnimationManager targetAnimations(&target);
  target.createObject<tsd::scene::Geometry>(
      tsd::scene::tokens::geometry::cylinder);
  auto destination =
      target.insertChildNode(target.defaultLayer()->root(), "destination");
  tsd::io::SubtreeIOOptions importOptions;
  importOptions.animationManager = &targetAnimations;
  auto imported = tsd::io::import_Subtree(target,
      filename.c_str(),
      destination,
      {"layer-subtree", tsd::io::schema::LAYER_SUBTREE, false},
      nullptr,
      importOptions);

  REQUIRE(imported);
  REQUIRE(targetAnimations.animations().size() == 1);
  auto &loaded = targetAnimations.animations().front();
  REQUIRE(loaded.name() == "dataset_animation");
  REQUIRE(loaded.objectParameterBindings().size() == 2);
  REQUIRE(loaded.objectParameterBindings().front().target());
  REQUIRE(loaded.objectParameterBindings().front().target()->name()
      == "animated_geometry");
  REQUIRE(loaded.transformBindings().size() == 1);
  REQUIRE(loaded.transformBindings().front().target());
  REQUIRE((*loaded.transformBindings().front().target())->name()
      == "animated_group");
  const auto &materialBinding = loaded.objectParameterBindings()[1];
  REQUIRE(materialBinding.data().size() == 2);
  const auto *materialIndices =
      static_cast<const size_t *>(materialBinding.data().data());
  REQUIRE(target.getObject(ANARI_MATERIAL, materialIndices[0]));
  REQUIRE(target.getObject(ANARI_MATERIAL, materialIndices[1])->name()
      == "alternate_material");

  removeTestFile(filename);
}

SCENARIO(
    "tsd::io rejects animations spanning layer subtrees", "[Serialization]")
{
  tsd::scene::Scene scene;
  tsd::animation::AnimationManager animations(&scene);
  auto first = scene.insertChildTransformNode(
      scene.defaultLayer()->root(), tsd::math::IDENTITY_MAT4, "first");
  auto second = scene.insertChildTransformNode(
      scene.defaultLayer()->root(), tsd::math::IDENTITY_MAT4, "second");
  auto &animation = animations.addAnimation("cross-dataset");
  animation.addTransformBinding(first);
  animation.addTransformBinding(second);

  tsd::io::SubtreeIOOptions options;
  options.animationManager = &animations;
  const auto filename = testFile("tsd_cross_subtree_animation.tsd");
  removeTestFile(filename);

  REQUIRE_FALSE(tsd::io::export_Subtree(filename.c_str(),
      first,
      {"layer-subtree", tsd::io::schema::LAYER_SUBTREE, false},
      {},
      options));
  REQUIRE_FALSE(std::filesystem::exists(filename));
}

SCENARIO("tsd::io validates layer subtree animation targets", "[Serialization]")
{
  tsd::scene::Scene scene;
  tsd::animation::AnimationManager animations(&scene);
  auto geometry = scene.createObject<tsd::scene::Geometry>(
      tsd::scene::tokens::geometry::sphere);
  auto material = scene.createObject<tsd::scene::Material>(
      tsd::scene::tokens::material::matte);
  auto surface = scene.createSurface("surface", geometry, material);
  auto root = scene.insertChildNode(scene.defaultLayer()->root(), "dataset");
  scene.insertChildObjectNode(root, surface, "surface");
  const float times[] = {0.f};
  const float values[] = {1.f};
  animations.addAnimation("radius").addObjectParameterBinding(
      geometry.data(), "radius", ANARI_FLOAT32, values, times, 1);

  const auto filename = testFile("tsd_invalid_subtree_animation.tsd");
  tsd::io::SubtreeIOOptions options;
  options.animationManager = &animations;
  REQUIRE(tsd::io::export_Subtree(filename.c_str(),
      root,
      {"layer-subtree", tsd::io::schema::LAYER_SUBTREE, false},
      {},
      options));

  tsd::core::DataTree tree;
  REQUIRE(tree.load(filename.c_str()));
  auto *animation = tree.root()["animations"].child(0);
  REQUIRE(animation);
  auto *binding = (*animation)["objectBindings"].child(0);
  REQUIRE(binding);
  (*binding)["targetIndex"] = size_t(999);
  auto validation = tsd::io::validate_LayerSubtreePayload(tree.root());
  REQUIRE_FALSE(validation.accepted());
  REQUIRE_FALSE(validation.message.empty());

  removeTestFile(filename);
}

SCENARIO("tsd::io save_Scene excludes light-rig subtrees", "[Serialization]")
{
  GIVEN("A scene with a retained surface and an excluded light-rig subtree")
  {
    tsd::scene::Scene source;

    auto positions = makeFloatArray(source, "positions", {0.f, 1.f, 2.f});
    auto geometry = source.createObject<tsd::scene::Geometry>(
        tsd::scene::tokens::geometry::triangle);
    geometry->setName("mesh_geometry");
    geometry->setParameterObject("vertex.position", *positions);
    auto material = source.createObject<tsd::scene::Material>(
        tsd::scene::tokens::material::matte);
    auto surface = source.createSurface("mesh_surface", geometry, material);

    auto *layer = source.addLayer("studio");
    auto datasets = source.insertChildNode(layer->root(), "datasets");
    source.insertChildObjectNode(datasets, surface, "surface_inst");

    auto rigsRoot = source.insertChildNode(layer->root(), "lightRigs");
    auto rigRoot = source.insertChildNode(rigsRoot, "rig0");
    auto light = source.createObject<tsd::scene::Light>("directional");
    light->setName("key_light");
    source.insertChildObjectNode(rigRoot, light, "key_light");

    WHEN("the scene is saved with the rig subtree excluded and reloaded")
    {
      tsd::io::SaveSceneOptions options;
      options.excludedSubtrees.push_back(rigRoot);

      tsd::core::DataTree tree;
      tsd::io::save_Scene(source, tree.root(), options);

      THEN("the manifest omits the light pool and the rig subtree node")
      {
        auto *objectDB = tree.root().child("objectDB");
        REQUIRE(objectDB);
        REQUIRE(objectDB->child("light") == nullptr);

        auto *rigsNode =
            tree.root()["layers"].child("studio")->child("children");
        REQUIRE(rigsNode);
        // lightRigs container is retained but its rig children are pruned.
        bool sawRig = false;
        rigsNode->foreach_child([&](tsd::core::DataNode &n) {
          if (n["name"].getValueOr<std::string>("") == "lightRigs") {
            if (auto *kids = n.child("children"))
              sawRig = kids->numChildren() > 0;
          }
        });
        REQUIRE_FALSE(sawRig);
      }

      THEN("the retained surface and its array survive the round trip")
      {
        tsd::scene::Scene target;
        tsd::io::load_Scene(target, tree.root());

        REQUIRE(target.numberOfObjects(ANARI_LIGHT) == 0);
        REQUIRE(target.numberOfObjects(ANARI_SURFACE) == 1);
        REQUIRE(target.numberOfObjects(ANARI_GEOMETRY) == 1);
        REQUIRE(target.numberOfObjects(ANARI_ARRAY) == 1);

        auto *targetGeometry = static_cast<tsd::scene::Geometry *>(
            target.getObject(ANARI_GEOMETRY, 0));
        REQUIRE(targetGeometry);
        auto *ref = targetGeometry->parameterValueAsObject<tsd::scene::Array>(
            "vertex.position");
        REQUIRE(ref != nullptr);
        REQUIRE(ref->type() == ANARI_ARRAY1D);
      }
    }
  }

  GIVEN("A light-only array and a shared array referenced by a retained object")
  {
    tsd::scene::Scene source;

    // Retained geometry referencing its own array.
    auto positions = makeFloatArray(source, "positions", {0.f, 1.f, 2.f});
    auto geometry = source.createObject<tsd::scene::Geometry>(
        tsd::scene::tokens::geometry::triangle);
    geometry->setParameterObject("vertex.position", *positions);
    auto material = source.createObject<tsd::scene::Material>(
        tsd::scene::tokens::material::matte);
    auto surface = source.createSurface("mesh_surface", geometry, material);

    // An array referenced only by the (excluded) light.
    auto lightOnly = makeFloatArray(source, "light_only", {9.f});
    // An array shared between the excluded light and the retained geometry.
    auto shared = makeFloatArray(source, "shared", {1.f, 2.f});
    geometry->setParameterObject("primitive.radius", *shared);

    auto *layer = source.addLayer("studio");
    source.insertChildObjectNode(layer->root(), surface, "surface_inst");

    auto rigRoot = source.insertChildNode(layer->root(), "rig0");
    auto light = source.createObject<tsd::scene::Light>("directional");
    light->setParameterObject("rig.lightOnly", *lightOnly);
    light->setParameterObject("rig.shared", *shared);
    source.insertChildObjectNode(rigRoot, light, "key_light");

    WHEN("the scene is saved with the rig excluded and reloaded")
    {
      tsd::io::SaveSceneOptions options;
      options.excludedSubtrees.push_back(rigRoot);

      tsd::core::DataTree tree;
      tsd::io::save_Scene(source, tree.root(), options);

      tsd::scene::Scene target;
      tsd::io::load_Scene(target, tree.root());

      THEN("the light-only array is dropped but the shared array is kept")
      {
        REQUIRE(target.numberOfObjects(ANARI_LIGHT) == 0);
        // positions + shared survive; light_only is dropped.
        REQUIRE(target.numberOfObjects(ANARI_ARRAY) == 2);

        auto *targetGeometry = static_cast<tsd::scene::Geometry *>(
            target.getObject(ANARI_GEOMETRY, 0));
        REQUIRE(targetGeometry);
        REQUIRE(targetGeometry->parameterValueAsObject<tsd::scene::Array>(
                    "vertex.position")
            != nullptr);
        REQUIRE(targetGeometry->parameterValueAsObject<tsd::scene::Array>(
                    "primitive.radius")
            != nullptr);
      }
    }
  }
}

SCENARIO("tsd::io save_Scene remaps animation bindings across exclusion",
    "[Serialization]")
{
  GIVEN("Animations whose targets shift when a light rig is excluded")
  {
    tsd::scene::Scene source;
    tsd::animation::AnimationManager animMgr(&source);

    // Array pool order: lightOnly @0 (excluded), positions @1 (retained, so it
    // shifts to @0 on reload). Exercises the object-index remap.
    auto lightOnly = makeFloatArray(source, "lightOnly", {9.f});
    auto positions = makeFloatArray(source, "positions", {0.f, 1.f, 2.f});

    auto geometry = source.createObject<tsd::scene::Geometry>(
        tsd::scene::tokens::geometry::triangle);
    geometry->setParameterObject("vertex.position", *positions);
    auto material = source.createObject<tsd::scene::Material>(
        tsd::scene::tokens::material::matte);
    auto surface = source.createSurface("surf", geometry, material);

    auto light = source.createObject<tsd::scene::Light>("directional");
    light->setParameterObject("rig.env", *lightOnly);

    // Layer order: excluded rig FIRST, retained "group" AFTER it, so group's
    // layer-node index shifts down on reload. Exercises the layer-node remap.
    auto *layer = source.addLayer("studio");
    auto rig0 = source.insertChildNode(layer->root(), "rig0");
    source.insertChildObjectNode(rig0, light, "key");
    auto group = source.insertChildTransformNode(
        layer->root(), tsd::math::IDENTITY_MAT4, "group");
    source.insertChildObjectNode(group, surface, "surf_inst");

    auto &anim = animMgr.addAnimation("test");
    const float timeBase[2] = {0.f, 1.f};
    const float scalarData[2] = {0.f, 1.f};
    anim.addObjectParameterBinding(
        positions.data(), "scale", ANARI_FLOAT32, scalarData, timeBase, 2);
    anim.addTransformBinding(group);

    WHEN("the scene is saved with the rig excluded and reloaded")
    {
      tsd::io::SaveSceneOptions options;
      options.animMgr = &animMgr;
      options.excludedSubtrees.push_back(rig0);

      tsd::core::DataTree tree;
      tsd::io::save_Scene(source, tree.root(), options);

      tsd::scene::Scene target;
      tsd::animation::AnimationManager targetMgr(&target);
      tsd::io::load_Scene(target, tree.root(), &targetMgr);

      THEN("binding targets resolve to the correct shifted objects/nodes")
      {
        REQUIRE(target.numberOfObjects(ANARI_LIGHT) == 0);
        REQUIRE(target.numberOfObjects(ANARI_ARRAY) == 1); // lightOnly dropped
        REQUIRE(targetMgr.animations().size() == 1);

        auto &loaded = targetMgr.animations().front();
        REQUIRE(loaded.objectParameterBindings().size() == 1);
        // Without the remap, targetIndex would stay @1 and resolve to nullptr.
        REQUIRE(loaded.objectParameterBindings().front().target() != nullptr);

        REQUIRE(loaded.transformBindings().size() == 1);
        auto tbTarget = loaded.transformBindings().front().target();
        REQUIRE(tbTarget);
        REQUIRE((*tbTarget)->name() == "group");
      }
    }
  }
}
