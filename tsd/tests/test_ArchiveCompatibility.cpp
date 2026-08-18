// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/animation/Animation.hpp"
#include "tsd/animation/AnimationManager.hpp"
#include "tsd/core/DataTree.hpp"
#include "tsd/core/DataTreeMetadata.hpp"
#include "tsd/io/archives.hpp"
#include "tsd/io/archives/detail/ArchivePlan.hpp"
#include "tsd/io/serialization/serialization_internal.hpp"
#include "tsd/scene/Scene.hpp"
#include "tsd/scene/objects/Geometry.hpp"
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

bool saveSubtreeArchiveContent(const char *filename,
    tsd::scene::LayerNodeRef root,
    const tsd::io::SubtreeArchiveContentDesc &desc,
    std::string_view displayName = {},
    const tsd::io::SubtreeArchiveContentOptions &options = {})
{
  tsd::core::DataTree tree;
  return tsd::io::serialize_SubtreeArchiveContent(
             root, tree.root(), desc, displayName, options)
      && tree.save(filename);
}

tsd::io::SubtreeArchiveResult loadSubtreeArchiveContent(
    tsd::scene::Scene &scene,
    const char *filename,
    tsd::scene::LayerNodeRef destination,
    const tsd::io::SubtreeArchiveContentDesc &desc,
    std::string *displayName = nullptr,
    const tsd::io::SubtreeArchiveContentOptions &options = {})
{
  tsd::core::DataTree tree;
  if (!tree.load(filename))
    return {};
  return tsd::io::deserialize_SubtreeArchiveContent(
      scene, tree.root(), destination, desc, displayName, options);
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

struct UnsupportedFileBinding : tsd::animation::FileBinding
{
  explicit UnsupportedFileBinding(tsd::scene::Scene *scene);

  std::string kind() const override;
  void toDataNode(tsd::core::DataNode &) const override;
  void update(float) override;

 private:
  void addCallbackToAnimation(tsd::animation::Animation &) override;
};

UnsupportedFileBinding::UnsupportedFileBinding(tsd::scene::Scene *scene)
    : FileBinding(scene)
{}

std::string UnsupportedFileBinding::kind() const
{
  return "unsupported";
}

void UnsupportedFileBinding::toDataNode(tsd::core::DataNode &) const {}

void UnsupportedFileBinding::update(float) {}

void UnsupportedFileBinding::addCallbackToAnimation(tsd::animation::Animation &)
{}

} // namespace

SCENARIO("tsd::io camera and renderer subset serialization",
    "[ArchiveCompatibility]")
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
      tsd::io::detail::serializeLegacyCameraRendererPayload(source, root);

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

        tsd::io::detail::tryDeserializeLegacyCameraRendererPayload(
            target, root);

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
      tsd::io::detail::tryDeserializeLegacyCameraRendererPayload(
          scene, tree.root());

      THEN("the scene still has a default camera")
      {
        REQUIRE(scene.defaultCamera());
        REQUIRE(scene.numberOfObjects(ANARI_CAMERA) == 1);
      }
    }
  }
}

SCENARIO("tsd::io scene payload metadata validation", "[ArchiveCompatibility]")
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
      REQUIRE(tsd::io::serialize_SceneArchive(source, tree.root()));

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
            tsd::io::detail::validateLegacyCameraRendererPayload(tree.root());
        REQUIRE(result.accepted());
        REQUIRE(result.status == tsd::io::ArchiveValidationStatus::Valid);
      }
    }

    WHEN("a camera and renderer subset is loaded as a full scene")
    {
      tsd::core::DataTree subsetTree;
      tsd::io::detail::serializeLegacyCameraRendererPayload(
          source, subsetTree.root());

      tsd::scene::Scene target;
      target.createObject<tsd::scene::Geometry>("sphere");
      target.addLayer("keep_me");

      THEN("validation rejects it before mutation")
      {
        auto result =
            tsd::io::detail::validateLegacyScenePayload(subsetTree.root());
        REQUIRE(!result.accepted());
        REQUIRE(result.status
            == tsd::io::ArchiveValidationStatus::IncompatibleSchema);

        tsd::io::detail::tryDeserializeLegacyScenePayload(
            target, subsetTree.root());
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
        auto result =
            tsd::io::detail::validateLegacyScenePayload(legacyTree.root());
        REQUIRE(result.accepted());
        REQUIRE(result.status
            == tsd::io::ArchiveValidationStatus::MissingMetadataAccepted);
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
        auto result =
            tsd::io::detail::validateLegacyScenePayload(invalidTree.root());
        REQUIRE(!result.accepted());
        REQUIRE(result.status
            == tsd::io::ArchiveValidationStatus::MissingRequiredNode);

        tsd::io::detail::tryDeserializeLegacyScenePayload(
            target, invalidTree.root());
        REQUIRE(target.numberOfObjects(ANARI_GEOMETRY) == 1);
        REQUIRE(target.numberOfLayers() == 1);
        REQUIRE(target.layer("keep_me") != nullptr);
      }
    }
  }
}

SCENARIO("tsd::io surface object serialization", "[ArchiveCompatibility]")
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

    WHEN("the Surface Object Archive is saved and loaded")
    {
      REQUIRE(tsd::io::save_ObjectArchive(*surface, filename.c_str()));

      tsd::core::DataTree savedTree;
      REQUIRE(savedTree.load(filename.c_str()));

      THEN("the payload is tagged as a surface object with local root index 0")
      {
        auto metadata = tsd::core::readDataTreeMetadata(savedTree.root());
        REQUIRE(
            metadata.status == tsd::core::DataTreeMetadataReadStatus::Found);
        REQUIRE(metadata.metadata);
        REQUIRE(metadata.metadata->fileType == "object");
        REQUIRE(metadata.metadata->schema
            == std::string(tsd::io::schema::OBJECT_SURFACE));

        auto *rootObject = savedTree.root().child("rootObject");
        REQUIRE(rootObject);
        REQUIRE(rootObject->getValue().type() == ANARI_SURFACE);
        REQUIRE(rootObject->getValue().getAsObjectIndex() == 0);

        auto *surfaceNode =
            savedTree.root().child("objectDB")->child("surface")->child(0);
        REQUIRE(surfaceNode);
        REQUIRE(surfaceNode->child("self")->getValue().type() == ANARI_SURFACE);
        REQUIRE(surfaceNode->child("self")->getValue().getAsObjectIndex() == 0);
      }

      tsd::scene::Scene target;
      auto existingGeometry = target.createObject<tsd::scene::Geometry>(
          tsd::scene::tokens::geometry::sphere);
      existingGeometry->setName("preexisting_geometry");
      target.addLayer("keep_me");

      auto *loaded = dynamic_cast<tsd::scene::Surface *>(
          tsd::io::load_ObjectArchive(target, filename.c_str()));

      THEN("loading appends objects without creating layers")
      {
        REQUIRE(loaded);
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
        REQUIRE(loaded->name() == "root_surface");
        REQUIRE(loaded->getMetadataValue("priority").getAs<int>() == 9);

        auto *loadedGeometry = loaded->geometry();
        auto *loadedMaterial = loaded->material();
        REQUIRE(loadedGeometry);
        REQUIRE(loadedMaterial);
        REQUIRE(loadedGeometry->name() == "mesh_geometry");
        REQUIRE(loadedMaterial->name() == "sampled_material");

        auto geometryMetadata = loaded->getMetadataValue("geometryRef");
        REQUIRE(geometryMetadata.holdsObject());
        REQUIRE(geometryMetadata.getAsObjectIndex() == loadedGeometry->index());

        auto *positionParam = loadedGeometry->parameter("vertex.position");
        REQUIRE(positionParam);
        REQUIRE(!positionParam->isEnabled());
        REQUIRE(positionParam->description() == "positions");

        auto *loadedPositions =
            loadedGeometry->parameterValueAsObject<tsd::scene::Array>(
                "vertex.position");
        REQUIRE(loadedPositions);
        REQUIRE(loadedPositions->name() == "positions");
        REQUIRE(loadedPositions->getMetadataValue("stride").getAs<int>() == 12);
        REQUIRE(loadedPositions->size() == 6);
        const auto *positionData = loadedPositions->dataAs<float>();
        REQUIRE(positionData[0] == 1.f);
        REQUIRE(positionData[5] == 6.f);

        auto *loadedSampler =
            loadedMaterial->parameterValueAsObject<tsd::scene::Sampler>(
                "color");
        REQUIRE(loadedSampler);
        auto *loadedTexture =
            loadedSampler->parameterValueAsObject<tsd::scene::Array>("image");
        REQUIRE(loadedTexture);
        REQUIRE(loadedTexture->name() == "texture");
        REQUIRE(loadedTexture->dataAs<float>()[2] == 0.75f);

        auto samplerMetadata = loadedMaterial->getMetadataValue("samplerRef");
        REQUIRE(samplerMetadata.holdsObject());
        REQUIRE(samplerMetadata.getAsObjectIndex() == loadedSampler->index());
      }
    }

    removeTestFile(filename);
  }
}

SCENARIO("tsd::io volume object serialization", "[ArchiveCompatibility]")
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

    WHEN("the Volume Object Archive is saved and loaded")
    {
      REQUIRE(tsd::io::save_ObjectArchive(*volume, filename.c_str()));

      tsd::core::DataTree savedTree;
      REQUIRE(savedTree.load(filename.c_str()));
      REQUIRE(tsd::io::validate_ObjectArchive(savedTree.root()).accepted());

      tsd::scene::Scene target;
      target.createObject<tsd::scene::SpatialField>(
          tsd::scene::tokens::spatial_field::structuredRegular);

      auto *loaded = dynamic_cast<tsd::scene::Volume *>(
          tsd::io::load_ObjectArchive(target, filename.c_str()));

      THEN("the loaded volume preserves field, arrays, metadata, and refs")
      {
        REQUIRE(loaded);
        REQUIRE(loaded->name() == "root_volume");
        REQUIRE(target.numberOfObjects(ANARI_VOLUME) == 1);
        REQUIRE(target.numberOfObjects(ANARI_SPATIAL_FIELD) == 2);
        REQUIRE(target.numberOfObjects(ANARI_ARRAY) == 3);

        auto *loadedField =
            loaded->parameterValueAsObject<tsd::scene::SpatialField>("value");
        REQUIRE(loadedField);
        REQUIRE(loadedField->name() == "density_field");

        auto fieldRef = loaded->getMetadataValue("fieldRef");
        REQUIRE(fieldRef.holdsObject());
        REQUIRE(fieldRef.getAsObjectIndex() == loadedField->index());

        auto *loadedData =
            loadedField->parameterValueAsObject<tsd::scene::Array>("data");
        REQUIRE(loadedData);
        REQUIRE(loadedData->dataAs<float>()[3] == 3.f);

        auto sourceData = loadedField->getMetadataValue("sourceData");
        REQUIRE(sourceData.holdsObject());
        REQUIRE(sourceData.getAsObjectIndex() == loadedData->index());

        auto *loadedSampler =
            loaded->parameterValueAsObject<tsd::scene::Sampler>("color");
        REQUIRE(loadedSampler);
        auto *loadedColors =
            loadedSampler->parameterValueAsObject<tsd::scene::Array>("image");
        REQUIRE(loadedColors);
        REQUIRE(loadedColors->name() == "tf_colors");
        REQUIRE(loadedColors->dataAs<float>()[0] == 1.f);

        auto *loadedOpacity =
            loaded->parameterValueAsObject<tsd::scene::Array>("opacity");
        REQUIRE(loadedOpacity);
        REQUIRE(loadedOpacity->name() == "tf_opacity");
      }
    }

    removeTestFile(filename);
  }
}

SCENARIO("tsd::io object payload validation failures", "[ArchiveCompatibility]")
{
  GIVEN("A full scene payload")
  {
    tsd::scene::Scene scene;
    tsd::core::DataTree tree;
    REQUIRE(tsd::io::serialize_SceneArchive(scene, tree.root()));
    const auto filename =
        testFile("tsd_full_scene_rejected_by_object_archive.tsd");
    removeTestFile(filename);
    REQUIRE(tree.save(filename.c_str()));

    THEN("Object Archive validation rejects it")
    {
      auto result = tsd::io::validate_ObjectArchive(tree.root());
      REQUIRE(!result.accepted());
      REQUIRE(result.status
          == tsd::io::ArchiveValidationStatus::IncompatibleSchema);

      tsd::scene::Scene target;
      const auto before = target.numberOfObjects(ANARI_MATERIAL);
      REQUIRE(tsd::io::load_ObjectArchive(target, filename.c_str()) == nullptr);
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
    REQUIRE(tsd::io::save_ObjectArchive(*surface, filename.c_str()));

    tsd::core::DataTree tree;
    REQUIRE(tree.load(filename.c_str()));

    WHEN("an extra unreferenced object is present")
    {
      auto &extra = tree.root()["objectDB"]["geometry"].append();
      tsd::io::serialize_Object(*geometry, extra);
      extra["self"] = tsd::core::Any(ANARI_GEOMETRY, size_t(1));

      THEN("validation rejects the payload")
      {
        auto result = tsd::io::validate_ObjectArchive(tree.root());
        REQUIRE(!result.accepted());
        REQUIRE(result.status
            == tsd::io::ArchiveValidationStatus::IncompatibleSchema);
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
      auto result = tsd::io::validate_ObjectArchive(root);
      REQUIRE(!result.accepted());
      REQUIRE(result.status
          == tsd::io::ArchiveValidationStatus::IncompatibleSchema);
    }
  }
}

SCENARIO("tsd::io Object Archive save failures", "[ArchiveCompatibility]")
{
  GIVEN("An unsupported root object type")
  {
    tsd::scene::Scene scene;
    auto geometry = scene.createObject<tsd::scene::Geometry>(
        tsd::scene::tokens::geometry::sphere);

    THEN("saving fails")
    {
      REQUIRE_FALSE(tsd::io::save_ObjectArchive(
          *geometry, testFile("tsd_unsupported_object.tsd").c_str()));
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

    THEN("saving fails because Object Archives must be self-contained")
    {
      REQUIRE_FALSE(tsd::io::save_ObjectArchive(
          *surface, testFile("tsd_proxy_array_object.tsd").c_str()));
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

    THEN("saving fails because object-valued array data cannot be remapped")
    {
      REQUIRE_FALSE(tsd::io::save_ObjectArchive(
          *surface, testFile("tsd_object_typed_array_object.tsd").c_str()));
    }
  }
}

SCENARIO("tsd::io layer subtree serialization", "[ArchiveCompatibility]")
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

    WHEN("the subtree Archive is saved")
    {
      REQUIRE(
          tsd::io::save_LayerSubtreeArchive(transformNode, filename.c_str()));

      tsd::core::DataTree savedTree;
      REQUIRE(savedTree.load(filename.c_str()));

      THEN(
          "the payload is tagged as a layer subtree with an objectDB and subtree")
      {
        auto metadata = tsd::core::readDataTreeMetadata(savedTree.root());
        REQUIRE(
            metadata.status == tsd::core::DataTreeMetadataReadStatus::Found);
        REQUIRE(metadata.metadata);
        REQUIRE(metadata.metadata->fileType == "layer-subtree");
        REQUIRE(metadata.metadata->schema
            == std::string(tsd::io::schema::LAYER_SUBTREE));
        REQUIRE(savedTree.root().child("objectDB"));
        REQUIRE(savedTree.root().child("subtree"));

        auto result = tsd::io::validate_LayerSubtreeArchive(savedTree.root());
        REQUIRE(result.accepted());
      }
    }

    WHEN("the subtree Archive is loaded under a destination node")
    {
      REQUIRE(
          tsd::io::save_LayerSubtreeArchive(transformNode, filename.c_str()));

      tsd::scene::Scene target;
      target.createObject<tsd::scene::Geometry>(
          tsd::scene::tokens::geometry::sphere);

      auto *targetLayer = target.defaultLayer();
      auto destination = target.insertChildTransformNode(
          targetLayer->root(), tsd::math::IDENTITY_MAT4, "mount");

      auto splicedRoot =
          tsd::io::load_LayerSubtreeArchive(destination, filename.c_str());

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

    WHEN("the subtree Archive is loaded into a fragmented scene")
    {
      REQUIRE(
          tsd::io::save_LayerSubtreeArchive(transformNode, filename.c_str()));

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
          tsd::io::load_LayerSubtreeArchive(destination, filename.c_str());

      THEN("the load succeeds and grafts the subtree")
      {
        REQUIRE(splicedRoot);
        REQUIRE(target.numberOfObjects(ANARI_SURFACE) == 1);
        REQUIRE(target.numberOfObjects(ANARI_LIGHT) == 1);
        REQUIRE((*splicedRoot).value().name() == "group");
      }
    }
  }
}

SCENARIO(
    "tsd::io layer subtree animations round trip", "[ArchiveCompatibility]")
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
  tsd::io::SubtreeArchiveContentOptions saveOptions;
  saveOptions.animationManager = &sourceAnimations;
  REQUIRE(saveSubtreeArchiveContent(filename.c_str(),
      group,
      {"layer-subtree",
          tsd::io::schema::LAYER_SUBTREE,
          tsd::io::ArchiveObjectPolicy::All},
      {},
      saveOptions));
  tsd::core::DataTree saved;
  REQUIRE(saved.load(filename.c_str()));
  auto *serializedAnimation = saved.root()["animations"].child(0);
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
  REQUIRE(saved.root()["objectDB"]["material"].numChildren() == 2);
  auto savedValidation = tsd::io::validate_SubtreeArchiveContent(saved.root(),
      {"layer-subtree",
          tsd::io::schema::LAYER_SUBTREE,
          tsd::io::ArchiveObjectPolicy::All});
  INFO(savedValidation.message);
  REQUIRE(savedValidation.accepted());

  tsd::scene::Scene target;
  tsd::animation::AnimationManager targetAnimations(&target);
  target.createObject<tsd::scene::Geometry>(
      tsd::scene::tokens::geometry::cylinder);
  auto destination =
      target.insertChildNode(target.defaultLayer()->root(), "destination");
  tsd::io::SubtreeArchiveContentOptions loadOptions;
  loadOptions.animationManager = &targetAnimations;
  auto loadedArchive = loadSubtreeArchiveContent(target,
      filename.c_str(),
      destination,
      {"layer-subtree",
          tsd::io::schema::LAYER_SUBTREE,
          tsd::io::ArchiveObjectPolicy::All},
      nullptr,
      loadOptions);

  REQUIRE(loadedArchive.root);
  REQUIRE(targetAnimations.animations().size() == 1);
  auto &loadedAnimation = targetAnimations.animations().front();
  REQUIRE(loadedAnimation.name() == "dataset_animation");
  REQUIRE(loadedAnimation.objectParameterBindings().size() == 2);
  REQUIRE(loadedAnimation.objectParameterBindings().front().target());
  REQUIRE(loadedAnimation.objectParameterBindings().front().target()->name()
      == "animated_geometry");
  REQUIRE(loadedAnimation.transformBindings().size() == 1);
  REQUIRE(loadedAnimation.transformBindings().front().target());
  REQUIRE((*loadedAnimation.transformBindings().front().target())->name()
      == "animated_group");
  const auto &materialBinding = loadedAnimation.objectParameterBindings()[1];
  REQUIRE(materialBinding.data().size() == 2);
  const auto *materialIndices =
      static_cast<const size_t *>(materialBinding.data().data());
  REQUIRE(target.getObject(ANARI_MATERIAL, materialIndices[0]));
  REQUIRE(target.getObject(ANARI_MATERIAL, materialIndices[1])->name()
      == "alternate_material");

  removeTestFile(filename);
}

SCENARIO("tsd::io plans subtree archive ownership", "[ArchiveCompatibility]")
{
  tsd::scene::Scene scene;
  tsd::animation::AnimationManager animations(&scene);

  auto geometry = scene.createObject<tsd::scene::Geometry>(
      tsd::scene::tokens::geometry::sphere);
  auto material = scene.createObject<tsd::scene::Material>(
      tsd::scene::tokens::material::matte);
  auto surface = scene.createSurface("surface", geometry, material);
  auto dataset = scene.insertChildNode(scene.defaultLayer()->root(), "dataset");
  scene.insertChildObjectNode(dataset, surface, "surface");
  auto outside = scene.insertChildTransformNode(
      scene.defaultLayer()->root(), tsd::math::IDENTITY_MAT4, "outside");

  const float times[] = {0.f};
  const float values[] = {1.f};
  animations.addAnimation("owned").addObjectParameterBinding(
      geometry.data(), "radius", ANARI_FLOAT32, values, times, 1);
  animations.addAnimation("outside").addTransformBinding(outside);

  tsd::io::ArchivePlanOptions options;
  options.animationManager = &animations;
  const auto result = tsd::io::plan_SubtreeArchive(scene, dataset, options);

  INFO(result.message);
  REQUIRE(result.accepted());
  REQUIRE(result.plan.nodes.size() == 2);
  REQUIRE(result.plan.objects.size() == 3);
  REQUIRE(result.plan.ownedAnimations == std::vector<size_t>{0});
  REQUIRE(result.plan.archivedAnimations == std::vector<size_t>{0});
  REQUIRE(result.plan.containsObject(geometry.data()));
  REQUIRE_FALSE(result.plan.containsObject(nullptr));
}

SCENARIO("tsd::io archive plans reject mixed animation ownership",
    "[ArchiveCompatibility]")
{
  tsd::scene::Scene scene;
  tsd::animation::AnimationManager animations(&scene);
  auto first = scene.insertChildTransformNode(
      scene.defaultLayer()->root(), tsd::math::IDENTITY_MAT4, "first");
  auto second = scene.insertChildTransformNode(
      scene.defaultLayer()->root(), tsd::math::IDENTITY_MAT4, "second");
  auto &animation = animations.addAnimation("mixed");
  animation.addTransformBinding(first);
  animation.addTransformBinding(second);

  tsd::io::ArchivePlanOptions options;
  options.animationManager = &animations;
  const auto result = tsd::io::plan_SubtreeArchive(scene, first, options);

  REQUIRE_FALSE(result.accepted());
  REQUIRE(result.status == tsd::io::ArchivePlanStatus::MixedAnimationTargets);
  REQUIRE_FALSE(result.message.empty());
}

SCENARIO("tsd::io archive plans reject invalid animation targets",
    "[ArchiveCompatibility]")
{
  tsd::scene::Scene scene;
  tsd::animation::AnimationManager animations(&scene);
  auto root = scene.insertChildNode(scene.defaultLayer()->root(), "dataset");
  animations.addAnimation("invalid").addEmptyObjectParameterBinding();

  tsd::io::ArchivePlanOptions options;
  options.animationManager = &animations;
  const auto result = tsd::io::plan_SubtreeArchive(scene, root, options);

  REQUIRE_FALSE(result.accepted());
  REQUIRE(result.status == tsd::io::ArchivePlanStatus::InvalidAnimationTarget);
}

SCENARIO("tsd::io archive plans reject unsupported file bindings",
    "[ArchiveCompatibility]")
{
  tsd::scene::Scene scene;
  tsd::animation::AnimationManager animations(&scene);
  auto root = scene.insertChildNode(scene.defaultLayer()->root(), "dataset");
  animations.addAnimation("unsupported")
      .emplaceFileBinding<UnsupportedFileBinding>(&scene);

  tsd::io::ArchivePlanOptions options;
  options.animationManager = &animations;
  options.fileBindings = tsd::io::FileBindingArchivePolicy::Omit;
  const auto result = tsd::io::plan_SubtreeArchive(scene, root, options);

  REQUIRE_FALSE(result.accepted());
  REQUIRE(result.status == tsd::io::ArchivePlanStatus::UnsupportedFileBinding);
}

SCENARIO("tsd::io accepts USD file bindings written before continuous time",
    "[ArchiveCompatibility]")
{
  // The `sampleTimes`/`timeBase` pair was a cache of what the Stage already
  // says, and was dropped when bindings started resolving at a Time Code
  // (ADR 0021). Archives that still carry it must keep validating: the fields
  // are ignored, not rejected, and no format version was bumped for them.
  GIVEN("An Animation Archive whose usdGeometry binding carries the old cache")
  {
    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animations(&scene);
    auto geometry = scene.createObject<tsd::scene::Geometry>(
        tsd::scene::tokens::geometry::triangle);

    tsd::core::DataTree tree;
    auto &archive = tree.root();
    archive["name"] = std::string("legacy");
    auto &binding = archive["fileBindings"].append();
    binding["kind"] = std::string("usdGeometry");
    binding["targetIndex"] = geometry->index();
    binding["stageFile"] = std::string("/data/blob.usd");
    binding["primPath"] = std::string("/Blob");
    binding["sampleTimes"].append() = 0.f;
    binding["sampleTimes"].append() = 2.f;
    binding["timeBase"].append() = 0.f;
    binding["timeBase"].append() = 1.f;

    THEN("It still validates against the scene")
    {
      std::string message;
      REQUIRE(tsd::io::validate_AnimationArchive(animations, archive, &message));
    }

    THEN("It deserializes, dropping the fields rather than failing on them")
    {
      auto *restored =
          tsd::io::deserialize_AnimationArchive(animations, archive);
      REQUIRE(restored != nullptr);
      REQUIRE(restored->fileBindings().size() == 1);
      REQUIRE(restored->fileBindings()[0]->kind() == "usdGeometry");

      tsd::core::DataTree rewritten;
      restored->fileBindings()[0]->toDataNode(rewritten.root());
      REQUIRE(rewritten.root().child("stageFile") != nullptr);
      REQUIRE(rewritten.root().child("sampleTimes") == nullptr);
      REQUIRE(rewritten.root().child("timeBase") == nullptr);
    }
  }

  GIVEN("An Animation Archive holding a usdInstancer binding")
  {
    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animations(&scene);
    auto transforms = scene.createArray(ANARI_FLOAT32_MAT4, 2);
    auto node = scene.insertChildTransformArrayNode(
        scene.defaultLayer()->root(), transforms.data(), "swarm");

    tsd::core::DataTree tree;
    auto &archive = tree.root();
    archive["name"] = std::string("swarm");
    auto &binding = archive["fileBindings"].append();
    binding["kind"] = std::string("usdInstancer");
    binding["layerName"] = std::string("default");
    binding["nodeIndex"] = node->index();
    binding["stageFile"] = std::string("/data/swarm.usd");
    binding["primPath"] = std::string("/Swarm");
    binding["prototypeIndex"] = uint64_t(0);

    THEN("The kind is a recognized part of the format")
    {
      std::string message;
      REQUIRE(tsd::io::validate_AnimationArchive(animations, archive, &message));
    }
  }
}

SCENARIO("tsd::io scene exclusion rejects mixed animation ownership",
    "[ArchiveCompatibility]")
{
  tsd::scene::Scene scene;
  tsd::animation::AnimationManager animations(&scene);
  auto first = scene.insertChildTransformNode(
      scene.defaultLayer()->root(), tsd::math::IDENTITY_MAT4, "first");
  auto second = scene.insertChildTransformNode(
      scene.defaultLayer()->root(), tsd::math::IDENTITY_MAT4, "second");
  auto &mixed = animations.addAnimation("mixed");
  mixed.addTransformBinding(first);
  mixed.addTransformBinding(second);

  tsd::io::detail::LegacySceneSerializationOptions options;
  options.animationManager = &animations;
  options.exclusion.roots.push_back(first);
  options.exclusion.objectPolicy = tsd::io::ArchiveObjectPolicy::All;
  options.exclusion.animations =
      tsd::io::detail::LegacyExcludedAnimationPolicy::OmitOwned;
  tsd::core::DataTree tree;
  tsd::io::detail::serializeLegacyScenePayload(scene, tree.root(), options);

  bool sawFirst = false;
  bool sawSecond = false;
  tree.root()["layers"].traverse([&](tsd::core::DataNode &node, int) {
    if (node.name() == "name") {
      const auto name = node.getValueOr<std::string>("");
      sawFirst |= name == "first";
      sawSecond |= name == "second";
    }
    return true;
  });
  REQUIRE(sawFirst);
  REQUIRE(sawSecond);
  REQUIRE(tree.root()["animations"]["objects"].numChildren() == 1);
}

SCENARIO("subtree Archive deserialization exposes exact rollback ownership",
    "[ArchiveCompatibility]")
{
  tsd::scene::Scene source;
  tsd::animation::AnimationManager sourceAnimations(&source);
  auto geometry = source.createObject<tsd::scene::Geometry>(
      tsd::scene::tokens::geometry::sphere);
  auto material = source.createObject<tsd::scene::Material>(
      tsd::scene::tokens::material::matte);
  auto surface = source.createSurface("surface", geometry, material);
  auto root = source.insertChildNode(source.defaultLayer()->root(), "dataset");
  source.insertChildObjectNode(root, surface, "surface");
  const float times[] = {0.f};
  const float values[] = {1.f};
  sourceAnimations.addAnimation("radius").addObjectParameterBinding(
      geometry.data(), "radius", ANARI_FLOAT32, values, times, 1);

  const auto filename = testFile("tsd_subtree_archive_ownership.tsd");
  tsd::io::SubtreeArchiveContentOptions saveOptions;
  saveOptions.animationManager = &sourceAnimations;
  REQUIRE(saveSubtreeArchiveContent(filename.c_str(),
      root,
      {"layer-subtree",
          tsd::io::schema::LAYER_SUBTREE,
          tsd::io::ArchiveObjectPolicy::All},
      {},
      saveOptions));

  tsd::scene::Scene target;
  tsd::animation::AnimationManager targetAnimations(&target);
  tsd::io::SubtreeArchiveContentOptions loadOptions;
  loadOptions.animationManager = &targetAnimations;
  auto loaded = loadSubtreeArchiveContent(target,
      filename.c_str(),
      target.defaultLayer()->root(),
      {"layer-subtree",
          tsd::io::schema::LAYER_SUBTREE,
          tsd::io::ArchiveObjectPolicy::All},
      nullptr,
      loadOptions);

  REQUIRE(loaded.valid());
  REQUIRE(loaded.root);
  REQUIRE(loaded.createdObjects.size() == 3);
  REQUIRE(loaded.createdAnimations == std::vector<size_t>{0});

  tsd::io::rollback_SubtreeArchiveContent(target, targetAnimations, loaded);
  REQUIRE_FALSE(loaded.valid());
  REQUIRE(targetAnimations.animations().empty());
  REQUIRE(target.numberOfObjects(ANARI_GEOMETRY) == 0);
  REQUIRE(target.numberOfObjects(ANARI_MATERIAL) == 1); // Scene default
  REQUIRE(target.numberOfObjects(ANARI_SURFACE) == 0);

  removeTestFile(filename);
}

SCENARIO("tsd::io rejects animations spanning layer subtrees",
    "[ArchiveCompatibility]")
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

  tsd::io::SubtreeArchiveContentOptions options;
  options.animationManager = &animations;
  const auto filename = testFile("tsd_cross_subtree_animation.tsd");
  removeTestFile(filename);

  REQUIRE_FALSE(saveSubtreeArchiveContent(filename.c_str(),
      first,
      {"layer-subtree",
          tsd::io::schema::LAYER_SUBTREE,
          tsd::io::ArchiveObjectPolicy::All},
      {},
      options));
  REQUIRE_FALSE(std::filesystem::exists(filename));
}

SCENARIO("tsd::io validates layer subtree animation targets",
    "[ArchiveCompatibility]")
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
  tsd::io::SubtreeArchiveContentOptions options;
  options.animationManager = &animations;
  REQUIRE(saveSubtreeArchiveContent(filename.c_str(),
      root,
      {"layer-subtree",
          tsd::io::schema::LAYER_SUBTREE,
          tsd::io::ArchiveObjectPolicy::All},
      {},
      options));

  tsd::core::DataTree tree;
  REQUIRE(tree.load(filename.c_str()));
  auto *animation = tree.root()["animations"].child(0);
  REQUIRE(animation);
  auto *binding = (*animation)["objectBindings"].child(0);
  REQUIRE(binding);
  (*binding)["targetIndex"] = size_t(999);
  auto validation = tsd::io::validate_SubtreeArchiveContent(tree.root(),
      {"layer-subtree",
          tsd::io::schema::LAYER_SUBTREE,
          tsd::io::ArchiveObjectPolicy::All});
  REQUIRE_FALSE(validation.accepted());
  REQUIRE_FALSE(validation.message.empty());

  removeTestFile(filename);
}

SCENARIO("legacy project payloads exclude light-rig subtrees",
    "[ArchiveCompatibility]")
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
      tsd::io::detail::LegacySceneSerializationOptions options;
      options.exclusion.roots.push_back(rigRoot);

      tsd::core::DataTree tree;
      tsd::io::detail::serializeLegacyScenePayload(
          source, tree.root(), options);

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
        tsd::io::detail::tryDeserializeLegacyScenePayload(target, tree.root());

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
      tsd::io::detail::LegacySceneSerializationOptions options;
      options.exclusion.roots.push_back(rigRoot);

      tsd::core::DataTree tree;
      tsd::io::detail::serializeLegacyScenePayload(
          source, tree.root(), options);

      tsd::scene::Scene target;
      tsd::io::detail::tryDeserializeLegacyScenePayload(target, tree.root());

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

SCENARIO("legacy project payloads remap animations across exclusion",
    "[ArchiveCompatibility]")
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
      tsd::io::detail::LegacySceneSerializationOptions options;
      options.animationManager = &animMgr;
      options.exclusion.roots.push_back(rig0);

      tsd::core::DataTree tree;
      tsd::io::detail::serializeLegacyScenePayload(
          source, tree.root(), options);

      tsd::scene::Scene target;
      tsd::animation::AnimationManager targetMgr(&target);
      tsd::io::detail::tryDeserializeLegacyScenePayload(
          target, tree.root(), nullptr, &targetMgr);

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

SCENARIO("tsd::io scene exclusion preserves retained animation dependencies",
    "[ArchiveCompatibility]")
{
  tsd::scene::Scene source;
  tsd::animation::AnimationManager animations(&source);

  auto removedMaterial = source.createObject<tsd::scene::Material>(
      tsd::scene::tokens::material::matte);
  removedMaterial->setName("removed");
  auto dependencyMaterial = source.createObject<tsd::scene::Material>(
      tsd::scene::tokens::material::matte);
  dependencyMaterial->setName("dependency");
  auto retainedMaterial = source.createObject<tsd::scene::Material>(
      tsd::scene::tokens::material::matte);
  retainedMaterial->setName("retained");

  auto removedGeometry = source.createObject<tsd::scene::Geometry>(
      tsd::scene::tokens::geometry::sphere);
  auto dependencyGeometry = source.createObject<tsd::scene::Geometry>(
      tsd::scene::tokens::geometry::sphere);
  auto retainedGeometry = source.createObject<tsd::scene::Geometry>(
      tsd::scene::tokens::geometry::sphere);
  auto removedSurface =
      source.createSurface("removed", removedGeometry, removedMaterial);
  auto dependencySurface = source.createSurface(
      "dependency", dependencyGeometry, dependencyMaterial);
  auto retainedSurface =
      source.createSurface("retained", retainedGeometry, retainedMaterial);

  auto excluded =
      source.insertChildNode(source.defaultLayer()->root(), "excluded");
  source.insertChildObjectNode(excluded, removedSurface, "removed");
  source.insertChildObjectNode(excluded, dependencySurface, "dependency");
  source.insertChildObjectNode(
      source.defaultLayer()->root(), retainedSurface, "retained");

  const float times[] = {0.f, 1.f};
  tsd::scene::Object *keyframes[] = {
      dependencyMaterial.data(), retainedMaterial.data()};
  animations.addAnimation("retained animation")
      .addObjectParameterBinding(retainedGeometry.data(),
          "material",
          ANARI_MATERIAL,
          keyframes,
          times,
          2);

  tsd::io::detail::LegacySceneSerializationOptions options;
  options.animationManager = &animations;
  options.exclusion.roots.push_back(excluded);
  options.exclusion.objectPolicy = tsd::io::ArchiveObjectPolicy::All;
  options.exclusion.animations =
      tsd::io::detail::LegacyExcludedAnimationPolicy::OmitOwned;
  tsd::core::DataTree tree;
  tsd::io::detail::serializeLegacyScenePayload(source, tree.root(), options);

  tsd::scene::Scene target;
  tsd::animation::AnimationManager targetAnimations(&target);
  tsd::io::detail::tryDeserializeLegacyScenePayload(
      target, tree.root(), nullptr, &targetAnimations);

  REQUIRE(target.numberOfObjects(ANARI_SURFACE) == 1);
  REQUIRE(target.numberOfObjects(ANARI_MATERIAL) == 3);
  REQUIRE(targetAnimations.animations().size() == 1);
  const auto &binding =
      targetAnimations.animations().front().objectParameterBindings().front();
  const auto *indices = static_cast<const size_t *>(binding.data().data());
  REQUIRE(binding.data().size() == 2);
  REQUIRE(target.getObject(ANARI_MATERIAL, indices[0])->name() == "dependency");
  REQUIRE(target.getObject(ANARI_MATERIAL, indices[1])->name() == "retained");
}
