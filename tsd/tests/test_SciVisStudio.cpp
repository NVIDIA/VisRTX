// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "catch.hpp"

#include "CameraRig.h"
#include "DatasetIO.h"
#include "LightRig.h"
#include "ProjectContext.h"
#include "ProjectSerialization.h"
#include "RenderShotCLI.h"

#include "tsd/app/Context.h"
#include "tsd/core/DataTree.hpp"
#include "tsd/core/DataTreeMetadata.hpp"
#include "tsd/io/animation/SpatialFieldFileBinding.hpp"
#include "tsd/io/serialization/serialization_internal.hpp"
#include "tsd/scene/UpdateDelegate.hpp"
#include "tsd/scene/objects/Geometry.hpp"
#include "tsd/scene/objects/Light.hpp"
#include "tsd/scene/objects/Material.hpp"
#include "tsd/scene/objects/SpatialField.hpp"
#include "tsd/scene/objects/Volume.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <utility>

using namespace tsd::scivis_studio;

namespace {

struct CountingLayerUpdateDelegate : public tsd::scene::EmptyUpdateDelegate
{
  void signalLayerStructureUpdated(const tsd::scene::Layer *l) override
  {
    lastLayer = l;
    layerStructureUpdates++;
  }

  const tsd::scene::Layer *lastLayer{nullptr};
  int layerStructureUpdates{0};
};

tsd::scene::LayerNodeRef findDirectChild(
    tsd::scene::LayerNodeRef parent, const std::string &name)
{
  auto child = parent->next();
  while (child && child != parent) {
    if ((*child)->name() == name)
      return child;
    child = child->sibling();
  }
  return {};
}

} // namespace

SCENARIO(
    "SciVis Studio static dataset assets are self-contained", "[SciVisStudio]")
{
  const auto file = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_static_dataset.tsd";
  std::filesystem::remove(file);

  tsd::scene::Scene source;
  tsd::animation::AnimationManager sourceAnimations(&source);
  auto geometry = source.createObject<tsd::scene::Geometry>(
      tsd::scene::tokens::geometry::sphere);
  geometry->setName("animated geometry");
  auto material = source.createObject<tsd::scene::Material>(
      tsd::scene::tokens::material::matte);
  auto surface = source.createSurface("surface", geometry, material);
  auto root =
      source.insertChildNode(source.defaultLayer()->root(), "dataset_0042");
  source.insertChildObjectNode(root, surface, "surface");

  const float times[] = {0.f, 1.f};
  const float radii[] = {0.5f, 1.5f};
  sourceAnimations.addAnimation("radius").addObjectParameterBinding(
      geometry.data(), "radius", ANARI_FLOAT32, radii, times, 2);

  Dataset dataset;
  dataset.id = "dataset_0042";
  dataset.name = "Example Dataset";
  dataset.sourceKind = DatasetSourceKind::Static;
  dataset.importerType = "OBJ";
  dataset.source.sourcePath = "../source/example.obj";
  dataset.source.importerSettings.set("flatten", "false");
  REQUIRE(exportDatasetAsset(dataset, root, sourceAnimations, file));

  auto validation = validateDatasetAsset(file);
  REQUIRE(validation.ok);
  REQUIRE(validation.dataset.id.empty());
  REQUIRE(validation.dataset.name == "Example Dataset");
  REQUIRE(validation.dataset.source.sourcePath == "../source/example.obj");
  REQUIRE(validation.dataset.source.importerSettings.size() == 1);
  REQUIRE(validation.dataset.source.importerSettings.at_index(0).first
      == "flatten");

  tsd::core::DataTree serialized;
  REQUIRE(serialized.load(file.string().c_str()));
  REQUIRE(serialized.root()["dataset"].child("id") == nullptr);
  REQUIRE(serialized.root()["subtree"]["name"].getValueAs<std::string>()
      == "Example Dataset");

  tsd::scene::Scene target;
  tsd::animation::AnimationManager targetAnimations(&target);
  auto destination =
      target.insertChildNode(target.defaultLayer()->root(), "datasets");
  Dataset imported;
  tsd::scene::LayerNodeRef importedRoot;
  REQUIRE(importDatasetAsset(
      target, targetAnimations, file, destination, imported, importedRoot));
  REQUIRE(importedRoot);
  REQUIRE(imported.status == DatasetStatus::Available);
  REQUIRE(targetAnimations.animations().size() == 1);
  REQUIRE(targetAnimations.animations()
              .front()
              .objectParameterBindings()
              .front()
              .target()
              ->name()
      == "animated geometry");

  Dataset secondImport;
  tsd::scene::LayerNodeRef secondRoot;
  REQUIRE(importDatasetAsset(
      target, targetAnimations, file, destination, secondImport, secondRoot));
  REQUIRE(target.numberOfObjects(ANARI_GEOMETRY) == 2);
  REQUIRE(target.getObject(ANARI_GEOMETRY, 0)
      != target.getObject(ANARI_GEOMETRY, 1));
  // One default material plus one independently loaded material per dataset.
  REQUIRE(target.numberOfObjects(ANARI_MATERIAL) == 3);

  std::filesystem::remove(file);
}

SCENARIO("SciVis Studio file-animation assets preserve opaque source paths",
    "[SciVisStudio]")
{
  const auto file = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_file_animation_dataset.tsd";
  const auto corruptFile = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_file_animation_dataset_corrupt.tsd";
  std::filesystem::remove(file);
  std::filesystem::remove(corruptFile);

  tsd::scene::Scene source;
  tsd::animation::AnimationManager sourceAnimations(&source);
  auto field = source.createObject<tsd::scene::SpatialField>(
      tsd::scene::tokens::spatial_field::structuredRegular);
  auto voxels = source.createArray(ANARI_FLOAT32, 1, 1, 1);
  *voxels->mapAs<float>() = 1.f;
  voxels->unmap();
  field->setParameterObject("data", *voxels);
  auto volume = source.createObject<tsd::scene::Volume>(
      tsd::scene::tokens::volume::transferFunction1D);
  volume->setParameterObject("value", *field);
  auto root =
      source.insertChildNode(source.defaultLayer()->root(), "dataset_0001");
  source.insertChildObjectNode(root, volume, "volume");

  const std::vector<std::string> paths = {
      "relative/frame 01.raw", "/missing/../opaque/frame02.raw"};
  sourceAnimations.addAnimation("runtime file animation")
      .emplaceFileBinding<tsd::io::SpatialFieldFileBinding>(
          &source, volume.data(), field, paths);

  Dataset dataset;
  dataset.id = "dataset_0001";
  dataset.name = "Opaque Frames";
  dataset.sourceKind = DatasetSourceKind::FileAnimation;
  dataset.importerType = "VOLUME_ANIMATION";
  for (const auto &path : paths)
    dataset.sourceFiles.push_back({path});
  Dataset invalidStatic = dataset;
  invalidStatic.sourceKind = DatasetSourceKind::Static;
  invalidStatic.importerType = "VOLUME";
  std::string invalidStaticError;
  REQUIRE_FALSE(exportDatasetAsset(
      invalidStatic, root, sourceAnimations, corruptFile, &invalidStaticError));
  REQUIRE(invalidStaticError.find("cannot own file animations")
      != std::string::npos);
  std::string exportError;
  const bool exported =
      exportDatasetAsset(dataset, root, sourceAnimations, file, &exportError);
  INFO(exportError);
  REQUIRE(exported);

  tsd::core::DataTree corrupt;
  REQUIRE(corrupt.load(file.string().c_str()));
  auto &derived = corrupt.root()["animations"].append();
  derived["name"] = "competing runtime authority";
  auto &fileBinding = derived["fileBindings"].append();
  fileBinding["kind"] = "spatialField";
  fileBinding["targetIndex"] = size_t(0);
  REQUIRE(corrupt.save(corruptFile.string().c_str()));
  auto corruptValidation = validateDatasetAsset(corruptFile);
  REQUIRE_FALSE(corruptValidation.ok);
  REQUIRE(corruptValidation.error.find("derived runtime file bindings")
      != std::string::npos);

  tsd::scene::Scene target;
  tsd::animation::AnimationManager targetAnimations(&target);
  Dataset imported;
  tsd::scene::LayerNodeRef importedRoot;
  REQUIRE(importDatasetAsset(target,
      targetAnimations,
      file,
      target.defaultLayer()->root(),
      imported,
      importedRoot));
  REQUIRE(imported.sourceFiles.size() == paths.size());
  REQUIRE(imported.sourceFiles[0].path == paths[0]);
  REQUIRE(imported.sourceFiles[1].path == paths[1]);
  REQUIRE(targetAnimations.animations().size() == 1);
  REQUIRE(targetAnimations.animations().front().fileBindings().size() == 1);

  tsd::core::DataTree binding;
  targetAnimations.animations().front().fileBindings().front()->toDataNode(
      binding.root());
  REQUIRE(
      binding.root()["files"].child(0)->getValueAs<std::string>() == paths[0]);
  REQUIRE(
      binding.root()["files"].child(1)->getValueAs<std::string>() == paths[1]);

  std::filesystem::remove(file);
  std::filesystem::remove(corruptFile);
}

SCENARIO("SciVis Studio project model serialization", "[SciVisStudio]")
{
  GIVEN("A project with datasets, shots, light rigs, and camera rigs")
  {
    Project project;
    project.name = "RoundTrip";
    project.projectDirectory = "/tmp/roundtrip";
    Dataset dataset;
    dataset.id = "dataset_0001";
    dataset.name = "Dataset";
    dataset.sourceKind = DatasetSourceKind::Static;
    dataset.importerType = "OBJ";
    dataset.source.sourcePath = "/tmp/data.obj";
    dataset.status = DatasetStatus::Available;
    dataset.rootNode = {"studio", 3};
    project.datasets.push_back(std::move(dataset));

    Shot shot;
    shot.id = "shot_0001";
    shot.name = "Shot 1";
    shot.datasetBindings.push_back({"dataset_0001", true});
    shot.lightRigId = "lightRig_0001";
    shot.cameraRigId = "cameraRig_0001";
    shot.camera = {ANARI_CAMERA, 2};
    shot.renderSettings.rendererLibrary = "dummy_test_device";
    shot.renderSettings.rendererObjectIndex = 7;
    shot.renderSettings.rendererSubtype = "dummy_test_renderer";

    CameraRig cameraRig;
    cameraRig.id = "cameraRig_0001";
    cameraRig.name = "Default Camera";
    CameraKeyframe keyframe;
    keyframe.frame = 12;
    keyframe.name = "mid";
    keyframe.manipulator.orbit.lookat = {1.f, 2.f, 3.f};
    keyframe.manipulator.orbit.azeldist = {10.f, 20.f, 30.f};
    keyframe.interpolationToNext = CameraInterpolation::EaseOutIn;
    cameraRig.keyframes.push_back(keyframe);
    project.activeShotId = shot.id;
    project.shots.push_back(shot);
    project.lightRigs.push_back({"lightRig_0001", "Default", {"studio", 5}});
    project.cameraRigs.push_back(std::move(cameraRig));

    tsd::core::DataTree tree;
    projectToNode(project, tree.root()["scivisStudio"]);
    auto &serialized = tree.root()["scivisStudio"];

    REQUIRE(serialized["datasets"].child(0)->child("rootNode") == nullptr);
    REQUIRE(serialized["datasets"].child(0)->child("sourceKind") == nullptr);
    REQUIRE(serialized["datasets"].child(0)->child("sourceFiles") == nullptr);
    REQUIRE(serialized["datasets"].child(0)->child("source") == nullptr);
    REQUIRE(serialized["shots"].child(0)->child("lightRigId") != nullptr);
    REQUIRE(serialized["shots"].child(0)->child("cameraRigId") != nullptr);
    REQUIRE(serialized["shots"].child(0)->child("cameraRig") == nullptr);
    REQUIRE(serialized["shots"].child(0)->child("camera") == nullptr);
    REQUIRE(serialized["lightRigs"].child(0)->child("rootNode") == nullptr);
    // v4: camera-rig keyframe data lives in cameras/<name>.tsd, not the
    // manifest.
    REQUIRE(serialized["cameraRigs"].child(0)->child("rig") == nullptr);
    REQUIRE(serialized["cameraRigs"].child(0)->child("name") != nullptr);

    Project loaded;
    REQUIRE(nodeToProject(serialized, loaded));

    THEN("IDs and bindings survive the manifest round trip")
    {
      REQUIRE(loaded.datasets.size() == 1);
      REQUIRE(loaded.datasets.front().id == "dataset_0001");
      REQUIRE(loaded.datasets.front().name == "Dataset");
      REQUIRE(loaded.datasets.front().sourceFiles.empty());
      REQUIRE(loaded.datasets.front().status == DatasetStatus::Unavailable);
      REQUIRE(loaded.shots.size() == 1);
      REQUIRE(loaded.shots.front().id == "shot_0001");
      REQUIRE(loaded.shots.front().lightRigId == "lightRig_0001");
      REQUIRE(loaded.shots.front().cameraRigId == "cameraRig_0001");
      REQUIRE(loaded.lightRigs.size() == 1);
      REQUIRE(loaded.lightRigs.front().id == "lightRig_0001");
      REQUIRE(loaded.cameraRigs.size() == 1);
      REQUIRE(loaded.cameraRigs.front().id == "cameraRig_0001");
      REQUIRE(loaded.cameraRigs.front().name == "Default Camera");
      REQUIRE(loaded.shots.front().renderSettings.rendererLibrary
          == "dummy_test_device");
      REQUIRE(loaded.shots.front().renderSettings.rendererObjectIndex == 7);
      REQUIRE(loaded.shots.front().renderSettings.rendererSubtype
          == "dummy_test_renderer");
    }
  }
}

SCENARIO(
    "SciVis Studio dataset IDs are not reused after removal", "[SciVisStudio]")
{
  Project project;
  Dataset first;
  first.id = project::nextDatasetId(project);
  project.datasets.push_back(first);
  Dataset second;
  second.id = project::nextDatasetId(project);
  const auto removedId = second.id;
  project.datasets.push_back(second);
  project.datasets.pop_back();

  const auto next = project::nextDatasetId(project);
  REQUIRE(next == "dataset_0003");
  REQUIRE(next != removedId);

  tsd::core::DataTree tree;
  projectToNode(project, tree.root());
  Project loaded;
  REQUIRE(nodeToProject(tree.root(), loaded));
  REQUIRE(project::nextDatasetId(loaded) == "dataset_0004");
}

SCENARIO(
    "SciVis Studio camera rig files round-trip keyframe data", "[SciVisStudio]")
{
  CameraRig rig;
  rig.name = "Hero Cam";
  CameraKeyframe keyframe;
  keyframe.frame = 12;
  keyframe.name = "mid";
  keyframe.manipulator.orbit.lookat = {1.f, 2.f, 3.f};
  keyframe.interpolationToNext = CameraInterpolation::EaseOutIn;
  rig.keyframes.push_back(keyframe);

  const auto file =
      std::filesystem::temp_directory_path() / "tsd_camera_rig_roundtrip.tsd";
  std::filesystem::remove(file);

  REQUIRE(camera_rig::exportCameraRigFile(rig, file));

  CameraRig loaded;
  REQUIRE(camera_rig::importCameraRigFile(file, loaded));

  REQUIRE(loaded.name == "Hero Cam");
  REQUIRE(loaded.keyframes.size() == 1);
  REQUIRE(loaded.keyframes.front().frame == 12);
  REQUIRE(loaded.keyframes.front().interpolationToNext
      == CameraInterpolation::EaseOutIn);

  std::filesystem::remove(file);
}

SCENARIO("SciVis Studio camera interpolation modes", "[SciVisStudio]")
{
  GIVEN("Camera interpolation modes")
  {
    THEN("String conversion round-trips all persisted values")
    {
      const CameraInterpolation modes[] = {CameraInterpolation::Hold,
          CameraInterpolation::Linear,
          CameraInterpolation::EaseOut,
          CameraInterpolation::EaseIn,
          CameraInterpolation::EaseOutIn};

      for (auto mode : modes)
        REQUIRE(camera_rig::interpolationFromString(camera_rig::toString(mode))
            == mode);

      REQUIRE(camera_rig::interpolationFromString("Unknown")
          == CameraInterpolation::Linear);
    }

    THEN("Sampling applies easing to the segment interpolation factor")
    {
      CameraRig rig;

      CameraKeyframe a;
      a.frame = 0;
      a.manipulator.orbit.lookat = {0.f, 0.f, 0.f};
      a.manipulator.orbit.azeldist = {0.f, 0.f, 0.f};
      a.manipulator.orbit.fixedDist = 0.f;

      CameraKeyframe b;
      b.frame = 100;
      b.manipulator.orbit.lookat = {100.f, 0.f, 0.f};
      b.manipulator.orbit.azeldist = {100.f, 0.f, 0.f};
      b.manipulator.orbit.fixedDist = 100.f;

      rig.keyframes = {a, b};

      rig.keyframes.front().interpolationToNext = CameraInterpolation::EaseOut;
      REQUIRE(
          camera_rig::sampleCameraRig(rig, 25).orbit.lookat.x == Approx(6.25f));

      rig.keyframes.front().interpolationToNext = CameraInterpolation::EaseIn;
      REQUIRE(camera_rig::sampleCameraRig(rig, 25).orbit.lookat.x
          == Approx(57.8125f));

      rig.keyframes.front().interpolationToNext =
          CameraInterpolation::EaseOutIn;
      REQUIRE(camera_rig::sampleCameraRig(rig, 25).orbit.lookat.x
          == Approx(10.3515625f));
      REQUIRE(camera_rig::sampleCameraRig(rig, 25).orbit.azeldist.x
          == Approx(10.3515625f));
      REQUIRE(camera_rig::sampleCameraRig(rig, 25).orbit.fixedDist
          == Approx(10.3515625f));
      REQUIRE(camera_rig::sampleCameraRig(rig, 75).orbit.lookat.x
          == Approx(89.6484375f));
    }
  }
}

SCENARIO("SciVis Studio project root validation", "[SciVisStudio]")
{
  const auto root =
      std::filesystem::temp_directory_path() / "tsd_scivis_studio_test_project";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);

  GIVEN("A valid metadata-tagged project manifest")
  {
    tsd::core::DataTree tree;
    tsd::core::writeDataTreeMetadata(tree.root(),
        {tsd::core::DATA_TREE_METADATA_ENVELOPE_VERSION,
            PROJECT_FILE_TYPE,
            PROJECT_SCHEMA,
            SCHEMA_VERSION});
    REQUIRE(tree.save((root / PROJECT_MANIFEST_FILENAME).string().c_str()));

    THEN("Validation succeeds")
    {
      auto result = validateProjectRoot(root);
      REQUIRE(result.ok);
    }
  }

  GIVEN("A valid legacy project manifest")
  {
    tsd::core::DataTree tree;
    tree.root()["projectKind"] = PROJECT_KIND;
    tree.root()["schemaVersion"] = 1;
    REQUIRE(tree.save((root / PROJECT_MANIFEST_FILENAME).string().c_str()));

    THEN("Validation succeeds")
    {
      auto result = validateProjectRoot(root);
      REQUIRE(result.ok);
    }
  }

  GIVEN("An invalid metadata schema")
  {
    tsd::core::DataTree tree;
    tsd::core::writeDataTreeMetadata(tree.root(),
        {tsd::core::DATA_TREE_METADATA_ENVELOPE_VERSION,
            "application-state",
            "tsd.viewer.state",
            1});
    REQUIRE(tree.save((root / PROJECT_MANIFEST_FILENAME).string().c_str()));

    THEN("Validation fails")
    {
      auto result = validateProjectRoot(root);
      REQUIRE_FALSE(result.ok);
    }
  }

  GIVEN("An invalid legacy project kind")
  {
    tsd::core::DataTree tree;
    tree.root()["projectKind"] = "Other";
    tree.root()["schemaVersion"] = SCHEMA_VERSION;
    REQUIRE(tree.save((root / PROJECT_MANIFEST_FILENAME).string().c_str()));

    THEN("Validation fails")
    {
      auto result = validateProjectRoot(root);
      REQUIRE_FALSE(result.ok);
    }
  }

  GIVEN("A future metadata-tagged project manifest")
  {
    tsd::core::DataTree tree;
    tsd::core::writeDataTreeMetadata(tree.root(),
        {tsd::core::DATA_TREE_METADATA_ENVELOPE_VERSION,
            PROJECT_FILE_TYPE,
            PROJECT_SCHEMA,
            SCHEMA_VERSION + 1});
    REQUIRE(tree.save((root / PROJECT_MANIFEST_FILENAME).string().c_str()));

    THEN("Validation fails")
    {
      auto result = validateProjectRoot(root);
      REQUIRE_FALSE(result.ok);
    }
  }

  std::filesystem::remove_all(root);
}

SCENARIO("SciVis Studio default project creation", "[SciVisStudio]")
{
  tsd::app::Context appContext;
  ProjectContext projectContext(&appContext);
  projectContext.createUnsavedProject();

  auto &project = projectContext.project();
  REQUIRE(project.name == "Untitled");
  REQUIRE(project.shots.size() == 1);
  REQUIRE(project.lightRigs.size() == 1);
  REQUIRE(project.lightRigs.front().name == "Default");
  REQUIRE(project.shots.front().lightRigId == project.lightRigs.front().id);
  REQUIRE(project.cameraRigs.size() == 1);
  REQUIRE(project.cameraRigs.front().name == "Default");
  REQUIRE(project.shots.front().cameraRigId == project.cameraRigs.front().id);
  REQUIRE(project.activeShotId == project.shots.front().id);
  REQUIRE(project.dirty == false);
  REQUIRE(appContext.tsd.scene.layer("studio") != nullptr);

  auto *layer = appContext.tsd.scene.layer("studio");
  auto lightRigsRoot = findDirectChild(layer->root(), "lightRigs");
  REQUIRE(lightRigsRoot);
  auto rigRoot = findDirectChild(lightRigsRoot, project.lightRigs.front().id);
  REQUIRE(rigRoot);
  REQUIRE(findDirectChild(rigRoot, "mainLight"));
}

SCENARIO("SciVis Studio new shots use the default light rig", "[SciVisStudio]")
{
  tsd::app::Context appContext;
  ProjectContext projectContext(&appContext);
  projectContext.createUnsavedProject();

  const auto defaultRigId = projectContext.project().lightRigs.front().id;
  const auto defaultCameraRigId =
      projectContext.project().cameraRigs.front().id;
  REQUIRE(projectContext.addShot());
  REQUIRE(project::activeShot(projectContext.project())->lightRigId
      == defaultRigId);
  REQUIRE(project::activeShot(projectContext.project())->cameraRigId
      == defaultCameraRigId);
}

SCENARIO(
    "SciVis Studio cloning a light rig deep-copies lights", "[SciVisStudio]")
{
  tsd::app::Context appContext;
  ProjectContext projectContext(&appContext);
  projectContext.createUnsavedProject();

  auto &project = projectContext.project();
  const auto sourceRigId = project.lightRigs.front().id;
  auto *sourceRig = light_rig::findLightRig(project, sourceRigId);
  REQUIRE(sourceRig != nullptr);

  auto sourceRoot = projectContext.resolveLightRigRoot(*sourceRig);
  REQUIRE(sourceRoot);
  auto sourceLightNode = findDirectChild(sourceRoot, "mainLight");
  REQUIRE(sourceLightNode);
  auto *sourceLight =
      dynamic_cast<tsd::scene::Light *>((*sourceLightNode)->getObject());
  REQUIRE(sourceLight != nullptr);
  sourceLight->setParameter("irradiance", 2.f);

  auto *cloneRig = projectContext.cloneLightRig(sourceRigId);
  REQUIRE(cloneRig != nullptr);
  REQUIRE(project.lightRigs.size() == 2);
  REQUIRE(cloneRig->id != sourceRigId);
  REQUIRE(cloneRig->name == "Default Copy");
  REQUIRE(project.shots.front().lightRigId == sourceRigId);

  auto cloneRoot = projectContext.resolveLightRigRoot(*cloneRig);
  REQUIRE(cloneRoot);
  auto cloneLightNode = findDirectChild(cloneRoot, "mainLight");
  REQUIRE(cloneLightNode);
  auto *cloneLight =
      dynamic_cast<tsd::scene::Light *>((*cloneLightNode)->getObject());
  REQUIRE(cloneLight != nullptr);
  REQUIRE(cloneLight != (*sourceLightNode)->getObject());
  REQUIRE(
      cloneLight->parameterValueAs<float>("irradiance").value() == Approx(2.f));

  cloneLight->setParameter("irradiance", 7.f);
  sourceLight =
      dynamic_cast<tsd::scene::Light *>((*sourceLightNode)->getObject());
  REQUIRE(sourceLight != nullptr);
  REQUIRE(sourceLight->parameterValueAs<float>("irradiance").value()
      == Approx(2.f));
}

SCENARIO("SciVis Studio shot dataset bindings update scene visibility",
    "[SciVisStudio]")
{
  tsd::app::Context appContext;
  ProjectContext projectContext(&appContext);
  projectContext.createUnsavedProject();

  auto &scene = appContext.tsd.scene;
  auto *layer = scene.layer("studio");
  REQUIRE(layer != nullptr);

  auto datasetRoot = scene.insertChildNode(layer->root(), "dataset_0001");
  REQUIRE(datasetRoot);

  auto &project = projectContext.project();
  project.datasets.push_back({"dataset_0001",
      "Dataset",
      DatasetSourceKind::Static,
      "OBJ",
      {},
      DatasetStatus::Available,
      projectContext.refFor("studio", datasetRoot)});

  auto &shot = *project::activeShot(project);
  shot::setDatasetBinding(shot, "dataset_0001", false);

  auto *delegate =
      scene.updateDelegate().emplace<CountingLayerUpdateDelegate>();

  projectContext.applyActiveShot();

  REQUIRE_FALSE((*datasetRoot)->isEnabled());
  REQUIRE(delegate->layerStructureUpdates == 1);
  REQUIRE(delegate->lastLayer == layer);
}

SCENARIO("SciVis Studio dataset binding resolves the dataset group by ID",
    "[SciVisStudio]")
{
  tsd::app::Context appContext;
  ProjectContext projectContext(&appContext);
  projectContext.createUnsavedProject();

  auto &scene = appContext.tsd.scene;
  auto *layer = scene.layer("studio");
  REQUIRE(layer != nullptr);

  auto datasetsRoot = findDirectChild(layer->root(), "datasets");
  REQUIRE(datasetsRoot);
  auto datasetRoot = scene.insertChildNode(datasetsRoot, "dataset_0001");
  auto importedFileRoot = scene.insertChildNode(datasetRoot, "imported.vtp");
  auto partRoot = scene.insertChildNode(importedFileRoot, "part_1");

  auto &project = projectContext.project();
  project.datasets.push_back({"dataset_0001",
      "Dataset",
      DatasetSourceKind::Static,
      "VTP",
      {},
      DatasetStatus::Available,
      projectContext.refFor("studio", partRoot)});

  auto &shot = *project::activeShot(project);
  shot::setDatasetBinding(shot, "dataset_0001", false);

  projectContext.applyActiveShot();

  REQUIRE_FALSE((*datasetRoot)->isEnabled());
  REQUIRE((*importedFileRoot)->isEnabled());
  REQUIRE((*partRoot)->isEnabled());
  REQUIRE(project.datasets.front().rootNode.nodeIndex == datasetRoot.index());
}

SCENARIO("SciVis Studio saved projects rebuild runtime refs from stable IDs",
    "[SciVisStudio]")
{
  const auto root =
      std::filesystem::temp_directory_path() / "tsd_scivis_studio_runtime_refs";
  std::filesystem::remove_all(root);

  {
    tsd::app::Context appContext;
    ProjectContext projectContext(&appContext);
    projectContext.createUnsavedProject();

    auto &scene = appContext.tsd.scene;
    auto *layer = scene.layer("studio");
    REQUIRE(layer != nullptr);
    auto datasetsRoot = findDirectChild(layer->root(), "datasets");
    REQUIRE(datasetsRoot);
    auto datasetRoot = scene.insertChildNode(datasetsRoot, "dataset_0001");
    scene.insertChildNode(datasetRoot, "imported.vtp");

    auto &project = projectContext.project();
    project.datasets.push_back({"dataset_0001",
        "Dataset",
        DatasetSourceKind::Static,
        "VTP",
        {},
        DatasetStatus::Available,
        projectContext.refFor("studio", datasetRoot)});
    shot::setDatasetBinding(
        *project::activeShot(project), "dataset_0001", false);

    REQUIRE(projectContext.saveProject(root));
  }

  {
    tsd::core::DataTree manifest;
    REQUIRE(manifest.load((root / PROJECT_MANIFEST_FILENAME).string().c_str()));
    auto metadata = tsd::core::readDataTreeMetadata(manifest.root());
    REQUIRE(metadata.status == tsd::core::DataTreeMetadataReadStatus::Found);
    REQUIRE(metadata.metadata);
    REQUIRE(metadata.metadata->fileType == PROJECT_FILE_TYPE);
    REQUIRE(metadata.metadata->schema == PROJECT_SCHEMA);
    REQUIRE(metadata.metadata->schemaVersion == SCHEMA_VERSION);
    REQUIRE(manifest.root().child("projectKind") == nullptr);
    REQUIRE(manifest.root().child("schemaVersion") == nullptr);

    auto &projectNode = manifest.root()["scivisStudio"];
    REQUIRE(projectNode["datasets"].child(0)->child("rootNode") == nullptr);
    REQUIRE(projectNode["shots"].child(0)->child("lightRigId") != nullptr);
    REQUIRE(projectNode["shots"].child(0)->child("cameraRigId") != nullptr);
    REQUIRE(projectNode["shots"].child(0)->child("cameraRig") == nullptr);
    REQUIRE(projectNode["shots"].child(0)->child("camera") == nullptr);
    REQUIRE(projectNode["lightRigs"].child(0)->child("rootNode") == nullptr);
    // v4: rig value data moved to standalone files.
    REQUIRE(projectNode["cameraRigs"].child(0)->child("rig") == nullptr);

    // The manifest's context must not carry the excluded light pool.
    REQUIRE(manifest.root()["context"]["objectDB"].child("light") == nullptr);

    // Each rig was written as its own portable file.
    const auto cameraName = projectNode["cameraRigs"]
                                .child(0)
                                ->child("name")
                                ->getValueAs<std::string>();
    const auto lightName = projectNode["lightRigs"]
                               .child(0)
                               ->child("name")
                               ->getValueAs<std::string>();
    REQUIRE(std::filesystem::exists(root / "cameras" / (cameraName + ".tsd")));
    REQUIRE(std::filesystem::exists(root / "lights" / (lightName + ".tsd")));
  }

  {
    tsd::app::Context appContext;
    ProjectContext projectContext(&appContext);
    REQUIRE(projectContext.openProject(root));
    REQUIRE(projectContext.project().cameraRigs.size() == 1);
    REQUIRE(projectContext.project().shots.front().cameraRigId
        == projectContext.project().cameraRigs.front().id);

    auto *layer = appContext.tsd.scene.layer("studio");
    REQUIRE(layer != nullptr);
    auto datasetsRoot = findDirectChild(layer->root(), "datasets");
    auto datasetRoot = findDirectChild(datasetsRoot, "dataset_0001");
    REQUIRE(datasetRoot);
    REQUIRE_FALSE((*datasetRoot)->isEnabled());
  }

  std::filesystem::remove_all(root);
}

SCENARIO("SciVis Studio v5 projects round-trip standalone datasets",
    "[SciVisStudio]")
{
  const auto root =
      std::filesystem::temp_directory_path() / "tsd_scivis_studio_v5_datasets";
  const auto copy = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_v5_datasets_copy";
  std::filesystem::remove_all(root);
  std::filesystem::remove_all(copy);

  {
    tsd::app::Context appContext;
    ProjectContext projectContext(&appContext);
    projectContext.createUnsavedProject();
    auto &project = projectContext.project();
    auto &scene = appContext.tsd.scene;
    auto *studio = scene.layer("studio");
    auto datasetsRoot = findDirectChild(studio->root(), "datasets");
    auto datasetRoot = scene.insertChildNode(datasetsRoot, "dataset_0001");

    auto geometry = scene.createObject<tsd::scene::Geometry>(
        tsd::scene::tokens::geometry::sphere);
    geometry->setName("dataset geometry");
    auto material = scene.createObject<tsd::scene::Material>(
        tsd::scene::tokens::material::matte);
    auto surface = scene.createSurface("dataset surface", geometry, material);
    scene.insertChildObjectNode(datasetRoot, surface, "surface");

    const float times[] = {0.f, 1.f};
    const float radii[] = {0.5f, 1.5f};
    appContext.tsd.animationMgr.addAnimation("dataset radius")
        .addObjectParameterBinding(
            geometry.data(), "radius", ANARI_FLOAT32, radii, times, 2);

    Dataset dataset;
    dataset.id = "dataset_0001";
    dataset.name = "Example";
    dataset.sourceKind = DatasetSourceKind::Static;
    dataset.importerType = "OBJ";
    dataset.source.sourcePath = "/source/example.obj";
    dataset.status = DatasetStatus::Available;
    dataset.rootNode = projectContext.refFor("studio", datasetRoot);
    project.datasets.push_back(std::move(dataset));
    shot::setDatasetBinding(project.shots.front(), "dataset_0001", false);
    project.markDirty();

    REQUIRE(projectContext.saveProject(root));
    REQUIRE(std::filesystem::exists(root / "datasets" / "Example.tsd"));
  }

  {
    tsd::core::DataTree manifest;
    REQUIRE(manifest.load((root / PROJECT_MANIFEST_FILENAME).string().c_str()));
    auto metadata = tsd::core::readDataTreeMetadata(manifest.root());
    REQUIRE(metadata.found());
    REQUIRE(metadata.metadata->schemaVersion == 5);
    auto *datasetNode = manifest.root()["scivisStudio"]["datasets"].child(0);
    REQUIRE(datasetNode);
    REQUIRE(datasetNode->child("id"));
    REQUIRE(datasetNode->child("name"));
    REQUIRE(datasetNode->child("sourceKind") == nullptr);
    REQUIRE(datasetNode->child("source") == nullptr);
    REQUIRE(datasetNode->child("sourceFiles") == nullptr);
    REQUIRE(manifest.root()["context"]["objectDB"].child("surface") == nullptr);
    REQUIRE(
        manifest.root()["context"]["animations"]["objects"].numChildren() == 0);
  }

  {
    tsd::app::Context appContext;
    ProjectContext projectContext(&appContext);
    REQUIRE(projectContext.openProject(root));
    auto &project = projectContext.project();
    REQUIRE(project.datasets.size() == 1);
    REQUIRE(project.datasets.front().id == "dataset_0001");
    REQUIRE(project.datasets.front().status == DatasetStatus::Available);
    REQUIRE(
        project.datasets.front().source.sourcePath == "/source/example.obj");
    REQUIRE_FALSE(project.shots.front().datasetBindings.front().enabled);
    REQUIRE(projectContext.resolveDatasetRoot(project.datasets.front()));
    REQUIRE(appContext.tsd.animationMgr.animations().size() == 1);
    REQUIRE(appContext.tsd.animationMgr.animations().front().name()
        == "dataset radius");

    const auto asset = root / "datasets" / "Example.tsd";
    const auto sentinel =
        std::filesystem::file_time_type::clock::now() - std::chrono::hours(1);
    std::filesystem::last_write_time(asset, sentinel);
    project.shots.front().name = "Unrelated Shot Edit";
    project.markDirty();
    REQUIRE(projectContext.saveProject(root));
    REQUIRE(std::filesystem::last_write_time(asset) == sentinel);
    appContext.tsd.animationMgr.setAnimationTime(0.5f);
    REQUIRE_FALSE(project.datasets.front().dirty);

    auto *geometry = appContext.tsd.animationMgr.animations()
                         .front()
                         .objectParameterBindings()
                         .front()
                         .target();
    REQUIRE(geometry);
    geometry->setParameter("radius", 3.f);
    REQUIRE(project.datasets.front().dirty);
    REQUIRE(projectContext.saveProject(root));
    REQUIRE(std::filesystem::last_write_time(asset) != sentinel);

    REQUIRE(projectContext.saveProject(copy));
    REQUIRE(std::filesystem::exists(copy / "datasets" / "Example.tsd"));
    auto copied = validateDatasetAsset(copy / "datasets" / "Example.tsd");
    REQUIRE(copied.ok);
    REQUIRE(copied.dataset.source.sourcePath == "/source/example.obj");
  }
  std::filesystem::remove_all(copy);

  std::filesystem::remove(root / "datasets" / "Example.tsd");
  {
    tsd::app::Context appContext;
    ProjectContext projectContext(&appContext);
    REQUIRE(projectContext.openProject(root));
    auto &project = projectContext.project();
    REQUIRE(project.datasets.size() == 1);
    REQUIRE(project.datasets.front().status == DatasetStatus::Unavailable);
    REQUIRE(project.shots.front().datasetBindings.size() == 1);
    REQUIRE(project.shots.front().datasetBindings.front().datasetId
        == "dataset_0001");
  }

  std::filesystem::remove_all(root);
}

SCENARIO("SciVis Studio extracts embedded v4 datasets only on save",
    "[SciVisStudio]")
{
  const auto root = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_v4_dataset_migration";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);

  {
    tsd::app::Context appContext;
    ProjectContext projectContext(&appContext);
    projectContext.createUnsavedProject();
    auto &project = projectContext.project();
    auto &scene = appContext.tsd.scene;
    auto datasetsRoot =
        findDirectChild(scene.layer("studio")->root(), "datasets");
    auto datasetRoot = scene.insertChildNode(datasetsRoot, "dataset_0042");
    auto geometry = scene.createObject<tsd::scene::Geometry>(
        tsd::scene::tokens::geometry::sphere);
    auto material = scene.createObject<tsd::scene::Material>(
        tsd::scene::tokens::material::matte);
    auto surface = scene.createSurface("legacy surface", geometry, material);
    scene.insertChildObjectNode(datasetRoot, surface, "surface");

    Dataset dataset;
    dataset.id = "dataset_0042";
    dataset.name = "Legacy Dataset";
    dataset.status = DatasetStatus::Available;
    dataset.rootNode = projectContext.refFor("studio", datasetRoot);
    project.datasets.push_back(dataset);
    shot::setDatasetBinding(project.shots.front(), dataset.id, true);

    tsd::core::DataTree tree;
    tsd::core::writeDataTreeMetadata(tree.root(),
        {tsd::core::DATA_TREE_METADATA_ENVELOPE_VERSION,
            PROJECT_FILE_TYPE,
            PROJECT_SCHEMA,
            4});
    projectToNode(project, tree.root()["scivisStudio"]);
    auto *datasetNode = tree.root()["scivisStudio"]["datasets"].child(0);
    REQUIRE(datasetNode);
    (*datasetNode)["sourceKind"] = "Static";
    (*datasetNode)["importerType"] = "OBJ";
    (*datasetNode)["status"] = "Available";
    (*datasetNode)["source"]["absolutePath"] = "/legacy/source.obj";
    tsd::io::save_Scene(
        scene, tree.root()["context"], false, &appContext.tsd.animationMgr);
    REQUIRE(tree.save((root / PROJECT_MANIFEST_FILENAME).string().c_str()));
  }

  {
    tsd::app::Context appContext;
    ProjectContext projectContext(&appContext);
    REQUIRE(projectContext.openProject(root));
    auto &dataset = projectContext.project().datasets.front();
    REQUIRE(dataset.id == "dataset_0042");
    REQUIRE(dataset.pendingExtraction);
    REQUIRE(dataset.status == DatasetStatus::Available);
    REQUIRE(dataset.source.sourcePath == "/legacy/source.obj");
    REQUIRE_FALSE(std::filesystem::exists(root / "datasets"));

    REQUIRE(projectContext.saveProject(root));
    REQUIRE(std::filesystem::exists(root / "datasets" / "Legacy Dataset.tsd"));
  }

  {
    tsd::core::DataTree manifest;
    REQUIRE(manifest.load((root / PROJECT_MANIFEST_FILENAME).string().c_str()));
    auto metadata = tsd::core::readDataTreeMetadata(manifest.root());
    REQUIRE(metadata.found());
    REQUIRE(metadata.metadata->schemaVersion == 5);
    auto *datasetNode = manifest.root()["scivisStudio"]["datasets"].child(0);
    REQUIRE(datasetNode);
    REQUIRE(datasetNode->child("sourceKind") == nullptr);
    REQUIRE(datasetNode->child("source") == nullptr);
  }

  {
    tsd::app::Context appContext;
    ProjectContext projectContext(&appContext);
    REQUIRE(projectContext.openProject(root));
    auto &project = projectContext.project();
    REQUIRE(project.datasets.front().id == "dataset_0042");
    REQUIRE(project.datasets.front().status == DatasetStatus::Available);
    REQUIRE(project.shots.front().datasetBindings.front().datasetId
        == "dataset_0042");
  }

  std::filesystem::remove_all(root);
}

SCENARIO("SciVis Studio Save As reports unavailable datasets", "[SciVisStudio]")
{
  const auto root = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_unavailable_source";
  const auto destination = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_unavailable_destination";
  std::filesystem::remove_all(root);
  std::filesystem::remove_all(destination);
  std::filesystem::create_directories(root);

  Project project;
  project.name = "Unavailable";
  project.projectDirectory = root;
  Dataset dataset;
  dataset.id = "dataset_0007";
  dataset.name = "Missing Dataset";
  dataset.dirty = false;
  dataset.persistedName = dataset.name;
  project.datasets.push_back(dataset);
  Shot shot;
  shot.id = "shot_0001";
  shot.datasetBindings.push_back({dataset.id, true});
  project.shots.push_back(shot);
  project.activeShotId = shot.id;

  tsd::core::DataTree tree;
  tsd::core::writeDataTreeMetadata(tree.root(),
      {tsd::core::DATA_TREE_METADATA_ENVELOPE_VERSION,
          PROJECT_FILE_TYPE,
          PROJECT_SCHEMA,
          SCHEMA_VERSION});
  projectToNode(project, tree.root()["scivisStudio"]);
  tsd::scene::Scene scene;
  tsd::animation::AnimationManager animations(&scene);
  tsd::io::save_Scene(scene, tree.root()["context"], false, &animations);
  REQUIRE(tree.save((root / PROJECT_MANIFEST_FILENAME).string().c_str()));

  tsd::app::Context appContext;
  ProjectContext projectContext(&appContext);
  REQUIRE(projectContext.openProject(root));
  REQUIRE(projectContext.project().datasets.front().status
      == DatasetStatus::Unavailable);
  REQUIRE(projectContext.saveProject(root));

  std::string error;
  REQUIRE_FALSE(
      projectContext.saveProject(destination, nullptr, "", nullptr, &error));
  REQUIRE(error.find("Missing Dataset") != std::string::npos);
  REQUIRE_FALSE(
      std::filesystem::exists(destination / PROJECT_MANIFEST_FILENAME));

  std::filesystem::remove_all(root);
  std::filesystem::remove_all(destination);
}

SCENARIO("SciVis Studio dataset lifecycle workflows preserve asset semantics",
    "[SciVisStudio]")
{
  const auto root = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_dataset_lifecycle";
  const auto source = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_dataset_lifecycle.obj";
  const auto exported = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_dataset_export.tsd";
  std::filesystem::remove_all(root);
  std::filesystem::remove(source);
  std::filesystem::remove(exported);
  {
    std::ofstream obj(source);
    obj << "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
  }

  tsd::app::Context appContext;
  ProjectContext projectContext(&appContext);
  projectContext.createUnsavedProject();
  auto *dataset = projectContext.addStaticDataset(
      "Mesh", source, tsd::io::ImporterType::OBJ);
  REQUIRE(dataset);
  REQUIRE(dataset->status == DatasetStatus::Available);
  const auto originalId = dataset->id;
  REQUIRE(projectContext.saveProject(root));
  REQUIRE(std::filesystem::exists(root / "datasets" / "Mesh.tsd"));

  REQUIRE(projectContext.exportDataset(originalId, exported));
  auto exportedValidation = validateDatasetAsset(exported);
  REQUIRE(exportedValidation.ok);
  REQUIRE(exportedValidation.dataset.id.empty());

  std::string error;
  auto originalRoot = projectContext.resolveDatasetRoot(*dataset);
  REQUIRE(originalRoot);
  std::filesystem::remove(source);
  REQUIRE_FALSE(projectContext.reimportStaticDataset(originalId, &error));
  REQUIRE(projectContext.resolveDatasetRoot(*dataset) == originalRoot);

  {
    std::ofstream obj(source);
    obj << "v 0 0 0\nv 2 0 0\nv 0 2 0\nf 1 2 3\n";
  }
  REQUIRE(projectContext.reimportStaticDataset(originalId, &error));
  REQUIRE(projectContext.project().datasets.front().id == originalId);

  REQUIRE(projectContext.renameDataset(originalId, "Renamed", &error));
  REQUIRE_FALSE(projectContext.renameDataset(originalId, "bad/name", &error));
  REQUIRE(projectContext.saveProject(root));
  REQUIRE(std::filesystem::exists(root / "datasets" / "Renamed.tsd"));
  REQUIRE_FALSE(std::filesystem::exists(root / "datasets" / "Mesh.tsd"));

  REQUIRE(projectContext.removeDataset(originalId, true, &error));
  REQUIRE(std::filesystem::exists(root / "datasets" / "Renamed.tsd"));
  REQUIRE(projectContext.project().shots.front().datasetBindings.empty());

  // A generic TSD scene in the flat directory is not a dataset candidate.
  tsd::scene::Scene genericScene;
  tsd::io::save_Scene(
      genericScene, (root / "datasets" / "generic.tsd").string().c_str());
  REQUIRE(projectContext.saveProject(root));
  REQUIRE(std::filesystem::exists(root / "datasets" / "Renamed.tsd"));
  REQUIRE(std::filesystem::exists(root / "datasets" / "generic.tsd"));
  auto candidates = projectContext.discoverDatasetCandidates();
  REQUIRE(candidates.size() == 1);
  REQUIRE(candidates.front().proposedName == "Renamed");

  auto *incorporated = projectContext.incorporateDatasetCandidate(
      candidates.front(), "Renamed", &error);
  REQUIRE(incorporated);
  REQUIRE(incorporated->id != originalId);
  REQUIRE_FALSE(incorporated->dirty);
  const auto incorporatedId = incorporated->id;
  REQUIRE(projectContext.removeDataset(incorporatedId, false, &error));
  REQUIRE_FALSE(std::filesystem::exists(root / "datasets" / "Renamed.tsd"));
  REQUIRE(std::filesystem::exists(root / "datasets" / "generic.tsd"));

  auto *imported = projectContext.importDataset(exported, &error);
  REQUIRE(imported);
  REQUIRE(imported->id != originalId);
  REQUIRE(imported->dirty);

  std::filesystem::remove_all(root);
  std::filesystem::remove(source);
  std::filesystem::remove(exported);
}

SCENARIO("SciVis Studio stages every dirty dataset before replacement",
    "[SciVisStudio]")
{
  const auto root = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_dataset_staging";
  std::filesystem::remove_all(root);

  tsd::app::Context appContext;
  ProjectContext projectContext(&appContext);
  projectContext.createUnsavedProject();
  auto &scene = appContext.tsd.scene;
  auto datasetsRoot =
      findDirectChild(scene.layer("studio")->root(), "datasets");

  std::vector<tsd::scene::Geometry *> geometries;
  for (int i = 0; i < 2; ++i) {
    const auto id = project::makeGeneratedId("dataset", i + 1);
    const auto name = i == 0 ? std::string("First") : std::string("Second");
    auto datasetRoot = scene.insertChildNode(datasetsRoot, id.c_str());
    auto geometry = scene.createObject<tsd::scene::Geometry>(
        tsd::scene::tokens::geometry::sphere);
    auto material = scene.createObject<tsd::scene::Material>(
        tsd::scene::tokens::material::matte);
    auto surface = scene.createSurface(name.c_str(), geometry, material);
    scene.insertChildObjectNode(datasetRoot, surface, "surface");
    geometries.push_back(geometry.data());

    Dataset dataset;
    dataset.id = id;
    dataset.name = name;
    dataset.sourceKind = DatasetSourceKind::Static;
    dataset.importerType = "OBJ";
    dataset.status = DatasetStatus::Available;
    dataset.rootNode = projectContext.refFor("studio", datasetRoot);
    projectContext.project().datasets.push_back(std::move(dataset));
  }
  REQUIRE(projectContext.saveProject(root));
  geometries.clear();
  geometries.push_back(
      static_cast<tsd::scene::Geometry *>(scene.getObject(ANARI_GEOMETRY, 0)));
  geometries.push_back(
      static_cast<tsd::scene::Geometry *>(scene.getObject(ANARI_GEOMETRY, 1)));
  REQUIRE(geometries[0]);
  REQUIRE(geometries[1]);

  const auto firstAsset = root / "datasets" / "First.tsd";
  auto readBytes = [](const std::filesystem::path &file) {
    std::ifstream input(file, std::ios::binary);
    return std::vector<char>(std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
  };
  const auto before = readBytes(firstAsset);
  REQUIRE_FALSE(before.empty());

  geometries[0]->setParameter("radius", 7.f);
  projectContext.project().datasets[0].dirty = true;
  auto proxy = scene.createArrayProxy(ANARI_FLOAT32, 4);
  geometries[1]->setParameterObject("invalid.proxy", *proxy);
  projectContext.project().datasets[1].dirty = true;

  std::string error;
  REQUIRE_FALSE(projectContext.saveProject(root, nullptr, "", nullptr, &error));
  REQUIRE(error.find("Second") != std::string::npos);
  REQUIRE(readBytes(firstAsset) == before);
  for (const auto &entry :
      std::filesystem::directory_iterator(root / "datasets")) {
    REQUIRE(
        entry.path().filename().string().find(".stage-") == std::string::npos);
  }

  std::filesystem::remove_all(root);
}

SCENARIO("SciVis Studio save collisions leave files and live names unchanged",
    "[SciVisStudio]")
{
  const auto root = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_save_collision";
  const auto destination = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_save_as_collision";
  const auto source = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_save_collision.obj";
  std::filesystem::remove_all(root);
  std::filesystem::remove_all(destination);
  std::filesystem::remove(source);
  {
    std::ofstream obj(source);
    obj << "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
  }

  tsd::app::Context appContext;
  ProjectContext projectContext(&appContext);
  projectContext.createUnsavedProject();
  auto *dataset = projectContext.addStaticDataset(
      "Mesh", source, tsd::io::ImporterType::OBJ);
  REQUIRE(dataset);
  REQUIRE(projectContext.saveProject(root));

  auto readBytes = [](const std::filesystem::path &file) {
    std::ifstream input(file, std::ios::binary);
    return std::vector<char>(std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
  };
  const auto manifest = root / PROJECT_MANIFEST_FILENAME;
  const auto managed = root / "datasets/Mesh.tsd";
  const auto collision = root / "datasets/bad_name.tsd";
  std::filesystem::copy_file(managed, collision);
  const auto manifestBefore = readBytes(manifest);
  const auto managedBefore = readBytes(managed);
  const auto collisionBefore = readBytes(collision);

  dataset = &projectContext.project().datasets.front();
  dataset->name = "bad/name";
  dataset->dirty = true;
  projectContext.project().markDirty();
  std::string error;
  REQUIRE_FALSE(projectContext.saveProject(root, nullptr, "", nullptr, &error));
  REQUIRE(error.find("unowned target") != std::string::npos);
  REQUIRE(dataset->name == "bad/name");
  REQUIRE(dataset->dirty);
  REQUIRE(projectContext.project().projectDirectory == root);
  REQUIRE(readBytes(manifest) == manifestBefore);
  REQUIRE(readBytes(managed) == managedBefore);
  REQUIRE(readBytes(collision) == collisionBefore);

  dataset->name = "Mesh";
  std::filesystem::create_directories(destination / "datasets");
  const auto destinationCollision = destination / "datasets/Mesh.tsd";
  std::filesystem::copy_file(managed, destinationCollision);
  const auto destinationBefore = readBytes(destinationCollision);
  error.clear();
  REQUIRE_FALSE(
      projectContext.saveProject(destination, nullptr, "", nullptr, &error));
  REQUIRE(error.find("unowned target") != std::string::npos);
  REQUIRE(projectContext.project().projectDirectory == root);
  REQUIRE(readBytes(destinationCollision) == destinationBefore);
  REQUIRE_FALSE(
      std::filesystem::exists(destination / PROJECT_MANIFEST_FILENAME));

  std::filesystem::remove_all(root);
  std::filesystem::remove_all(destination);
  std::filesystem::remove(source);
}

SCENARIO(
    "SciVis Studio active shot toggles light rig visibility", "[SciVisStudio]")
{
  tsd::app::Context appContext;
  ProjectContext projectContext(&appContext);
  projectContext.createUnsavedProject();

  auto &project = projectContext.project();
  auto &firstShot = project.shots.front();
  auto *defaultRig = light_rig::findLightRig(project, firstShot.lightRigId);
  REQUIRE(defaultRig != nullptr);
  auto defaultRoot = projectContext.resolveLightRigRoot(*defaultRig);
  REQUIRE(defaultRoot);

  auto *secondRig = projectContext.createLightRig("Second");
  REQUIRE(secondRig != nullptr);
  auto secondRoot = projectContext.resolveLightRigRoot(*secondRig);
  REQUIRE(secondRoot);

  projectContext.addShot("Second Shot");
  auto &secondShot = *project::activeShot(project);
  secondShot.lightRigId = secondRig->id;
  projectContext.applyActiveShot();

  REQUIRE_FALSE((*defaultRoot)->isEnabled());
  REQUIRE((*secondRoot)->isEnabled());

  secondShot.lightRigId.clear();
  projectContext.applyActiveShot();
  REQUIRE_FALSE((*defaultRoot)->isEnabled());
  REQUIRE_FALSE((*secondRoot)->isEnabled());

  secondShot.lightRigId = "missing";
  projectContext.applyActiveShot();
  REQUIRE_FALSE((*defaultRoot)->isEnabled());
  REQUIRE_FALSE((*secondRoot)->isEnabled());
}

SCENARIO(
    "SciVis Studio active shot samples assigned camera rig", "[SciVisStudio]")
{
  tsd::app::Context appContext;
  ProjectContext projectContext(&appContext);
  projectContext.createUnsavedProject();

  auto &project = projectContext.project();
  auto &shot = *project::activeShot(project);
  auto *secondRig = projectContext.createCameraRig("Second Camera");
  REQUIRE(secondRig != nullptr);

  CameraKeyframe keyframe;
  keyframe.frame = 5;
  keyframe.manipulator.orbit.lookat = {7.f, 8.f, 9.f};
  keyframe.manipulator.orbit.azeldist = {10.f, 20.f, 12.f};
  keyframe.manipulator.orbit.fixedDist = 12.f;
  secondRig->keyframes.push_back(keyframe);

  shot.cameraRigId = secondRig->id;
  shot.currentFrame = 5;
  appContext.view.manipulator.setConfig({0.f, 0.f, 0.f}, 1.f);
  projectContext.applyActiveShot();

  REQUIRE(appContext.view.manipulator.at().x == Approx(7.f));
  REQUIRE(appContext.view.manipulator.at().y == Approx(8.f));
  REQUIRE(appContext.view.manipulator.at().z == Approx(9.f));
  REQUIRE(appContext.view.manipulator.azel().x == Approx(10.f));
  REQUIRE(appContext.view.manipulator.azel().y == Approx(20.f));
  REQUIRE(appContext.view.manipulator.distance() == Approx(12.f));

  shot.cameraRigId.clear();
  appContext.view.manipulator.setConfig({1.f, 2.f, 3.f}, 4.f);
  projectContext.applyActiveShot();
  REQUIRE(appContext.view.manipulator.at().x == Approx(1.f));
  REQUIRE(appContext.view.manipulator.at().y == Approx(2.f));
  REQUIRE(appContext.view.manipulator.at().z == Approx(3.f));
  REQUIRE(appContext.view.manipulator.distance() == Approx(4.f));
}

SCENARIO("SciVis Studio removing a light rig clears shot references",
    "[SciVisStudio]")
{
  tsd::app::Context appContext;
  ProjectContext projectContext(&appContext);
  projectContext.createUnsavedProject();

  auto &project = projectContext.project();
  const auto rigId = project.lightRigs.front().id;
  auto *rig = light_rig::findLightRig(project, rigId);
  REQUIRE(rig != nullptr);
  auto root = projectContext.resolveLightRigRoot(*rig);
  REQUIRE(root);

  REQUIRE(projectContext.removeLightRig(rigId));
  REQUIRE(project.lightRigs.empty());
  REQUIRE(project.shots.front().lightRigId.empty());
  auto *layer = appContext.tsd.scene.layer("studio");
  auto lightRigsRoot = findDirectChild(layer->root(), "lightRigs");
  REQUIRE_FALSE(findDirectChild(lightRigsRoot, rigId));
}

SCENARIO("SciVis Studio removing a camera rig clears shot references",
    "[SciVisStudio]")
{
  tsd::app::Context appContext;
  ProjectContext projectContext(&appContext);
  projectContext.createUnsavedProject();

  auto &project = projectContext.project();
  const auto rigId = project.cameraRigs.front().id;
  REQUIRE(projectContext.removeCameraRig(rigId));
  REQUIRE(project.cameraRigs.empty());
  REQUIRE(project.shots.front().cameraRigId.empty());
}

SCENARIO("SciVis Studio v1 shot lights migrate to light rigs", "[SciVisStudio]")
{
  const auto root =
      std::filesystem::temp_directory_path() / "tsd_scivis_studio_v1_migrate";
  std::filesystem::remove_all(root);

  {
    tsd::app::Context appContext;
    ProjectContext projectContext(&appContext);
    projectContext.createUnsavedProject();
    auto &project = projectContext.project();
    project.lightRigs.clear();
    project.shots.front().lightRigId.clear();

    auto *layer = appContext.tsd.scene.layer("studio");
    auto shotsRoot = findDirectChild(layer->root(), "shots");
    auto shotRoot = findDirectChild(shotsRoot, project.shots.front().id);
    auto legacyLights =
        appContext.tsd.scene.insertChildNode(shotRoot, "lights");
    auto light = appContext.tsd.scene.createObject<tsd::scene::Light>(
        tsd::scene::tokens::light::directional);
    light->setName("legacyLight");
    appContext.tsd.scene.insertChildObjectNode(
        legacyLights, light, "legacyLight");

    tsd::core::DataTree tree;
    tsd::core::writeDataTreeMetadata(tree.root(),
        {tsd::core::DATA_TREE_METADATA_ENVELOPE_VERSION,
            PROJECT_FILE_TYPE,
            PROJECT_SCHEMA,
            1});
    projectToNode(project, tree.root()["scivisStudio"]);
    tsd::io::save_Scene(appContext.tsd.scene,
        tree.root()["context"],
        false,
        &appContext.tsd.animationMgr);
    std::filesystem::create_directories(root);
    REQUIRE(tree.save((root / PROJECT_MANIFEST_FILENAME).string().c_str()));
  }

  {
    tsd::app::Context appContext;
    ProjectContext projectContext(&appContext);
    REQUIRE(projectContext.openProject(root));
    auto &project = projectContext.project();
    REQUIRE(project.lightRigs.size() == 1);
    REQUIRE(project.shots.front().lightRigId == project.lightRigs.front().id);

    auto rigRoot =
        projectContext.resolveLightRigRoot(project.lightRigs.front());
    REQUIRE(rigRoot);
    REQUIRE(findDirectChild(rigRoot, "legacyLight"));
  }

  std::filesystem::remove_all(root);
}

SCENARIO("SciVis Studio v2 shot camera rigs migrate to camera rigs",
    "[SciVisStudio]")
{
  const auto root =
      std::filesystem::temp_directory_path() / "tsd_scivis_studio_v2_migrate";
  std::filesystem::remove_all(root);

  {
    tsd::app::Context appContext;
    ProjectContext projectContext(&appContext);
    projectContext.createUnsavedProject();
    auto &project = projectContext.project();
    project.cameraRigs.clear();
    project.shots.front().cameraRigId.clear();

    tsd::core::DataTree tree;
    tsd::core::writeDataTreeMetadata(tree.root(),
        {tsd::core::DATA_TREE_METADATA_ENVELOPE_VERSION,
            PROJECT_FILE_TYPE,
            PROJECT_SCHEMA,
            2});
    projectToNode(project, tree.root()["scivisStudio"]);

    auto *shotNode = tree.root()["scivisStudio"]["shots"].child(0);
    REQUIRE(shotNode != nullptr);
    auto &cameraRig = (*shotNode)["cameraRig"];

    tsd::rendering::CameraPose currentPose;
    currentPose.lookat = {1.f, 2.f, 3.f};
    currentPose.azeldist = {4.f, 5.f, 6.f};
    currentPose.fixedDist = 6.f;
    tsd::io::serialize_CameraPose(currentPose, cameraRig["current"]["orbit"]);

    tsd::rendering::CameraPose keyframePose;
    keyframePose.lookat = {7.f, 8.f, 9.f};
    keyframePose.azeldist = {10.f, 20.f, 30.f};
    keyframePose.fixedDist = 30.f;
    auto &keyframe = cameraRig["keyframes"].append();
    keyframe["frame"] = 11;
    keyframe["name"] = "legacy";
    keyframe["interpolationToNext"] = "Ease Out + In";
    tsd::io::serialize_CameraPose(
        keyframePose, keyframe["manipulator"]["orbit"]);

    tsd::io::save_Scene(appContext.tsd.scene,
        tree.root()["context"],
        false,
        &appContext.tsd.animationMgr);
    std::filesystem::create_directories(root);
    REQUIRE(tree.save((root / PROJECT_MANIFEST_FILENAME).string().c_str()));
  }

  {
    tsd::app::Context appContext;
    ProjectContext projectContext(&appContext);
    REQUIRE(projectContext.openProject(root));
    auto &project = projectContext.project();
    REQUIRE(project.cameraRigs.size() == 1);
    REQUIRE(project.shots.front().cameraRigId == project.cameraRigs.front().id);
    REQUIRE(project.cameraRigs.front().name == "Shot 1 Camera");
    REQUIRE(project.cameraRigs.front().current.orbit.lookat.x == Approx(1.f));
    REQUIRE(project.cameraRigs.front().keyframes.size() == 1);
    REQUIRE(project.cameraRigs.front().keyframes.front().frame == 11);
    REQUIRE(project.cameraRigs.front().keyframes.front().name == "legacy");
    REQUIRE(project.cameraRigs.front().keyframes.front().interpolationToNext
        == CameraInterpolation::EaseOutIn);
  }

  std::filesystem::remove_all(root);
}

SCENARIO("SciVis Studio shot time is driven by the animation manager",
    "[SciVisStudio]")
{
  tsd::app::Context appContext;
  ProjectContext projectContext(&appContext);
  projectContext.createUnsavedProject();

  auto &shot = *project::activeShot(projectContext.project());
  shot.frameCount = 24;
  shot.fps = 12.f;
  shot.currentFrame = 4;
  shot.loop = false;
  projectContext.syncAnimationManagerToActiveShot();

  auto &animMgr = appContext.tsd.animationMgr;
  REQUIRE(animMgr.getAnimationTotalFrames() == 24);
  REQUIRE(animMgr.getAnimationFPS() == Approx(12.f));
  REQUIRE(animMgr.getAnimationFrame() == 4);
  REQUIRE_FALSE(animMgr.isLoop());

  animMgr.setAnimationFrame(9);
  REQUIRE(shot.currentFrame == 9);
}

SCENARIO("SciVis Studio render-shot CLI parses command line", "[SciVisStudio]")
{
  RenderShotCommandLine commandLine;
  std::string error;

  REQUIRE(parseRenderShotCommandLine(
      {"scivisStudioRenderShot", "/tmp/project", "--shot", "shot_0002"},
      commandLine,
      error));
  REQUIRE(
      commandLine.projectDirectory == std::filesystem::path("/tmp/project"));
  REQUIRE(commandLine.shotId == "shot_0002");
  REQUIRE_FALSE(commandLine.showHelp);

  REQUIRE(parseRenderShotCommandLine(
      {"scivisStudioRenderShot", "--help"}, commandLine, error));
  REQUIRE(commandLine.showHelp);

  REQUIRE_FALSE(parseRenderShotCommandLine(
      {"scivisStudioRenderShot", "/tmp/project", "--shot"},
      commandLine,
      error));
  REQUIRE(error.find("--shot requires") != std::string::npos);
}

SCENARIO("SciVis Studio render-shot CLI selects shots", "[SciVisStudio]")
{
  Project project;
  project.shots.push_back({"shot_0001", "Overview"});
  project.shots.push_back({"shot_0002", "Detail"});

  std::string error;
  std::istringstream emptyInput;
  std::ostringstream output;

  auto *shot = selectShotForRender(
      project, "shot_0002", false, emptyInput, output, error);
  REQUIRE(shot != nullptr);
  REQUIRE(shot->id == "shot_0002");

  shot =
      selectShotForRender(project, "missing", false, emptyInput, output, error);
  REQUIRE(shot == nullptr);
  REQUIRE(error.find("unknown shot ID: missing") != std::string::npos);
  REQUIRE(error.find("shot_0001") != std::string::npos);

  shot = selectShotForRender(project, "", false, emptyInput, output, error);
  REQUIRE(shot == nullptr);
  REQUIRE(error.find("multiple shots found") != std::string::npos);
  REQUIRE(error.find("--shot <shot-id>") != std::string::npos);

  std::istringstream selectionInput("2\n");
  output.str("");
  output.clear();
  shot = selectShotForRender(project, "", true, selectionInput, output, error);
  REQUIRE(shot != nullptr);
  REQUIRE(shot->id == "shot_0002");
  REQUIRE(output.str().find("Select shot [1-2]") != std::string::npos);

  std::istringstream invalidInput("3\n");
  shot = selectShotForRender(project, "", true, invalidInput, output, error);
  REQUIRE(shot == nullptr);
  REQUIRE(error.find("invalid shot selection: 3") != std::string::npos);
}

SCENARIO(
    "SciVis Studio render-shot CLI auto-selects one shot", "[SciVisStudio]")
{
  Project project;
  project.shots.push_back({"shot_0001", "Only Shot"});

  std::string error;
  std::istringstream input;
  std::ostringstream output;
  auto *shot = selectShotForRender(project, "", false, input, output, error);
  REQUIRE(shot != nullptr);
  REQUIRE(shot->id == "shot_0001");
}

SCENARIO("SciVis Studio rig name validation", "[SciVisStudio]")
{
  GIVEN("Rig name format rules")
  {
    std::string error;
    REQUIRE(validateRigName("Key Light", &error));
    REQUIRE(validateRigName("rig-01 (imported)", &error));
    REQUIRE(validateRigName("Default Copy", &error));

    REQUIRE_FALSE(validateRigName("", &error));
    REQUIRE_FALSE(error.empty());
    REQUIRE_FALSE(validateRigName("bad/name", &error));
    REQUIRE_FALSE(validateRigName("path\\sep", &error));
    REQUIRE_FALSE(validateRigName(" leading", &error));
    REQUIRE_FALSE(validateRigName("trailing ", &error));
    REQUIRE_FALSE(validateRigName(".", &error));
    REQUIRE_FALSE(validateRigName("with*star", &error));
  }

  GIVEN("Sanitization of arbitrary strings")
  {
    REQUIRE(sanitizeRigName("a/b*c") == "a_b_c");
    REQUIRE(sanitizeRigName("  spaced  ") == "spaced");
    REQUIRE(validateRigName(sanitizeRigName("weird:name?")));
    REQUIRE(sanitizeRigName("") == "rig");
    REQUIRE(sanitizeRigName("   ") == "rig");
    REQUIRE(validateRigName(sanitizeRigName("///")));
  }
}

SCENARIO(
    "SciVis Studio rig rename enforces format and uniqueness", "[SciVisStudio]")
{
  tsd::app::Context appContext;
  ProjectContext projectContext(&appContext);
  projectContext.createUnsavedProject();

  auto &project = projectContext.project();
  auto *second = projectContext.createLightRig("Second");
  REQUIRE(second != nullptr);
  const auto defaultId = project.lightRigs.front().id;
  const auto secondId = second->id;

  std::string error;
  WHEN("renaming to a valid, unused name")
  {
    REQUIRE(projectContext.renameLightRig(defaultId, "Studio Key", &error));
    REQUIRE(light_rig::findLightRig(project, defaultId)->name == "Studio Key");
  }

  WHEN("renaming to a name used by another rig (case-insensitive)")
  {
    REQUIRE_FALSE(projectContext.renameLightRig(defaultId, "second", &error));
    REQUIRE_FALSE(error.empty());
    REQUIRE(light_rig::findLightRig(project, defaultId)->name == "Default");
  }

  WHEN("renaming a rig to its own current name")
  {
    REQUIRE(projectContext.renameLightRig(secondId, "Second", &error));
    REQUIRE(light_rig::findLightRig(project, secondId)->name == "Second");
  }

  WHEN("renaming to an invalid format")
  {
    REQUIRE_FALSE(projectContext.renameLightRig(defaultId, "bad/name", &error));
    REQUIRE_FALSE(error.empty());
  }
}

SCENARIO("SciVis Studio programmatic rig names are sanitized and unique",
    "[SciVisStudio]")
{
  tsd::app::Context appContext;
  ProjectContext projectContext(&appContext);
  projectContext.createUnsavedProject();

  auto *a = projectContext.createLightRig("weird/name");
  REQUIRE(a != nullptr);
  REQUIRE(validateRigName(a->name));

  auto *b = projectContext.createLightRig("weird/name");
  REQUIRE(b != nullptr);
  REQUIRE(validateRigName(b->name));
  REQUIRE(a->name != b->name);
}

SCENARIO("SciVis Studio v4 projects round-trip rigs through standalone files",
    "[SciVisStudio]")
{
  const auto root =
      std::filesystem::temp_directory_path() / "tsd_scivis_studio_v4_rigs";
  std::filesystem::remove_all(root);

  std::string defaultLightId;
  std::string cameraRigId;

  {
    tsd::app::Context appContext;
    ProjectContext projectContext(&appContext);
    projectContext.createUnsavedProject();
    auto &project = projectContext.project();

    // A second light rig with two lights, plus keyframe data on a camera rig.
    auto *second = projectContext.createLightRig("Rim");
    REQUIRE(second != nullptr);
    projectContext.addLightToRig(*second, "directional");
    projectContext.addLightToRig(*second, "point");

    defaultLightId = project.lightRigs.front().id;

    auto *cameraRig = &project.cameraRigs.front();
    cameraRigId = cameraRig->id;
    CameraKeyframe kf;
    kf.frame = 7;
    kf.manipulator.orbit.lookat = {4.f, 5.f, 6.f};
    cameraRig->keyframes.push_back(kf);

    REQUIRE(projectContext.saveProject(root));

    THEN("one file is written per rig")
    {
      REQUIRE(std::filesystem::exists(root / "cameras"));
      REQUIRE(std::filesystem::exists(root / "lights"));
      size_t lightFiles = 0;
      for (auto &e : std::filesystem::directory_iterator(root / "lights"))
        if (e.path().extension() == ".tsd")
          ++lightFiles;
      REQUIRE(lightFiles == project.lightRigs.size());
    }
  }

  WHEN("the project is reopened")
  {
    tsd::app::Context appContext;
    ProjectContext projectContext(&appContext);
    REQUIRE(projectContext.openProject(root));
    auto &project = projectContext.project();

    THEN("light rigs and their lights are restored from files")
    {
      REQUIRE(project.lightRigs.size() == 2);
      LightRig *rim = nullptr;
      for (auto &r : project.lightRigs) {
        if (r.name == "Rim")
          rim = &r;
      }
      REQUIRE(rim != nullptr);
      auto rimRoot = projectContext.resolveLightRigRoot(*rim);
      REQUIRE(rimRoot);
      int lightCount = 0;
      auto *layer = (*rimRoot)->layer();
      layer->traverse(rimRoot, [&](auto &node, int) {
        if (node->isObject() && node->type() == ANARI_LIGHT)
          ++lightCount;
        return true;
      });
      REQUIRE(lightCount == 2);
    }

    THEN("camera rig keyframes are restored from files")
    {
      auto *cameraRig = camera_rig::findCameraRig(project, cameraRigId);
      REQUIRE(cameraRig != nullptr);
      REQUIRE(cameraRig->keyframes.size() == 1);
      REQUIRE(cameraRig->keyframes.front().frame == 7);
    }
  }

  WHEN("a rig is renamed and the project is saved again")
  {
    std::string oldLightName;
    const auto unlisted = root / "lights/Unlisted.tsd";
    {
      tsd::app::Context appContext;
      ProjectContext projectContext(&appContext);
      REQUIRE(projectContext.openProject(root));
      auto &project = projectContext.project();
      auto *defaultRig = light_rig::findLightRig(project, defaultLightId);
      REQUIRE(defaultRig != nullptr);
      oldLightName = defaultRig->name;
      std::filesystem::copy_file(
          root / "lights" / (oldLightName + ".tsd"), unlisted);
      REQUIRE(projectContext.renameLightRig(defaultLightId, "Key Light"));
      REQUIRE(projectContext.saveProject(root));
    }

    THEN("only the explicitly superseded rig file is removed")
    {
      REQUIRE(std::filesystem::exists(root / "lights" / "Key Light.tsd"));
      REQUIRE_FALSE(
          std::filesystem::exists(root / "lights" / (oldLightName + ".tsd")));
      REQUIRE(std::filesystem::exists(unlisted));
    }
  }

  std::filesystem::remove_all(root);
}

SCENARIO("SciVis Studio tolerates a missing light rig file on open",
    "[SciVisStudio]")
{
  const auto root =
      std::filesystem::temp_directory_path() / "tsd_scivis_studio_missing_rig";
  std::filesystem::remove_all(root);

  std::string rimId;
  {
    tsd::app::Context appContext;
    ProjectContext projectContext(&appContext);
    projectContext.createUnsavedProject();
    auto *rim = projectContext.createLightRig("Rim");
    REQUIRE(rim != nullptr);
    projectContext.addLightToRig(*rim, "directional");
    rimId = rim->id;
    REQUIRE(projectContext.saveProject(root));
  }

  // Corrupt the project by deleting one rig's file.
  std::filesystem::remove(root / "lights" / "Rim.tsd");

  {
    tsd::app::Context appContext;
    ProjectContext projectContext(&appContext);
    REQUIRE(projectContext.openProject(root));
    auto &project = projectContext.project();

    THEN("the missing rig is skipped and the rest of the project opens")
    {
      REQUIRE(light_rig::findLightRig(project, rimId) == nullptr);
      REQUIRE_FALSE(project.lightRigs.empty()); // default rig survived
    }
  }

  std::filesystem::remove_all(root);
}
