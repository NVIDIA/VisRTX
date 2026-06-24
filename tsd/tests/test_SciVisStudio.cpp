// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "catch.hpp"

#include "ProjectContext.h"
#include "ProjectSerialization.h"
#include "RenderShotCLI.h"

#include "tsd/app/Context.h"
#include "tsd/core/DataTree.hpp"
#include "tsd/core/DataTreeMetadata.hpp"
#include "tsd/io/serialization.hpp"
#include "tsd/scene/UpdateDelegate.hpp"
#include "tsd/scene/objects/Light.hpp"

#include <filesystem>
#include <fstream>
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

SCENARIO("SciVis Studio project model serialization", "[SciVisStudio]")
{
  GIVEN("A project with datasets, shots, light rigs, and camera rigs")
  {
    Project project;
    project.name = "RoundTrip";
    project.projectDirectory = "/tmp/roundtrip";
    project.datasets.push_back({"dataset_0001",
        "Dataset",
        DatasetSourceKind::Static,
        "OBJ",
        {"/tmp/data.obj", "data.obj", 100, 42},
        DatasetStatus::Available,
        {"studio", 3}});
    project.datasets.front().sourceFiles.push_back(
        {"/tmp/frame_0001.raw", "frames/frame_0001.raw", 101, 43});
    project.datasets.front().sourceFiles.push_back(
        {"/tmp/frame_0002.raw", "frames/frame_0002.raw", 102, 44});

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
    cameraRig.rig.keyframes.push_back(keyframe);
    project.activeShotId = shot.id;
    project.shots.push_back(shot);
    project.lightRigs.push_back({"lightRig_0001", "Default", {"studio", 5}});
    project.cameraRigs.push_back(std::move(cameraRig));

    tsd::core::DataTree tree;
    projectToNode(project, tree.root()["scivisStudio"]);
    auto &serialized = tree.root()["scivisStudio"];

    REQUIRE(serialized["datasets"].child(0)->child("rootNode") == nullptr);
    REQUIRE(serialized["datasets"].child(0)->child("sourceFiles") != nullptr);
    REQUIRE(serialized["shots"].child(0)->child("lightRigId") != nullptr);
    REQUIRE(serialized["shots"].child(0)->child("cameraRigId") != nullptr);
    REQUIRE(serialized["shots"].child(0)->child("cameraRig") == nullptr);
    REQUIRE(serialized["shots"].child(0)->child("camera") == nullptr);
    REQUIRE(serialized["lightRigs"].child(0)->child("rootNode") == nullptr);
    // v4: camera-rig keyframe data lives in cameras/<name>.tsd, not the manifest.
    REQUIRE(serialized["cameraRigs"].child(0)->child("rig") == nullptr);
    REQUIRE(serialized["cameraRigs"].child(0)->child("name") != nullptr);

    Project loaded;
    REQUIRE(nodeToProject(serialized, loaded));

    THEN("IDs and bindings survive the manifest round trip")
    {
      REQUIRE(loaded.datasets.size() == 1);
      REQUIRE(loaded.datasets.front().id == "dataset_0001");
      REQUIRE(loaded.datasets.front().sourceFiles.size() == 2);
      REQUIRE(loaded.datasets.front().sourceFiles.front().projectRelativePath
          == "frames/frame_0001.raw");
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

SCENARIO("SciVis Studio camera rig files round-trip keyframe data",
    "[SciVisStudio]")
{
  ShotCameraRig rig;
  CameraKeyframe keyframe;
  keyframe.frame = 12;
  keyframe.name = "mid";
  keyframe.manipulator.orbit.lookat = {1.f, 2.f, 3.f};
  keyframe.interpolationToNext = CameraInterpolation::EaseOutIn;
  rig.keyframes.push_back(keyframe);

  const auto file =
      std::filesystem::temp_directory_path() / "tsd_camera_rig_roundtrip.tsd";
  std::filesystem::remove(file);

  REQUIRE(exportCameraRigFile("Hero Cam", rig, file));

  std::string name;
  ShotCameraRig loaded;
  REQUIRE(importCameraRigFile(file, name, loaded));

  REQUIRE(name == "Hero Cam");
  REQUIRE(loaded.keyframes.size() == 1);
  REQUIRE(loaded.keyframes.front().frame == 12);
  REQUIRE(loaded.keyframes.front().interpolationToNext
      == CameraInterpolation::EaseOutIn);

  std::filesystem::remove(file);
}

SCENARIO("SciVis Studio source file resolution prefers project-relative paths",
    "[SciVisStudio]")
{
  const auto root =
      std::filesystem::temp_directory_path() / "tsd_scivis_studio_source_paths";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root / "frames");

  const auto relativeFrame = root / "frames" / "frame_0001.raw";
  {
    std::ofstream out(relativeFrame);
    out << "frame";
  }

  ProjectContext projectContext;
  projectContext.project().projectDirectory = root;
  DatasetSourceFile sourceFile;
  sourceFile.absolutePath = "/missing/frame_0001.raw";
  sourceFile.projectRelativePath = "frames/frame_0001.raw";

  REQUIRE(projectContext.resolveSourceFilePath(sourceFile) == relativeFrame);
  REQUIRE(projectContext.sourceFileIsRegular(sourceFile));

  std::filesystem::remove_all(root);
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
        REQUIRE(shot_camera_rig::interpolationFromString(
                    shot_camera_rig::toString(mode))
            == mode);

      REQUIRE(shot_camera_rig::interpolationFromString("Unknown")
          == CameraInterpolation::Linear);
    }

    THEN("Sampling applies easing to the segment interpolation factor")
    {
      ShotCameraRig rig;

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
      REQUIRE(shot_camera_rig::sampleCameraRig(rig, 25).orbit.lookat.x
          == Approx(6.25f));

      rig.keyframes.front().interpolationToNext = CameraInterpolation::EaseIn;
      REQUIRE(shot_camera_rig::sampleCameraRig(rig, 25).orbit.lookat.x
          == Approx(57.8125f));

      rig.keyframes.front().interpolationToNext =
          CameraInterpolation::EaseOutIn;
      REQUIRE(shot_camera_rig::sampleCameraRig(rig, 25).orbit.lookat.x
          == Approx(10.3515625f));
      REQUIRE(shot_camera_rig::sampleCameraRig(rig, 25).orbit.azeldist.x
          == Approx(10.3515625f));
      REQUIRE(shot_camera_rig::sampleCameraRig(rig, 25).orbit.fixedDist
          == Approx(10.3515625f));
      REQUIRE(shot_camera_rig::sampleCameraRig(rig, 75).orbit.lookat.x
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

SCENARIO("SciVis Studio cloning a light rig deep-copies lights",
    "[SciVisStudio]")
{
  tsd::app::Context appContext;
  ProjectContext projectContext(&appContext);
  projectContext.createUnsavedProject();

  auto &project = projectContext.project();
  const auto sourceRigId = project.lightRigs.front().id;
  auto *sourceRig = project::findLightRig(project, sourceRigId);
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
  REQUIRE(cloneLight->parameterValueAs<float>("irradiance").value()
      == Approx(2.f));

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
    const auto cameraName =
        projectNode["cameraRigs"].child(0)->child("name")->getValueAs<std::string>();
    const auto lightName =
        projectNode["lightRigs"].child(0)->child("name")->getValueAs<std::string>();
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

SCENARIO(
    "SciVis Studio active shot toggles light rig visibility", "[SciVisStudio]")
{
  tsd::app::Context appContext;
  ProjectContext projectContext(&appContext);
  projectContext.createUnsavedProject();

  auto &project = projectContext.project();
  auto &firstShot = project.shots.front();
  auto *defaultRig = project::findLightRig(project, firstShot.lightRigId);
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
  secondRig->rig.keyframes.push_back(keyframe);

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
  auto *rig = project::findLightRig(project, rigId);
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
    tsd::io::cameraPoseToNode(currentPose, cameraRig["current"]["orbit"]);

    tsd::rendering::CameraPose keyframePose;
    keyframePose.lookat = {7.f, 8.f, 9.f};
    keyframePose.azeldist = {10.f, 20.f, 30.f};
    keyframePose.fixedDist = 30.f;
    auto &keyframe = cameraRig["keyframes"].append();
    keyframe["frame"] = 11;
    keyframe["name"] = "legacy";
    keyframe["interpolationToNext"] = "Ease Out + In";
    tsd::io::cameraPoseToNode(keyframePose, keyframe["manipulator"]["orbit"]);

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
    REQUIRE(
        project.cameraRigs.front().rig.current.orbit.lookat.x == Approx(1.f));
    REQUIRE(project.cameraRigs.front().rig.keyframes.size() == 1);
    REQUIRE(project.cameraRigs.front().rig.keyframes.front().frame == 11);
    REQUIRE(project.cameraRigs.front().rig.keyframes.front().name == "legacy");
    REQUIRE(project.cameraRigs.front().rig.keyframes.front().interpolationToNext
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

SCENARIO("SciVis Studio rig rename enforces format and uniqueness",
    "[SciVisStudio]")
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
    REQUIRE(project::findLightRig(project, defaultId)->name == "Studio Key");
  }

  WHEN("renaming to a name used by another rig (case-insensitive)")
  {
    REQUIRE_FALSE(projectContext.renameLightRig(defaultId, "second", &error));
    REQUIRE_FALSE(error.empty());
    REQUIRE(project::findLightRig(project, defaultId)->name == "Default");
  }

  WHEN("renaming a rig to its own current name")
  {
    REQUIRE(projectContext.renameLightRig(secondId, "Second", &error));
    REQUIRE(project::findLightRig(project, secondId)->name == "Second");
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
    cameraRig->rig.keyframes.push_back(kf);

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
      auto *cameraRig = project::findCameraRig(project, cameraRigId);
      REQUIRE(cameraRig != nullptr);
      REQUIRE(cameraRig->rig.keyframes.size() == 1);
      REQUIRE(cameraRig->rig.keyframes.front().frame == 7);
    }
  }

  WHEN("a rig is renamed and the project is saved again")
  {
    std::string oldLightName;
    {
      tsd::app::Context appContext;
      ProjectContext projectContext(&appContext);
      REQUIRE(projectContext.openProject(root));
      auto &project = projectContext.project();
      auto *defaultRig = project::findLightRig(project, defaultLightId);
      REQUIRE(defaultRig != nullptr);
      oldLightName = defaultRig->name;
      REQUIRE(projectContext.renameLightRig(defaultLightId, "Key Light"));
      REQUIRE(projectContext.saveProject(root));
    }

    THEN("the stale rig file is reconciled away")
    {
      REQUIRE(std::filesystem::exists(root / "lights" / "Key Light.tsd"));
      REQUIRE_FALSE(
          std::filesystem::exists(root / "lights" / (oldLightName + ".tsd")));
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
      REQUIRE(project::findLightRig(project, rimId) == nullptr);
      REQUIRE_FALSE(project.lightRigs.empty()); // default rig survived
    }
  }

  std::filesystem::remove_all(root);
}
