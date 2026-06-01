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
  GIVEN("A project with datasets, shots, light rigs, and camera keyframes")
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
    shot.camera = {ANARI_CAMERA, 2};
    shot.renderSettings.rendererLibrary = "dummy_test_device";
    shot.renderSettings.rendererObjectIndex = 7;
    shot.renderSettings.rendererSubtype = "dummy_test_renderer";
    CameraKeyframe keyframe;
    keyframe.frame = 12;
    keyframe.name = "mid";
    keyframe.manipulator.orbit.lookat = {1.f, 2.f, 3.f};
    keyframe.manipulator.orbit.azeldist = {10.f, 20.f, 30.f};
    keyframe.interpolationToNext = CameraInterpolation::EaseOutIn;
    shot.cameraRig.keyframes.push_back(keyframe);
    project.activeShotId = shot.id;
    project.shots.push_back(shot);
    project.lightRigs.push_back({"lightRig_0001", "Default", {"studio", 5}});

    tsd::core::DataTree tree;
    projectToNode(project, tree.root()["scivisStudio"]);
    auto &serialized = tree.root()["scivisStudio"];

    REQUIRE(serialized["datasets"].child(0)->child("rootNode") == nullptr);
    REQUIRE(serialized["datasets"].child(0)->child("sourceFiles") != nullptr);
    REQUIRE(serialized["shots"].child(0)->child("lightRigId") != nullptr);
    REQUIRE(serialized["shots"].child(0)->child("camera") == nullptr);
    REQUIRE(serialized["lightRigs"].child(0)->child("rootNode") == nullptr);

    Project loaded;
    REQUIRE(nodeToProject(serialized, loaded));

    THEN("IDs and keyframes survive round trip")
    {
      REQUIRE(loaded.datasets.size() == 1);
      REQUIRE(loaded.datasets.front().id == "dataset_0001");
      REQUIRE(loaded.datasets.front().sourceFiles.size() == 2);
      REQUIRE(loaded.datasets.front().sourceFiles.front().projectRelativePath
          == "frames/frame_0001.raw");
      REQUIRE(loaded.shots.size() == 1);
      REQUIRE(loaded.shots.front().id == "shot_0001");
      REQUIRE(loaded.shots.front().lightRigId == "lightRig_0001");
      REQUIRE(loaded.lightRigs.size() == 1);
      REQUIRE(loaded.lightRigs.front().id == "lightRig_0001");
      REQUIRE(loaded.shots.front().renderSettings.rendererLibrary
          == "dummy_test_device");
      REQUIRE(loaded.shots.front().renderSettings.rendererObjectIndex == 7);
      REQUIRE(loaded.shots.front().renderSettings.rendererSubtype
          == "dummy_test_renderer");
      REQUIRE(loaded.shots.front().cameraRig.keyframes.size() == 1);
      REQUIRE(loaded.shots.front().cameraRig.keyframes.front().frame == 12);
      REQUIRE(
          loaded.shots.front().cameraRig.keyframes.front().interpolationToNext
          == CameraInterpolation::EaseOutIn);
    }
  }
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
      REQUIRE(shot_camera_rig::sampleCameraRig(rig, 25).orbit.lookat.x == Approx(6.25f));

      rig.keyframes.front().interpolationToNext = CameraInterpolation::EaseIn;
      REQUIRE(shot_camera_rig::sampleCameraRig(rig, 25).orbit.lookat.x == Approx(57.8125f));

      rig.keyframes.front().interpolationToNext =
          CameraInterpolation::EaseOutIn;
      REQUIRE(shot_camera_rig::sampleCameraRig(rig, 25).orbit.lookat.x == Approx(10.3515625f));
      REQUIRE(shot_camera_rig::sampleCameraRig(rig, 25).orbit.azeldist.x == Approx(10.3515625f));
      REQUIRE(shot_camera_rig::sampleCameraRig(rig, 25).orbit.fixedDist == Approx(10.3515625f));
      REQUIRE(shot_camera_rig::sampleCameraRig(rig, 75).orbit.lookat.x == Approx(89.6484375f));
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
  REQUIRE(projectContext.addShot());
  REQUIRE(project::activeShot(projectContext.project())->lightRigId == defaultRigId);
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
    shot::setDatasetBinding(*project::activeShot(project), "dataset_0001", false);

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
    REQUIRE(projectNode["shots"].child(0)->child("camera") == nullptr);
    REQUIRE(projectNode["lightRigs"].child(0)->child("rootNode") == nullptr);
  }

  {
    tsd::app::Context appContext;
    ProjectContext projectContext(&appContext);
    REQUIRE(projectContext.openProject(root));

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
