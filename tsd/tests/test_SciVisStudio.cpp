// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "catch.hpp"

#include "ProjectContext.h"
#include "ProjectSerialization.h"
#include "RenderShotCLI.h"

#include "tsd/app/Context.h"
#include "tsd/core/DataTree.hpp"
#include "tsd/scene/UpdateDelegate.hpp"

#include <filesystem>
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
  GIVEN("A project with datasets, shots, and camera keyframes")
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

    Shot shot;
    shot.id = "shot_0001";
    shot.name = "Shot 1";
    shot.datasetBindings.push_back({"dataset_0001", true});
    shot.lightGroup = {"studio", 5};
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

    tsd::core::DataTree tree;
    projectToNode(project, tree.root()["scivisStudio"]);
    auto &serialized = tree.root()["scivisStudio"];

    REQUIRE(serialized["datasets"].child(0)->child("rootNode") == nullptr);
    REQUIRE(serialized["shots"].child(0)->child("lightGroup") == nullptr);
    REQUIRE(serialized["shots"].child(0)->child("camera") == nullptr);

    Project loaded;
    REQUIRE(nodeToProject(serialized, loaded));

    THEN("IDs and keyframes survive round trip")
    {
      REQUIRE(loaded.datasets.size() == 1);
      REQUIRE(loaded.datasets.front().id == "dataset_0001");
      REQUIRE(loaded.shots.size() == 1);
      REQUIRE(loaded.shots.front().id == "shot_0001");
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
        REQUIRE(cameraInterpolationFromString(toString(mode)) == mode);

      REQUIRE(cameraInterpolationFromString("Unknown")
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
      REQUIRE(sampleCameraRig(rig, 25).orbit.lookat.x == Approx(6.25f));

      rig.keyframes.front().interpolationToNext = CameraInterpolation::EaseIn;
      REQUIRE(sampleCameraRig(rig, 25).orbit.lookat.x == Approx(43.75f));

      rig.keyframes.front().interpolationToNext =
          CameraInterpolation::EaseOutIn;
      REQUIRE(sampleCameraRig(rig, 25).orbit.lookat.x == Approx(15.625f));
      REQUIRE(sampleCameraRig(rig, 25).orbit.azeldist.x == Approx(15.625f));
      REQUIRE(sampleCameraRig(rig, 25).orbit.fixedDist == Approx(15.625f));
    }
  }
}

SCENARIO("SciVis Studio project root validation", "[SciVisStudio]")
{
  const auto root =
      std::filesystem::temp_directory_path() / "tsd_scivis_studio_test_project";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);

  GIVEN("A valid project manifest")
  {
    tsd::core::DataTree tree;
    tree.root()["projectKind"] = PROJECT_KIND;
    tree.root()["schemaVersion"] = SCHEMA_VERSION;
    REQUIRE(tree.save((root / PROJECT_MANIFEST_FILENAME).string().c_str()));

    THEN("Validation succeeds")
    {
      auto result = validateProjectRoot(root);
      REQUIRE(result.ok);
    }
  }

  GIVEN("An invalid project kind")
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
  REQUIRE(project.activeShotId == project.shots.front().id);
  REQUIRE(project.dirty == false);
  REQUIRE(appContext.tsd.scene.layer("studio") != nullptr);
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

  auto &shot = *activeShot(project);
  setDatasetBinding(shot, "dataset_0001", false);

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

  auto &shot = *activeShot(project);
  setDatasetBinding(shot, "dataset_0001", false);

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
    setDatasetBinding(*activeShot(project), "dataset_0001", false);

    REQUIRE(projectContext.saveProject(root));
  }

  {
    tsd::core::DataTree manifest;
    REQUIRE(manifest.load((root / PROJECT_MANIFEST_FILENAME).string().c_str()));
    auto &projectNode = manifest.root()["scivisStudio"];
    REQUIRE(projectNode["datasets"].child(0)->child("rootNode") == nullptr);
    REQUIRE(projectNode["shots"].child(0)->child("lightGroup") == nullptr);
    REQUIRE(projectNode["shots"].child(0)->child("camera") == nullptr);
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

SCENARIO("SciVis Studio shot time is driven by the animation manager",
    "[SciVisStudio]")
{
  tsd::app::Context appContext;
  ProjectContext projectContext(&appContext);
  projectContext.createUnsavedProject();

  auto &shot = *activeShot(projectContext.project());
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
