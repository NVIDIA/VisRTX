// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "catch.hpp"

#include "CameraRig.h"
#include "DatasetIO.h"
#include "LightRig.h"
#include "ProjectContext.h"
#include "ProjectPersistence.h"
#include "ProjectSerialization.h"
#include "RenderShot.h"
#include "RenderShotCLI.h"

#include "tsd/app/ApplicationDump.h"
#include "tsd/app/Context.h"
#include "tsd/app/LegacyApplicationContext.h"
#include "tsd/core/DataTree.hpp"
#include "tsd/core/DataTreeMetadata.hpp"
#include "tsd/io/animation/SpatialFieldFileBinding.hpp"
#include "tsd/io/archives/CameraArchive.hpp"
#include "tsd/io/archives/RendererArchive.hpp"
#include "tsd/io/archives/SceneArchive.hpp"
#include "tsd/io/serialization/serialization_internal.hpp"
#include "tsd/scene/UpdateDelegate.hpp"
#include "tsd/scene/objects/Camera.hpp"
#include "tsd/scene/objects/Geometry.hpp"
#include "tsd/scene/objects/Light.hpp"
#include "tsd/scene/objects/Material.hpp"
#include "tsd/scene/objects/Renderer.hpp"
#include "tsd/scene/objects/SpatialField.hpp"
#include "tsd/scene/objects/Volume.hpp"

#include <algorithm>
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

// Build the minimal file-animation dataset runtime — a volume with an initial
// spatial field plus one runtime file animation over the given paths — and
// return its dataset metadata.
Dataset makeFileAnimationDatasetRuntime(tsd::scene::Scene &scene,
    tsd::animation::AnimationManager &animations,
    tsd::scene::LayerNodeRef parent,
    const std::string &id,
    const std::string &name,
    const std::vector<std::string> &paths,
    tsd::scene::LayerNodeRef *rootOut = nullptr)
{
  auto field = scene.createObject<tsd::scene::SpatialField>(
      tsd::scene::tokens::spatial_field::structuredRegular);
  auto voxels = scene.createArray(ANARI_FLOAT32, 1, 1, 1);
  *voxels->mapAs<float>() = 1.f;
  voxels->unmap();
  field->setParameterObject("data", *voxels);
  auto volume = scene.createObject<tsd::scene::Volume>(
      tsd::scene::tokens::volume::transferFunction1D);
  volume->setParameterObject("value", *field);
  auto root = scene.insertChildNode(parent, id.c_str());
  scene.insertChildObjectNode(root, volume, "volume");
  animations.addAnimation(name + " file animation")
      .emplaceFileBinding<tsd::io::SpatialFieldFileBinding>(
          &scene, volume.data(), field, paths);
  if (rootOut)
    *rootOut = root;

  Dataset dataset;
  dataset.id = id;
  dataset.name = name;
  dataset.sourceKind = DatasetSourceKind::FileAnimation;
  dataset.importerType = "VOLUME_ANIMATION";
  for (const auto &path : paths)
    dataset.sourceFiles.push_back({path});
  return dataset;
}

std::string fileContents(const std::filesystem::path &file)
{
  std::ifstream in(file, std::ios::binary);
  std::stringstream contents;
  contents << in.rdbuf();
  return contents.str();
}

} // namespace

SCENARIO("SciVis Studio persistence plans a complete new project save",
    "[SciVisStudio][ProjectPersistence]")
{
  const auto root = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_persistence_save_plan";
  std::filesystem::remove_all(root);

  Project project;
  tsd::scene::Scene scene;
  tsd::animation::AnimationManager animations(&scene);
  ProjectSaveResult result;
  std::string error;

  REQUIRE(buildProjectSavePlan(
      ProjectSaveRequest(project, scene, animations, root), result, &error));
  REQUIRE_FALSE(std::filesystem::exists(root));
  REQUIRE(result.project.name == root.filename().string());
  REQUIRE(result.project.projectDirectory == root);
  REQUIRE_FALSE(result.project.dirty);
  REQUIRE(result.plan.directory == root);
  REQUIRE(result.plan.directories
      == std::vector<std::filesystem::path>{
          "renders", "datasets", "cameras", "lights", "scene"});
  REQUIRE(result.plan.assets.size() == 2);
  REQUIRE(result.plan.assets[0].target
      == std::filesystem::path("scene/cameras.tsd"));
  REQUIRE(result.plan.assets[1].target
      == std::filesystem::path("scene/renderers.tsd"));
  REQUIRE(result.plan.manifest.target == PROJECT_MANIFEST_FILENAME);
}

SCENARIO("SciVis Studio persistence stages a project before applying it",
    "[SciVisStudio][ProjectPersistence]")
{
  const auto root = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_persistence_open_stage";
  std::filesystem::remove_all(root);

  {
    tsd::app::Context appContext;
    ProjectContext context(&appContext);
    context.createUnsavedProject();
    context.project().name = "Staged Project";
    REQUIRE(context.saveProject(root));
  }

  ProjectOpenStage stage;
  std::string error;
  REQUIRE(stageProjectOpen(root, stage, {}, &error));
  REQUIRE(stage.project.name == "Staged Project");
  REQUIRE(stage.project.projectDirectory == root);

  // Applying uses the decoded stage rather than reopening project files.
  std::filesystem::remove_all(root);

  tsd::scene::Scene target;
  tsd::animation::AnimationManager targetAnimations(&target);
  const auto cameraCountBefore = target.numberOfObjects(ANARI_CAMERA);
  REQUIRE(cameraCountBefore == 1);
  REQUIRE(applyProjectOpen(stage, target, targetAnimations, &error));
  REQUIRE(target.numberOfObjects(ANARI_CAMERA) >= cameraCountBefore);
  REQUIRE(target.layer("studio") != nullptr);
  REQUIRE(stage.project.cameraRigs.size() == 1);
  REQUIRE(stage.project.lightRigs.size() == 1);

  std::filesystem::remove_all(root);
}

SCENARIO("SciVis Studio static Dataset Archives are self-contained",
    "[SciVisStudio]")
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
  REQUIRE(saveDatasetArchiveFile(dataset, root, sourceAnimations, file));

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
  Dataset loadedDataset;
  tsd::scene::LayerNodeRef loadedRoot;
  REQUIRE(loadDatasetArchiveFile(
      target, targetAnimations, file, destination, loadedDataset, loadedRoot));
  REQUIRE(loadedRoot);
  REQUIRE(loadedDataset.status == DatasetStatus::Available);
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
  REQUIRE(loadDatasetArchiveFile(
      target, targetAnimations, file, destination, secondImport, secondRoot));
  REQUIRE(target.numberOfObjects(ANARI_GEOMETRY) == 2);
  REQUIRE(target.getObject(ANARI_GEOMETRY, 0)
      != target.getObject(ANARI_GEOMETRY, 1));
  // One default material plus one independently loaded material per dataset.
  REQUIRE(target.numberOfObjects(ANARI_MATERIAL) == 3);

  std::filesystem::remove(file);
}

SCENARIO("SciVis Studio file-animation Dataset Archives externalize their "
         "source list",
    "[SciVisStudio]")
{
  const auto file = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_file_animation_dataset.tsd";
  const auto corruptFile = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_file_animation_dataset_corrupt.tsd";
  std::filesystem::remove(file);
  std::filesystem::remove(sourceListFilePath(file));
  std::filesystem::remove(corruptFile);

  tsd::scene::Scene source;
  tsd::animation::AnimationManager sourceAnimations(&source);
  const std::vector<std::string> paths = {
      "relative/frame 01.raw", "/missing/../opaque/frame02.raw"};
  tsd::scene::LayerNodeRef root;
  auto dataset = makeFileAnimationDatasetRuntime(source,
      sourceAnimations,
      source.defaultLayer()->root(),
      "dataset_0001",
      "Opaque Frames",
      paths,
      &root);
  Dataset invalidStatic = dataset;
  invalidStatic.sourceKind = DatasetSourceKind::Static;
  invalidStatic.importerType = "VOLUME";
  std::string invalidStaticError;
  REQUIRE_FALSE(saveDatasetArchiveFile(
      invalidStatic, root, sourceAnimations, corruptFile, &invalidStaticError));
  REQUIRE(invalidStaticError.find("cannot own file animations")
      != std::string::npos);
  std::string saveError;
  const bool saved =
      saveDatasetArchiveFile(dataset, root, sourceAnimations, file, &saveError);
  INFO(saveError);
  REQUIRE(saved);

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

  // New-format dataset files persist importer settings but no frame paths.
  tsd::core::DataTree serialized;
  REQUIRE(serialized.load(file.string().c_str()));
  REQUIRE(serialized.root()["dataset"].child("sourceFiles") == nullptr);
  REQUIRE(datasetArchiveUsesSourceListFile(serialized.root()));

  // Dataset Archive Load fails cleanly without the sibling Source List File.
  {
    tsd::scene::Scene target;
    tsd::animation::AnimationManager targetAnimations(&target);
    Dataset missingDataset;
    tsd::scene::LayerNodeRef missingRoot;
    std::string missingError;
    REQUIRE_FALSE(loadDatasetArchiveFile(target,
        targetAnimations,
        file,
        target.defaultLayer()->root(),
        missingDataset,
        missingRoot,
        &missingError));
    REQUIRE(missingError.find("Source List File") != std::string::npos);
    REQUIRE(target.numberOfObjects(ANARI_VOLUME) == 0);
    REQUIRE_FALSE(validateDatasetAsset(file).ok);
  }

  REQUIRE(writeSourceListFile(sourceListFilePath(file), dataset.sourceFiles));
  auto pairValidation = validateDatasetAsset(file);
  REQUIRE(pairValidation.ok);
  REQUIRE(pairValidation.dataset.sourceFiles.size() == paths.size());

  tsd::scene::Scene target;
  tsd::animation::AnimationManager targetAnimations(&target);
  Dataset loadedDataset;
  tsd::scene::LayerNodeRef importedRoot;
  REQUIRE(loadDatasetArchiveFile(target,
      targetAnimations,
      file,
      target.defaultLayer()->root(),
      loadedDataset,
      importedRoot));
  REQUIRE_FALSE(loadedDataset.pendingSourceListMigration);
  REQUIRE(loadedDataset.sourceFiles.size() == paths.size());
  REQUIRE(loadedDataset.sourceFiles[0].path == paths[0]);
  REQUIRE(loadedDataset.sourceFiles[1].path == paths[1]);
  // The relative entry is anchored once, at read, to the Source List File's
  // directory; the absolute entry stays opaque.
  const auto anchored =
      (std::filesystem::temp_directory_path() / paths[0]).string();
  REQUIRE(loadedDataset.sourceFiles[0].resolvedPath == anchored);
  REQUIRE(loadedDataset.sourceFiles[1].resolvedPath.empty());
  REQUIRE(targetAnimations.animations().size() == 1);
  REQUIRE(targetAnimations.animations().front().fileBindings().size() == 1);

  tsd::core::DataTree binding;
  targetAnimations.animations().front().fileBindings().front()->toDataNode(
      binding.root());
  REQUIRE(
      binding.root()["files"].child(0)->getValueAs<std::string>() == anchored);
  REQUIRE(
      binding.root()["files"].child(1)->getValueAs<std::string>() == paths[1]);

  std::filesystem::remove(file);
  std::filesystem::remove(sourceListFilePath(file));
  std::filesystem::remove(corruptFile);
}

SCENARIO("SciVis Studio Source List Files hold one trimmed path per line",
    "[SciVisStudio]")
{
  const auto directory = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_source_list_files";
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const auto file = directory / "Frames.sources";

  GIVEN("A hand-written Source List File")
  {
    {
      std::ofstream out(file);
      out << "\n";
      out << "  relative/frame 01.raw \t\n";
      out << "\t\n";
      out << "/absolute/frame02.raw\n";
      out << "trailing/frame03.raw"; // no final newline
    }

    THEN("Entries read back trimmed, in line order, skipping blanks")
    {
      std::vector<DatasetSourceFile> entries;
      REQUIRE(readSourceListFile(file, entries));
      REQUIRE(entries.size() == 3);
      REQUIRE(entries[0].path == "relative/frame 01.raw");
      REQUIRE(entries[1].path == "/absolute/frame02.raw");
      REQUIRE(entries[2].path == "trailing/frame03.raw");
      REQUIRE(entries[0].resolvedPath
          == (directory / "relative/frame 01.raw").string());
      REQUIRE(entries[1].resolvedPath.empty());

      AND_THEN("Writing the entries back reproduces the raw lines verbatim")
      {
        const auto rewritten = directory / "Rewritten.sources";
        REQUIRE(writeSourceListFile(rewritten, entries));
        std::ifstream in(rewritten);
        std::stringstream contents;
        contents << in.rdbuf();
        REQUIRE(contents.str()
            == "relative/frame 01.raw\n/absolute/frame02.raw\n"
               "trailing/frame03.raw\n");
      }
    }
  }

  GIVEN("A missing, empty, or blank Source List File")
  {
    THEN("Reading fails with a clear error")
    {
      std::vector<DatasetSourceFile> entries;
      std::string error;
      REQUIRE_FALSE(
          readSourceListFile(directory / "Missing.sources", entries, &error));
      REQUIRE_FALSE(error.empty());

      const auto blank = directory / "Blank.sources";
      {
        std::ofstream out(blank);
        out << " \n\t\n";
      }
      error.clear();
      REQUIRE_FALSE(readSourceListFile(blank, entries, &error));
      REQUIRE(error.find("empty") != std::string::npos);
    }
  }

  GIVEN("An empty File Animation Source List")
  {
    THEN("Writing refuses to produce an unloadable dataset")
    {
      std::string error;
      REQUIRE_FALSE(writeSourceListFile(file, {}, &error));
    }
  }

  std::filesystem::remove_all(directory);
}

SCENARIO(
    "SciVis Studio legacy embedded sourceFiles load and mark migration",
    "[SciVisStudio]")
{
  const auto file = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_legacy_embedded_dataset.tsd";
  std::filesystem::remove(file);
  std::filesystem::remove(sourceListFilePath(file));

  tsd::scene::Scene source;
  tsd::animation::AnimationManager sourceAnimations(&source);
  const std::vector<std::string> paths = {
      "legacy/frame01.raw", "/legacy/frame02.raw"};
  tsd::scene::LayerNodeRef root;
  auto dataset = makeFileAnimationDatasetRuntime(source,
      sourceAnimations,
      source.defaultLayer()->root(),
      "dataset_0001",
      "Legacy Frames",
      paths,
      &root);
  REQUIRE(saveDatasetArchiveFile(dataset, root, sourceAnimations, file));

  // Recreate the legacy on-disk format: the dataset file embeds the list.
  {
    tsd::core::DataTree legacy;
    REQUIRE(legacy.load(file.string().c_str()));
    auto &files = legacy.root()["dataset"]["sourceFiles"];
    for (const auto &path : paths)
      files.append() = path;
    REQUIRE_FALSE(datasetArchiveUsesSourceListFile(legacy.root()));
    REQUIRE(legacy.save(file.string().c_str()));
  }

  auto validation = validateDatasetAsset(file);
  REQUIRE(validation.ok);
  REQUIRE(validation.dataset.pendingSourceListMigration);
  REQUIRE(validation.dataset.sourceFiles.size() == paths.size());

  tsd::scene::Scene target;
  tsd::animation::AnimationManager targetAnimations(&target);
  Dataset loadedDataset;
  tsd::scene::LayerNodeRef importedRoot;
  REQUIRE(loadDatasetArchiveFile(target,
      targetAnimations,
      file,
      target.defaultLayer()->root(),
      loadedDataset,
      importedRoot));
  REQUIRE(loadedDataset.pendingSourceListMigration);
  REQUIRE(loadedDataset.sourceFiles.size() == paths.size());
  REQUIRE(loadedDataset.sourceFiles[0].path == paths[0]);
  // Legacy entries stay opaque: no read-time anchoring is applied.
  REQUIRE(loadedDataset.sourceFiles[0].resolvedPath.empty());
  REQUIRE(targetAnimations.animations().size() == 1);

  tsd::core::DataTree binding;
  targetAnimations.animations().front().fileBindings().front()->toDataNode(
      binding.root());
  REQUIRE(
      binding.root()["files"].child(0)->getValueAs<std::string>() == paths[0]);

  std::filesystem::remove(file);
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

SCENARIO("SciVis Studio dataset residency round-trips through the manifest",
    "[SciVisStudio]")
{
  GIVEN("A project with a Loaded and an Unloaded dataset")
  {
    Project project;
    Dataset resident;
    resident.id = "dataset_0001";
    resident.name = "Resident";
    project.datasets.push_back(std::move(resident));
    Dataset parked;
    parked.id = "dataset_0002";
    parked.name = "Parked";
    parked.residency = DatasetResidency::Unloaded;
    project.datasets.push_back(std::move(parked));

    tsd::core::DataTree tree;
    projectToNode(project, tree.root());

    THEN("Residency survives the manifest round trip")
    {
      Project loaded;
      REQUIRE(nodeToProject(tree.root(), loaded));
      REQUIRE(loaded.datasets.size() == 2);
      REQUIRE(loaded.datasets[0].residency == DatasetResidency::Loaded);
      REQUIRE(loaded.datasets[1].residency == DatasetResidency::Unloaded);
    }
  }

  GIVEN("A manifest whose datasets predate residency")
  {
    tsd::core::DataTree legacy;
    auto &d = legacy.root()["datasets"].append();
    d["id"] = std::string("dataset_0001");
    d["name"] = std::string("Old");

    THEN("An absent residency field means Loaded")
    {
      Project loaded;
      REQUIRE(nodeToProject(legacy.root(), loaded));
      REQUIRE(loaded.datasets.size() == 1);
      REQUIRE(loaded.datasets.front().residency == DatasetResidency::Loaded);
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

SCENARIO("SciVis Studio Camera Rig Archives round-trip keyframe data",
    "[SciVisStudio]")
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

  REQUIRE(camera_rig::saveCameraRigArchiveFile(rig, file));

  CameraRig loaded;
  REQUIRE(camera_rig::loadCameraRigArchiveFile(file, loaded));

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
    std::filesystem::create_directories(root / "scene");
    tsd::scene::Scene scene;
    REQUIRE(tsd::io::save_CameraArchive(
        scene, (root / "scene/cameras.tsd").string().c_str()));
    REQUIRE(tsd::io::save_RendererArchive(
        scene, (root / "scene/renderers.tsd").string().c_str()));

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

SCENARIO("SciVis Studio requires valid scene pool Archives", "[SciVisStudio]")
{
  const auto root = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_required_scene_pools";
  std::filesystem::remove_all(root);

  tsd::app::Context appContext;
  ProjectContext projectContext(&appContext);
  projectContext.createUnsavedProject();
  REQUIRE(projectContext.saveProject(root));

  std::filesystem::remove(root / "scene/cameras.tsd");
  auto validation = validateProjectRoot(root);
  REQUIRE_FALSE(validation.ok);
  REQUIRE(validation.error.find("cameras.tsd") != std::string::npos);

  std::string error;
  REQUIRE_FALSE(
      projectContext.openProject(root, nullptr, nullptr, nullptr, &error));
  REQUIRE(error.find("cameras.tsd") != std::string::npos);

  REQUIRE(projectContext.saveProject(root));
  {
    std::ofstream corrupt(
        root / "scene/renderers.tsd", std::ios::binary | std::ios::trunc);
    corrupt << "not a Renderer Archive";
  }
  validation = validateProjectRoot(root);
  REQUIRE_FALSE(validation.ok);
  REQUIRE(validation.error.find("renderers.tsd") != std::string::npos);
  REQUIRE_FALSE(
      projectContext.openProject(root, nullptr, nullptr, nullptr, &error));
  REQUIRE(error.find("renderers.tsd") != std::string::npos);

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
      DatasetResidency::Loaded,
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
      DatasetResidency::Loaded,
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
        DatasetResidency::Loaded,
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
    // v4: rig value data moved to standalone Archives.
    REQUIRE(projectNode["cameraRigs"].child(0)->child("rig") == nullptr);

    REQUIRE(manifest.root().child("context") == nullptr);

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

SCENARIO("SciVis Studio projects store scene pools in required Archives",
    "[SciVisStudio]")
{
  const auto root = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_scene_pool_archives";
  std::filesystem::remove_all(root);

  {
    tsd::app::Context appContext;
    ProjectContext projectContext(&appContext);
    projectContext.createUnsavedProject();
    REQUIRE(projectContext.addShot("Second Shot"));
    projectContext.project().activeShotId =
        projectContext.project().shots.front().id;

    auto renderer =
        appContext.tsd.scene.createRenderer("test_device", "pathtracer");
    renderer->setName("selected renderer");
    renderer->setParameter("pixelSamples", 7);
    auto &renderSettings =
        projectContext.project().shots.front().renderSettings;
    renderSettings.rendererLibrary = "test_device";
    renderSettings.rendererObjectIndex = renderer->index();
    renderSettings.rendererSubtype = "pathtracer";

    REQUIRE(projectContext.saveProject(root));
  }

  {
    tsd::core::DataTree manifest;
    REQUIRE(manifest.load((root / PROJECT_MANIFEST_FILENAME).string().c_str()));
    auto metadata = tsd::core::readDataTreeMetadata(manifest.root());
    REQUIRE(metadata.found());
    REQUIRE(metadata.metadata->schemaVersion == SCHEMA_VERSION);
    REQUIRE(SCHEMA_VERSION == 8);
    REQUIRE(manifest.root().child("context") == nullptr);

    tsd::core::DataTree cameras;
    tsd::core::DataTree renderers;
    REQUIRE(cameras.load((root / "scene/cameras.tsd").string().c_str()));
    REQUIRE(renderers.load((root / "scene/renderers.tsd").string().c_str()));
    REQUIRE(tsd::io::validate_CameraArchive(cameras.root()).accepted());
    REQUIRE(tsd::io::validate_RendererArchive(renderers.root()).accepted());
  }

  {
    tsd::app::Context appContext;
    ProjectContext projectContext(&appContext);
    REQUIRE(projectContext.openProject(root));

    auto &shot = projectContext.project().shots.front();
    REQUIRE(projectContext.project().shots.size() == 2);
    REQUIRE(projectContext.project().activeShotId == shot.id);
    auto *camera = projectContext.resolveShotCamera(shot);
    REQUIRE(camera);
    REQUIRE(camera->name() == shot.id + "_camera");
    REQUIRE(shot.renderSettings.rendererObjectIndex != TSD_INVALID_INDEX);
    auto renderer = appContext.tsd.scene.getObject<tsd::scene::Renderer>(
        shot.renderSettings.rendererObjectIndex);
    REQUIRE(renderer);
    REQUIRE(renderer->name() == "selected renderer");
    REQUIRE(renderer->rendererDeviceName() == "test_device");
    REQUIRE(renderer->parameter("pixelSamples"));
    REQUIRE(renderer->parameter("pixelSamples")->value().getAs<int>() == 7);
    auto &secondShot = projectContext.project().shots.back();
    REQUIRE(projectContext.resolveShotCamera(secondShot));
  }

  std::filesystem::remove_all(root);
}

SCENARIO("SciVis Studio writes empty scene pool Archives", "[SciVisStudio]")
{
  const auto root = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_empty_scene_pool_archives";
  std::filesystem::remove_all(root);

  {
    tsd::app::Context appContext;
    ProjectContext projectContext(&appContext);
    projectContext.createUnsavedProject();
    projectContext.project() = {};
    projectContext.project().name = "Empty Pools";
    appContext.tsd.scene.removeAllObjects();
    appContext.tsd.scene.defaultMaterial();

    REQUIRE(appContext.tsd.scene.numberOfObjects(ANARI_CAMERA) == 0);
    REQUIRE(appContext.tsd.scene.numberOfObjects(ANARI_RENDERER) == 0);
    REQUIRE(projectContext.saveProject(root));
  }

  tsd::core::DataTree cameras;
  tsd::core::DataTree renderers;
  REQUIRE(cameras.load((root / "scene/cameras.tsd").string().c_str()));
  REQUIRE(renderers.load((root / "scene/renderers.tsd").string().c_str()));
  REQUIRE(tsd::io::validate_CameraArchive(cameras.root()).accepted());
  REQUIRE(tsd::io::validate_RendererArchive(renderers.root()).accepted());
  REQUIRE(cameras.root()["objectDB"].child("camera") == nullptr);
  REQUIRE(renderers.root()["objectDB"].child("renderer") == nullptr);

  tsd::app::Context appContext;
  ProjectContext projectContext(&appContext);
  REQUIRE(projectContext.openProject(root));
  REQUIRE(appContext.tsd.scene.numberOfObjects(ANARI_CAMERA) == 0);
  REQUIRE(appContext.tsd.scene.numberOfObjects(ANARI_RENDERER) == 0);
  REQUIRE(appContext.tsd.scene.layer("studio"));

  std::filesystem::remove_all(root);
}

SCENARIO(
    "SciVis Studio projects round-trip standalone datasets", "[SciVisStudio]")
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
    REQUIRE(metadata.metadata->schemaVersion == SCHEMA_VERSION);
    auto *datasetNode = manifest.root()["scivisStudio"]["datasets"].child(0);
    REQUIRE(datasetNode);
    REQUIRE(datasetNode->child("id"));
    REQUIRE(datasetNode->child("name"));
    REQUIRE(datasetNode->child("sourceKind") == nullptr);
    REQUIRE(datasetNode->child("source") == nullptr);
    REQUIRE(datasetNode->child("sourceFiles") == nullptr);
    REQUIRE(manifest.root().child("context") == nullptr);
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

SCENARIO("SciVis Studio projects persist file-animation source lists in "
         "sibling Source List Files",
    "[SciVisStudio]")
{
  const auto root = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_source_list_pair";
  const auto copy = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_source_list_pair_copy";
  std::filesystem::remove_all(root);
  std::filesystem::remove_all(copy);

  const std::vector<std::string> paths = {"frames/a.raw", "frames/b.raw"};
  const auto datasetFile = root / "datasets" / "Frames.tsd";
  const auto sourcesFile = root / "datasets" / "Frames.sources";

  {
    tsd::app::Context appContext;
    ProjectContext projectContext(&appContext);
    projectContext.createUnsavedProject();
    auto &project = projectContext.project();
    auto &scene = appContext.tsd.scene;
    auto datasetsRoot =
        findDirectChild(scene.layer("studio")->root(), "datasets");
    auto dataset = makeFileAnimationDatasetRuntime(scene,
        appContext.tsd.animationMgr,
        datasetsRoot,
        "dataset_0001",
        "Frames",
        paths);
    dataset.status = DatasetStatus::Available;
    dataset.rootNode = projectContext.refFor(
        "studio", findDirectChild(datasetsRoot, "dataset_0001"));
    project.datasets.push_back(std::move(dataset));
    shot::setDatasetBinding(project.shots.front(), "dataset_0001", true);
    project.markDirty();

    // The ADR 0004 transaction stages the pair together: the plan carries
    // both files, and a failure while installing leaves neither behind.
    {
      struct : AssetTransactionFailureInjector
      {
        bool fail(AssetTransactionPhase phase,
            const std::filesystem::path &,
            std::string &message) override
        {
          if (phase != AssetTransactionPhase::ManifestInstall)
            return false;
          message = "injected manifest failure";
          return true;
        }
      } injector;
      ProjectSaveRequest request(
          project, scene, appContext.tsd.animationMgr, root);
      ProjectSaveResult result;
      REQUIRE(buildProjectSavePlan(request, result));
      const auto hasTarget = [&](const char *target) {
        return std::any_of(result.plan.assets.begin(),
            result.plan.assets.end(),
            [&](const ProjectAssetWrite &write) {
              return write.target == std::filesystem::path(target);
            });
      };
      REQUIRE(hasTarget("datasets/Frames.tsd"));
      REQUIRE(hasTarget("datasets/Frames.sources"));
      AssetTransaction transaction(&injector);
      std::string error;
      REQUIRE_FALSE(transaction.commit(result.plan, &error));
      REQUIRE_FALSE(std::filesystem::exists(datasetFile));
      REQUIRE_FALSE(std::filesystem::exists(sourcesFile));
    }

    REQUIRE(projectContext.saveProject(root));
    REQUIRE(std::filesystem::exists(datasetFile));
    REQUIRE(fileContents(sourcesFile) == "frames/a.raw\nframes/b.raw\n");
    tsd::core::DataTree assetTree;
    REQUIRE(assetTree.load(datasetFile.string().c_str()));
    REQUIRE(assetTree.root()["dataset"].child("sourceFiles") == nullptr);
  }

  {
    tsd::app::Context appContext;
    ProjectContext projectContext(&appContext);
    REQUIRE(projectContext.openProject(root));
    auto &project = projectContext.project();
    REQUIRE(project.datasets.front().status == DatasetStatus::Available);
    REQUIRE(project.datasets.front().sourceFiles.size() == 2);
    REQUIRE(project.datasets.front().sourceFiles[0].path == "frames/a.raw");
    REQUIRE(project.datasets.front().sourceFiles[0].resolvedPath
        == (root / "datasets" / "frames/a.raw").string());

    // Saves that do not touch the source list never rewrite the sibling, so
    // external edits survive: an unrelated project edit...
    const auto sentinel =
        std::filesystem::file_time_type::clock::now() - std::chrono::hours(1);
    std::filesystem::last_write_time(sourcesFile, sentinel);
    project.shots.front().name = "Unrelated Shot Edit";
    project.markDirty();
    REQUIRE(projectContext.saveProject(root));
    REQUIRE(std::filesystem::last_write_time(sourcesFile) == sentinel);

    // ...and a save that rewrites a dirty dataset file for unrelated reasons.
    project.datasets.front().dirty = true;
    project.markDirty();
    REQUIRE(projectContext.saveProject(root));
    REQUIRE(std::filesystem::last_write_time(sourcesFile) == sentinel);
  }

  {
    // A hand-edited Source List File takes effect on the next Dataset Load.
    std::ofstream out(sourcesFile, std::ios::trunc);
    out << "\n  frames/b.raw\nframes/c.raw\nframes/a.raw\n";
  }

  {
    tsd::app::Context appContext;
    ProjectContext projectContext(&appContext);
    REQUIRE(projectContext.openProject(root));
    auto &project = projectContext.project();
    const auto &dataset = project.datasets.front();
    REQUIRE(dataset.status == DatasetStatus::Available);
    REQUIRE(dataset.sourceFiles.size() == 3);
    REQUIRE(dataset.sourceFiles[0].path == "frames/b.raw");
    REQUIRE(dataset.sourceFiles[1].path == "frames/c.raw");
    REQUIRE(dataset.sourceFiles[2].path == "frames/a.raw");

    // Save As writes the pair into the new project.
    REQUIRE(projectContext.saveProject(copy));
    REQUIRE(std::filesystem::exists(copy / "datasets" / "Frames.tsd"));
    REQUIRE(fileContents(copy / "datasets" / "Frames.sources")
        == "frames/b.raw\nframes/c.raw\nframes/a.raw\n");
  }
  std::filesystem::remove_all(copy);

  {
    tsd::app::Context appContext;
    ProjectContext projectContext(&appContext);
    REQUIRE(projectContext.openProject(root));
    auto &project = projectContext.project();
    const auto id = project.datasets.front().id;

    // Renaming the dataset renames the pair. A rename is not a source-list
    // edit, so an external edit made after the load is carried over verbatim
    // rather than clobbered by the in-memory entries.
    {
      std::ofstream out(sourcesFile, std::ios::trunc);
      out << "frames/hand-edited-after-load.raw\n";
    }
    REQUIRE(projectContext.renameDataset(id, "Renamed Frames"));
    REQUIRE(projectContext.saveProject(root));
    REQUIRE(std::filesystem::exists(root / "datasets" / "Renamed Frames.tsd"));
    REQUIRE(fileContents(root / "datasets" / "Renamed Frames.sources")
        == "frames/hand-edited-after-load.raw\n");
    REQUIRE_FALSE(std::filesystem::exists(datasetFile));
    REQUIRE_FALSE(std::filesystem::exists(sourcesFile));

    // Dataset Removal deletes the pair with the asset.
    REQUIRE(projectContext.removeDataset(id));
    REQUIRE_FALSE(
        std::filesystem::exists(root / "datasets" / "Renamed Frames.tsd"));
    REQUIRE_FALSE(
        std::filesystem::exists(root / "datasets" / "Renamed Frames.sources"));
  }

  std::filesystem::remove_all(root);
}

SCENARIO(
    "SciVis Studio migrates legacy embedded sourceFiles on explicit save",
    "[SciVisStudio]")
{
  const auto root = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_source_list_migration";
  std::filesystem::remove_all(root);

  const std::vector<std::string> paths = {"legacy/a.raw", "/legacy/b.raw"};
  const auto datasetFile = root / "datasets" / "Frames.tsd";
  const auto sourcesFile = root / "datasets" / "Frames.sources";

  {
    tsd::app::Context appContext;
    ProjectContext projectContext(&appContext);
    projectContext.createUnsavedProject();
    auto &project = projectContext.project();
    auto &scene = appContext.tsd.scene;
    auto datasetsRoot =
        findDirectChild(scene.layer("studio")->root(), "datasets");
    auto dataset = makeFileAnimationDatasetRuntime(scene,
        appContext.tsd.animationMgr,
        datasetsRoot,
        "dataset_0001",
        "Frames",
        paths);
    dataset.status = DatasetStatus::Available;
    dataset.rootNode = projectContext.refFor(
        "studio", findDirectChild(datasetsRoot, "dataset_0001"));
    project.datasets.push_back(std::move(dataset));
    shot::setDatasetBinding(project.shots.front(), "dataset_0001", true);
    project.markDirty();
    REQUIRE(projectContext.saveProject(root));
  }

  // Rewrite the managed asset into the legacy embedded-sourceFiles format.
  {
    tsd::core::DataTree legacy;
    REQUIRE(legacy.load(datasetFile.string().c_str()));
    auto &files = legacy.root()["dataset"]["sourceFiles"];
    for (const auto &path : paths)
      files.append() = path;
    REQUIRE(legacy.save(datasetFile.string().c_str()));
    std::filesystem::remove(sourcesFile);
  }

  {
    tsd::app::Context appContext;
    ProjectContext projectContext(&appContext);
    REQUIRE(projectContext.openProject(root));
    auto &project = projectContext.project();
    REQUIRE(project.datasets.front().status == DatasetStatus::Available);
    REQUIRE(project.datasets.front().pendingSourceListMigration);
    REQUIRE_FALSE(project.datasets.front().dirty);
    // Merely opening never migrates.
    REQUIRE_FALSE(std::filesystem::exists(sourcesFile));

    // The next explicit save writes the Source List File verbatim and
    // rewrites the dataset file without paths.
    REQUIRE(projectContext.saveProject(root));
    REQUIRE(fileContents(sourcesFile) == "legacy/a.raw\n/legacy/b.raw\n");
    tsd::core::DataTree migrated;
    REQUIRE(migrated.load(datasetFile.string().c_str()));
    REQUIRE(migrated.root()["dataset"].child("sourceFiles") == nullptr);
    REQUIRE_FALSE(project.datasets.front().pendingSourceListMigration);

    // The migrated pair is now stable across further saves.
    const auto sentinel =
        std::filesystem::file_time_type::clock::now() - std::chrono::hours(1);
    std::filesystem::last_write_time(sourcesFile, sentinel);
    std::filesystem::last_write_time(datasetFile, sentinel);
    project.markDirty();
    REQUIRE(projectContext.saveProject(root));
    REQUIRE(std::filesystem::last_write_time(sourcesFile) == sentinel);
    REQUIRE(std::filesystem::last_write_time(datasetFile) == sentinel);
  }

  std::filesystem::remove_all(root);
}

SCENARIO("SciVis Studio opens legacy projects without rewriting them",
    "[SciVisStudio]")
{
  const auto base = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_legacy_project_versions";
  std::filesystem::remove_all(base);

  for (int version = 1; version <= 5; ++version) {
    const auto root = base / std::to_string(version);
    std::filesystem::create_directories(root);

    Project legacyProject;
    legacyProject.name = "Legacy " + std::to_string(version);
    Shot shot;
    shot.id = "shot_0001";
    shot.name = "Legacy Shot";
    legacyProject.shots.push_back(shot);
    legacyProject.activeShotId = shot.id;

    tsd::app::Context legacyContext;
    auto camera = legacyContext.tsd.scene.createObject<tsd::scene::Camera>(
        tsd::scene::tokens::camera::perspective);
    camera->setName(shot.id + "_camera");

    tsd::core::DataTree tree;
    tsd::core::writeDataTreeMetadata(tree.root(),
        {tsd::core::DATA_TREE_METADATA_ENVELOPE_VERSION,
            PROJECT_FILE_TYPE,
            PROJECT_SCHEMA,
            version});
    projectToNode(legacyProject, tree.root()["scivisStudio"]);
    tsd::app::detail::serializeLegacyApplicationContext(
        legacyContext, tree.root()["context"]);
    REQUIRE(tree.save((root / PROJECT_MANIFEST_FILENAME).string().c_str()));

    tsd::app::Context appContext;
    ProjectContext projectContext(&appContext);
    REQUIRE(projectContext.openProject(root));
    REQUIRE(projectContext.project().name == legacyProject.name);
    REQUIRE(projectContext.resolveShotCamera(
        projectContext.project().shots.front()));

    tsd::core::DataTree unchanged;
    REQUIRE(
        unchanged.load((root / PROJECT_MANIFEST_FILENAME).string().c_str()));
    auto metadata = tsd::core::readDataTreeMetadata(unchanged.root());
    REQUIRE(metadata.found());
    REQUIRE(metadata.metadata->schemaVersion == version);
    REQUIRE(unchanged.root().child("context"));
    REQUIRE_FALSE(std::filesystem::exists(root / "scene"));

    if (version == 5) {
      REQUIRE(projectContext.saveProject(root));
      tsd::core::DataTree migrated;
      REQUIRE(
          migrated.load((root / PROJECT_MANIFEST_FILENAME).string().c_str()));
      metadata = tsd::core::readDataTreeMetadata(migrated.root());
      REQUIRE(metadata.found());
      REQUIRE(metadata.metadata->schemaVersion == SCHEMA_VERSION);
      REQUIRE(migrated.root().child("context") == nullptr);
      REQUIRE(std::filesystem::exists(root / "scene/cameras.tsd"));
      REQUIRE(std::filesystem::exists(root / "scene/renderers.tsd"));
    }
  }

  std::filesystem::remove_all(base);
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
    tsd::app::detail::serializeLegacyApplicationContext(
        appContext, tree.root()["context"]);
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
    REQUIRE(metadata.metadata->schemaVersion == SCHEMA_VERSION);
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
          5});
  projectToNode(project, tree.root()["scivisStudio"]);
  tsd::scene::Scene scene;
  tsd::animation::AnimationManager animations(&scene);
  tsd::io::detail::LegacySceneSerializationOptions options;
  options.animationManager = &animations;
  tsd::io::detail::serializeLegacyScenePayload(
      scene, tree.root()["context"], options);
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
  REQUIRE(error.find("Save As requires every dataset to be available")
      != std::string::npos);
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
  const auto savedArchive = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_dataset_archive.tsd";
  std::filesystem::remove_all(root);
  std::filesystem::remove(source);
  std::filesystem::remove(savedArchive);
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

  REQUIRE(projectContext.saveDatasetArchive(originalId, savedArchive));
  auto savedValidation = validateDatasetAsset(savedArchive);
  REQUIRE(savedValidation.ok);
  REQUIRE(savedValidation.dataset.id.empty());

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
  tsd::io::save_SceneArchive(
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

  auto *loaded = projectContext.loadDatasetArchive(savedArchive, &error);
  REQUIRE(loaded);
  REQUIRE(loaded->id != originalId);
  REQUIRE(loaded->dirty);

  std::filesystem::remove_all(root);
  std::filesystem::remove(source);
  std::filesystem::remove(savedArchive);
}

SCENARIO("SciVis Studio treats the file-animation pair as one dataset asset",
    "[SciVisStudio]")
{
  const auto root = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_source_list_asset";
  const auto archiveDir = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_source_list_archives";
  std::filesystem::remove_all(root);
  std::filesystem::remove_all(archiveDir);
  std::filesystem::create_directories(archiveDir);

  tsd::app::Context appContext;
  ProjectContext projectContext(&appContext);
  projectContext.createUnsavedProject();
  auto &project = projectContext.project();
  auto &scene = appContext.tsd.scene;
  auto datasetsRoot =
      findDirectChild(scene.layer("studio")->root(), "datasets");
  auto dataset = makeFileAnimationDatasetRuntime(scene,
      appContext.tsd.animationMgr,
      datasetsRoot,
      "dataset_0001",
      "Frames",
      {"frames/a.raw", "frames/b.raw"});
  dataset.status = DatasetStatus::Available;
  dataset.rootNode = projectContext.refFor(
      "studio", findDirectChild(datasetsRoot, "dataset_0001"));
  project.datasets.push_back(std::move(dataset));
  shot::setDatasetBinding(project.shots.front(), "dataset_0001", true);
  project.markDirty();
  REQUIRE(projectContext.saveProject(root));
  const auto id = project.datasets.front().id;

  // Dataset Archive Save writes the pair.
  const auto archiveFile = archiveDir / "Archived.tsd";
  REQUIRE(projectContext.saveDatasetArchive(id, archiveFile));
  REQUIRE(std::filesystem::exists(archiveFile));
  REQUIRE(fileContents(archiveDir / "Archived.sources")
      == "frames/a.raw\nframes/b.raw\n");

  // Dataset Archive Load incorporates the pair with a fresh identity...
  std::string error;
  auto *incorporated = projectContext.loadDatasetArchive(archiveFile, &error);
  REQUIRE(incorporated);
  REQUIRE(incorporated->sourceFiles.size() == 2);
  REQUIRE(incorporated->sourceFiles[0].resolvedPath
      == (archiveDir / "frames/a.raw").string());
  REQUIRE(projectContext.removeDataset(incorporated->id));

  // ...and fails cleanly without the sibling.
  std::filesystem::remove(archiveDir / "Archived.sources");
  const auto datasetCount = project.datasets.size();
  REQUIRE(projectContext.loadDatasetArchive(archiveFile, &error) == nullptr);
  REQUIRE(error.find("Source List File") != std::string::npos);
  REQUIRE(project.datasets.size() == datasetCount);

  // Dataset Load re-reads the Source List File, so a hand edit made while the
  // dataset was Unloaded takes effect when it is loaded back.
  REQUIRE(projectContext.unloadDataset(id, &error));
  {
    std::ofstream out(root / "datasets" / "Frames.sources", std::ios::trunc);
    out << "frames/z.raw\n";
  }
  REQUIRE(projectContext.loadDataset(id, &error));
  REQUIRE(project::findDataset(project, id)->sourceFiles.size() == 1);
  REQUIRE(project::findDataset(project, id)->sourceFiles[0].path
      == "frames/z.raw");

  // Discovery scans only dataset files, and a file-animation dataset file
  // without its sibling is not a valid Dataset Candidate.
  {
    std::error_code ec;
    std::filesystem::copy_file(
        archiveFile, root / "datasets" / "Orphan.tsd", ec);
    REQUIRE_FALSE(ec);
    std::ofstream stray(root / "datasets" / "Stray.sources");
    stray << "frames/a.raw\n";
  }
  REQUIRE(projectContext.discoverDatasetCandidates().empty());
  {
    std::ofstream sibling(root / "datasets" / "Orphan.sources");
    sibling << "frames/a.raw\n";
  }
  auto candidates = projectContext.discoverDatasetCandidates();
  REQUIRE(candidates.size() == 1);
  REQUIRE(candidates.front().file.filename() == "Orphan.tsd");

  // A missing Source List File makes the dataset Unavailable at open; the
  // project itself still opens.
  std::filesystem::remove(root / "datasets" / "Frames.sources");
  {
    tsd::app::Context reopenedContext;
    ProjectContext reopened(&reopenedContext);
    REQUIRE(reopened.openProject(root));
    REQUIRE(reopened.project().datasets.front().status
        == DatasetStatus::Unavailable);
  }

  // Save As copies an Unloaded dataset's pair without loading it.
  const auto destination = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_source_list_asset_save_as";
  std::filesystem::remove_all(destination);
  {
    std::ofstream out(root / "datasets" / "Frames.sources", std::ios::trunc);
    out << "frames/z.raw\n";
  }
  REQUIRE(projectContext.unloadDataset(id, &error));
  REQUIRE(
      projectContext.saveProject(destination, nullptr, "", nullptr, &error));
  REQUIRE(validateDatasetAsset(destination / "datasets" / "Frames.tsd").ok);
  REQUIRE(fileContents(destination / "datasets" / "Frames.sources")
      == "frames/z.raw\n");

  std::filesystem::remove_all(root);
  std::filesystem::remove_all(destination);
  std::filesystem::remove_all(archiveDir);
}

SCENARIO("SciVis Studio unloads a clean dataset without touching its asset",
    "[SciVisStudio]")
{
  const auto root = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_dataset_unload";
  const auto source = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_dataset_unload.obj";
  std::filesystem::remove_all(root);
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
  const auto datasetId = dataset->id;
  REQUIRE(projectContext.saveProject(root));

  auto &project = projectContext.project();
  auto &record = project.datasets.front();
  REQUIRE_FALSE(record.dirty);
  REQUIRE_FALSE(project.dirty);
  const auto assetFile = root / "datasets" / "Mesh.tsd";
  const auto assetWriteTime = std::filesystem::last_write_time(assetFile);
  const auto geometryCount =
      appContext.tsd.scene.numberOfObjects(ANARI_GEOMETRY);
  REQUIRE(geometryCount > 0);

  std::string error;
  REQUIRE(projectContext.unloadDataset(datasetId, &error));

  THEN("The runtime representation is gone but the inventory entry remains")
  {
    REQUIRE(record.residency == DatasetResidency::Unloaded);
    REQUIRE(record.status == DatasetStatus::Available);
    REQUIRE_FALSE(projectContext.resolveDatasetRoot(record));
    REQUIRE(appContext.tsd.scene.numberOfObjects(ANARI_GEOMETRY)
        < geometryCount);
    REQUIRE(record.id == datasetId);
    REQUIRE_FALSE(record.dirty);
    REQUIRE(
        shot::findDatasetBinding(*project::activeShot(project), datasetId));
  }

  THEN("Unload marks the project dirty and never writes to disk")
  {
    REQUIRE(project.dirty);
    REQUIRE(std::filesystem::last_write_time(assetFile) == assetWriteTime);
  }

  std::filesystem::remove_all(root);
  std::filesystem::remove(source);
}

SCENARIO("SciVis Studio Dataset Load recreates the runtime from the asset",
    "[SciVisStudio]")
{
  const auto root = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_dataset_load";
  const auto source = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_dataset_load.obj";
  std::filesystem::remove_all(root);
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
  const auto datasetId = dataset->id;
  REQUIRE(projectContext.saveProject(root));

  auto &project = projectContext.project();
  auto &record = project.datasets.front();
  const auto geometryCount =
      appContext.tsd.scene.numberOfObjects(ANARI_GEOMETRY);
  const auto savedFrameCount = project::activeShot(project)->frameCount;
  const auto savedCurrentFrame = project::activeShot(project)->currentFrame;

  std::string error;
  REQUIRE(projectContext.unloadDataset(datasetId, &error));
  REQUIRE(projectContext.loadDataset(datasetId, &error));

  THEN("The dataset is resident again with its identity intact")
  {
    REQUIRE(record.residency == DatasetResidency::Loaded);
    REQUIRE(record.status == DatasetStatus::Available);
    REQUIRE(record.id == datasetId);
    REQUIRE(record.name == "Mesh");
    REQUIRE_FALSE(record.dirty);
    auto datasetRoot = projectContext.resolveDatasetRoot(record);
    REQUIRE(datasetRoot);
    REQUIRE((*datasetRoot)->isEnabled());
    REQUIRE(appContext.tsd.scene.numberOfObjects(ANARI_GEOMETRY)
        == geometryCount);
    REQUIRE(
        shot::findDatasetBinding(*project::activeShot(project), datasetId));
    REQUIRE(project.dirty);
  }

  THEN("Dataset Load never mutates shot state")
  {
    REQUIRE(project::activeShot(project)->frameCount == savedFrameCount);
    REQUIRE(project::activeShot(project)->currentFrame == savedCurrentFrame);
  }

  THEN("Loading an already-loaded dataset is a no-op success")
  {
    REQUIRE(projectContext.loadDataset(datasetId, &error));
    REQUIRE(appContext.tsd.scene.numberOfObjects(ANARI_GEOMETRY)
        == geometryCount);
  }

  std::filesystem::remove_all(root);
  std::filesystem::remove(source);
}

SCENARIO("SciVis Studio residency guards keep unloaded datasets read-only",
    "[SciVisStudio]")
{
  const auto root = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_residency_guards";
  const auto source = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_residency_guards.obj";
  std::filesystem::remove_all(root);
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
  const auto datasetId = dataset->id;
  REQUIRE(projectContext.saveProject(root));
  auto &project = projectContext.project();
  auto &record = project.datasets.front();

  GIVEN("A dirty dataset")
  {
    record.dirty = true;

    THEN("Unload refuses rather than discard unsaved changes")
    {
      std::string error;
      REQUIRE_FALSE(projectContext.unloadDataset(datasetId, &error));
      REQUIRE(error.find("save") != std::string::npos);
      REQUIRE(record.residency == DatasetResidency::Loaded);
      REQUIRE(projectContext.resolveDatasetRoot(record));
    }
  }

  GIVEN("A dataset that is importing")
  {
    record.status = DatasetStatus::Importing;

    THEN("Unload refuses")
    {
      std::string error;
      REQUIRE_FALSE(projectContext.unloadDataset(datasetId, &error));
      REQUIRE(record.residency == DatasetResidency::Loaded);
    }
  }

  GIVEN("An unloaded dataset")
  {
    std::string error;
    REQUIRE(projectContext.unloadDataset(datasetId, &error));

    THEN("Operations that touch the asset require loading first")
    {
      REQUIRE_FALSE(projectContext.renameDataset(datasetId, "Other", &error));
      REQUIRE(error.find("load") != std::string::npos);
      REQUIRE(record.name == "Mesh");
      REQUIRE_FALSE(projectContext.reimportStaticDataset(datasetId, &error));
      const auto archive = root / "standalone.tsd";
      REQUIRE_FALSE(
          projectContext.saveDatasetArchive(datasetId, archive, &error));
      REQUIRE_FALSE(std::filesystem::exists(archive));
    }

    THEN("An in-place asset rewrite requires loading first")
    {
      record.dirty = true;
      REQUIRE_FALSE(
          projectContext.saveProject(root, nullptr, "", nullptr, &error));
      REQUIRE(error.find("read-only") != std::string::npos);
    }

    THEN("Shot bindings and Dataset Removal remain available")
    {
      auto *shot = project::activeShot(project);
      shot::setDatasetBinding(*shot, datasetId, false);
      REQUIRE_FALSE(shot::findDatasetBinding(*shot, datasetId)->enabled);

      REQUIRE(projectContext.removeDataset(datasetId, false, &error));
      REQUIRE(project.datasets.empty());
      REQUIRE_FALSE(std::filesystem::exists(root / "datasets" / "Mesh.tsd"));
    }

    THEN("A failed load changes nothing except revealing unavailability")
    {
      std::filesystem::remove(root / "datasets" / "Mesh.tsd");
      const auto objectCount =
          appContext.tsd.scene.numberOfObjects(ANARI_GEOMETRY);
      REQUIRE_FALSE(projectContext.loadDataset(datasetId, &error));
      REQUIRE(record.residency == DatasetResidency::Unloaded);
      REQUIRE(record.status == DatasetStatus::Unavailable);
      REQUIRE_FALSE(projectContext.resolveDatasetRoot(record));
      REQUIRE(appContext.tsd.scene.numberOfObjects(ANARI_GEOMETRY)
          == objectCount);
    }
  }

  GIVEN("A loaded dataset whose asset file has vanished")
  {
    std::filesystem::remove(root / "datasets" / "Mesh.tsd");

    THEN("Unload refuses rather than discard the only copy of the data")
    {
      std::string error;
      REQUIRE_FALSE(projectContext.unloadDataset(datasetId, &error));
      REQUIRE(error.find("missing") != std::string::npos);
      REQUIRE(record.residency == DatasetResidency::Loaded);
      REQUIRE(projectContext.resolveDatasetRoot(record));
    }
  }

  std::filesystem::remove_all(root);
  std::filesystem::remove(source);
}

SCENARIO("SciVis Studio dataset residency survives save and open",
    "[SciVisStudio]")
{
  const auto root = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_residency_roundtrip";
  const auto source = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_residency_roundtrip.obj";
  std::filesystem::remove_all(root);
  std::filesystem::remove(source);
  {
    std::ofstream obj(source);
    obj << "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
  }

  DatasetID datasetId;
  {
    tsd::app::Context appContext;
    ProjectContext projectContext(&appContext);
    projectContext.createUnsavedProject();
    auto *dataset = projectContext.addStaticDataset(
        "Mesh", source, tsd::io::ImporterType::OBJ);
    REQUIRE(dataset);
    datasetId = dataset->id;
    REQUIRE(projectContext.saveProject(root));
    REQUIRE(projectContext.unloadDataset(datasetId));
    // Saving with an unloaded dataset persists residency without needing the
    // runtime.
    REQUIRE(projectContext.saveProject(root));
    REQUIRE_FALSE(projectContext.project().dirty);
  }

  {
    tsd::app::Context appContext;
    ProjectContext projectContext(&appContext);
    REQUIRE(projectContext.openProject(root));
    auto &project = projectContext.project();
    REQUIRE(project.datasets.size() == 1);
    auto &record = project.datasets.front();

    THEN("Opening hydrates only resident datasets")
    {
      REQUIRE(record.residency == DatasetResidency::Unloaded);
      REQUIRE(record.status == DatasetStatus::Available);
      REQUIRE_FALSE(record.dirty);
      REQUIRE_FALSE(project.dirty);
      REQUIRE_FALSE(projectContext.resolveDatasetRoot(record));
      REQUIRE(appContext.tsd.scene.numberOfObjects(ANARI_GEOMETRY) == 0);
    }

    THEN("An explicit Dataset Load brings it back into the scene")
    {
      REQUIRE(projectContext.loadDataset(datasetId));
      REQUIRE(record.residency == DatasetResidency::Loaded);
      REQUIRE(appContext.tsd.scene.numberOfObjects(ANARI_GEOMETRY) > 0);
    }
  }

  std::filesystem::remove(root / "datasets" / "Mesh.tsd");
  {
    tsd::app::Context appContext;
    ProjectContext projectContext(&appContext);
    REQUIRE(projectContext.openProject(root));
    auto &record = projectContext.project().datasets.front();

    THEN("A missing asset makes an unloaded dataset definitively Unavailable")
    {
      REQUIRE(record.residency == DatasetResidency::Unloaded);
      REQUIRE(record.status == DatasetStatus::Unavailable);
    }
  }

  std::filesystem::remove_all(root);
  std::filesystem::remove(source);
}

SCENARIO("SciVis Studio Save As copies unloaded datasets", "[SciVisStudio]")
{
  const auto root = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_save_as_residency";
  const auto destination = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_save_as_residency_destination";
  const auto source = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_save_as_residency.obj";
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
  auto *parked = projectContext.addStaticDataset(
      "Parked", source, tsd::io::ImporterType::OBJ);
  REQUIRE(parked);
  const auto parkedId = parked->id;
  REQUIRE(projectContext.addStaticDataset(
      "Resident", source, tsd::io::ImporterType::OBJ));
  REQUIRE(projectContext.saveProject(root));

  const auto readBytes = [](const std::filesystem::path &file) {
    std::ifstream input(file, std::ios::binary);
    return std::vector<char>(std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
  };
  const auto sourceArchive = root / "datasets/Parked.tsd";
  const auto sourceBytes = readBytes(sourceArchive);
  REQUIRE_FALSE(sourceBytes.empty());
  REQUIRE(projectContext.unloadDataset(parkedId));

  std::string error;
  REQUIRE(
      projectContext.saveProject(destination, nullptr, "", nullptr, &error));
  REQUIRE(readBytes(destination / "datasets/Parked.tsd") == sourceBytes);
  REQUIRE(validateDatasetAsset(destination / "datasets/Parked.tsd").ok);
  REQUIRE(validateDatasetAsset(destination / "datasets/Resident.tsd").ok);

  tsd::app::Context reopenedAppContext;
  ProjectContext reopened(&reopenedAppContext);
  REQUIRE(reopened.openProject(destination));
  REQUIRE(reopened.project().datasets.size() == 2);
  REQUIRE(
      reopened.project().datasets[0].residency == DatasetResidency::Unloaded);
  REQUIRE(reopened.project().datasets[0].status == DatasetStatus::Available);
  REQUIRE(reopened.project().datasets[1].residency == DatasetResidency::Loaded);
  REQUIRE(reopened.project().datasets[1].status == DatasetStatus::Available);

  std::filesystem::remove_all(root);
  std::filesystem::remove_all(destination);
  std::filesystem::remove(source);
}

SCENARIO("SciVis Studio Save As renames colliding unloaded datasets",
    "[SciVisStudio]")
{
  const auto root = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_save_as_unloaded_collision";
  const auto destination = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_save_as_unloaded_collision_destination";
  const auto source = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_save_as_unloaded_collision.obj";
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
  auto *first = projectContext.addStaticDataset(
      "First", source, tsd::io::ImporterType::OBJ);
  REQUIRE(first);
  const auto firstId = first->id;
  auto *second = projectContext.addStaticDataset(
      "Second", source, tsd::io::ImporterType::OBJ);
  REQUIRE(second);
  const auto secondId = second->id;
  REQUIRE(projectContext.saveProject(root));
  REQUIRE(projectContext.unloadDataset(firstId));
  REQUIRE(projectContext.unloadDataset(secondId));

  auto &datasets = projectContext.project().datasets;
  datasets[0].name = "Duplicate";
  datasets[1].name = "Duplicate";
  projectContext.project().markDirty();

  std::string error;
  REQUIRE(
      projectContext.saveProject(destination, nullptr, "", nullptr, &error));
  const auto firstArchive =
      validateDatasetAsset(destination / "datasets/Duplicate.tsd");
  REQUIRE(firstArchive.ok);
  REQUIRE(firstArchive.dataset.name == "Duplicate");
  const auto secondArchive =
      validateDatasetAsset(destination / "datasets/Duplicate (2).tsd");
  REQUIRE(secondArchive.ok);
  REQUIRE(secondArchive.dataset.name == "Duplicate (2)");

  tsd::app::Context reopenedAppContext;
  ProjectContext reopened(&reopenedAppContext);
  REQUIRE(reopened.openProject(destination));
  REQUIRE(reopened.loadDataset(firstId));
  REQUIRE(reopened.loadDataset(secondId));

  std::filesystem::remove_all(root);
  std::filesystem::remove_all(destination);
  std::filesystem::remove(source);
}

SCENARIO("SciVis Studio --openUnloaded overrides initial residency",
    "[SciVisStudio]")
{
  GIVEN("The application command line")
  {
    tsd::app::Context appContext;
    std::vector<std::string> args{
        "scivisStudio", "--openUnloaded", "/some/project"};
    appContext.parseCommandLine(args);

    THEN("--openUnloaded is recognized alongside the project directory")
    {
      REQUIRE(appContext.commandLine.openUnloaded);
      REQUIRE(appContext.commandLine.stateFile == "/some/project");
    }
  }

  const auto root = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_open_unloaded";
  const auto source = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_open_unloaded.obj";
  std::filesystem::remove_all(root);
  std::filesystem::remove(source);
  {
    std::ofstream obj(source);
    obj << "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
  }

  {
    tsd::app::Context appContext;
    ProjectContext projectContext(&appContext);
    projectContext.createUnsavedProject();
    REQUIRE(projectContext.addStaticDataset(
        "Mesh", source, tsd::io::ImporterType::OBJ));
    REQUIRE(projectContext.saveProject(root));
  }

  ProjectOpenOptions openUnloaded;
  openUnloaded.openUnloaded = true;

  {
    tsd::app::Context appContext;
    ProjectContext projectContext(&appContext);
    std::string error;
    REQUIRE(projectContext.openProject(
        root, nullptr, nullptr, nullptr, &error, openUnloaded));
    auto &project = projectContext.project();
    auto &record = project.datasets.front();

    THEN("The override changes initial residency and dirties the project")
    {
      REQUIRE(record.residency == DatasetResidency::Unloaded);
      REQUIRE(record.status == DatasetStatus::Available);
      REQUIRE_FALSE(projectContext.resolveDatasetRoot(record));
      REQUIRE(appContext.tsd.scene.numberOfObjects(ANARI_GEOMETRY) == 0);
      REQUIRE(project.dirty);
    }

    // Session residency is the single source of truth: saving persists it.
    REQUIRE(projectContext.saveProject(root));
  }

  {
    tsd::app::Context appContext;
    ProjectContext projectContext(&appContext);
    std::string error;
    REQUIRE(projectContext.openProject(
        root, nullptr, nullptr, nullptr, &error, openUnloaded));

    THEN("An override that changes nothing leaves the project clean")
    {
      auto &project = projectContext.project();
      REQUIRE(
          project.datasets.front().residency == DatasetResidency::Unloaded);
      REQUIRE_FALSE(project.dirty);
    }
  }

  GIVEN("A pre-v5 legacy project with an embedded dataset")
  {
    const auto legacyRoot = std::filesystem::temp_directory_path()
        / "tsd_scivis_studio_open_unloaded_legacy";
    std::filesystem::remove_all(legacyRoot);
    std::filesystem::create_directories(legacyRoot);
    {
      tsd::app::Context legacyContext;
      ProjectContext legacyProject(&legacyContext);
      legacyProject.createUnsavedProject();
      auto &project = legacyProject.project();
      auto &scene = legacyContext.tsd.scene;
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
      dataset.rootNode = legacyProject.refFor("studio", datasetRoot);
      project.datasets.push_back(dataset);

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
      tsd::app::detail::serializeLegacyApplicationContext(
          legacyContext, tree.root()["context"]);
      REQUIRE(
          tree.save((legacyRoot / PROJECT_MANIFEST_FILENAME).string().c_str()));
    }

    THEN("--openUnloaded is ignored and the project remains saveable")
    {
      tsd::app::Context appContext;
      ProjectContext projectContext(&appContext);
      std::string error;
      REQUIRE(projectContext.openProject(
          legacyRoot, nullptr, nullptr, nullptr, &error, openUnloaded));
      auto &record = projectContext.project().datasets.front();
      REQUIRE(record.residency == DatasetResidency::Loaded);
      REQUIRE(record.status == DatasetStatus::Available);
      REQUIRE(projectContext.saveProject(legacyRoot));
    }
    std::filesystem::remove_all(legacyRoot);
  }

  std::filesystem::remove_all(root);
  std::filesystem::remove(source);
}

SCENARIO("SciVis Studio shot rendering materializes bound datasets",
    "[SciVisStudio]")
{
  const auto root = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_render_residency";
  const auto source = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_render_residency.obj";
  std::filesystem::remove_all(root);
  std::filesystem::remove(source);
  {
    std::ofstream obj(source);
    obj << "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
  }

  tsd::app::Context appContext;
  ProjectContext projectContext(&appContext);
  projectContext.createUnsavedProject();
  const auto firstId =
      projectContext.addStaticDataset("First", source, tsd::io::ImporterType::OBJ)
          ->id;
  const auto secondId =
      projectContext.addStaticDataset("Second", source, tsd::io::ImporterType::OBJ)
          ->id;
  const auto thirdId =
      projectContext.addStaticDataset("Third", source, tsd::io::ImporterType::OBJ)
          ->id;
  REQUIRE(projectContext.saveProject(root));
  REQUIRE(projectContext.unloadDataset(secondId));
  REQUIRE(projectContext.unloadDataset(thirdId));
  REQUIRE(projectContext.saveProject(root));

  auto &project = projectContext.project();
  auto *shot = project::activeShot(project);
  shot::setDatasetBinding(*shot, thirdId, false);
  auto *second = project::findDataset(project, secondId);
  auto *third = project::findDataset(project, thirdId);

  GIVEN("A bound, enabled dataset that is unloaded")
  {
    ShotDatasetResidencyRestore restore;
    std::string error;
    REQUIRE(makeShotDatasetsResident(projectContext, *shot, restore, &error));

    THEN("It is materialized for the render and restored afterward")
    {
      REQUIRE(project::findDataset(project, firstId)->residency
          == DatasetResidency::Loaded);
      REQUIRE(second->residency == DatasetResidency::Loaded);
      REQUIRE(second->status == DatasetStatus::Available);
      // A disabled binding is not part of the shot's rendered intent.
      REQUIRE(third->residency == DatasetResidency::Unloaded);
      REQUIRE(restore.loadedForRender == std::vector<DatasetID>{secondId});

      restoreShotDatasetResidency(projectContext, restore);
      REQUIRE(second->residency == DatasetResidency::Unloaded);
      REQUIRE_FALSE(projectContext.resolveDatasetRoot(*second));
      // Temporary render loads never change what a save would persist.
      REQUIRE_FALSE(project.dirty);
    }
  }

  GIVEN("A bound, enabled dataset that cannot be made resident")
  {
    std::filesystem::remove(root / "datasets" / "Second.tsd");

    THEN("Materialization hard-errors up front and restores what it loaded")
    {
      ShotDatasetResidencyRestore restore;
      std::string error;
      REQUIRE_FALSE(
          makeShotDatasetsResident(projectContext, *shot, restore, &error));
      REQUIRE(error.find("Second") != std::string::npos);
      REQUIRE(second->residency == DatasetResidency::Unloaded);
      REQUIRE_FALSE(project.dirty);
    }
  }

  GIVEN("A bound, enabled dataset that is missing from the project")
  {
    const DatasetID missingId = "dataset_missing";
    shot::setDatasetBinding(*shot, missingId, true);

    THEN("Materialization hard-errors up front and restores what it loaded")
    {
      ShotDatasetResidencyRestore restore;
      std::string error;
      REQUIRE_FALSE(
          makeShotDatasetsResident(projectContext, *shot, restore, &error));
      REQUIRE(error
          == "enabled shot binding references a missing dataset: " + missingId);
      REQUIRE(second->residency == DatasetResidency::Unloaded);
      REQUIRE_FALSE(project.dirty);
    }
  }

  GIVEN("A bound, disabled dataset that is missing from the project")
  {
    shot::setDatasetBinding(*shot, "dataset_missing", false);

    THEN("It does not prevent materialization")
    {
      ShotDatasetResidencyRestore restore;
      std::string error;
      REQUIRE(makeShotDatasetsResident(projectContext, *shot, restore, &error));
      REQUIRE(error.empty());
      REQUIRE(second->residency == DatasetResidency::Loaded);

      restoreShotDatasetResidency(projectContext, restore);
      REQUIRE(second->residency == DatasetResidency::Unloaded);
      REQUIRE_FALSE(project.dirty);
    }
  }

  std::filesystem::remove_all(root);
  std::filesystem::remove(source);
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

SCENARIO("SciVis Studio pool Archive failures preserve the previous project",
    "[SciVisStudio]")
{
  const auto root = std::filesystem::temp_directory_path()
      / "tsd_scivis_studio_pool_archive_rollback";
  std::filesystem::remove_all(root);

  tsd::app::Context appContext;
  ProjectContext projectContext(&appContext);
  projectContext.createUnsavedProject();
  REQUIRE(projectContext.saveProject(root));

  auto readBytes = [](const std::filesystem::path &file) {
    std::ifstream input(file, std::ios::binary);
    return std::vector<char>(std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
  };
  const auto manifest = root / PROJECT_MANIFEST_FILENAME;
  const auto cameras = root / "scene/cameras.tsd";
  const auto renderers = root / "scene/renderers.tsd";
  const auto manifestBefore = readBytes(manifest);
  const auto camerasBefore = readBytes(cameras);
  const auto renderersBefore = readBytes(renderers);

  auto camera = appContext.tsd.scene.getObject<tsd::scene::Camera>(0);
  REQUIRE(camera);
  auto geometry = appContext.tsd.scene.createObject<tsd::scene::Geometry>(
      tsd::scene::tokens::geometry::sphere);
  camera->setParameterObject("invalidPoolDependency", *geometry);
  projectContext.project().shots.front().name = "must not be committed";
  projectContext.project().markDirty();

  std::string error;
  REQUIRE_FALSE(projectContext.saveProject(root, nullptr, "", nullptr, &error));
  REQUIRE(error.find("Camera pool Archive") != std::string::npos);
  REQUIRE(readBytes(manifest) == manifestBefore);
  REQUIRE(readBytes(cameras) == camerasBefore);
  REQUIRE(readBytes(renderers) == renderersBefore);
  for (const auto &entry :
      std::filesystem::directory_iterator(root / "scene")) {
    REQUIRE(
        entry.path().filename().string().find(".stage-") == std::string::npos);
    REQUIRE(
        entry.path().filename().string().find(".backup-") == std::string::npos);
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
    tsd::app::detail::serializeLegacyApplicationContext(
        appContext, tree.root()["context"]);
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
    tsd::app::serialize_CameraPose(currentPose, cameraRig["current"]["orbit"]);

    tsd::rendering::CameraPose keyframePose;
    keyframePose.lookat = {7.f, 8.f, 9.f};
    keyframePose.azeldist = {10.f, 20.f, 30.f};
    keyframePose.fixedDist = 30.f;
    auto &keyframe = cameraRig["keyframes"].append();
    keyframe["frame"] = 11;
    keyframe["name"] = "legacy";
    keyframe["interpolationToNext"] = "Ease Out + In";
    tsd::app::serialize_CameraPose(
        keyframePose, keyframe["manipulator"]["orbit"]);

    tsd::app::detail::serializeLegacyApplicationContext(
        appContext, tree.root()["context"]);
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
    REQUIRE(validateRigName("rig-01 Copy", &error));
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

SCENARIO("SciVis Studio v4 projects round-trip standalone Rig Archives",
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

    THEN("one Archive is written per rig")
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

    THEN("light rigs and their lights are restored from Archives")
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

    THEN("camera rig keyframes are restored from Archives")
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

    THEN("only the explicitly superseded Rig Archive is removed")
    {
      REQUIRE(std::filesystem::exists(root / "lights" / "Key Light.tsd"));
      REQUIRE_FALSE(
          std::filesystem::exists(root / "lights" / (oldLightName + ".tsd")));
      REQUIRE(std::filesystem::exists(unlisted));
    }
  }

  std::filesystem::remove_all(root);
}

SCENARIO("SciVis Studio tolerates a missing Light Rig Archive on open",
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
