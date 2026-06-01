// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ProjectContext.h"

#include "ProjectSerialization.h"

#include "tsd/core/DataTreeMetadata.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/io/serialization.hpp"
#include "tsd/rendering/view/ManipulatorToTSD.hpp"
#include "tsd/scene/objects/Camera.hpp"
#include "tsd/scene/objects/Light.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <vector>

namespace tsd::scivis_studio {


static tsd::scene::LayerNodeRef findDirectChild(
    tsd::scene::LayerNodeRef parent, const std::string &name)
{
  if (!parent)
    return {};

  auto child = parent->next();
  while (child && child != parent) {
    if ((*child)->name() == name)
      return child;
    child = child->sibling();
  }

  return {};
}

static bool hasChildNodes(tsd::scene::LayerNodeRef parent)
{
  if (!parent)
    return false;

  auto child = parent->next();
  return child && child != parent;
}

ProjectContext::ProjectContext(tsd::app::Context *ctx) : m_ctx(ctx)
{
  installAnimationManagerCallback();
}

void ProjectContext::setAppContext(tsd::app::Context *ctx)
{
  m_ctx = ctx;
  installAnimationManagerCallback();
}

tsd::app::Context *ProjectContext::appContext() const
{
  return m_ctx;
}

Project &ProjectContext::project()
{
  return m_project;
}

const Project &ProjectContext::project() const
{
  return m_project;
}

void ProjectContext::resetScene()
{
  if (!m_ctx)
    return;
  m_ctx->clearSelected();
  m_ctx->tsd.animationMgr.removeAllAnimations();
  m_ctx->tsd.scene.removeAllObjects();
  m_ctx->tsd.scene.defaultMaterial();
  m_ctx->tsd.scene.defaultCamera();
}

void ProjectContext::installAnimationManagerCallback()
{
  if (!m_ctx)
    return;

  m_ctx->tsd.animationMgr.setTimeChangedCallback(
      [this](float) { updateActiveShotFromAnimationTime(); });
}

tsd::scene::LayerNodeRef ProjectContext::ensureChild(
    tsd::scene::LayerNodeRef parent, const char *name)
{
  if (auto found = findDirectChild(parent, name))
    return found;

  return m_ctx->tsd.scene.insertChildNode(parent, name);
}

tsd::scene::LayerNodeRef ProjectContext::ensureStudioRoot()
{
  auto *layer = m_ctx->tsd.scene.addLayer("studio");
  return layer ? layer->root() : tsd::scene::LayerNodeRef{};
}

tsd::scene::LayerNodeRef ProjectContext::ensureDatasetsRoot()
{
  return ensureChild(ensureStudioRoot(), "datasets");
}

tsd::scene::LayerNodeRef ProjectContext::ensureShotsRoot()
{
  return ensureChild(ensureStudioRoot(), "shots");
}

tsd::scene::LayerNodeRef ProjectContext::ensureLightRigsRoot()
{
  return ensureChild(ensureStudioRoot(), "lightRigs");
}

SceneNodeRef ProjectContext::refFor(
    const std::string &layerName, tsd::scene::LayerNodeRef ref) const
{
  return {layerName, ref ? ref.index() : TSD_INVALID_INDEX};
}

tsd::scene::LayerNodeRef ProjectContext::resolve(const SceneNodeRef &ref) const
{
  if (!m_ctx || ref.layerName.empty() || ref.nodeIndex == TSD_INVALID_INDEX)
    return {};

  auto *layer = m_ctx->tsd.scene.layer(ref.layerName.c_str());
  return layer ? layer->at(ref.nodeIndex) : tsd::scene::LayerNodeRef{};
}

tsd::scene::Object *ProjectContext::resolve(const SceneObjectRef &ref) const
{
  if (!m_ctx || ref.type == ANARI_UNKNOWN
      || ref.objectIndex == TSD_INVALID_INDEX)
    return nullptr;
  return m_ctx->tsd.scene.getObject(ref.type, ref.objectIndex);
}

tsd::scene::LayerNodeRef ProjectContext::resolveDatasetRoot(Dataset &dataset)
{
  if (!m_ctx)
    return {};

  auto *layer = m_ctx->tsd.scene.layer("studio");
  if (layer) {
    auto datasetsRoot = findDirectChild(layer->root(), "datasets");
    auto datasetRoot = findDirectChild(datasetsRoot, dataset.id);
    if (datasetRoot) {
      dataset.rootNode = refFor("studio", datasetRoot);
      return datasetRoot;
    }
  }

  return resolve(dataset.rootNode);
}

tsd::scene::LayerNodeRef ProjectContext::resolveLightRigRoot(LightRig &rig)
{
  if (!m_ctx)
    return {};

  auto *layer = m_ctx->tsd.scene.layer("studio");
  if (layer) {
    auto lightRigsRoot = findDirectChild(layer->root(), "lightRigs");
    auto rigRoot = findDirectChild(lightRigsRoot, rig.id);
    if (rigRoot) {
      rig.rootNode = refFor("studio", rigRoot);
      return rigRoot;
    }
  }

  return resolve(rig.rootNode);
}

tsd::scene::Object *ProjectContext::resolveShotCamera(Shot &shot)
{
  if (!m_ctx)
    return nullptr;

  const auto cameraName = shot.id + "_camera";
  const auto &cameras = m_ctx->tsd.scene.objectDB().camera;
  tsd::core::foreach_item_const(cameras, [&](const tsd::scene::Camera *camera) {
    if (camera && camera->name() == cameraName)
      shot.camera = {ANARI_CAMERA, camera->index()};
  });

  return resolve(shot.camera);
}

void ProjectContext::ensureRendererDefaults(Shot &shot)
{
  if (!shot.renderSettings.rendererLibrary.empty())
    return;

  if (!m_ctx)
    return;

  for (const auto &lib : m_ctx->anari.libraryList()) {
    if (lib != "{none}") {
      shot.renderSettings.rendererLibrary = lib;
      break;
    }
  }
}

LightRig *ProjectContext::createLightRig(const std::string &name)
{
  if (!m_ctx)
    return nullptr;

  LightRig rig;
  rig.id = project::nextLightRigId(m_project);
  rig.name = name.empty()
      ? ("Light Rig " + std::to_string(m_project.lightRigs.size() + 1))
      : name;

  auto rigRoot = ensureChild(ensureLightRigsRoot(), rig.id.c_str());
  rig.rootNode = refFor("studio", rigRoot);
  m_project.lightRigs.push_back(std::move(rig));
  m_project.markDirty();
  return &m_project.lightRigs.back();
}

tsd::scene::LayerNodeRef ProjectContext::addLightToRig(
    LightRig &rig, const std::string &subtype)
{
  if (!m_ctx)
    return {};

  auto rigRoot = resolveLightRigRoot(rig);
  if (!rigRoot)
    return {};

  const auto lightSubtype =
      subtype.empty() ? std::string("directional") : subtype;
  auto light = m_ctx->tsd.scene.createObject<tsd::scene::Light>(lightSubtype);
  const auto lightName =
      lightSubtype + "Light_" + std::to_string(light->index());
  light->setName(lightName);
  if (lightSubtype == "directional") {
    light->setParameter("direction", tsd::math::float2(0.f, 240.f));
    light->setParameter("irradiance", 1.f);
  }

  auto node =
      m_ctx->tsd.scene.insertChildObjectNode(rigRoot, light, lightName.c_str());
  m_project.markDirty();
  applyActiveShot();
  return node;
}

bool ProjectContext::removeLightFromRig(
    LightRig &rig, tsd::scene::LayerNodeRef lightNode)
{
  if (!m_ctx || !lightNode)
    return false;

  auto rigRoot = resolveLightRigRoot(rig);
  if (!rigRoot)
    return false;

  auto *layer = (*rigRoot)->layer();
  if (!layer || !layer->isAncestorOf(rigRoot, lightNode)
      || !(*lightNode)->isObject() || (*lightNode)->type() != ANARI_LIGHT)
    return false;

  m_ctx->tsd.scene.removeNode(lightNode, true);
  m_project.markDirty();
  applyActiveShot();
  return true;
}

int ProjectContext::shotUseCount(const LightRigID &id) const
{
  return static_cast<int>(std::count_if(m_project.shots.begin(),
      m_project.shots.end(),
      [&](const Shot &shot) { return shot.lightRigId == id; }));
}

bool ProjectContext::removeLightRig(const LightRigID &id)
{
  if (!m_ctx)
    return false;

  auto itr = std::find_if(m_project.lightRigs.begin(),
      m_project.lightRigs.end(),
      [&](const LightRig &rig) { return rig.id == id; });
  if (itr == m_project.lightRigs.end())
    return false;

  auto rigRoot = resolveLightRigRoot(*itr);
  if (rigRoot)
    m_ctx->tsd.scene.removeNode(rigRoot, true);

  for (auto &shot : m_project.shots) {
    if (shot.lightRigId == id)
      shot.lightRigId.clear();
  }

  m_project.lightRigs.erase(itr);
  m_project.markDirty();
  applyActiveShot();
  return true;
}

LightRig *ProjectContext::ensureDefaultLightRig()
{
  if (!m_project.lightRigs.empty())
    return &m_project.lightRigs.front();

  auto *rig = createLightRig("Default");
  if (!rig)
    return nullptr;

  addLightToRig(*rig, "directional");
  if (auto root = resolveLightRigRoot(*rig)) {
    auto *layer = (*root)->layer();
    layer->traverse(root, [&](auto &node, int) {
      if (node->isObject() && node->type() == ANARI_LIGHT) {
        if (auto *light = node->getObject())
          light->setName("mainLight");
        node->name() = "mainLight";
        return false;
      }
      return true;
    });
  }
  return rig;
}

void ProjectContext::createUnsavedProject()
{
  resetScene();

  m_project = {};
  m_project.name = "Untitled";

  auto datasetsRoot = ensureDatasetsRoot();
  auto shotsRoot = ensureShotsRoot();
  auto *defaultRig = ensureDefaultLightRig();
  (void)datasetsRoot;

  Shot shot;
  shot.id = project::nextShotId(m_project);
  shot.name = "Shot 1";
  shot.renderSettings.outputFilePrefix = shot.id;
  ensureRendererDefaults(shot);

  auto camera = m_ctx->tsd.scene.createObject<tsd::scene::Camera>(
      tsd::scene::tokens::camera::perspective);
  camera->setName(shot.id + "_camera");
  shot.camera = {ANARI_CAMERA, camera.index()};
  shot.cameraRig.current =
      shot_camera_rig::manipulatorStateFromManipulator(m_ctx->view.manipulator);
  tsd::rendering::updateCameraObject(*camera, m_ctx->view.manipulator);

  ensureChild(shotsRoot, shot.id.c_str());
  if (defaultRig)
    shot.lightRigId = defaultRig->id;

  m_project.shots.push_back(std::move(shot));
  m_project.activeShotId = m_project.shots.front().id;
  m_project.markClean();
  syncAnimationManagerToActiveShot();
  applyActiveShot();
}

bool ProjectContext::addShot(const std::string &name)
{
  if (!m_ctx)
    return false;

  Shot shot;
  shot.id = project::nextShotId(m_project);
  shot.name = name.empty()
      ? ("Shot " + std::to_string(m_project.shots.size() + 1))
      : name;
  shot.renderSettings.outputFilePrefix = shot.id;
  ensureRendererDefaults(shot);

  for (const auto &dataset : m_project.datasets) {
    if (dataset.status == DatasetStatus::Available)
      shot.datasetBindings.push_back({dataset.id, true});
  }

  auto camera = m_ctx->tsd.scene.createObject<tsd::scene::Camera>(
      tsd::scene::tokens::camera::perspective);
  camera->setName(shot.id + "_camera");
  shot.camera = {ANARI_CAMERA, camera.index()};
  shot.cameraRig.current =
      shot_camera_rig::manipulatorStateFromManipulator(m_ctx->view.manipulator);
  tsd::rendering::updateCameraObject(*camera, m_ctx->view.manipulator);

  ensureChild(ensureShotsRoot(), shot.id.c_str());
  if (auto *defaultRig = ensureDefaultLightRig())
    shot.lightRigId = defaultRig->id;

  m_project.activeShotId = shot.id;
  m_project.shots.push_back(std::move(shot));
  m_project.markDirty();
  syncAnimationManagerToActiveShot();
  applyActiveShot();
  return true;
}

static DatasetSourceMetadata collectSourceMetadata(
    const std::filesystem::path &sourcePath,
    const std::filesystem::path &projectDirectory)
{
  DatasetSourceMetadata metadata;
  std::error_code ec;
  auto absolute = std::filesystem::absolute(sourcePath, ec);
  if (ec)
    absolute = sourcePath;
  absolute = absolute.lexically_normal();
  metadata.absolutePath = absolute.string();

  if (!projectDirectory.empty()) {
    auto relative = std::filesystem::relative(absolute, projectDirectory, ec);
    if (!ec)
      metadata.projectRelativePath = relative.string();
  }

  metadata.fileSize = std::filesystem::is_regular_file(absolute, ec)
      ? std::filesystem::file_size(absolute, ec)
      : 0;

  auto modified = std::filesystem::last_write_time(absolute, ec);
  if (!ec) {
    metadata.modifiedTime = modified.time_since_epoch().count();
  }

  return metadata;
}

static DatasetSourceFile sourceFileFromMetadata(
    const DatasetSourceMetadata &metadata)
{
  return {metadata.absolutePath,
      metadata.projectRelativePath,
      metadata.fileSize,
      metadata.modifiedTime};
}

Dataset *ProjectContext::addStaticDataset(const std::string &name,
    const std::filesystem::path &sourcePath,
    tsd::io::ImporterType importerType)
{
  if (!m_ctx)
    return nullptr;

  Dataset dataset;
  dataset.id = project::nextDatasetId(m_project);
  dataset.name = name.empty() ? dataset.id : name;
  dataset.sourceKind = DatasetSourceKind::Static;
  dataset.importerType = toString(importerType);
  dataset.source =
      collectSourceMetadata(sourcePath, m_project.projectDirectory);
  dataset.status = DatasetStatus::Importing;

  auto datasetRoot = ensureChild(ensureDatasetsRoot(), dataset.id.c_str());
  dataset.rootNode = refFor("studio", datasetRoot);

  auto datasetIndex = m_project.datasets.size();
  m_project.datasets.push_back(std::move(dataset));
  auto &record = m_project.datasets.back();

  try {
    tsd::io::import_file(m_ctx->tsd.scene,
        m_ctx->tsd.animationMgr,
        {importerType, sourcePath.string()},
        datasetRoot);
    record.status = DatasetStatus::Available;
    for (auto &shot : m_project.shots)
      shot::setDatasetBinding(shot, record.id, &shot == project::activeShot(m_project));
  } catch (const std::exception &e) {
    record.status = DatasetStatus::ImportFailed;
    tsd::core::logError("[SciVisStudio] Dataset import failed for '%s': %s",
        sourcePath.string().c_str(),
        e.what());
  } catch (...) {
    record.status = DatasetStatus::ImportFailed;
    tsd::core::logError("[SciVisStudio] Dataset import failed for '%s'",
        sourcePath.string().c_str());
  }

  (void)datasetIndex;
  m_project.markDirty();
  applyActiveShot();
  return &record;
}

Dataset *ProjectContext::addFileAnimationDataset(const std::string &name,
    const std::vector<std::filesystem::path> &sourcePaths,
    tsd::io::ImporterType importerType,
    const FileAnimationDatasetOptions &options)
{
  if (!m_ctx || sourcePaths.empty())
    return nullptr;

  Dataset dataset;
  dataset.id = project::nextDatasetId(m_project);
  dataset.name = name.empty() ? dataset.id : name;
  dataset.sourceKind = DatasetSourceKind::TimeSeries;
  dataset.importerType = toString(importerType);
  dataset.status = DatasetStatus::Importing;
  dataset.source =
      collectSourceMetadata(sourcePaths.front(), m_project.projectDirectory);

  std::vector<std::string> importPaths;
  importPaths.reserve(sourcePaths.size());
  dataset.sourceFiles.reserve(sourcePaths.size());
  for (const auto &path : sourcePaths) {
    auto metadata = collectSourceMetadata(path, m_project.projectDirectory);
    dataset.sourceFiles.push_back(sourceFileFromMetadata(metadata));
    importPaths.push_back(metadata.absolutePath);
  }

  auto datasetRoot = ensureChild(ensureDatasetsRoot(), dataset.id.c_str());
  dataset.rootNode = refFor("studio", datasetRoot);

  m_project.datasets.push_back(std::move(dataset));
  auto &record = m_project.datasets.back();

  try {
    tsd::core::logStatus(
        "[SciVisStudio] Importing file animation dataset '%s' with %zu frames",
        record.name.c_str(),
        importPaths.size());
    tsd::io::import_animations(m_ctx->tsd.scene,
        m_ctx->tsd.animationMgr,
        {{importerType, importPaths}},
        datasetRoot);

    if (!hasChildNodes(datasetRoot)) {
      record.status = DatasetStatus::ImportFailed;
      tsd::core::logError(
          "[SciVisStudio] File animation dataset import created no scene objects for '%s'",
          record.name.c_str());
    } else {
      record.status = DatasetStatus::Available;
      if (auto *activeShot = project::activeShot(m_project)) {
        for (const auto &dataset : m_project.datasets) {
          if (dataset.id == record.id
              || dataset.sourceKind != DatasetSourceKind::TimeSeries)
            continue;
          const auto *binding =
              shot::findDatasetBinding(*activeShot, dataset.id);
          if (binding && binding->enabled
              && dataset.sourceFiles.size() != sourcePaths.size()) {
            tsd::core::logWarning(
                "[SciVisStudio] Enabled file animation datasets have different frame counts: '%s' has %zu frames, '%s' has %zu frames",
                dataset.name.c_str(),
                dataset.sourceFiles.size(),
                record.name.c_str(),
                sourcePaths.size());
          }
        }
      }
      for (auto &shot : m_project.shots)
        shot::setDatasetBinding(
            shot, record.id, &shot == project::activeShot(m_project));

      if (auto *activeShot = project::activeShot(m_project)) {
        if (options.setActiveShotFrameCount)
          activeShot->frameCount = static_cast<int>(sourcePaths.size());
        activeShot->currentFrame = 0;
        activeShot->playing = false;
      }
      syncAnimationManagerToActiveShot();
      m_ctx->tsd.animationMgr.setAnimationFrame(0);
      tsd::core::logStatus(
          "[SciVisStudio] Imported file animation dataset '%s' (%zu frames)",
          record.name.c_str(),
          importPaths.size());
    }
  } catch (const std::exception &e) {
    record.status = DatasetStatus::ImportFailed;
    tsd::core::logError(
        "[SciVisStudio] File animation dataset import failed for '%s': %s",
        record.name.c_str(),
        e.what());
  } catch (...) {
    record.status = DatasetStatus::ImportFailed;
    tsd::core::logError(
        "[SciVisStudio] File animation dataset import failed for '%s'",
        record.name.c_str());
  }

  m_project.markDirty();
  applyActiveShot();
  return &record;
}

void ProjectContext::applyActiveShot()
{
  if (!m_ctx)
    return;

  auto *shot = project::activeShot(m_project);
  if (!shot)
    return;

  std::vector<const tsd::scene::Layer *> changedLayers;
  auto setNodeEnabled = [&](tsd::scene::LayerNodeRef node, bool enabled) {
    if (node) {
      if ((*node)->isEnabled() == enabled)
        return;

      (*node)->setEnabled(enabled);
      auto *layer = (*node)->layer();
      if (layer
          && std::find(changedLayers.begin(), changedLayers.end(), layer)
              == changedLayers.end())
        changedLayers.push_back(layer);
    }
  };

  for (auto &rig : m_project.lightRigs) {
    setNodeEnabled(resolveLightRigRoot(rig), rig.id == shot->lightRigId);
  }

  for (auto &dataset : m_project.datasets) {
    bool enabled = false;
    if (const auto *binding = shot::findDatasetBinding(*shot, dataset.id))
      enabled = binding->enabled;
    setNodeEnabled(resolveDatasetRoot(dataset), enabled);
  }

  for (auto *layer : changedLayers)
    m_ctx->tsd.scene.signalLayerStructureChanged(layer);

  auto sampled = shot_camera_rig::sampleCameraRig(shot->cameraRig, shot->currentFrame);
  shot_camera_rig::applyManipulatorState(m_ctx->view.manipulator, sampled);

  if (auto *obj = resolveShotCamera(*shot)) {
    auto *camera = static_cast<tsd::scene::Camera *>(obj);
    tsd::rendering::updateCameraObject(*camera, m_ctx->view.manipulator);
  }
}

void ProjectContext::syncAnimationManagerToActiveShot()
{
  if (!m_ctx)
    return;

  auto *shot = project::activeShot(m_project);
  if (!shot)
    return;

  shot->frameCount = std::max(1, shot->frameCount);
  shot->currentFrame = std::clamp(shot->currentFrame, 0, shot->frameCount - 1);
  shot->fps = std::max(1.f, shot->fps);

  m_syncingAnimationManager = true;

  auto &animMgr = m_ctx->tsd.animationMgr;
  animMgr.setAnimationTotalFrames(std::max(2, shot->frameCount));
  animMgr.setAnimationFPS(shot->fps);
  animMgr.setLoop(shot->loop);
  animMgr.setAnimationFrame(shot->currentFrame);
  if (shot->playing)
    animMgr.play();
  else
    animMgr.stop();

  m_syncingAnimationManager = false;
}

void ProjectContext::updateActiveShotFromAnimationTime()
{
  if (!m_ctx || m_syncingAnimationManager)
    return;

  auto *shot = project::activeShot(m_project);
  if (!shot)
    return;

  const auto &animMgr = m_ctx->tsd.animationMgr;
  shot->frameCount = std::max(1, shot->frameCount);
  shot->currentFrame =
      std::clamp(animMgr.getAnimationFrame(), 0, shot->frameCount - 1);
  shot->playing = animMgr.isPlaying();
  applyActiveShot();
}

bool ProjectContext::saveProject(const std::filesystem::path &directory,
    tsd::core::DataNode *windows,
    const std::string &layout,
    tsd::core::DataNode *settings,
    std::string *error)
{
  if (!m_ctx) {
    if (error)
      *error = "missing TSD application context";
    return false;
  }

  std::error_code ec;
  const auto manifest = directory / PROJECT_MANIFEST_FILENAME;
  const bool savingCurrent = !m_project.projectDirectory.empty()
      && std::filesystem::equivalent(directory, m_project.projectDirectory, ec);

  if (std::filesystem::exists(manifest) && !savingCurrent) {
    if (error)
      *error = "target directory already contains project.tsd";
    return false;
  }

  std::filesystem::create_directories(directory, ec);
  if (ec) {
    if (error)
      *error = "failed to create project directory: " + ec.message();
    return false;
  }

  std::filesystem::create_directories(directory / "renders", ec);
  if (ec) {
    if (error)
      *error = "failed to create renders directory: " + ec.message();
    return false;
  }

  m_project.projectDirectory = directory;
  if (m_project.name.empty() || m_project.name == "Untitled")
    m_project.name = directory.filename().string().empty()
        ? std::string("Untitled")
        : directory.filename().string();

  tsd::core::DataTree tree;
  auto &root = tree.root();
  tsd::core::writeDataTreeMetadata(root,
      {tsd::core::DATA_TREE_METADATA_ENVELOPE_VERSION,
          PROJECT_FILE_TYPE,
          PROJECT_SCHEMA,
          SCHEMA_VERSION});
  projectToNode(m_project, root["scivisStudio"]);
  tsd::io::save_Scene(
      m_ctx->tsd.scene, root["context"], false, &m_ctx->tsd.animationMgr);

  if (windows)
    root["windows"] = *windows;
  if (!layout.empty())
    root["layout"] = layout;
  if (settings)
    root["settings"] = *settings;

  if (!tree.save(manifest.string().c_str())) {
    if (error)
      *error = "failed to write project.tsd";
    return false;
  }

  m_project.markClean();
  tsd::core::logStatus(
      "[SciVisStudio] Saved project '%s'", directory.string().c_str());
  return true;
}

bool ProjectContext::openProject(const std::filesystem::path &directory,
    tsd::core::DataNode *windowsOut,
    std::string *layoutOut,
    tsd::core::DataNode *settingsOut,
    std::string *error)
{
  auto validation = validateProjectRoot(directory);
  if (!validation.ok) {
    if (error)
      *error = validation.error;
    return false;
  }

  tsd::core::DataTree tree;
  if (!tree.load(validation.manifestPath.string().c_str())) {
    if (error)
      *error = "failed to load project.tsd";
    return false;
  }

  Project loadedProject;
  if (auto *model = tree.root().child("scivisStudio"))
    nodeToProject(*model, loadedProject);
  else {
    if (error)
      *error = "project.tsd is missing scivisStudio section";
    return false;
  }

  auto &root = tree.root();
  resetScene();
  m_syncingAnimationManager = true;
  if (auto *context = root.child("context"))
    tsd::io::load_Scene(m_ctx->tsd.scene, *context, &m_ctx->tsd.animationMgr);
  m_syncingAnimationManager = false;

  loadedProject.projectDirectory = directory;
  loadedProject.markClean();
  m_project = std::move(loadedProject);
  auto manifestMetadata = tsd::core::readDataTreeMetadata(root);
  const int loadedSchemaVersion = manifestMetadata.found()
      ? manifestMetadata.metadata->schemaVersion
      : root["schemaVersion"].getValueOr<int>(1);
  if (loadedSchemaVersion < 2)
    migrateLegacyShotLightsToLightRigs();
  markMissingDatasets();
  refreshRuntimeRefs();
  syncAnimationManagerToActiveShot();

  if (windowsOut) {
    windowsOut->reset();
    if (auto *windows = root.child("windows"))
      *windowsOut = *windows;
  }

  if (layoutOut) {
    layoutOut->clear();
    if (auto *layout = root.child("layout"))
      *layoutOut = layout->getValueAs<std::string>();
  }

  if (settingsOut) {
    settingsOut->reset();
    if (auto *settings = root.child("settings"))
      *settingsOut = *settings;
  }

  applyActiveShot();
  tsd::core::logStatus(
      "[SciVisStudio] Opened project '%s'", directory.string().c_str());
  return true;
}

void ProjectContext::markMissingDatasets()
{
  for (auto &dataset : m_project.datasets) {
    if (dataset.sourceKind == DatasetSourceKind::TimeSeries) {
      const bool missingSource = dataset.sourceFiles.empty()
          || std::any_of(dataset.sourceFiles.begin(),
              dataset.sourceFiles.end(),
              [this](const DatasetSourceFile &sourceFile) {
                return !sourceFileIsRegular(sourceFile);
              });
      if (missingSource)
        dataset.status = DatasetStatus::Missing;
      continue;
    }

    if (dataset.sourceKind != DatasetSourceKind::Static)
      continue;

    DatasetSourceFile sourceFile{dataset.source.absolutePath,
        dataset.source.projectRelativePath,
        dataset.source.fileSize,
        dataset.source.modifiedTime};
    if (!sourceFile.absolutePath.empty() && !sourceFileIsRegular(sourceFile))
      dataset.status = DatasetStatus::Missing;
  }
}

std::filesystem::path ProjectContext::resolveSourceFilePath(
    const DatasetSourceFile &sourceFile) const
{
  if (!sourceFile.projectRelativePath.empty()
      && !m_project.projectDirectory.empty()) {
    auto relativePath =
        (m_project.projectDirectory / sourceFile.projectRelativePath)
            .lexically_normal();
    std::error_code ec;
    if (std::filesystem::is_regular_file(relativePath, ec) && !ec)
      return relativePath;
  }

  if (!sourceFile.absolutePath.empty())
    return std::filesystem::path(sourceFile.absolutePath).lexically_normal();

  return {};
}

bool ProjectContext::sourceFileIsRegular(
    const DatasetSourceFile &sourceFile) const
{
  auto path = resolveSourceFilePath(sourceFile);
  if (path.empty())
    return false;

  std::error_code ec;
  return std::filesystem::is_regular_file(path, ec) && !ec;
}

void ProjectContext::refreshRuntimeRefs()
{
  for (auto &dataset : m_project.datasets)
    resolveDatasetRoot(dataset);

  for (auto &rig : m_project.lightRigs)
    resolveLightRigRoot(rig);

  for (auto &shot : m_project.shots) {
    resolveShotCamera(shot);
  }
}

void ProjectContext::migrateLegacyShotLightsToLightRigs()
{
  if (!m_ctx || !m_project.lightRigs.empty())
    return;

  auto *layer = m_ctx->tsd.scene.layer("studio");
  if (!layer)
    return;

  auto lightRigsRoot = ensureLightRigsRoot();
  auto shotsRoot = findDirectChild(layer->root(), "shots");
  for (auto &shot : m_project.shots) {
    auto shotRoot = findDirectChild(shotsRoot, shot.id);
    auto legacyLights = findDirectChild(shotRoot, "lights");
    if (!legacyLights)
      continue;

    LightRig rig;
    rig.id = project::nextLightRigId(m_project);
    rig.name =
        shot.name.empty() ? (shot.id + " Lights") : (shot.name + " Lights");
    if (auto existing = findDirectChild(lightRigsRoot, rig.id))
      m_ctx->tsd.scene.removeNode(existing, true);

    legacyLights->container()->move_subtree(legacyLights, lightRigsRoot);
    (*legacyLights)->name() = rig.id;
    rig.rootNode = refFor("studio", legacyLights);
    shot.lightRigId = rig.id;
    m_project.lightRigs.push_back(std::move(rig));
  }
  m_ctx->tsd.scene.signalLayerStructureChanged(layer);
}

const char *toString(tsd::io::ImporterType importerType)
{
  switch (importerType) {
  case tsd::io::ImporterType::AGX:
    return "AGX";
  case tsd::io::ImporterType::ASSIMP:
    return "ASSIMP";
  case tsd::io::ImporterType::ASSIMP_FLAT:
    return "ASSIMP_FLAT";
  case tsd::io::ImporterType::AXYZ:
    return "AXYZ";
  case tsd::io::ImporterType::DLAF:
    return "DLAF";
  case tsd::io::ImporterType::E57XYZ:
    return "E57XYZ";
  case tsd::io::ImporterType::ENSIGHT:
    return "ENSIGHT";
  case tsd::io::ImporterType::GLTF:
    return "GLTF";
  case tsd::io::ImporterType::HDRI:
    return "HDRI";
  case tsd::io::ImporterType::HSMESH:
    return "HSMESH";
  case tsd::io::ImporterType::NBODY:
    return "NBODY";
  case tsd::io::ImporterType::OBJ:
    return "OBJ";
  case tsd::io::ImporterType::PDB:
    return "PDB";
  case tsd::io::ImporterType::PLY:
    return "PLY";
  case tsd::io::ImporterType::POINTSBIN_MULTIFILE:
    return "POINTSBIN_MULTIFILE";
  case tsd::io::ImporterType::PT:
    return "PT";
  case tsd::io::ImporterType::SILO:
    return "SILO";
  case tsd::io::ImporterType::SMESH:
    return "SMESH";
  case tsd::io::ImporterType::SMESH_ANIMATION:
    return "SMESH_ANIMATION";
  case tsd::io::ImporterType::SWC:
    return "SWC";
  case tsd::io::ImporterType::TRK:
    return "TRK";
  case tsd::io::ImporterType::USD:
    return "USD";
  case tsd::io::ImporterType::VTP:
    return "VTP";
  case tsd::io::ImporterType::VTU:
    return "VTU";
  case tsd::io::ImporterType::XYZDP:
    return "XYZDP";
  case tsd::io::ImporterType::VOLUME:
    return "VOLUME";
  case tsd::io::ImporterType::VOLUME_ANIMATION:
    return "VOLUME_ANIMATION";
  case tsd::io::ImporterType::TSD:
    return "TSD";
  case tsd::io::ImporterType::XF:
    return "XF";
  case tsd::io::ImporterType::BLANK:
    return "BLANK";
  case tsd::io::ImporterType::NONE:
    return "NONE";
  }
  return "NONE";
}

tsd::io::ImporterType importerTypeFromString(const std::string &s)
{
  for (int i = 0; i <= static_cast<int>(tsd::io::ImporterType::NONE); ++i) {
    auto type = static_cast<tsd::io::ImporterType>(i);
    if (s == toString(type))
      return type;
  }
  return tsd::io::ImporterType::NONE;
}

} // namespace tsd::scivis_studio
