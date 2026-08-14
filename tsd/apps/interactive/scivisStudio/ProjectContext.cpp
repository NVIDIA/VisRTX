// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ProjectContext.h"

#include "DatasetIO.h"
#include "LightRigIO.h"
#include "ProjectAssetTransaction.h"
#include "ProjectPersistence.h"
#include "ProjectSerialization.h"

#include "tsd/core/DataTree.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/io/archives/LayerSubtreeArchive.hpp"
#include "tsd/rendering/view/ManipulatorToTSD.hpp"
#include "tsd/scene/objects/Array.hpp"
#include "tsd/scene/objects/Camera.hpp"
#include "tsd/scene/objects/Light.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <vector>

namespace tsd::scivis_studio {

// Marker importerType for a Static Dataset whose runtime came from a TSD Layer
// Subtree Archive rather than a foreign-format importer. It is recorded as the
// dataset's provenance and routes reimport back through the subtree loader.
static constexpr const char *SUBTREE_IMPORTER_TYPE = "TSD_SUBTREE";

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

static bool hasObjectNodes(tsd::scene::LayerNodeRef root)
{
  if (!root)
    return false;
  bool found = false;
  auto *layer = (*root).value().layer();
  layer->traverse(root, [&](tsd::scene::LayerNode &node, int) {
    found |= node->isObject();
    return !found;
  });
  return found;
}

static bool hasFileBindingAnimations(
    const tsd::animation::AnimationManager &manager, size_t firstAnimation)
{
  const auto &animations = manager.animations();
  for (size_t i = firstAnimation; i < animations.size(); ++i) {
    if (!animations[i].fileBindings().empty())
      return true;
  }
  return false;
}

static bool fail(const std::string &message, std::string *error)
{
  if (error)
    *error = message;
  return false;
}

struct DatasetDirtyDelegate : tsd::scene::EmptyUpdateDelegate
{
  explicit DatasetDirtyDelegate(ProjectContext *context) : context(context) {}

  void signalParameterUpdated(
      const tsd::scene::Object *object, const tsd::scene::Parameter *) override
  {
    context->markDatasetDirtyForObject(object);
  }

  void signalParameterRemoved(
      const tsd::scene::Object *object, const tsd::scene::Parameter *) override
  {
    context->markDatasetDirtyForObject(object);
  }

  void signalParameterBatchUpdated(const tsd::scene::Object *object,
      const std::vector<const tsd::scene::Parameter *> &) override
  {
    context->markDatasetDirtyForObject(object);
  }

  void signalArrayMapped(const tsd::scene::Array *array) override
  {
    context->markDatasetDirtyForObject(array);
  }

  void signalArrayUnmapped(const tsd::scene::Array *array) override
  {
    context->markDatasetDirtyForObject(array);
  }

  void signalObjectRemoved(const tsd::scene::Object *object) override
  {
    context->markDatasetDirtyForObject(object);
  }

  ProjectContext *context{nullptr};
};

// Managed asset names are compared case-insensitively because they map to
// on-disk filenames (which collide case-insensitively on Windows/macOS).
static bool assetNamesEqual(const std::string &a, const std::string &b)
{
  if (a.size() != b.size())
    return false;
  return std::equal(a.begin(), a.end(), b.begin(), [](char x, char y) {
    return std::tolower(static_cast<unsigned char>(x))
        == std::tolower(static_cast<unsigned char>(y));
  });
}

// Produce an asset name that does not collide (case-insensitively) with any
// existing item, mirroring the " Copy" convention used when cloning rigs.
template <typename AssetT>
static std::string uniqueAssetName(
    const std::vector<AssetT> &assets, const std::string &desired)
{
  auto taken = [&](const std::string &candidate) {
    return std::any_of(assets.begin(), assets.end(), [&](const AssetT &asset) {
      return assetNamesEqual(asset.name, candidate);
    });
  };

  if (!taken(desired))
    return desired;

  std::string candidate = desired + " Copy";
  for (int n = 2; taken(candidate); ++n)
    candidate = desired + " Copy " + std::to_string(n);
  return candidate;
}

// Combine sanitize + de-duplication so programmatic names are always valid
// filenames that do not collide with an existing item.
template <typename AssetT>
static std::string makeValidUniqueAssetName(
    const std::vector<AssetT> &assets, const std::string &desired)
{
  return uniqueAssetName(assets, sanitizeRigName(desired));
}

// Shared rename logic: validate the format and reject names already taken by a
// *different* item in the same collection (case-insensitive).
template <typename AssetT, typename IdT>
static bool renameAssetImpl(std::vector<AssetT> &assets,
    const IdT &id,
    const std::string &newName,
    const char *assetKind,
    std::string *error)
{
  auto itr = std::find_if(assets.begin(),
      assets.end(),
      [&](const AssetT &asset) { return asset.id == id; });
  if (itr == assets.end()) {
    if (error)
      *error = std::string(assetKind) + " not found";
    return false;
  }

  if (!validateRigName(newName, error))
    return false;

  for (const auto &asset : assets) {
    if (&asset != &*itr && assetNamesEqual(asset.name, newName)) {
      if (error)
        *error =
            std::string("another ") + assetKind + " already uses that name";
      return false;
    }
  }

  itr->name = newName;
  return true;
}

ProjectContext::ProjectContext(tsd::app::Context *ctx) : m_ctx(ctx)
{
  installAnimationManagerCallback();
  installDatasetDirtyDelegate();
}

ProjectContext::~ProjectContext()
{
  if (m_ctx && m_datasetDirtyDelegate)
    m_ctx->tsd.scene.updateDelegate().erase(m_datasetDirtyDelegate);
}

void ProjectContext::setAppContext(tsd::app::Context *ctx)
{
  if (m_ctx && m_datasetDirtyDelegate)
    m_ctx->tsd.scene.updateDelegate().erase(m_datasetDirtyDelegate);
  m_datasetDirtyDelegate = nullptr;
  m_ctx = ctx;
  installAnimationManagerCallback();
  installDatasetDirtyDelegate();
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

void ProjectContext::installDatasetDirtyDelegate()
{
  if (!m_ctx)
    return;
  m_datasetDirtyDelegate =
      m_ctx->tsd.scene.updateDelegate().emplace<DatasetDirtyDelegate>(this);
}

void ProjectContext::markDatasetDirtyForObject(const tsd::scene::Object *object)
{
  if (!m_ctx || !object || m_syncingAnimationManager
      || m_mutatingDatasetRuntime
      || m_ctx->tsd.animationMgr.isApplyingAnimations())
    return;
  for (auto &dataset : m_project.datasets) {
    auto root = resolveDatasetRoot(dataset);
    if (!root || !datasetRuntimeContainsObject(m_ctx->tsd.scene, root, object))
      continue;
    dataset.dirty = true;
    m_project.markDirty();
  }
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
  rig.id = light_rig::nextLightRigId(m_project);
  rig.name = makeValidUniqueAssetName(m_project.lightRigs,
      name.empty()
          ? ("Light Rig " + std::to_string(m_project.lightRigs.size() + 1))
          : name);

  auto rigRoot = ensureChild(ensureLightRigsRoot(), rig.id.c_str());
  rig.rootNode = refFor("studio", rigRoot);
  m_project.lightRigs.push_back(std::move(rig));
  m_project.markDirty();
  return &m_project.lightRigs.back();
}

LightRig *ProjectContext::cloneLightRig(const LightRigID &id)
{
  if (!m_ctx)
    return nullptr;

  auto *source = light_rig::findLightRig(m_project, id);
  if (!source)
    return nullptr;

  auto sourceRoot = resolveLightRigRoot(*source);
  if (!sourceRoot)
    return nullptr;

  LightRig clone;
  clone.id = light_rig::nextLightRigId(m_project);
  clone.name = makeValidUniqueAssetName(m_project.lightRigs,
      source->name.empty() ? "Light Rig Copy" : source->name + " Copy");

  auto &scene = m_ctx->tsd.scene;
  auto lightRigsRoot = ensureLightRigsRoot();
  scene.beginLayerEditBatch();
  auto cloneRoot = scene.cloneLayerSubtree(sourceRoot, lightRigsRoot, true);
  if (cloneRoot)
    (*cloneRoot)->name() = clone.id;
  scene.endLayerEditBatch();
  if (!cloneRoot)
    return nullptr;

  clone.rootNode = refFor("studio", cloneRoot);
  m_project.lightRigs.push_back(std::move(clone));
  m_project.markDirty();
  applyActiveShot();
  return &m_project.lightRigs.back();
}

CameraRig *ProjectContext::createCameraRig(const std::string &name)
{
  CameraRig rig;
  rig.id = camera_rig::nextCameraRigId(m_project);
  rig.name = makeValidUniqueAssetName(m_project.cameraRigs,
      name.empty()
          ? ("Camera Rig " + std::to_string(m_project.cameraRigs.size() + 1))
          : name);
  if (m_ctx)
    rig.current =
        camera_rig::manipulatorStateFromManipulator(m_ctx->view.manipulator);

  m_project.cameraRigs.push_back(std::move(rig));
  m_project.markDirty();
  return &m_project.cameraRigs.back();
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

int ProjectContext::cameraRigUseCount(const CameraRigID &id) const
{
  return static_cast<int>(std::count_if(m_project.shots.begin(),
      m_project.shots.end(),
      [&](const Shot &shot) { return shot.cameraRigId == id; }));
}

CameraRig *ProjectContext::activeShotCameraRig()
{
  auto *shot = project::activeShot(m_project);
  if (!shot || shot->cameraRigId.empty())
    return nullptr;

  return camera_rig::findCameraRig(m_project, shot->cameraRigId);
}

bool ProjectContext::saveCameraRigArchive(const CameraRigID &id,
    const std::filesystem::path &file,
    std::string *error)
{
  auto *rig = camera_rig::findCameraRig(m_project, id);
  if (!rig) {
    if (error)
      *error = "camera rig not found";
    return false;
  }
  return camera_rig::saveCameraRigArchiveFile(*rig, file, error);
}

CameraRig *ProjectContext::loadCameraRigArchive(
    const std::filesystem::path &file, std::string *error)
{
  CameraRig rig;
  if (!camera_rig::loadCameraRigArchiveFile(file, rig, error))
    return nullptr;

  const std::string loadedName = std::move(rig.name);
  rig.id = camera_rig::nextCameraRigId(m_project);
  rig.name = makeValidUniqueAssetName(m_project.cameraRigs,
      loadedName.empty() ? "Loaded Camera Rig" : loadedName);
  m_project.cameraRigs.push_back(std::move(rig));
  m_project.markDirty();
  return &m_project.cameraRigs.back();
}

bool ProjectContext::saveLightRigArchive(
    const LightRigID &id, const std::filesystem::path &file, std::string *error)
{
  if (!m_ctx) {
    if (error)
      *error = "missing TSD application context";
    return false;
  }

  auto *rig = light_rig::findLightRig(m_project, id);
  if (!rig) {
    if (error)
      *error = "light rig not found";
    return false;
  }

  auto rigRoot = resolveLightRigRoot(*rig);
  if (!rigRoot) {
    if (error)
      *error = "light rig has no scene node";
    return false;
  }

  if (!saveLightRigArchiveFile(rigRoot, file, rig->name)) {
    if (error)
      *error = "failed to save Light Rig Archive (see log for details)";
    return false;
  }
  return true;
}

LightRig *ProjectContext::loadLightRigArchive(
    const std::filesystem::path &file, std::string *error)
{
  if (!m_ctx) {
    if (error)
      *error = "missing TSD application context";
    return nullptr;
  }

  auto &scene = m_ctx->tsd.scene;
  auto lightRigsRoot = ensureLightRigsRoot();
  std::string name;
  LightRig rig;
  scene.beginLayerEditBatch();
  auto splicedRoot = loadLightRigArchiveFile(scene, file, lightRigsRoot, &name);
  if (splicedRoot) {
    rig.id = light_rig::nextLightRigId(m_project);
    (*splicedRoot)->name() = rig.id; // resolveLightRigRoot keys on node==rig.id
  }
  scene.endLayerEditBatch();

  if (!splicedRoot) {
    if (error)
      *error = "failed to load Light Rig Archive (see log for details)";
    return nullptr;
  }

  rig.name = makeValidUniqueAssetName(
      m_project.lightRigs, name.empty() ? "Loaded Light Rig" : name);
  rig.rootNode = refFor("studio", splicedRoot);
  m_project.lightRigs.push_back(std::move(rig));
  m_project.markDirty();
  applyActiveShot(); // A loaded rig is unbound, so it starts hidden.
  return &m_project.lightRigs.back();
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

  if (!itr->persistedName.empty()) {
    m_pendingAssetRemovals.push_back(
        std::filesystem::path("lights") / (itr->persistedName + ".tsd"));
  }

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

bool ProjectContext::removeCameraRig(const CameraRigID &id)
{
  auto itr = std::find_if(m_project.cameraRigs.begin(),
      m_project.cameraRigs.end(),
      [&](const CameraRig &rig) { return rig.id == id; });
  if (itr == m_project.cameraRigs.end())
    return false;

  if (!itr->persistedName.empty()) {
    m_pendingAssetRemovals.push_back(
        std::filesystem::path("cameras") / (itr->persistedName + ".tsd"));
  }

  for (auto &shot : m_project.shots) {
    if (shot.cameraRigId == id)
      shot.cameraRigId.clear();
  }

  m_project.cameraRigs.erase(itr);
  m_project.markDirty();
  applyActiveShot();
  return true;
}

bool ProjectContext::renameLightRig(
    const LightRigID &id, const std::string &newName, std::string *error)
{
  if (!renameAssetImpl(m_project.lightRigs, id, newName, "light rig", error))
    return false;
  m_project.markDirty();
  return true;
}

bool ProjectContext::renameCameraRig(
    const CameraRigID &id, const std::string &newName, std::string *error)
{
  if (!renameAssetImpl(m_project.cameraRigs, id, newName, "camera rig", error))
    return false;
  m_project.markDirty();
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

CameraRig *ProjectContext::ensureDefaultCameraRig()
{
  if (!m_project.cameraRigs.empty())
    return &m_project.cameraRigs.front();

  return createCameraRig("Default");
}

void ProjectContext::createUnsavedProject()
{
  resetScene();

  m_project = {};
  m_pendingAssetRemovals.clear();
  m_project.name = "Untitled";

  auto datasetsRoot = ensureDatasetsRoot();
  auto shotsRoot = ensureShotsRoot();
  auto *defaultRig = ensureDefaultLightRig();
  auto *defaultCameraRig = ensureDefaultCameraRig();
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
  tsd::rendering::updateCameraObject(*camera, m_ctx->view.manipulator);

  ensureChild(shotsRoot, shot.id.c_str());
  if (defaultRig)
    shot.lightRigId = defaultRig->id;
  if (defaultCameraRig)
    shot.cameraRigId = defaultCameraRig->id;

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
  tsd::rendering::updateCameraObject(*camera, m_ctx->view.manipulator);

  ensureChild(ensureShotsRoot(), shot.id.c_str());
  if (auto *defaultRig = ensureDefaultLightRig())
    shot.lightRigId = defaultRig->id;
  if (auto *defaultCameraRig = ensureDefaultCameraRig())
    shot.cameraRigId = defaultCameraRig->id;

  m_project.activeShotId = shot.id;
  m_project.shots.push_back(std::move(shot));
  m_project.markDirty();
  syncAnimationManagerToActiveShot();
  applyActiveShot();
  return true;
}

static DatasetSourceMetadata collectSourceMetadata(
    const std::filesystem::path &sourcePath)
{
  DatasetSourceMetadata metadata;
  metadata.sourcePath = sourcePath.string();
  return metadata;
}

static DatasetSourceFile sourceFileFromMetadata(
    const DatasetSourceMetadata &metadata)
{
  return {metadata.sourcePath};
}

Dataset *ProjectContext::addStaticDataset(const std::string &name,
    const std::filesystem::path &sourcePath,
    tsd::io::ImporterType importerType)
{
  if (!m_ctx)
    return nullptr;

  Dataset dataset;
  dataset.id = project::nextDatasetId(m_project);
  dataset.name = makeValidUniqueAssetName(
      m_project.datasets, name.empty() ? dataset.id : name);
  dataset.sourceKind = DatasetSourceKind::Static;
  dataset.importerType = toString(importerType);
  dataset.source = collectSourceMetadata(sourcePath);
  dataset.status = DatasetStatus::Importing;

  auto datasetRoot = ensureChild(ensureDatasetsRoot(), dataset.id.c_str());
  dataset.rootNode = refFor("studio", datasetRoot);

  m_project.datasets.push_back(std::move(dataset));
  auto &record = m_project.datasets.back();
  const auto firstImportedAnimation =
      m_ctx->tsd.animationMgr.animations().size();

  try {
    tsd::io::import_file(m_ctx->tsd.scene,
        m_ctx->tsd.animationMgr,
        {importerType, sourcePath.string()},
        datasetRoot);
    if (hasFileBindingAnimations(
            m_ctx->tsd.animationMgr, firstImportedAnimation)) {
      removeDatasetRuntime(
          m_ctx->tsd.scene, m_ctx->tsd.animationMgr, datasetRoot);
      datasetRoot = ensureChild(ensureDatasetsRoot(), record.id.c_str());
      record.rootNode = refFor("studio", datasetRoot);
      record.status = DatasetStatus::ImportFailed;
      tsd::core::logError(
          "[SciVisStudio] Static dataset import created a file animation for '%s'",
          sourcePath.string().c_str());
    } else if (!hasObjectNodes(datasetRoot)) {
      record.status = DatasetStatus::ImportFailed;
      tsd::core::logError(
          "[SciVisStudio] Dataset import created no scene objects for '%s'",
          sourcePath.string().c_str());
    } else {
      record.status = DatasetStatus::Available;
      for (auto &shot : m_project.shots)
        shot::setDatasetBinding(
            shot, record.id, &shot == project::activeShot(m_project));
    }
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

  record.dirty = record.status == DatasetStatus::Available;
  m_project.markDirty();
  applyActiveShot();
  return &record;
}

Dataset *ProjectContext::addStaticDatasetFromSubtree(
    const std::string &name, const std::filesystem::path &sourcePath)
{
  if (!m_ctx)
    return nullptr;

  Dataset dataset;
  dataset.id = project::nextDatasetId(m_project);
  dataset.name = makeValidUniqueAssetName(
      m_project.datasets, name.empty() ? dataset.id : name);
  dataset.sourceKind = DatasetSourceKind::Static;
  dataset.importerType = SUBTREE_IMPORTER_TYPE;
  dataset.source = collectSourceMetadata(sourcePath);
  dataset.status = DatasetStatus::Importing;

  auto datasetRoot = ensureChild(ensureDatasetsRoot(), dataset.id.c_str());
  dataset.rootNode = refFor("studio", datasetRoot);

  m_project.datasets.push_back(std::move(dataset));
  auto &record = m_project.datasets.back();

  try {
    // A Layer Subtree Archive deserializes its subtree as a child of the
    // dataset root, mirroring how tsdViewer's LayerTree loads one.
    auto subtreeRoot = tsd::io::load_LayerSubtreeArchive(
        datasetRoot, sourcePath.string().c_str());
    if (!subtreeRoot || !hasObjectNodes(datasetRoot)) {
      record.status = DatasetStatus::ImportFailed;
      tsd::core::logError(
          "[SciVisStudio] Subtree dataset load created no scene objects for '%s'",
          sourcePath.string().c_str());
    } else {
      record.status = DatasetStatus::Available;
      for (auto &shot : m_project.shots)
        shot::setDatasetBinding(
            shot, record.id, &shot == project::activeShot(m_project));
    }
  } catch (const std::exception &e) {
    record.status = DatasetStatus::ImportFailed;
    tsd::core::logError(
        "[SciVisStudio] Subtree dataset load failed for '%s': %s",
        sourcePath.string().c_str(),
        e.what());
  } catch (...) {
    record.status = DatasetStatus::ImportFailed;
    tsd::core::logError("[SciVisStudio] Subtree dataset load failed for '%s'",
        sourcePath.string().c_str());
  }

  record.dirty = record.status == DatasetStatus::Available;
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
  dataset.name = makeValidUniqueAssetName(
      m_project.datasets, name.empty() ? dataset.id : name);
  dataset.sourceKind = DatasetSourceKind::FileAnimation;
  dataset.importerType = toString(importerType);
  dataset.status = DatasetStatus::Importing;
  dataset.source = collectSourceMetadata(sourcePaths.front());

  std::vector<std::string> importPaths;
  importPaths.reserve(sourcePaths.size());
  dataset.sourceFiles.reserve(sourcePaths.size());
  for (const auto &path : sourcePaths) {
    auto metadata = collectSourceMetadata(path);
    dataset.sourceFiles.push_back(sourceFileFromMetadata(metadata));
    importPaths.push_back(metadata.sourcePath);
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
      applyFileAnimationShotSemantics(record, sourcePaths.size(), options);
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
  record.dirty = record.status == DatasetStatus::Available;
  applyActiveShot();
  return &record;
}

void ProjectContext::applyFileAnimationShotSemantics(const Dataset &record,
    size_t frameCount,
    const FileAnimationDatasetOptions &options)
{
  if (auto *activeShot = project::activeShot(m_project)) {
    for (const auto &dataset : m_project.datasets) {
      if (dataset.id == record.id
          || dataset.sourceKind != DatasetSourceKind::FileAnimation)
        continue;
      const auto *binding = shot::findDatasetBinding(*activeShot, dataset.id);
      if (binding && binding->enabled
          && dataset.sourceFiles.size() != frameCount) {
        tsd::core::logWarning(
            "[SciVisStudio] Enabled file animation datasets have different frame counts: '%s' has %zu frames, '%s' has %zu frames",
            dataset.name.c_str(),
            dataset.sourceFiles.size(),
            record.name.c_str(),
            frameCount);
      }
    }
  }
  for (auto &shot : m_project.shots)
    shot::setDatasetBinding(
        shot, record.id, &shot == project::activeShot(m_project));
  if (auto *activeShot = project::activeShot(m_project)) {
    if (options.setActiveShotFrameCount)
      activeShot->frameCount = static_cast<int>(frameCount);
    activeShot->currentFrame = 0;
    activeShot->playing = false;
  }
  syncAnimationManagerToActiveShot();
  m_ctx->tsd.animationMgr.setAnimationFrame(0);
}

Dataset *ProjectContext::addDeclaredFileAnimationDataset(
    const std::string &name,
    const std::vector<std::string> &sourceList,
    tsd::io::ImporterType importerType,
    const FileAnimationDatasetOptions &options)
{
  if (!m_ctx || sourceList.empty())
    return nullptr;

  Dataset dataset;
  dataset.id = project::nextDatasetId(m_project);
  dataset.name = makeValidUniqueAssetName(
      m_project.datasets, name.empty() ? dataset.id : name);
  dataset.sourceKind = DatasetSourceKind::FileAnimation;
  dataset.importerType = toString(importerType);
  dataset.source.sourcePath = sourceList.front();
  dataset.sourceFiles.reserve(sourceList.size());
  for (const auto &entry : sourceList)
    dataset.sourceFiles.push_back({entry});
  // Entries are opaque and never preflighted, so availability cannot be
  // assessed here; the authoritative assessment is the load attempt. The
  // dataset records Unloaded residency until materialized.
  dataset.status = DatasetStatus::Available;
  dataset.residency = DatasetResidency::Unloaded;
  dataset.declared = true;
  dataset.dirty = true;

  m_project.datasets.push_back(std::move(dataset));
  auto &record = m_project.datasets.back();

  // Shot semantics mirror the eager create; the frame count is known from
  // the Source List length without reading a single file.
  applyFileAnimationShotSemantics(record, record.sourceFiles.size(), options);

  m_project.markDirty();
  applyActiveShot();
  tsd::core::logStatus(
      "[SciVisStudio] Declared file animation dataset '%s' (%zu frames)",
      record.name.c_str(),
      record.sourceFiles.size());
  return &record;
}

bool ProjectContext::renameDataset(
    const DatasetID &id, const std::string &newName, std::string *error)
{
  // The asset stores the name and open cross-checks it, so renaming must be
  // able to rewrite the asset: unloaded datasets are read-only.
  if (auto *dataset = project::findDataset(m_project, id);
      dataset && dataset->residency == DatasetResidency::Unloaded)
    return fail("dataset must be loaded to rename it", error);
  if (!renameAssetImpl(m_project.datasets, id, newName, "dataset", error))
    return false;
  auto *dataset = project::findDataset(m_project, id);
  dataset->dirty = dataset->status == DatasetStatus::Available
      || dataset->name != dataset->persistedName;
  m_project.markDirty();
  return true;
}

bool ProjectContext::loadDataset(const DatasetID &id, std::string *error)
{
  if (!m_ctx)
    return fail("missing TSD application context", error);
  auto *dataset = project::findDataset(m_project, id);
  if (!dataset)
    return fail("dataset not found", error);
  if (dataset->residency == DatasetResidency::Loaded)
    return true;
  if (m_project.projectDirectory.empty() || dataset->persistedName.empty())
    return fail("dataset has no saved asset to load", error);

  const auto file = m_project.projectDirectory / "datasets"
      / (dataset->persistedName + ".tsd");
  Dataset loaded;
  tsd::scene::LayerNodeRef loadedRoot;
  std::string loadError;
  m_mutatingDatasetRuntime = true;
  const bool loadedAsset = loadDatasetArchiveFile(m_ctx->tsd.scene,
      m_ctx->tsd.animationMgr,
      file,
      ensureDatasetsRoot(),
      loaded,
      loadedRoot,
      &loadError);
  if (!loadedAsset) {
    m_mutatingDatasetRuntime = false;
    // A failed load changes nothing: the dataset stays Unloaded and is now
    // known to be Unavailable until its asset is restored.
    dataset->status = DatasetStatus::Unavailable;
    return fail(
        "failed to load dataset '" + dataset->name + "': " + loadError, error);
  }

  if (loaded.name != dataset->name) {
    removeDatasetRuntime(m_ctx->tsd.scene, m_ctx->tsd.animationMgr, loadedRoot);
    m_mutatingDatasetRuntime = false;
    dataset->status = DatasetStatus::Unavailable;
    return fail("dataset asset name '" + loaded.name
            + "' does not match inventory name '" + dataset->name + "'",
        error);
  }
  m_mutatingDatasetRuntime = false;

  loaded.id = dataset->id;
  loaded.persistedName = dataset->persistedName;
  // loaded.dirty stays as deserialized: clean for an ordinary asset, dirty
  // when a Declared Dataset just materialized (ADR 0014).
  loaded.residency = DatasetResidency::Loaded;
  (*loadedRoot)->name() = loaded.id;
  loaded.rootNode = refFor("studio", loadedRoot);
  *dataset = std::move(loaded);
  m_project.markDirty();
  syncAnimationManagerToActiveShot();
  applyActiveShot();
  return true;
}

bool ProjectContext::unloadDataset(const DatasetID &id, std::string *error)
{
  if (!m_ctx)
    return fail("missing TSD application context", error);
  auto *dataset = project::findDataset(m_project, id);
  if (!dataset)
    return fail("dataset not found", error);
  if (dataset->residency == DatasetResidency::Unloaded)
    return true;
  if (dataset->status == DatasetStatus::Importing)
    return fail("dataset is importing and cannot be unloaded", error);
  if (dataset->dirty)
    return fail(
        "dataset has unsaved changes; save the project before unloading",
        error);
  if (dataset->persistedName.empty())
    return fail("dataset has no saved asset to load it back from", error);

  // Never discard the only copy of the data: the managed asset must still be
  // on disk before the runtime representation is destroyed.
  {
    std::error_code ec;
    const bool assetExists = !m_project.projectDirectory.empty()
        && std::filesystem::exists(m_project.projectDirectory / "datasets"
                / (dataset->persistedName + ".tsd"),
            ec)
        && !ec;
    if (!assetExists) {
      dataset->status = DatasetStatus::Unavailable;
      return fail("dataset asset '" + dataset->persistedName
              + ".tsd' is missing on disk; unloading would discard the only "
                "copy of the data",
          error);
    }
  }

  if (auto root = resolveDatasetRoot(*dataset)) {
    // Selection holds plain layer-node refs that would dangle once the
    // subtree is gone; only a selection inside this dataset is affected.
    const auto &selected = m_ctx->tsd.selectedNodes;
    auto *layer = (*root)->layer();
    const bool selectionInDataset = std::any_of(
        selected.begin(), selected.end(), [&](tsd::scene::LayerNodeRef node) {
          return node && (*node)->layer() == layer
              && (node == root || layer->isAncestorOf(root, node));
        });
    if (selectionInDataset)
      m_ctx->clearSelected();

    m_mutatingDatasetRuntime = true;
    const bool removed =
        removeDatasetRuntime(m_ctx->tsd.scene, m_ctx->tsd.animationMgr, root);
    m_mutatingDatasetRuntime = false;
    if (!removed) {
      return fail(
          "failed to release the dataset's runtime representation "
          "(see log for details)",
          error);
    }
  }
  dataset->rootNode = {};
  dataset->residency = DatasetResidency::Unloaded;
  m_project.markDirty();
  applyActiveShot();
  return true;
}

void ProjectContext::refreshUnloadedDatasetAvailability(Dataset &dataset) const
{
  if (dataset.residency != DatasetResidency::Unloaded
      || m_project.projectDirectory.empty() || dataset.persistedName.empty())
    return;
  std::error_code ec;
  const bool exists = std::filesystem::exists(m_project.projectDirectory
          / "datasets" / (dataset.persistedName + ".tsd"),
      ec);
  if (!exists || ec)
    dataset.status = DatasetStatus::Unavailable;
}

bool ProjectContext::removeDataset(
    const DatasetID &id, bool keepAssetFile, std::string *error)
{
  auto itr = std::find_if(m_project.datasets.begin(),
      m_project.datasets.end(),
      [&](const Dataset &dataset) { return dataset.id == id; });
  if (itr == m_project.datasets.end())
    return fail("dataset not found", error);

  if (!keepAssetFile && !m_project.projectDirectory.empty()
      && !itr->persistedName.empty()) {
    const auto assetFile =
        m_project.projectDirectory / "datasets" / (itr->persistedName + ".tsd");
    std::error_code ec;
    std::filesystem::remove(assetFile, ec);
    if (ec)
      return fail("failed to remove Dataset Archive: " + ec.message(), error);
    std::filesystem::remove(sourceListFilePath(assetFile), ec);
    if (ec)
      return fail("failed to remove Source List File: " + ec.message(), error);
  }

  if (m_ctx) {
    if (auto root = resolveDatasetRoot(*itr)) {
      m_mutatingDatasetRuntime = true;
      removeDatasetRuntime(m_ctx->tsd.scene, m_ctx->tsd.animationMgr, root);
      m_mutatingDatasetRuntime = false;
    }
  }
  for (auto &shot : m_project.shots) {
    shot.datasetBindings.erase(std::remove_if(shot.datasetBindings.begin(),
                                   shot.datasetBindings.end(),
                                   [&](const DatasetBinding &binding) {
                                     return binding.datasetId == id;
                                   }),
        shot.datasetBindings.end());
  }
  m_project.datasets.erase(itr);
  m_project.markDirty();
  applyActiveShot();
  return true;
}

bool ProjectContext::saveDatasetArchive(
    const DatasetID &id, const std::filesystem::path &file, std::string *error)
{
  if (!m_ctx)
    return fail("missing TSD application context", error);
  auto *dataset = project::findDataset(m_project, id);
  if (!dataset)
    return fail("dataset not found", error);
  if (dataset->residency == DatasetResidency::Unloaded)
    return fail("dataset must be loaded to save an Archive", error);
  if (dataset->status != DatasetStatus::Available)
    return fail("dataset is unavailable", error);
  auto root = resolveDatasetRoot(*dataset);
  if (!root)
    return fail("dataset has no scene subtree", error);
  if (dataset->sourceKind != DatasetSourceKind::FileAnimation) {
    return saveDatasetArchiveFile(
        *dataset, root, m_ctx->tsd.animationMgr, file, error);
  }

  // For a File Animation Dataset the Archive is the pair: stage both files
  // and install them together, so a failed save cannot leave half a pair or
  // destroy a valid pair already at the target.
  const auto sources = sourceListFilePath(file);
  const auto stageName = [](const std::filesystem::path &target) {
    return target.parent_path()
        / ("." + target.filename().string() + ".stage");
  };
  const auto stagedFile = stageName(file);
  const auto stagedSources = stageName(sources);
  auto discardStages = [&]() {
    std::error_code ec;
    std::filesystem::remove(stagedFile, ec);
    std::filesystem::remove(stagedSources, ec);
  };

  if (!saveDatasetArchiveFile(
          *dataset, root, m_ctx->tsd.animationMgr, stagedFile, error)
      || !writeSourceListFile(stagedSources, dataset->sourceFiles, error)) {
    discardStages();
    return false;
  }

  std::error_code ec;
  std::filesystem::rename(stagedFile, file, ec);
  if (!ec)
    std::filesystem::rename(stagedSources, sources, ec);
  if (ec) {
    discardStages();
    return fail("failed to install Dataset Archive: " + ec.message(), error);
  }
  return true;
}

Dataset *ProjectContext::loadDatasetArchiveImpl(
    const std::filesystem::path &file,
    const std::string &name,
    bool alreadyManaged,
    std::string *error)
{
  if (!m_ctx) {
    fail("missing TSD application context", error);
    return nullptr;
  }

  auto validation = validateDatasetAsset(file);
  if (!validation.ok) {
    fail(validation.error, error);
    return nullptr;
  }

  std::string finalName;
  if (name.empty()) {
    finalName =
        makeValidUniqueAssetName(m_project.datasets, validation.dataset.name);
  } else {
    if (!validateRigName(name, error))
      return nullptr;
    const bool taken = std::any_of(m_project.datasets.begin(),
        m_project.datasets.end(),
        [&](const Dataset &dataset) {
          return assetNamesEqual(dataset.name, name);
        });
    if (taken) {
      fail("another dataset already uses that name", error);
      return nullptr;
    }
    finalName = name;
  }

  Dataset loaded;
  tsd::scene::LayerNodeRef root;
  if (!loadDatasetArchiveFile(m_ctx->tsd.scene,
          m_ctx->tsd.animationMgr,
          file,
          ensureDatasetsRoot(),
          loaded,
          root,
          error))
    return nullptr;

  loaded.id = project::nextDatasetId(m_project);
  loaded.name = finalName;
  (*root)->name() = loaded.id;
  loaded.rootNode = refFor("studio", root);
  loaded.dirty = !alreadyManaged || finalName != validation.dataset.name;
  loaded.persistedName = loaded.dirty ? std::string{} : finalName;
  m_project.datasets.push_back(std::move(loaded));
  auto &record = m_project.datasets.back();
  for (auto &shot : m_project.shots)
    shot::setDatasetBinding(
        shot, record.id, &shot == project::activeShot(m_project));
  m_project.markDirty();
  applyActiveShot();
  return &record;
}

Dataset *ProjectContext::loadDatasetArchive(
    const std::filesystem::path &file, std::string *error)
{
  return loadDatasetArchiveImpl(file, {}, false, error);
}

std::vector<DatasetCandidate> ProjectContext::discoverDatasetCandidates() const
{
  std::vector<DatasetCandidate> candidates;
  if (m_project.projectDirectory.empty())
    return candidates;

  const auto datasetsDir = m_project.projectDirectory / "datasets";
  std::error_code ec;
  for (const auto &entry :
      std::filesystem::directory_iterator(datasetsDir, ec)) {
    if (ec)
      break;
    if (!entry.is_regular_file(ec) || entry.path().extension() != ".tsd")
      continue;
    const auto stem = entry.path().stem().string();
    const bool managed = std::any_of(m_project.datasets.begin(),
        m_project.datasets.end(),
        [&](const Dataset &dataset) {
          return assetNamesEqual(dataset.name, stem);
        });
    if (managed)
      continue;
    auto validation = validateDatasetAsset(entry.path());
    if (!validation.ok)
      continue;
    candidates.push_back({entry.path(), validation.dataset.name});
  }
  std::sort(candidates.begin(),
      candidates.end(),
      [](const DatasetCandidate &a, const DatasetCandidate &b) {
        return a.file.filename() < b.file.filename();
      });
  return candidates;
}

Dataset *ProjectContext::incorporateDatasetCandidate(
    const DatasetCandidate &candidate,
    const std::string &name,
    std::string *error)
{
  const bool sameManagedPath = !m_project.projectDirectory.empty()
      && candidate.file.parent_path().lexically_normal()
          == (m_project.projectDirectory / "datasets").lexically_normal()
      && candidate.file.stem().string() == name
      && candidate.proposedName == name;
  return loadDatasetArchiveImpl(candidate.file, name, sameManagedPath, error);
}

bool ProjectContext::reimportStaticDataset(
    const DatasetID &id, std::string *error)
{
  if (!m_ctx)
    return fail("missing TSD application context", error);
  auto *dataset = project::findDataset(m_project, id);
  if (!dataset)
    return fail("dataset not found", error);
  if (dataset->residency == DatasetResidency::Unloaded)
    return fail("dataset must be loaded to reimport it", error);
  if (dataset->sourceKind != DatasetSourceKind::Static)
    return fail("only static datasets can be reimported", error);
  if (dataset->source.sourcePath.empty())
    return fail("dataset has no provenance source path", error);

  tsd::scene::Scene stagedScene;
  tsd::animation::AnimationManager stagedAnimations(&stagedScene);
  auto stagedRoot = stagedScene.insertChildNode(
      stagedScene.defaultLayer()->root(), "dataset");
  try {
    if (dataset->importerType == SUBTREE_IMPORTER_TYPE) {
      tsd::io::load_LayerSubtreeArchive(
          stagedRoot, dataset->source.sourcePath.c_str());
    } else {
      tsd::io::import_file(stagedScene,
          stagedAnimations,
          {importerTypeFromString(dataset->importerType),
              dataset->source.sourcePath},
          stagedRoot);
    }
  } catch (const std::exception &e) {
    return fail(std::string("dataset reimport failed: ") + e.what(), error);
  } catch (...) {
    return fail("dataset reimport failed", error);
  }
  if (hasFileBindingAnimations(stagedAnimations, 0))
    return fail("static dataset reimport created a file animation", error);
  if (!hasObjectNodes(stagedRoot))
    return fail("dataset reimport created no scene objects", error);

  tsd::core::logStatus(
      "[SciVisStudio] Serializing reimported dataset '%s' for staging",
      dataset->name.c_str());
  const auto stageFile = std::filesystem::temp_directory_path()
      / ("scivis-dataset-reimport-"
          + std::to_string(
              std::chrono::steady_clock::now().time_since_epoch().count())
          + ".tsd");
  Dataset replacementMetadata = *dataset;
  std::string stageError;
  if (!saveDatasetArchiveFile(replacementMetadata,
          stagedRoot,
          stagedAnimations,
          stageFile,
          &stageError)) {
    std::error_code ec;
    std::filesystem::remove(stageFile, ec);
    return fail("dataset reimport staging failed: " + stageError, error);
  }

  tsd::core::logStatus(
      "[SciVisStudio] Loading staged replacement for dataset '%s'",
      dataset->name.c_str());
  Dataset replacement;
  tsd::scene::LayerNodeRef replacementRoot;
  if (!loadDatasetArchiveFile(m_ctx->tsd.scene,
          m_ctx->tsd.animationMgr,
          stageFile,
          ensureDatasetsRoot(),
          replacement,
          replacementRoot,
          &stageError)) {
    std::error_code ec;
    std::filesystem::remove(stageFile, ec);
    return fail("dataset reimport staging failed: " + stageError, error);
  }
  std::error_code ec;
  std::filesystem::remove(stageFile, ec);

  tsd::core::logStatus("[SciVisStudio] Installing replacement for dataset '%s'",
      dataset->name.c_str());
  const auto persistedName = dataset->persistedName;
  if (auto oldRoot = resolveDatasetRoot(*dataset))
    removeDatasetRuntime(m_ctx->tsd.scene, m_ctx->tsd.animationMgr, oldRoot);
  replacement.id = id;
  replacement.name = dataset->name;
  replacement.source = dataset->source;
  replacement.persistedName = persistedName;
  replacement.dirty = true;
  (*replacementRoot)->name() = id;
  replacement.rootNode = refFor("studio", replacementRoot);
  *dataset = std::move(replacement);
  m_project.markDirty();
  applyActiveShot();
  tsd::core::logStatus(
      "[SciVisStudio] Reimported dataset '%s'", dataset->name.c_str());
  return true;
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

  if (auto *cameraRig = activeShotCameraRig()) {
    auto sampled = camera_rig::sampleCameraRig(*cameraRig, shot->currentFrame);
    camera_rig::applyManipulatorState(m_ctx->view.manipulator, sampled);
  }

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
  if (!m_ctx)
    return fail("missing TSD application context", error);

  ProjectSaveRequest request(
      m_project, m_ctx->tsd.scene, m_ctx->tsd.animationMgr, directory);
  request.pendingAssetRemovals = m_pendingAssetRemovals;
  request.windows = windows;
  request.layout = layout;
  request.settings = settings;

  ProjectSaveResult save;
  if (!buildProjectSavePlan(request, save, error))
    return false;

  AssetTransaction transaction;
  if (!transaction.commit(save.plan, error))
    return false;

  m_project = std::move(save.project);
  m_pendingAssetRemovals.clear();
  tsd::core::logStatus(
      "[SciVisStudio] Saved project '%s'", directory.string().c_str());
  return true;
}

bool ProjectContext::openProject(const std::filesystem::path &directory,
    tsd::core::DataNode *windowsOut,
    std::string *layoutOut,
    tsd::core::DataNode *settingsOut,
    std::string *error,
    const ProjectOpenOptions &options)
{
  if (!m_ctx)
    return fail("missing TSD application context", error);

  ProjectOpenStage stage;
  if (!stageProjectOpen(directory, stage, options, error))
    return false;

  m_ctx->clearSelected();
  m_syncingAnimationManager = true;
  const bool applied =
      applyProjectOpen(stage, m_ctx->tsd.scene, m_ctx->tsd.animationMgr, error);
  m_syncingAnimationManager = false;
  if (!applied)
    return false;

  m_project = std::move(stage.project);
  m_pendingAssetRemovals.clear();
  if (!m_project.shots.empty()) {
    if (m_project.cameraRigs.empty()) {
      CameraRig rig;
      rig.id = camera_rig::nextCameraRigId(m_project);
      rig.name = "Default";
      rig.current =
          camera_rig::manipulatorStateFromManipulator(m_ctx->view.manipulator);
      m_project.cameraRigs.push_back(std::move(rig));
    }
    for (auto &shot : m_project.shots) {
      if (shot.cameraRigId.empty())
        shot.cameraRigId = m_project.cameraRigs.front().id;
    }
  }
  syncAnimationManagerToActiveShot();

  if (windowsOut) {
    windowsOut->reset();
    if (auto *windows = stage.ui.root().child("windows"))
      *windowsOut = *windows;
  }
  if (layoutOut) {
    layoutOut->clear();
    if (auto *layout = stage.ui.root().child("layout"))
      *layoutOut = layout->getValueAs<std::string>();
  }
  if (settingsOut) {
    settingsOut->reset();
    if (auto *settings = stage.ui.root().child("settings"))
      *settingsOut = *settings;
  }

  applyActiveShot();
  tsd::core::logStatus(
      "[SciVisStudio] Opened project '%s'", directory.string().c_str());
  return true;
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
  case tsd::io::ImporterType::PBRT:
    return "PBRT";
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
  case tsd::io::ImporterType::SWC_SDF:
    return "SWC_SDF";
  case tsd::io::ImporterType::TRK:
    return "TRK";
  case tsd::io::ImporterType::USD:
    return "USD";
  case tsd::io::ImporterType::USD_MTLX:
    return "USD_MTLX";
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
