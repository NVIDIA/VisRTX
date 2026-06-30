// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ProjectContext.h"

#include "DatasetIO.h"
#include "ProjectSerialization.h"

#include "tsd/core/DataTreeMetadata.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/io/serialization.hpp"
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

  std::string candidate = desired + " (imported)";
  for (int n = 2; taken(candidate); ++n)
    candidate = desired + " (imported " + std::to_string(n) + ")";
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

// Coerce every asset name to a valid, case-insensitively-unique filename stem.
// Idempotent for already-valid names; rescues legacy/free-form names on save.
template <typename AssetT>
static void normalizeAssetNames(std::vector<AssetT> &assets)
{
  std::vector<std::string> assigned;
  for (auto &asset : assets) {
    const std::string base = sanitizeRigName(asset.name);
    std::string candidate = base;
    for (int n = 2; std::any_of(assigned.begin(),
             assigned.end(),
             [&](const std::string &a) {
               return assetNamesEqual(a, candidate);
             });
         ++n)
      candidate = base + " (" + std::to_string(n) + ")";
    asset.name = candidate;
    assigned.push_back(candidate);
  }
}

// True if file is a TSD rig file tagged with the given fileType/schema. Used to
// avoid deleting unrelated ".tsd" files during reconcile.
static bool isRigFileOfType(const std::filesystem::path &file,
    const char *expectedFileType,
    const char *expectedSchema)
{
  tsd::core::DataTree tree;
  if (!tree.load(file.string().c_str()))
    return false;
  auto metadata = tsd::core::readDataTreeMetadata(tree.root());
  if (!metadata.found())
    return false;
  const auto &m = *metadata.metadata;
  return m.fileType == expectedFileType && m.schema == expectedSchema;
}

// Delete rig files in dir whose stem is not in keepStems (case-insensitive).
// Only files we recognize as our own rig type are ever removed, so unrelated
// ".tsd" files dropped in the folder are left untouched. Callers must have
// already written the current rig files (write-before-delete).
static void removeStaleRigFiles(const std::filesystem::path &dir,
    const std::vector<std::string> &keepStems,
    const char *expectedFileType,
    const char *expectedSchema)
{
  std::error_code ec;
  for (const auto &entry : std::filesystem::directory_iterator(dir, ec)) {
    if (ec)
      break;
    const auto &path = entry.path();
    if (!entry.is_regular_file(ec) || path.extension() != ".tsd")
      continue;
    const auto stem = path.stem().string();
    const bool keep = std::any_of(keepStems.begin(),
        keepStems.end(),
        [&](const std::string &s) { return assetNamesEqual(s, stem); });
    if (keep)
      continue;
    if (!isRigFileOfType(path, expectedFileType, expectedSchema)) {
      tsd::core::logWarning(
          "[SciVisStudio] Leaving unrecognized file in rig directory: %s",
          path.string().c_str());
      continue;
    }
    std::filesystem::remove(path, ec);
  }
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

bool ProjectContext::exportCameraRig(const CameraRigID &id,
    const std::filesystem::path &file,
    std::string *error)
{
  auto *rig = camera_rig::findCameraRig(m_project, id);
  if (!rig) {
    if (error)
      *error = "camera rig not found";
    return false;
  }
  return camera_rig::exportCameraRigFile(*rig, file, error);
}

CameraRig *ProjectContext::importCameraRig(
    const std::filesystem::path &file, std::string *error)
{
  CameraRig rig;
  if (!camera_rig::importCameraRigFile(file, rig, error))
    return nullptr;

  const std::string importedName = std::move(rig.name);
  rig.id = camera_rig::nextCameraRigId(m_project);
  rig.name = makeValidUniqueAssetName(m_project.cameraRigs,
      importedName.empty() ? "Imported Camera Rig" : importedName);
  m_project.cameraRigs.push_back(std::move(rig));
  m_project.markDirty();
  return &m_project.cameraRigs.back();
}

bool ProjectContext::exportLightRig(
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

  const tsd::io::SubtreeIODesc desc{LIGHT_RIG_FILE_TYPE,
      LIGHT_RIG_SCHEMA,
      tsd::io::ArchiveObjectPolicy::LightsOnly};
  if (!tsd::io::export_Subtree(
          file.string().c_str(), rigRoot, desc, rig->name)) {
    if (error)
      *error = "failed to export light rig (see log for details)";
    return false;
  }
  return true;
}

LightRig *ProjectContext::importLightRig(
    const std::filesystem::path &file, std::string *error)
{
  if (!m_ctx) {
    if (error)
      *error = "missing TSD application context";
    return nullptr;
  }

  auto &scene = m_ctx->tsd.scene;
  auto lightRigsRoot = ensureLightRigsRoot();
  const tsd::io::SubtreeIODesc desc{LIGHT_RIG_FILE_TYPE,
      LIGHT_RIG_SCHEMA,
      tsd::io::ArchiveObjectPolicy::LightsOnly};

  std::string name;
  LightRig rig;
  scene.beginLayerEditBatch();
  auto splicedRoot = tsd::io::import_Subtree(
      scene, file.string().c_str(), lightRigsRoot, desc, &name);
  if (splicedRoot) {
    rig.id = light_rig::nextLightRigId(m_project);
    (*splicedRoot)->name() = rig.id; // resolveLightRigRoot keys on node==rig.id
  }
  scene.endLayerEditBatch();

  if (!splicedRoot) {
    if (error)
      *error = "failed to import light rig (see log for details)";
    return nullptr;
  }

  rig.name = makeValidUniqueAssetName(
      m_project.lightRigs, name.empty() ? "Imported Light Rig" : name);
  rig.rootNode = refFor("studio", splicedRoot);
  m_project.lightRigs.push_back(std::move(rig));
  m_project.markDirty();
  applyActiveShot(); // imported rig is unbound, so it starts hidden
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
      if (auto *activeShot = project::activeShot(m_project)) {
        for (const auto &dataset : m_project.datasets) {
          if (dataset.id == record.id
              || dataset.sourceKind != DatasetSourceKind::FileAnimation)
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
  record.dirty = record.status == DatasetStatus::Available;
  applyActiveShot();
  return &record;
}

bool ProjectContext::renameDataset(
    const DatasetID &id, const std::string &newName, std::string *error)
{
  if (!renameAssetImpl(m_project.datasets, id, newName, "dataset", error))
    return false;
  auto *dataset = project::findDataset(m_project, id);
  dataset->dirty = dataset->status == DatasetStatus::Available
      || dataset->name != dataset->persistedName;
  m_project.markDirty();
  return true;
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
    std::error_code ec;
    std::filesystem::remove(
        m_project.projectDirectory / "datasets" / (itr->persistedName + ".tsd"),
        ec);
    if (ec)
      return fail("failed to remove dataset asset: " + ec.message(), error);
  }

  if (m_ctx) {
    if (auto root = resolveDatasetRoot(*itr))
      removeDatasetRuntime(m_ctx->tsd.scene, m_ctx->tsd.animationMgr, root);
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

bool ProjectContext::exportDataset(
    const DatasetID &id, const std::filesystem::path &file, std::string *error)
{
  if (!m_ctx)
    return fail("missing TSD application context", error);
  auto *dataset = project::findDataset(m_project, id);
  if (!dataset)
    return fail("dataset not found", error);
  if (dataset->status != DatasetStatus::Available)
    return fail("dataset is unavailable", error);
  auto root = resolveDatasetRoot(*dataset);
  if (!root)
    return fail("dataset has no scene subtree", error);
  return exportDatasetAsset(
      *dataset, root, m_ctx->tsd.animationMgr, file, error);
}

Dataset *ProjectContext::importDatasetImpl(const std::filesystem::path &file,
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
  if (!importDatasetAsset(m_ctx->tsd.scene,
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

Dataset *ProjectContext::importDataset(
    const std::filesystem::path &file, std::string *error)
{
  return importDatasetImpl(file, {}, false, error);
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
  return importDatasetImpl(candidate.file, name, sameManagedPath, error);
}

bool ProjectContext::reimportStaticDataset(
    const DatasetID &id, std::string *error)
{
  if (!m_ctx)
    return fail("missing TSD application context", error);
  auto *dataset = project::findDataset(m_project, id);
  if (!dataset)
    return fail("dataset not found", error);
  if (dataset->sourceKind != DatasetSourceKind::Static)
    return fail("only static datasets can be reimported", error);
  if (dataset->source.sourcePath.empty())
    return fail("dataset has no provenance source path", error);

  tsd::scene::Scene stagedScene;
  tsd::animation::AnimationManager stagedAnimations(&stagedScene);
  auto stagedRoot = stagedScene.insertChildNode(
      stagedScene.defaultLayer()->root(), "dataset");
  try {
    tsd::io::import_file(stagedScene,
        stagedAnimations,
        {importerTypeFromString(dataset->importerType),
            dataset->source.sourcePath},
        stagedRoot);
  } catch (const std::exception &e) {
    return fail(std::string("dataset reimport failed: ") + e.what(), error);
  } catch (...) {
    return fail("dataset reimport failed", error);
  }
  if (hasFileBindingAnimations(stagedAnimations, 0))
    return fail("static dataset reimport created a file animation", error);
  if (!hasObjectNodes(stagedRoot))
    return fail("dataset reimport created no scene objects", error);

  const auto stageFile = std::filesystem::temp_directory_path()
      / ("scivis-dataset-reimport-"
          + std::to_string(
              std::chrono::steady_clock::now().time_since_epoch().count())
          + ".tsd");
  Dataset replacementMetadata = *dataset;
  std::string stageError;
  if (!exportDatasetAsset(replacementMetadata,
          stagedRoot,
          stagedAnimations,
          stageFile,
          &stageError)) {
    std::error_code ec;
    std::filesystem::remove(stageFile, ec);
    return fail("dataset reimport staging failed: " + stageError, error);
  }

  Dataset replacement;
  tsd::scene::LayerNodeRef replacementRoot;
  if (!importDatasetAsset(m_ctx->tsd.scene,
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
  if (!m_ctx) {
    if (error)
      *error = "missing TSD application context";
    return false;
  }

  std::error_code ec;
  const auto manifest = directory / PROJECT_MANIFEST_FILENAME;
  bool savingCurrent = false;
  if (!m_project.projectDirectory.empty()) {
    savingCurrent = directory.lexically_normal()
        == m_project.projectDirectory.lexically_normal();
    if (!savingCurrent) {
      savingCurrent = std::filesystem::equivalent(
          directory, m_project.projectDirectory, ec);
      ec.clear();
    }
  }

  if (std::filesystem::exists(manifest) && !savingCurrent) {
    if (error)
      *error = "target directory already contains project.tsd";
    return false;
  }

  if (!savingCurrent) {
    std::vector<std::string> unavailable;
    for (const auto &dataset : m_project.datasets) {
      if (dataset.status != DatasetStatus::Available)
        unavailable.push_back(dataset.name);
    }
    if (!unavailable.empty()) {
      if (error) {
        *error = "Save As requires every dataset to be available:";
        for (const auto &name : unavailable)
          *error += "\n- " + name;
      }
      return false;
    }
  }

  std::filesystem::create_directories(directory, ec);
  if (ec) {
    if (error)
      *error = "failed to create project directory: " + ec.message();
    return false;
  }

  for (const auto *subdirectory :
      {"renders", "datasets", "cameras", "lights"}) {
    std::filesystem::create_directories(directory / subdirectory, ec);
    if (ec) {
      if (error)
        *error = std::string("failed to create ") + subdirectory
            + " directory: " + ec.message();
      return false;
    }
  }

  const auto savedProjectName =
      (m_project.name.empty() || m_project.name == "Untitled")
      ? (directory.filename().string().empty() ? std::string("Untitled")
                                               : directory.filename().string())
      : m_project.name;

  // Each rig is stored as its own portable "<name>.tsd". Normalize names so
  // they map to safe, unique filenames before any file is written.
  normalizeAssetNames(m_project.datasets);
  normalizeAssetNames(m_project.cameraRigs);
  normalizeAssetNames(m_project.lightRigs);

  struct StagedDataset
  {
    Dataset *dataset{nullptr};
    std::filesystem::path stage;
    std::filesystem::path target;
    std::filesystem::path backup;
    bool hadTarget{false};
    bool installed{false};
  };

  const auto transactionTag = std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count());
  const auto datasetsDir = directory / "datasets";
  std::vector<StagedDataset> stagedDatasets;
  for (auto &dataset : m_project.datasets) {
    if (savingCurrent && !dataset.dirty && !dataset.pendingExtraction)
      continue;
    if (dataset.status != DatasetStatus::Available) {
      if (error)
        *error =
            "dataset '" + dataset.name + "' is unavailable and cannot be saved";
      return false;
    }
    auto root = resolveDatasetRoot(dataset);
    if (!root) {
      if (error)
        *error = "dataset '" + dataset.name + "' has no scene subtree";
      return false;
    }

    StagedDataset staged;
    staged.dataset = &dataset;
    staged.target = datasetsDir / (dataset.name + ".tsd");
    staged.stage = datasetsDir
        / ("." + dataset.name + ".stage-" + transactionTag + ".tsd");
    staged.backup = datasetsDir
        / ("." + dataset.name + ".backup-" + transactionTag + ".tsd");
    std::string datasetError;
    if (!exportDatasetAsset(dataset,
            root,
            m_ctx->tsd.animationMgr,
            staged.stage,
            &datasetError)) {
      for (const auto &pending : stagedDatasets)
        std::filesystem::remove(pending.stage, ec);
      std::filesystem::remove(staged.stage, ec);
      if (error)
        *error =
            "failed to stage dataset '" + dataset.name + "': " + datasetError;
      return false;
    }
    stagedDatasets.push_back(std::move(staged));
  }

  const auto camerasDir = directory / "cameras";
  const auto lightsDir = directory / "lights";

  // Write every rig file first and abort the whole save on any failure, before
  // touching the existing manifest or deleting stale files.
  std::vector<std::string> cameraStems;
  for (const auto &rig : m_project.cameraRigs) {
    const auto file = camerasDir / (rig.name + ".tsd");
    std::string rigError;
    if (!camera_rig::exportCameraRigFile(rig, file, &rigError)) {
      if (error)
        *error = "failed to write camera rig '" + rig.name + "': " + rigError;
      for (const auto &pending : stagedDatasets)
        std::filesystem::remove(pending.stage, ec);
      return false;
    }
    cameraStems.push_back(rig.name);
  }

  std::vector<std::string> lightStems;
  std::vector<tsd::scene::LayerNodeRef> lightRigRoots;
  for (auto &rig : m_project.lightRigs) {
    auto rigRoot = resolveLightRigRoot(rig);
    if (!rigRoot) {
      if (error)
        *error = "light rig '" + rig.name + "' has no scene node";
      for (const auto &pending : stagedDatasets)
        std::filesystem::remove(pending.stage, ec);
      return false;
    }
    const auto file = lightsDir / (rig.name + ".tsd");
    const tsd::io::SubtreeIODesc desc{LIGHT_RIG_FILE_TYPE,
        LIGHT_RIG_SCHEMA,
        tsd::io::ArchiveObjectPolicy::LightsOnly};
    if (!tsd::io::export_Subtree(
            file.string().c_str(), rigRoot, desc, rig.name)) {
      if (error)
        *error = "failed to write light rig '" + rig.name
            + "' (see log for details)";
      for (const auto &pending : stagedDatasets)
        std::filesystem::remove(pending.stage, ec);
      return false;
    }
    lightStems.push_back(rig.name);
    lightRigRoots.push_back(rigRoot);
  }

  // Build the manifest. Camera-rig value data and light-rig subtrees now live
  // in their own files, and datasets own their subtrees and animations.
  tsd::core::DataTree tree;
  auto &root = tree.root();
  tsd::core::writeDataTreeMetadata(root,
      {tsd::core::DATA_TREE_METADATA_ENVELOPE_VERSION,
          PROJECT_FILE_TYPE,
          PROJECT_SCHEMA,
          SCHEMA_VERSION});
  Project manifestProject = m_project;
  manifestProject.name = savedProjectName;
  manifestProject.projectDirectory = directory;
  manifestProject.markClean();
  projectToNode(manifestProject, root["scivisStudio"]);

  tsd::io::SaveSceneOptions sceneOptions;
  sceneOptions.animationManager = &m_ctx->tsd.animationMgr;
  sceneOptions.exclusion.roots = lightRigRoots;
  for (auto &dataset : m_project.datasets) {
    if (auto datasetRoot = resolveDatasetRoot(dataset))
      sceneOptions.exclusion.roots.push_back(datasetRoot);
  }
  sceneOptions.exclusion.objectPolicy = tsd::io::ArchiveObjectPolicy::All;
  sceneOptions.exclusion.animations =
      tsd::io::ExcludedAnimationPolicy::OmitOwned;
  tsd::io::save_Scene(m_ctx->tsd.scene, root["context"], sceneOptions);

  if (windows)
    root["windows"] = *windows;
  if (!layout.empty())
    root["layout"] = layout;
  if (settings)
    root["settings"] = *settings;

  const auto manifestStage =
      directory / (".project.stage-" + transactionTag + ".tsd");
  const auto manifestBackup =
      directory / (".project.backup-" + transactionTag + ".tsd");
  if (!tree.save(manifestStage.string().c_str())) {
    for (const auto &pending : stagedDatasets)
      std::filesystem::remove(pending.stage, ec);
    if (error)
      *error = "failed to stage project.tsd";
    return false;
  }

  auto rollbackDatasets = [&]() {
    for (auto it = stagedDatasets.rbegin(); it != stagedDatasets.rend(); ++it) {
      if (it->installed)
        std::filesystem::remove(it->target, ec);
      if (it->hadTarget)
        std::filesystem::rename(it->backup, it->target, ec);
      std::filesystem::remove(it->stage, ec);
    }
  };

  for (auto &staged : stagedDatasets) {
    staged.hadTarget = std::filesystem::exists(staged.target);
    if (staged.hadTarget) {
      std::filesystem::rename(staged.target, staged.backup, ec);
      if (ec) {
        const auto failure = ec.message();
        rollbackDatasets();
        std::filesystem::remove(manifestStage, ec);
        if (error)
          *error = "failed to prepare dataset replacement: " + failure;
        return false;
      }
    }
    std::filesystem::rename(staged.stage, staged.target, ec);
    if (ec) {
      const auto failure = ec.message();
      rollbackDatasets();
      std::filesystem::remove(manifestStage, ec);
      if (error)
        *error = "failed to install staged dataset: " + failure;
      return false;
    }
    staged.installed = true;
  }

  const bool hadManifest = std::filesystem::exists(manifest);
  if (hadManifest) {
    std::filesystem::rename(manifest, manifestBackup, ec);
    if (ec) {
      const auto failure = ec.message();
      rollbackDatasets();
      std::filesystem::remove(manifestStage, ec);
      if (error)
        *error = "failed to prepare project.tsd replacement: " + failure;
      return false;
    }
  }
  std::filesystem::rename(manifestStage, manifest, ec);
  if (ec) {
    const auto failure = ec.message();
    if (hadManifest)
      std::filesystem::rename(manifestBackup, manifest, ec);
    rollbackDatasets();
    if (error)
      *error = "failed to install project.tsd: " + failure;
    return false;
  }

  if (hadManifest)
    std::filesystem::remove(manifestBackup, ec);
  for (auto &staged : stagedDatasets) {
    if (staged.hadTarget)
      std::filesystem::remove(staged.backup, ec);
  }

  // Only explicit renames remove superseded managed dataset paths. Unlisted
  // dataset files remain untouched for discovery.
  std::vector<std::filesystem::path> currentDatasetPaths;
  for (const auto &dataset : m_project.datasets)
    currentDatasetPaths.push_back(datasetsDir / (dataset.name + ".tsd"));
  for (auto &dataset : m_project.datasets) {
    if (!dataset.persistedName.empty()
        && dataset.persistedName != dataset.name) {
      const auto oldPath = datasetsDir / (dataset.persistedName + ".tsd");
      if (std::find(
              currentDatasetPaths.begin(), currentDatasetPaths.end(), oldPath)
          == currentDatasetPaths.end())
        std::filesystem::remove(oldPath, ec);
    }
    dataset.persistedName = dataset.name;
    dataset.pendingExtraction = false;
    dataset.dirty = false;
  }

  // Reconcile standalone rig files only after the manifest commits.
  removeStaleRigFiles(
      camerasDir, cameraStems, CAMERA_RIG_FILE_TYPE, CAMERA_RIG_SCHEMA);
  removeStaleRigFiles(
      lightsDir, lightStems, LIGHT_RIG_FILE_TYPE, LIGHT_RIG_SCHEMA);

  m_project.projectDirectory = directory;
  m_project.name = savedProjectName;
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
  if (loadedSchemaVersion >= 4) {
    // v4 stores rigs as standalone files; the manifest only carries ids+names.
    loadCameraRigFiles(directory / "cameras");
    loadLightRigFiles(directory / "lights");
  }
  if (loadedSchemaVersion >= 5)
    loadDatasetFiles(directory / "datasets");
  if (!m_project.shots.empty()) {
    CameraRig *defaultCameraRig = nullptr;
    if (m_project.cameraRigs.empty()) {
      CameraRig rig;
      rig.id = camera_rig::nextCameraRigId(m_project);
      rig.name = "Default";
      if (m_ctx)
        rig.current = camera_rig::manipulatorStateFromManipulator(
            m_ctx->view.manipulator);
      m_project.cameraRigs.push_back(std::move(rig));
    }
    defaultCameraRig = &m_project.cameraRigs.front();
    for (auto &shot : m_project.shots) {
      if (shot.cameraRigId.empty())
        shot.cameraRigId = defaultCameraRig->id;
    }
  }
  if (loadedSchemaVersion < 5)
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

void ProjectContext::loadCameraRigFiles(const std::filesystem::path &camerasDir)
{
  std::vector<CameraRig> kept;
  kept.reserve(m_project.cameraRigs.size());
  for (auto &rig : m_project.cameraRigs) {
    const auto file = camerasDir / (rig.name + ".tsd");
    CameraRig data;
    std::string err;
    if (!camera_rig::importCameraRigFile(file, data, &err)) {
      tsd::core::logWarning("[SciVisStudio] Skipping camera rig '%s': %s",
          rig.name.c_str(),
          err.c_str());
      for (auto &shot : m_project.shots) {
        if (shot.cameraRigId == rig.id)
          shot.cameraRigId.clear();
      }
      continue;
    }
    rig.current = std::move(data.current);
    rig.keyframes = std::move(data.keyframes);
    kept.push_back(std::move(rig));
  }
  m_project.cameraRigs = std::move(kept);
}

void ProjectContext::loadDatasetFiles(const std::filesystem::path &datasetsDir)
{
  if (!m_ctx)
    return;

  auto destination = ensureDatasetsRoot();
  for (auto &inventoryEntry : m_project.datasets) {
    const auto file = datasetsDir / (inventoryEntry.name + ".tsd");
    Dataset loaded;
    tsd::scene::LayerNodeRef loadedRoot;
    std::string datasetError;
    if (!importDatasetAsset(m_ctx->tsd.scene,
            m_ctx->tsd.animationMgr,
            file,
            destination,
            loaded,
            loadedRoot,
            &datasetError)) {
      inventoryEntry.status = DatasetStatus::Unavailable;
      inventoryEntry.dirty = false;
      inventoryEntry.persistedName = inventoryEntry.name;
      tsd::core::logWarning("[SciVisStudio] Dataset '%s' is unavailable: %s",
          inventoryEntry.name.c_str(),
          datasetError.c_str());
      continue;
    }

    if (loaded.name != inventoryEntry.name) {
      removeDatasetRuntime(
          m_ctx->tsd.scene, m_ctx->tsd.animationMgr, loadedRoot);
      inventoryEntry.status = DatasetStatus::Unavailable;
      inventoryEntry.dirty = false;
      inventoryEntry.persistedName = inventoryEntry.name;
      tsd::core::logWarning(
          "[SciVisStudio] Dataset '%s' is unavailable: asset name is '%s'",
          inventoryEntry.name.c_str(),
          loaded.name.c_str());
      continue;
    }

    loaded.id = inventoryEntry.id;
    loaded.name = inventoryEntry.name;
    loaded.persistedName = inventoryEntry.name;
    loaded.dirty = false;
    (*loadedRoot)->name() = loaded.id;
    loaded.rootNode = refFor("studio", loadedRoot);
    inventoryEntry = std::move(loaded);
  }
}

void ProjectContext::loadLightRigFiles(const std::filesystem::path &lightsDir)
{
  if (!m_ctx)
    return;

  auto &scene = m_ctx->tsd.scene;
  const tsd::io::SubtreeIODesc desc{LIGHT_RIG_FILE_TYPE,
      LIGHT_RIG_SCHEMA,
      tsd::io::ArchiveObjectPolicy::LightsOnly};

  std::vector<LightRig> kept;
  kept.reserve(m_project.lightRigs.size());
  for (auto &rig : m_project.lightRigs) {
    const auto file = lightsDir / (rig.name + ".tsd");
    std::error_code ec;
    const bool readable = std::filesystem::is_regular_file(file, ec) && !ec;

    tsd::scene::LayerNodeRef spliced;
    if (readable) {
      auto lightRigsRoot = ensureLightRigsRoot();
      scene.beginLayerEditBatch();
      std::string name;
      spliced = tsd::io::import_Subtree(
          scene, file.string().c_str(), lightRigsRoot, desc, &name);
      if (spliced)
        (*spliced)->name() = rig.id; // resolveLightRigRoot keys on node==rig.id
      scene.endLayerEditBatch();
    }

    if (!spliced) {
      tsd::core::logWarning("[SciVisStudio] Skipping light rig '%s': %s",
          rig.name.c_str(),
          readable ? "failed to load light rig file"
                   : "light rig file is missing");
      for (auto &shot : m_project.shots) {
        if (shot.lightRigId == rig.id)
          shot.lightRigId.clear();
      }
      continue;
    }

    rig.rootNode = refFor("studio", spliced);
    kept.push_back(std::move(rig));
  }
  m_project.lightRigs = std::move(kept);
}

void ProjectContext::markMissingDatasets()
{
  for (auto &dataset : m_project.datasets)
    dataset.status = resolveDatasetRoot(dataset) ? DatasetStatus::Available
                                                 : DatasetStatus::Unavailable;
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
    rig.id = light_rig::nextLightRigId(m_project);
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
