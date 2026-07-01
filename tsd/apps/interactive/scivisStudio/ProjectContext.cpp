// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ProjectContext.h"

#include "DatasetIO.h"
#include "ProjectAssetTransaction.h"
#include "ProjectSerialization.h"

#include "tsd/core/DataTree.hpp"
#include "tsd/core/DataTreeMetadata.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/io/archives/CameraArchive.hpp"
#include "tsd/io/archives/RendererArchive.hpp"
#include "tsd/io/archives/SubtreeArchiveContent.hpp"
#include "tsd/io/serialization/serialization_internal.hpp"
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

static const tsd::io::SubtreeArchiveContentDesc LIGHT_RIG_ARCHIVE_DESC{
    LIGHT_RIG_FILE_TYPE,
    LIGHT_RIG_SCHEMA,
    tsd::io::ArchiveObjectPolicy::LightsOnly};

static bool saveLightRigArchiveFile(tsd::scene::LayerNodeRef root,
    const std::filesystem::path &file,
    std::string_view displayName)
{
  tsd::core::DataTree tree;
  return tsd::io::serialize_SubtreeArchiveContent(
             root, tree.root(), LIGHT_RIG_ARCHIVE_DESC, displayName)
      && tree.save(file.string().c_str());
}

static tsd::scene::LayerNodeRef loadLightRigArchiveFile(
    tsd::scene::Scene &scene,
    const std::filesystem::path &file,
    tsd::scene::LayerNodeRef destination,
    std::string *displayName = nullptr)
{
  tsd::core::DataTree tree;
  if (!tree.load(file.string().c_str()))
    return {};
  return tsd::io::deserialize_SubtreeArchiveContent(
      scene, tree.root(), destination, LIGHT_RIG_ARCHIVE_DESC, displayName)
      .root;
}

static bool deserializeLegacyProjectContext(tsd::scene::Scene &scene,
    tsd::animation::AnimationManager &animationManager,
    tsd::core::DataNode &node)
{
  return tsd::io::detail::tryDeserializeLegacyScenePayload(
      scene, node, nullptr, &animationManager);
}

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

using PoolArchiveSaver = bool (*)(const tsd::scene::Scene &, const char *);
using PoolArchiveValidator = tsd::io::ArchiveValidationResult (*)(
    tsd::core::DataNode &);

static ProjectAssetWrite makePoolArchiveWrite(const tsd::scene::Scene &scene,
    bool replaceExisting,
    const char *description,
    const std::filesystem::path &target,
    PoolArchiveSaver saveArchive,
    PoolArchiveValidator validateArchive,
    std::string_view expectedSchema)
{
  ProjectAssetWrite write;
  write.description = description;
  write.target = target;
  if (replaceExisting)
    write.ownedTarget = target;
  write.writer = [&scene, saveArchive, description](
                     const std::filesystem::path &file,
                     std::string *writeError) {
    if (saveArchive(scene, file.string().c_str()))
      return true;
    return fail(std::string("failed to save ") + description, writeError);
  };
  write.validator = [validateArchive, expectedSchema](
                        const std::filesystem::path &file,
                        std::string *validationError) {
    tsd::core::DataTree archive;
    if (!archive.load(file.string().c_str()))
      return fail("failed to load staged Archive", validationError);

    const auto metadata = tsd::core::readDataTreeMetadata(archive.root());
    const auto validation = validateArchive(archive.root());
    if (metadata.found() && metadata.metadata->schema == expectedSchema
        && validation.accepted()) {
      return true;
    }
    return fail(validation.message.empty() ? "Archive has unexpected metadata"
                                           : validation.message,
        validationError);
  };
  return write;
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

// Reopen a staged asset and verify its type before the transaction can commit.
static bool validateAssetMetadata(const std::filesystem::path &file,
    const char *expectedFileType,
    const char *expectedSchema,
    std::string *error)
{
  tsd::core::DataTree tree;
  if (!tree.load(file.string().c_str())) {
    if (error)
      *error = "failed to reopen staged file";
    return false;
  }
  auto metadata = tsd::core::readDataTreeMetadata(tree.root());
  if (!metadata.found()) {
    if (error)
      *error = metadata.malformed() ? metadata.message : "missing metadata";
    return false;
  }
  const auto &m = *metadata.metadata;
  if (m.fileType != expectedFileType || m.schema != expectedSchema) {
    if (error)
      *error = "unexpected asset metadata";
    return false;
  }
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
      return fail("failed to remove Dataset Archive: " + ec.message(), error);
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

bool ProjectContext::saveDatasetArchive(
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
  return saveDatasetArchiveFile(
      *dataset, root, m_ctx->tsd.animationMgr, file, error);
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
    return fail("target directory already contains project.tsd", error);
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

  const auto savedProjectName =
      (m_project.name.empty() || m_project.name == "Untitled")
      ? (directory.filename().string().empty() ? std::string("Untitled")
                                               : directory.filename().string())
      : m_project.name;

  Project savedProject = m_project;
  savedProject.name = savedProjectName;
  savedProject.projectDirectory = directory;
  normalizeAssetNames(savedProject.datasets);
  normalizeAssetNames(savedProject.cameraRigs);
  normalizeAssetNames(savedProject.lightRigs);
  savedProject.markClean();

  auto resolveDatasetForSave = [&](const Dataset &dataset) {
    if (auto *layer = m_ctx->tsd.scene.layer("studio")) {
      auto datasetsRoot = findDirectChild(layer->root(), "datasets");
      if (auto datasetRoot = findDirectChild(datasetsRoot, dataset.id))
        return datasetRoot;
    }
    return resolve(dataset.rootNode);
  };
  auto resolveLightRigForSave = [&](const LightRig &rig) {
    if (auto *layer = m_ctx->tsd.scene.layer("studio")) {
      auto lightRigsRoot = findDirectChild(layer->root(), "lightRigs");
      if (auto rigRoot = findDirectChild(lightRigsRoot, rig.id))
        return rigRoot;
    }
    return resolve(rig.rootNode);
  };

  ProjectSavePlan savePlan;
  savePlan.directory = directory;
  savePlan.directories = {"renders", "datasets", "cameras", "lights", "scene"};
  savePlan.assets.push_back(makePoolArchiveWrite(m_ctx->tsd.scene,
      savingCurrent,
      "Camera pool Archive",
      std::filesystem::path("scene") / "cameras.tsd",
      tsd::io::save_CameraArchive,
      tsd::io::validate_CameraArchive,
      tsd::io::schema::SCENE_CAMERAS));
  savePlan.assets.push_back(makePoolArchiveWrite(m_ctx->tsd.scene,
      savingCurrent,
      "Renderer pool Archive",
      std::filesystem::path("scene") / "renderers.tsd",
      tsd::io::save_RendererArchive,
      tsd::io::validate_RendererArchive,
      tsd::io::schema::SCENE_RENDERERS));

  std::vector<tsd::scene::LayerNodeRef> datasetRoots(
      savedProject.datasets.size());
  for (size_t i = 0; i < savedProject.datasets.size(); ++i) {
    auto &savedDataset = savedProject.datasets[i];
    const auto &liveDataset = m_project.datasets[i];
    const bool needsWrite = !savingCurrent || liveDataset.dirty
        || liveDataset.pendingExtraction
        || liveDataset.persistedName != savedDataset.name;
    datasetRoots[i] = resolveDatasetForSave(liveDataset);
    if (!needsWrite)
      continue;
    if (savedDataset.status != DatasetStatus::Available) {
      return fail("dataset '" + savedDataset.name
              + "' is unavailable and cannot be saved",
          error);
    }
    if (!datasetRoots[i]) {
      return fail(
          "dataset '" + savedDataset.name + "' has no scene subtree", error);
    }

    savedDataset.persistedName = savedDataset.name;
    savedDataset.pendingExtraction = false;
    savedDataset.dirty = false;
    ProjectAssetWrite write;
    write.description = "dataset '" + savedDataset.name + "'";
    write.target =
        std::filesystem::path("datasets") / (savedDataset.name + ".tsd");
    if (savingCurrent && !liveDataset.persistedName.empty()) {
      write.ownedTarget = std::filesystem::path("datasets")
          / (liveDataset.persistedName + ".tsd");
    }
    const auto *dataset = &savedDataset;
    const auto datasetRoot = datasetRoots[i];
    auto *animationManager = &m_ctx->tsd.animationMgr;
    write.writer = [dataset, datasetRoot, animationManager](
                       const std::filesystem::path &file,
                       std::string *writeError) {
      return saveDatasetArchiveFile(
          *dataset, datasetRoot, *animationManager, file, writeError);
    };
    const auto expectedName = savedDataset.name;
    write.validator = [expectedName](const std::filesystem::path &file,
                          std::string *validationError) {
      auto validation = validateDatasetAsset(file);
      if (!validation.ok) {
        if (validationError)
          *validationError = validation.error;
        return false;
      }
      if (validation.dataset.name != expectedName) {
        if (validationError)
          *validationError = "dataset name does not match target";
        return false;
      }
      return true;
    };
    savePlan.assets.push_back(std::move(write));
  }

  for (size_t i = 0; i < savedProject.cameraRigs.size(); ++i) {
    auto &savedRig = savedProject.cameraRigs[i];
    const auto &liveRig = m_project.cameraRigs[i];
    savedRig.persistedName = savedRig.name;

    ProjectAssetWrite write;
    write.description = "camera rig '" + savedRig.name + "'";
    write.target = std::filesystem::path("cameras") / (savedRig.name + ".tsd");
    if (savingCurrent && !liveRig.persistedName.empty()) {
      write.ownedTarget =
          std::filesystem::path("cameras") / (liveRig.persistedName + ".tsd");
    }
    const auto *rig = &savedRig;
    write.writer = [rig](const std::filesystem::path &file,
                       std::string *writeError) {
      return camera_rig::saveCameraRigArchiveFile(*rig, file, writeError);
    };
    const auto expectedName = savedRig.name;
    write.validator = [expectedName](const std::filesystem::path &file,
                          std::string *validationError) {
      CameraRig staged;
      if (!camera_rig::loadCameraRigArchiveFile(file, staged, validationError))
        return false;
      if (staged.name != expectedName) {
        if (validationError)
          *validationError = "camera rig name does not match target";
        return false;
      }
      return true;
    };
    savePlan.assets.push_back(std::move(write));
  }

  for (size_t i = 0; i < savedProject.lightRigs.size(); ++i) {
    auto &savedRig = savedProject.lightRigs[i];
    const auto &liveRig = m_project.lightRigs[i];
    auto rigRoot = resolveLightRigForSave(liveRig);
    if (!rigRoot)
      return fail("light rig '" + savedRig.name + "' has no scene node", error);
    savedRig.persistedName = savedRig.name;

    ProjectAssetWrite write;
    write.description = "light rig '" + savedRig.name + "'";
    write.target = std::filesystem::path("lights") / (savedRig.name + ".tsd");
    if (savingCurrent && !liveRig.persistedName.empty()) {
      write.ownedTarget =
          std::filesystem::path("lights") / (liveRig.persistedName + ".tsd");
    }
    const auto expectedName = savedRig.name;
    write.writer = [rigRoot, expectedName](const std::filesystem::path &file,
                       std::string *writeError) {
      if (saveLightRigArchiveFile(rigRoot, file, expectedName))
        return true;
      if (writeError)
        *writeError = "failed to serialize light rig (see log for details)";
      return false;
    };
    write.validator = [](const std::filesystem::path &file,
                          std::string *validationError) {
      return validateAssetMetadata(
          file, LIGHT_RIG_FILE_TYPE, LIGHT_RIG_SCHEMA, validationError);
    };
    savePlan.assets.push_back(std::move(write));
  }

  // Build the manifest. Scene-owned pools and project assets live in their
  // required Archives; the manifest contains only project and UI state.
  tsd::core::DataTree tree;
  auto &root = tree.root();
  tsd::core::writeDataTreeMetadata(root,
      {tsd::core::DATA_TREE_METADATA_ENVELOPE_VERSION,
          PROJECT_FILE_TYPE,
          PROJECT_SCHEMA,
          SCHEMA_VERSION});
  projectToNode(savedProject, root["scivisStudio"]);

  if (windows)
    root["windows"] = *windows;
  if (!layout.empty())
    root["layout"] = layout;
  if (settings)
    root["settings"] = *settings;

  savePlan.manifest.description = "project manifest";
  savePlan.manifest.target = PROJECT_MANIFEST_FILENAME;
  if (savingCurrent)
    savePlan.manifest.ownedTarget = PROJECT_MANIFEST_FILENAME;
  savePlan.manifest.writer = [&tree](const std::filesystem::path &file,
                                 std::string *writeError) {
    if (tree.save(file.string().c_str()))
      return true;
    if (writeError)
      *writeError = "failed to serialize project manifest";
    return false;
  };
  savePlan.manifest.validator = [](const std::filesystem::path &file,
                                    std::string *validationError) {
    return validateAssetMetadata(
        file, PROJECT_FILE_TYPE, PROJECT_SCHEMA, validationError);
  };

  auto addRemoval = [&](const std::filesystem::path &path) {
    const bool alreadyAdded = std::any_of(savePlan.removals.begin(),
        savePlan.removals.end(),
        [&](const auto &removal) {
          return assetNamesEqual(
              removal.generic_string(), path.generic_string());
        });
    if (!alreadyAdded)
      savePlan.removals.push_back(path);
  };
  if (savingCurrent) {
    for (const auto &removal : m_pendingAssetRemovals)
      addRemoval(removal);
    for (size_t i = 0; i < savedProject.datasets.size(); ++i) {
      const auto &oldName = m_project.datasets[i].persistedName;
      if (!oldName.empty()
          && !assetNamesEqual(oldName, savedProject.datasets[i].name)) {
        addRemoval(std::filesystem::path("datasets") / (oldName + ".tsd"));
      }
    }
    for (size_t i = 0; i < savedProject.cameraRigs.size(); ++i) {
      const auto &oldName = m_project.cameraRigs[i].persistedName;
      if (!oldName.empty()
          && !assetNamesEqual(oldName, savedProject.cameraRigs[i].name)) {
        addRemoval(std::filesystem::path("cameras") / (oldName + ".tsd"));
      }
    }
    for (size_t i = 0; i < savedProject.lightRigs.size(); ++i) {
      const auto &oldName = m_project.lightRigs[i].persistedName;
      if (!oldName.empty()
          && !assetNamesEqual(oldName, savedProject.lightRigs[i].name)) {
        addRemoval(std::filesystem::path("lights") / (oldName + ".tsd"));
      }
    }
  }

  AssetTransaction transaction;
  if (!transaction.commit(savePlan, error))
    return false;

  for (size_t i = 0; i < m_project.datasets.size(); ++i) {
    m_project.datasets[i].name = savedProject.datasets[i].name;
    m_project.datasets[i].persistedName = savedProject.datasets[i].name;
    m_project.datasets[i].pendingExtraction = false;
    m_project.datasets[i].dirty = false;
  }
  for (size_t i = 0; i < m_project.cameraRigs.size(); ++i) {
    m_project.cameraRigs[i].name = savedProject.cameraRigs[i].name;
    m_project.cameraRigs[i].persistedName = savedProject.cameraRigs[i].name;
  }
  for (size_t i = 0; i < m_project.lightRigs.size(); ++i) {
    m_project.lightRigs[i].name = savedProject.lightRigs[i].name;
    m_project.lightRigs[i].persistedName = savedProject.lightRigs[i].name;
  }

  m_project.projectDirectory = directory;
  m_project.name = savedProjectName;
  m_project.markClean();
  m_pendingAssetRemovals.clear();
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
  if (!m_ctx)
    return fail("missing TSD application context", error);

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
  auto manifestMetadata = tsd::core::readDataTreeMetadata(root);
  const int loadedSchemaVersion = manifestMetadata.found()
      ? manifestMetadata.metadata->schemaVersion
      : root["schemaVersion"].getValueOr<int>(1);
  resetScene();
  m_syncingAnimationManager = true;
  if (loadedSchemaVersion >= DECOMPOSED_SCENE_SCHEMA_VERSION) {
    auto shotsRoot = ensureShotsRoot();
    ensureDatasetsRoot();
    ensureLightRigsRoot();
    for (const auto &shot : loadedProject.shots)
      ensureChild(shotsRoot, shot.id.c_str());

    const auto cameras = directory / "scene/cameras.tsd";
    const auto renderers = directory / "scene/renderers.tsd";
    if (!tsd::io::load_CameraArchive(m_ctx->tsd.scene, cameras.string().c_str())
        || !tsd::io::load_RendererArchive(
            m_ctx->tsd.scene, renderers.string().c_str())) {
      m_syncingAnimationManager = false;
      return fail("failed to load required scene pool Archives", error);
    }
  } else if (auto *context = root.child("context")) {
    deserializeLegacyProjectContext(
        m_ctx->tsd.scene, m_ctx->tsd.animationMgr, *context);
  }
  m_syncingAnimationManager = false;

  loadedProject.projectDirectory = directory;
  loadedProject.markClean();
  m_project = std::move(loadedProject);
  m_pendingAssetRemovals.clear();
  if (loadedSchemaVersion < 2)
    migrateLegacyShotLightsToLightRigs();
  if (loadedSchemaVersion >= 4) {
    // v4 stores rigs as standalone Archives; the manifest carries ids+names.
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
    if (!camera_rig::loadCameraRigArchiveFile(file, data, &err)) {
      tsd::core::logWarning(
          "[SciVisStudio] Skipping Camera Rig Archive '%s': %s",
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
    rig.persistedName = rig.name;
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
    if (!loadDatasetArchiveFile(m_ctx->tsd.scene,
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
      spliced = loadLightRigArchiveFile(scene, file, lightRigsRoot, &name);
      if (spliced)
        (*spliced)->name() = rig.id; // resolveLightRigRoot keys on node==rig.id
      scene.endLayerEditBatch();
    }

    if (!spliced) {
      tsd::core::logWarning(
          "[SciVisStudio] Skipping Light Rig Archive '%s': %s",
          rig.name.c_str(),
          readable ? "failed to load Archive" : "Archive is missing");
      for (auto &shot : m_project.shots) {
        if (shot.lightRigId == rig.id)
          shot.lightRigId.clear();
      }
      continue;
    }

    rig.rootNode = refFor("studio", spliced);
    rig.persistedName = rig.name;
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
