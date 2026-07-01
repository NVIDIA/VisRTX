// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ApplicationDump.h"

#include "Context.h"
#include "LegacyApplicationContext.h"
// tsd_core
#include "tsd/core/DataTree.hpp"
#include "tsd/core/Logging.hpp"
// tsd_io
#include "tsd/io/archives/AnimationManagerArchive.hpp"
#include "tsd/io/archives/SceneArchive.hpp"
#include "tsd/io/archives/detail/AnimationRemap.hpp"

namespace tsd::app {

namespace {

template <typename OBJECT_POOL_T>
size_t denseObjectIndex(const OBJECT_POOL_T &pool, size_t sourceIndex)
{
  size_t denseIndex = 0;
  size_t result = core::INVALID_INDEX;
  foreach_item_const(pool, [&](const auto *object) {
    if (!object)
      return;
    if (object->index() == sourceIndex)
      result = denseIndex;
    ++denseIndex;
  });
  return result;
}

size_t sceneArchiveObjectIndex(
    const scene::Scene &scene, anari::DataType type, size_t sourceIndex)
{
  const auto &db = scene.objectDB();
  if (anari::isArray(type))
    return denseObjectIndex(db.array, sourceIndex);

  switch (type) {
  case ANARI_SAMPLER:
    return denseObjectIndex(db.sampler, sourceIndex);
  case ANARI_MATERIAL:
    return denseObjectIndex(db.material, sourceIndex);
  case ANARI_GEOMETRY:
    return denseObjectIndex(db.geometry, sourceIndex);
  case ANARI_SURFACE:
    return denseObjectIndex(db.surface, sourceIndex);
  case ANARI_SPATIAL_FIELD:
    return denseObjectIndex(db.field, sourceIndex);
  case ANARI_VOLUME:
    return denseObjectIndex(db.volume, sourceIndex);
  case ANARI_LIGHT:
    return denseObjectIndex(db.light, sourceIndex);
  case ANARI_CAMERA:
    return denseObjectIndex(db.camera, sourceIndex);
  case ANARI_RENDERER:
    return denseObjectIndex(db.renderer, sourceIndex);
  default:
    return core::INVALID_INDEX;
  }
}

size_t sceneArchiveLayerNodeIndex(
    const scene::Scene &scene, const std::string &layerName, size_t sourceIndex)
{
  auto *layer = scene.layer(core::Token(layerName));
  if (!layer)
    return core::INVALID_INDEX;

  size_t archiveIndex = 0;
  size_t result = core::INVALID_INDEX;
  layer->traverse_const(layer->root(), [&](const scene::LayerNode &node, int) {
    if (node.index() == sourceIndex)
      result = archiveIndex;
    ++archiveIndex;
    return true;
  });
  return result;
}

void deserializeStableContextSettings(Context &context, core::DataNode &root)
{
  if (auto *deviceManager = root.child("ANARIDeviceManager"))
    context.anari.loadSettings(*deviceManager);
  if (auto *offlineRendering = root.child("offlineRendering"))
    context.offline.loadSettings(*offlineRendering);
  if (auto *settings = root.child("settings")) {
    bool value = context.logVerbose();
    if (auto *logVerbose = settings->child("logVerbose");
        logVerbose && logVerbose->getValue(ANARI_BOOL, &value)) {
      context.setLogVerbose(value);
    }
    value = context.logEchoOutput();
    if (auto *logEchoOutput = settings->child("logEchoOutput");
        logEchoOutput && logEchoOutput->getValue(ANARI_BOOL, &value)) {
      context.setLogEchoOutput(value);
    }
  }

  context.view.poses.clear();
  if (auto *cameraPoses = root.child("cameraPoses")) {
    cameraPoses->foreach_child([&](core::DataNode &node) {
      rendering::CameraPose pose;
      deserialize_CameraPose(node, pose);
      context.view.poses.push_back(std::move(pose));
    });
  }
}

} // namespace

void serialize_CameraPose(
    const rendering::CameraPose &pose, core::DataNode &node)
{
  node["name"] = pose.name;
  node["lookat"] = pose.lookat;
  node["azeldist"] = pose.azeldist;
  node["fixedDist"] = pose.fixedDist;
  node["upAxis"] = pose.upAxis;
  node["mode"] = pose.mode;
}

void deserialize_CameraPose(core::DataNode &node, rendering::CameraPose &pose)
{
  node["name"].getValue(ANARI_STRING, &pose.name);
  node["lookat"].getValue(ANARI_FLOAT32_VEC3, &pose.lookat);
  node["azeldist"].getValue(ANARI_FLOAT32_VEC3, &pose.azeldist);
  node["fixedDist"].getValue(ANARI_FLOAT32, &pose.fixedDist);
  node["upAxis"].getValue(ANARI_INT32, &pose.upAxis);
  node["mode"].getValue(ANARI_INT32, &pose.mode);
}

bool serialize_ApplicationDump(const Context &context, core::DataNode &root)
{
  auto &archives = root["archives"];
  if (!io::serialize_SceneArchive(context.tsd.scene, archives["scene"])
      || !io::serialize_AnimationManagerArchive(
          context.tsd.animationMgr, archives["animationManager"])) {
    return false;
  }

  std::string remapError;
  if (!io::detail::remapSceneAnimations(
          archives["animationManager"],
          [&](anari::DataType type, size_t index) {
            return sceneArchiveObjectIndex(context.tsd.scene, type, index);
          },
          [&](const std::string &layerName, size_t index) {
            return sceneArchiveLayerNodeIndex(
                context.tsd.scene, layerName, index);
          },
          remapError)) {
    core::logError("[serialize_ApplicationDump] %s", remapError.c_str());
    return false;
  }

  context.anari.saveSettings(root["ANARIDeviceManager"]);
  context.offline.saveSettings(root["offlineRendering"]);

  auto &settings = root["settings"];
  settings["logVerbose"] = context.logVerbose();
  settings["logEchoOutput"] = context.logEchoOutput();

  auto &cameraPoses = root["cameraPoses"];
  cameraPoses.reset();
  for (const auto &pose : context.view.poses)
    serialize_CameraPose(pose, cameraPoses.append());

  return true;
}

bool deserialize_ApplicationDump(Context &context, core::DataNode &root)
{
  auto *archives = root.child("archives");
  if (!archives) {
    auto *legacyContext = root.child("context");
    auto &legacyPayload = legacyContext ? *legacyContext : root;
    if (!io::validate_SceneArchive(legacyPayload).accepted())
      return false;

    auto &animationManager = context.tsd.animationMgr;
    animationManager.stop();
    animationManager.removeAllAnimations();
    if (!detail::deserializeLegacyApplicationContext(context, legacyPayload)) {
      return false;
    }
    deserializeStableContextSettings(context, root);
    return true;
  }

  auto *sceneArchive = archives->child("scene");
  auto *animationManagerArchive = archives->child("animationManager");
  if (!sceneArchive || !animationManagerArchive) {
    core::logError(
        "[deserialize_ApplicationDump] Application Dump requires "
        "archives/scene and archives/animationManager");
    return false;
  }
  if (!io::validate_SceneArchive(*sceneArchive).accepted())
    return false;

  auto &animationManager = context.tsd.animationMgr;
  animationManager.stop();
  animationManager.removeAllAnimations();
  if (!io::deserialize_SceneArchive(context.tsd.scene, *sceneArchive))
    return false;
  if (!io::deserialize_AnimationManagerArchive(
          animationManager, *animationManagerArchive)) {
    return false;
  }

  deserializeStableContextSettings(context, root);
  return true;
}

} // namespace tsd::app
