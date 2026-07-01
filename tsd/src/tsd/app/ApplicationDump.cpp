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

namespace tsd::app {

namespace {

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

bool deserializeApplicationArchives(TSDState &state,
    core::DataNode &sceneArchive,
    core::DataNode &animationManagerArchive)
{
  state.animationMgr.stop();
  state.animationMgr.removeAllAnimations();
  return io::deserialize_SceneArchive(state.scene, sceneArchive)
      && io::deserialize_AnimationManagerArchive(
          state.animationMgr, animationManagerArchive);
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
  if (!io::serialize_SceneAndAnimationManagerArchives(context.tsd.scene,
          context.tsd.animationMgr,
          archives["scene"],
          archives["animationManager"])) {
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
  TSDState stagedState;
  if (!deserializeApplicationArchives(
          stagedState, *sceneArchive, *animationManagerArchive)) {
    return false;
  }

  if (!deserializeApplicationArchives(
          context.tsd, *sceneArchive, *animationManagerArchive)) {
    core::logError(
        "[deserialize_ApplicationDump] staged Archives failed during commit");
    return false;
  }

  deserializeStableContextSettings(context, root);
  return true;
}

} // namespace tsd::app
