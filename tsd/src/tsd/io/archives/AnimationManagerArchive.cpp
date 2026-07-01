// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/archives/AnimationManagerArchive.hpp"
// tsd_animation
#include "tsd/animation/AnimationManager.hpp"
// tsd_core
#include "tsd/core/DataTree.hpp"
#include "tsd/core/Logging.hpp"
// tsd_io
#include "tsd/io/archives/AnimationArchive.hpp"

namespace tsd::io {

namespace {

struct ManagerState
{
  float time{0.f};
  float increment{0.01f};
  float fps{30.f};
  int totalFrames{100};
  bool loop{true};
};

bool readManagerState(
    core::DataNode &archive, ManagerState &state, std::string &message)
{
  auto *time = archive.child("time");
  auto *increment = archive.child("increment");
  auto *totalFrames = archive.child("totalFrames");
  auto *fps = archive.child("fps");
  if (!time || !increment || !totalFrames || !fps
      || !time->getValue(ANARI_FLOAT32, &state.time)
      || !increment->getValue(ANARI_FLOAT32, &state.increment)
      || !totalFrames->getValue(ANARI_INT32, &state.totalFrames)
      || !fps->getValue(ANARI_FLOAT32, &state.fps)) {
    message = "Animation Manager Archive is missing timeline state";
    return false;
  }
  if (state.totalFrames < 2 || state.fps <= 0.f) {
    message = "Animation Manager Archive has invalid timeline state";
    return false;
  }
  if (auto *loop = archive.child("loop");
      loop != nullptr && !loop->getValue(ANARI_BOOL, &state.loop)) {
    message = "Animation Manager Archive has invalid loop state";
    return false;
  }
  return true;
}

bool validateManagerArchive(animation::AnimationManager &manager,
    core::DataNode &archive,
    ManagerState &state,
    std::string &message)
{
  if (!readManagerState(archive, state, message))
    return false;
  if (auto *animations = archive.child("objects")) {
    bool valid = true;
    animations->foreach_child([&](core::DataNode &animation) {
      if (valid)
        valid = validate_AnimationArchive(manager, animation, &message);
    });
    return valid;
  }
  return true;
}

} // namespace

bool serialize_AnimationManagerArchive(
    const animation::AnimationManager &manager, core::DataNode &archive)
{
  archive.reset();
  archive["time"] = manager.getAnimationTime();
  archive["increment"] = manager.getAnimationIncrement();
  archive["totalFrames"] = manager.getAnimationTotalFrames();
  archive["fps"] = manager.getAnimationFPS();
  archive["loop"] = manager.isLoop();
  auto &animations = archive["objects"];
  for (const auto &animation : manager.animations()) {
    if (!serialize_AnimationArchive(animation, animations.append())) {
      archive.reset();
      return false;
    }
  }
  return true;
}

bool deserialize_AnimationManagerArchive(
    animation::AnimationManager &manager, core::DataNode &archive)
{
  ManagerState state;
  std::string message;
  if (!validateManagerArchive(manager, archive, state, message)) {
    core::logError("[deserialize_AnimationManagerArchive] %s", message.c_str());
    return false;
  }

  manager.stop();
  manager.removeAllAnimations();
  manager.setAnimationIncrement(state.increment);
  manager.setAnimationTotalFrames(state.totalFrames);
  manager.setAnimationFPS(state.fps);
  manager.setLoop(state.loop);
  bool loaded = true;
  if (auto *animations = archive.child("objects")) {
    animations->foreach_child([&](core::DataNode &animation) {
      if (loaded)
        loaded = deserialize_AnimationArchive(manager, animation) != nullptr;
    });
  }
  if (!loaded) {
    manager.removeAllAnimations();
    manager.stop();
    return false;
  }
  manager.setAnimationTime(state.time);
  manager.stop();
  return true;
}

bool save_AnimationManagerArchive(
    const animation::AnimationManager &manager, const char *filename)
{
  if (!filename)
    return false;
  core::DataTree tree;
  return serialize_AnimationManagerArchive(manager, tree.root())
      && tree.save(filename);
}

bool load_AnimationManagerArchive(
    animation::AnimationManager &manager, const char *filename)
{
  if (!filename)
    return false;
  core::DataTree tree;
  return tree.load(filename)
      && deserialize_AnimationManagerArchive(manager, tree.root());
}

} // namespace tsd::io
