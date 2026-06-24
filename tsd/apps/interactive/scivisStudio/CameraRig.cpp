// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "CameraRig.h"

#include "Project.h"
#include "ProjectSerialization.h"

#include "tsd/core/DataTreeMetadata.hpp"
#include "tsd/io/serialization.hpp"
#include "tsd/rendering/view/CameraPath.h"

#include <algorithm>
#include <cmath>

namespace tsd::scivis_studio::camera_rig {

// Collection lookups ////////////////////////////////////////////////////////

CameraRigID nextCameraRigId(const Project &project)
{
  return project::makeGeneratedId("cameraRig", project.cameraRigs.size() + 1);
}

CameraRig *findCameraRig(Project &project, const CameraRigID &id)
{
  auto itr = std::find_if(project.cameraRigs.begin(),
      project.cameraRigs.end(),
      [&](const CameraRig &r) { return r.id == id; });
  return itr == project.cameraRigs.end() ? nullptr : &*itr;
}

const CameraRig *findCameraRig(const Project &project, const CameraRigID &id)
{
  auto itr = std::find_if(project.cameraRigs.begin(),
      project.cameraRigs.end(),
      [&](const CameraRig &r) { return r.id == id; });
  return itr == project.cameraRigs.end() ? nullptr : &*itr;
}

// Interpolation enum <-> string //////////////////////////////////////////////

const char *toString(CameraInterpolation interpolation)
{
  switch (interpolation) {
  case CameraInterpolation::Hold:
    return "Hold";
  case CameraInterpolation::Linear:
    return "Linear";
  case CameraInterpolation::EaseOut:
    return "Ease Out";
  case CameraInterpolation::EaseIn:
    return "Ease In";
  case CameraInterpolation::EaseOutIn:
    return "Ease Out + In";
  }
  return "Linear";
}

CameraInterpolation interpolationFromString(const std::string &s)
{
  if (s == "Hold")
    return CameraInterpolation::Hold;
  if (s == "Ease Out")
    return CameraInterpolation::EaseOut;
  if (s == "Ease In")
    return CameraInterpolation::EaseIn;
  if (s == "Ease Out + In")
    return CameraInterpolation::EaseOutIn;
  return CameraInterpolation::Linear;
}

// Manipulator <-> stored pose ////////////////////////////////////////////////

ManipulatorState manipulatorStateFromManipulator(
    const tsd::rendering::Manipulator &m)
{
  ManipulatorState state;
  state.orbit.lookat = m.at();
  state.orbit.azeldist =
      tsd::math::float3(m.azel().x, m.azel().y, m.distance());
  state.orbit.fixedDist = m.fixedDistance();
  state.orbit.upAxis = static_cast<int>(m.axis());
  state.orbit.mode = static_cast<int>(m.mode());
  return state;
}

void applyManipulatorState(
    tsd::rendering::Manipulator &m, const ManipulatorState &state)
{
  m.setConfig(state.orbit);
  m.setFixedDistance(state.orbit.fixedDist);
}

// Keyframe animation /////////////////////////////////////////////////////////

void sortKeyframes(CameraRig &rig)
{
  std::stable_sort(rig.keyframes.begin(),
      rig.keyframes.end(),
      [](const CameraKeyframe &a, const CameraKeyframe &b) {
        return a.frame < b.frame;
      });
}

static float lerp(float t, float a, float b)
{
  return a + t * (b - a);
}

static tsd::math::float3 lerpVec3(
    float t, const tsd::math::float3 &a, const tsd::math::float3 &b)
{
  return tsd::math::float3{
      lerp(t, a.x, b.x), lerp(t, a.y, b.y), lerp(t, a.z, b.z)};
}

static float applyInterpolation(CameraInterpolation interpolation, float t)
{
  switch (interpolation) {
  case CameraInterpolation::Hold:
  case CameraInterpolation::Linear:
    return t;
  case CameraInterpolation::EaseOut:
    return t * t;
  case CameraInterpolation::EaseIn:
    return 1.f - (1.f - t) * (1.f - t) * (1.f - t);
  case CameraInterpolation::EaseOutIn:
    return t * t * t * (t * (6.f * t - 15.f) + 10.f);
  }
  return t;
}

ManipulatorState sampleCameraRig(const CameraRig &rig, int frame)
{
  if (rig.keyframes.empty())
    return rig.current;

  if (rig.keyframes.size() == 1)
    return rig.keyframes.front().manipulator;

  auto keyframes = rig.keyframes;
  std::stable_sort(keyframes.begin(),
      keyframes.end(),
      [](const CameraKeyframe &a, const CameraKeyframe &b) {
        return a.frame < b.frame;
      });

  if (frame <= keyframes.front().frame)
    return keyframes.front().manipulator;

  for (size_t i = 0; i + 1 < keyframes.size(); ++i) {
    const auto &a = keyframes[i];
    const auto &b = keyframes[i + 1];
    if (frame > b.frame)
      continue;

    if (a.interpolationToNext == CameraInterpolation::Hold
        || b.frame == a.frame)
      return a.manipulator;

    const float t = static_cast<float>(frame - a.frame)
        / static_cast<float>(b.frame - a.frame);
    const float interpolatedT = applyInterpolation(a.interpolationToNext, t);

    ManipulatorState out;
    out.orbit = a.manipulator.orbit;
    out.orbit.lookat = lerpVec3(
        interpolatedT, a.manipulator.orbit.lookat, b.manipulator.orbit.lookat);
    out.orbit.azeldist = tsd::rendering::lerpAzElDist(interpolatedT,
        a.manipulator.orbit.azeldist,
        b.manipulator.orbit.azeldist);
    out.orbit.fixedDist = lerp(interpolatedT,
        a.manipulator.orbit.fixedDist,
        b.manipulator.orbit.fixedDist);
    out.orbit.upAxis = a.manipulator.orbit.upAxis;
    out.orbit.mode = a.manipulator.orbit.mode;
    return out;
  }

  return keyframes.back().manipulator;
}

// Serialization //////////////////////////////////////////////////////////////

static void manipulatorStateToNode(
    const ManipulatorState &state, tsd::core::DataNode &node)
{
  tsd::io::cameraPoseToNode(state.orbit, node["orbit"]);
}

static void nodeToManipulatorState(
    tsd::core::DataNode &node, ManipulatorState &state)
{
  if (auto *orbit = node.child("orbit"))
    tsd::io::nodeToCameraPose(*orbit, state.orbit);
}

void cameraRigToNode(const CameraRig &rig, tsd::core::DataNode &node)
{
  manipulatorStateToNode(rig.current, node["current"]);
  auto &keyframes = node["keyframes"];
  for (const auto &keyframe : rig.keyframes) {
    auto &kf = keyframes.append();
    kf["frame"] = keyframe.frame;
    kf["name"] = keyframe.name;
    kf["interpolationToNext"] = toString(keyframe.interpolationToNext);
    manipulatorStateToNode(keyframe.manipulator, kf["manipulator"]);
  }
}

void nodeToCameraRig(tsd::core::DataNode &node, CameraRig &rig)
{
  if (auto *current = node.child("current"))
    nodeToManipulatorState(*current, rig.current);

  rig.keyframes.clear();
  if (auto *keyframes = node.child("keyframes")) {
    keyframes->foreach_child([&](tsd::core::DataNode &kf) {
      CameraKeyframe keyframe;
      keyframe.frame = kf["frame"].getValueOr<int>(0);
      keyframe.name = kf["name"].getValueOr<std::string>("");
      keyframe.interpolationToNext = interpolationFromString(
          kf["interpolationToNext"].getValueOr<std::string>("Linear"));
      if (auto *manip = kf.child("manipulator"))
        nodeToManipulatorState(*manip, keyframe.manipulator);
      rig.keyframes.push_back(std::move(keyframe));
    });
  }
  sortKeyframes(rig);
}

bool exportCameraRigFile(const CameraRig &rig,
    const std::filesystem::path &file,
    std::string *error)
{
  tsd::core::DataTree tree;
  auto &root = tree.root();
  tsd::core::writeDataTreeMetadata(root,
      {tsd::core::DATA_TREE_METADATA_ENVELOPE_VERSION,
          CAMERA_RIG_FILE_TYPE,
          CAMERA_RIG_SCHEMA,
          RIG_SCHEMA_VERSION});
  root["name"] = rig.name;
  cameraRigToNode(rig, root["rig"]);

  if (!tree.save(file.string().c_str())) {
    if (error)
      *error = "failed to write camera rig file";
    return false;
  }
  return true;
}

bool importCameraRigFile(const std::filesystem::path &file,
    CameraRig &rigOut,
    std::string *error)
{
  tsd::core::DataTree tree;
  if (!tree.load(file.string().c_str())) {
    if (error)
      *error = "failed to load camera rig file";
    return false;
  }

  auto &root = tree.root();
  auto metadata = tsd::core::readDataTreeMetadata(root);
  if (metadata.malformed()) {
    if (error)
      *error = "malformed __tsd_metadata: " + metadata.message;
    return false;
  }
  if (!metadata.found()) {
    if (error)
      *error = "file is missing __tsd_metadata";
    return false;
  }

  const auto &m = *metadata.metadata;
  if (m.envelopeVersion != tsd::core::DATA_TREE_METADATA_ENVELOPE_VERSION) {
    if (error)
      *error = "unsupported metadata envelopeVersion";
    return false;
  }
  if (m.fileType != CAMERA_RIG_FILE_TYPE || m.schema != CAMERA_RIG_SCHEMA) {
    if (error)
      *error = "file is not a SciVis Studio camera rig";
    return false;
  }
  if (m.schemaVersion < 1 || m.schemaVersion > RIG_SCHEMA_VERSION) {
    if (error)
      *error = "unsupported camera rig schemaVersion";
    return false;
  }

  rigOut = {};
  rigOut.name = root["name"].getValueOr<std::string>("");
  if (auto *rigNode = root.child("rig"))
    nodeToCameraRig(*rigNode, rigOut);
  return true;
}

} // namespace tsd::scivis_studio::camera_rig
