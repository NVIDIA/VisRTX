// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Dataset.h"

#include "tsd/core/DataTree.hpp"
#include "tsd/rendering/view/Manipulator.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace tsd::scivis_studio {

struct Project;

enum class CameraInterpolation
{
  Hold,
  Linear,
  EaseOut,
  EaseIn,
  EaseOutIn
};

struct ManipulatorState
{
  tsd::rendering::CameraPose orbit;
};

struct CameraKeyframe
{
  int frame{0};
  std::string name;
  ManipulatorState manipulator;
  CameraInterpolation interpolationToNext{CameraInterpolation::Linear};
};

struct CameraRig
{
  CameraRigID id;
  std::string name;
  ManipulatorState current;
  std::vector<CameraKeyframe> keyframes;

  // Runtime-only name of the asset path owned by this rig.
  std::string persistedName;
};

namespace camera_rig {

// Collection lookups within a Project.
CameraRigID nextCameraRigId(const Project &project);
CameraRig *findCameraRig(Project &project, const CameraRigID &id);
const CameraRig *findCameraRig(const Project &project, const CameraRigID &id);

// Interpolation enum <-> persisted string.
const char *toString(CameraInterpolation interpolation);
CameraInterpolation interpolationFromString(const std::string &s);

// Manipulator <-> stored camera pose.
ManipulatorState manipulatorStateFromManipulator(
    const tsd::rendering::Manipulator &m);
void applyManipulatorState(
    tsd::rendering::Manipulator &m, const ManipulatorState &state);

// Keyframe animation.
void sortKeyframes(CameraRig &rig);
ManipulatorState sampleCameraRig(const CameraRig &rig, int frame);

// Standalone camera-rig file IO. The rig's runtime-only id is intentionally not
// written; only the portable name and value data (current pose + keyframes) are
// stored. importCameraRigFile fills rigOut.name and value data, leaving
// rigOut.id empty for the caller to assign.
bool exportCameraRigFile(const CameraRig &rig,
    const std::filesystem::path &file,
    std::string *error = nullptr);
bool importCameraRigFile(const std::filesystem::path &file,
    CameraRig &rigOut,
    std::string *error = nullptr);

// DataTree node <-> camera rig value data (current pose + keyframes). Exposed
// so the legacy (pre-v4) inline-manifest read path can reuse it.
void cameraRigToNode(const CameraRig &rig, tsd::core::DataNode &node);
void nodeToCameraRig(tsd::core::DataNode &node, CameraRig &rig);

} // namespace camera_rig

} // namespace tsd::scivis_studio
