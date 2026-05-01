// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/rendering/view/Manipulator.hpp"

#include <string>
#include <vector>

namespace tsd::scivis_studio {

enum class CameraInterpolation
{
  Hold,
  Linear
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

struct ShotCameraRig
{
  ManipulatorState current;
  std::vector<CameraKeyframe> keyframes;
};

const char *toString(CameraInterpolation interpolation);
CameraInterpolation cameraInterpolationFromString(const std::string &s);

ManipulatorState manipulatorStateFromManipulator(
    const tsd::rendering::Manipulator &m);
void applyManipulatorState(
    tsd::rendering::Manipulator &m, const ManipulatorState &state);

void sortKeyframes(ShotCameraRig &rig);
ManipulatorState sampleCameraRig(const ShotCameraRig &rig, int frame);

} // namespace tsd::scivis_studio
