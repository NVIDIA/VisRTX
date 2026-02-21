// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Manipulator.hpp"
// tsd_core
#include "tsd/core/scene/objects/Camera.hpp"

namespace tsd::rendering {

void updateCameraObject(tsd::core::Camera &c,
    const Manipulator &m,
    bool includeManipulatorMetadata = true);

void updateManipulatorFromCamera(Manipulator &m, const tsd::core::Camera &c);

} // namespace tsd::rendering
