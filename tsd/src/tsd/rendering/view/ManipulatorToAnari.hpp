// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Manipulator.hpp"
// anari
#include <anari/anari_cpp.hpp>

namespace tsd::rendering {

void updateCameraParametersPerspective(
    anari::Device d, anari::Camera c, const Manipulator &m);

void updateCameraParametersOrthographic(
    anari::Device d, anari::Camera c, const Manipulator &m);

} // namespace tsd::rendering
