// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Manipulator.hpp"
// anari
#include <anari/anari_cpp.hpp>

namespace tsd::manipulators {

void updateCameraParametersPerspective(
    anari::Device d, anari::Camera c, const Manipulator &m);

void updateCameraParametersOrthographic(
    anari::Device d, anari::Camera c, const Manipulator &m);

} // namespace tsd::manipulators
