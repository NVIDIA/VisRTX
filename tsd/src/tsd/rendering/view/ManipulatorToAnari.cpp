// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ManipulatorToAnari.hpp"

namespace tsd::rendering {

void updateCameraParametersPerspective(
    anari::Device d, anari::Camera c, const Manipulator &m)
{
  anari::setParameter(d, c, "position", m.eye());
  anari::setParameter(d, c, "direction", m.dir());
  anari::setParameter(d, c, "up", m.up());
}

void updateCameraParametersOrthographic(
    anari::Device d, anari::Camera c, const Manipulator &m)
{
  anari::setParameter(d, c, "position", m.eye_FixedDistance());
  anari::setParameter(d, c, "direction", m.dir());
  anari::setParameter(d, c, "up", m.up());
  anari::setParameter(d, c, "height", m.distance() * 0.75f);
}

} // namespace tsd::rendering
