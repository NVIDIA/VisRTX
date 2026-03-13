// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/scene/Scene.hpp"
#include "tsd/scene/objects/SpatialField.hpp"
#include "tsd/scene/objects/Volume.hpp"

namespace tsd::io {

using namespace tsd::scene;

struct VorticityOptions
{
  bool lambda2{true};
  bool qCriterion{true};
  bool vorticity{true};
  bool helicity{false};
};

struct VorticityResult
{
  VolumeRef lambda2;
  VolumeRef qCriterion;
  VolumeRef vorticity;
  VolumeRef helicity;
};

VorticityResult computeVorticity(Scene &scene,
    const SpatialField *u,
    const SpatialField *v,
    const SpatialField *w,
    LayerNodeRef location = {},
    VorticityOptions opts = {});

} // namespace tsd::io
