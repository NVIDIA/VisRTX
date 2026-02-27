// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/scene/Scene.hpp"
#include "tsd/core/scene/objects/SpatialField.hpp"
#include "tsd/core/scene/objects/Volume.hpp"

namespace tsd::io {

using namespace tsd::core;

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
    SpatialField *u,
    SpatialField *v,
    SpatialField *w,
    LayerNodeRef location = {},
    VorticityOptions opts = {});

} // namespace tsd::io
