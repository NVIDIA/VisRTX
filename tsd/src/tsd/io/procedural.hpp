// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/scene/Scene.hpp"

namespace tsd::io {

using namespace tsd::core;

// clang-format off

void generate_cylinders(Scene &scene, LayerNodeRef location = {}, bool useDefaultMaterial = false);
void generate_default_lights(Scene &scene);
void generate_hdri_dome(Scene &scene, LayerNodeRef location = {});
void generate_hdri_test_image(Scene &scene, LayerNodeRef location = {});
void generate_material_orb(Scene &scene, LayerNodeRef location = {});
void generate_monkey(Scene &scene, LayerNodeRef location = {});
VolumeRef generate_noiseVolume(Scene &scene, LayerNodeRef location = {}, ArrayRef colors = {}, ArrayRef opacities = {});
void generate_randomSpheres(Scene &scene, LayerNodeRef location = {}, bool useDefaultMaterial = false);
void generate_rtow(Scene &scene, LayerNodeRef location = {});
void generate_sphereSetVolume(Scene &scene, LayerNodeRef location = {});

// clang-format on

} // namespace tsd::io
