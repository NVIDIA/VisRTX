/*
 * Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "ui_scenes.h"
// match3D
#include <match3D/match3D.h>

static void ui_config(Config &config)
{
  ImGui::Checkbox("addPlane", &config.addPlane);
}

static void ui_objFileConfig(ObjFileConfig &config)
{
  ui_config(config);
  ImGui::Text("filename: %s", config.filename.c_str());
}

static void ui_spheresConfig(SpheresConfig &config)
{
  ui_config(config);
  ImGui::DragInt("particles", &config.numSpheres, 1.f, 1, 1e5);
  ImGui::InputFloat("radius", &config.radius);
  ImGui::DragFloat("opacity", &config.opacity, 0.001f, 0.f, 1.f);
}

static void ui_cylindersConfig(CylindersConfig &config)
{
  ui_config(config);
  ImGui::DragInt("numCylinders", &config.numCylinders, 1.f, 1, 100);
  ImGui::InputFloat("position range", &config.positionRange);
  ImGui::InputFloat("radius", &config.radius);
  ImGui::Checkbox("caps", &config.caps);
  ImGui::Checkbox("randomize seed", &config.useRandomSeed);
  ImGui::DragFloat("opacity", &config.opacity, 0.001f, 0.f, 1.f);
}

static void ui_conesConfig(ConesConfig &config)
{
  ui_config(config);
  ImGui::DragInt("numCones", &config.numCones, 1.f, 1, 100);
  ImGui::InputFloat("position range", &config.positionRange);
  ImGui::InputFloat("arrow radius", &config.arrowRadius);
  ImGui::Checkbox("caps", &config.caps);
  ImGui::Checkbox("randomize seed", &config.useRandomSeed);
  ImGui::DragFloat("opacity", &config.opacity, 0.001f, 0.f, 1.f);
}

static void ui_curvesConfig(CurvesConfig &config)
{
  ui_config(config);
}

static void ui_volumeConfig(VolumeConfig &config)
{
  ui_config(config);
  ImGui::DragInt("size", &config.size, 1.f, 10, 256);
  ImGui::DragFloat("density", &config.density, 0.001f, 0.f, 1.f);
}

static void ui_noiseVolumeConfig(NoiseVolumeConfig &config)
{
  ui_volumeConfig(config);
  ImGui::Checkbox("instance volume", &config.instanceVolume);
  if (config.instanceVolume)
    config.addPlane = false;
}

static void ui_gravityVolumeConfig(GravityVolumeConfig &config)
{
  ui_volumeConfig(config);
  ImGui::DragInt("wells", &config.numWells, 1.f, 1, 25);
  ImGui::Checkbox("show volume", &config.withVolume);
  ImGui::Checkbox("show bodies", &config.withBodies);
}

bool ui_scenes(SpheresConfig &spheresConfig,
    CylindersConfig &cylindersConfig,
    ConesConfig &conesConfig,
    CurvesConfig &curvesConfig,
    NoiseVolumeConfig &noiseVolumeConfig,
    GravityVolumeConfig &gravityVolumeConfig,
    ObjFileConfig &objFileConfig,
    int &whichScene)
{
  ImGui::Text("Scene:");

  ImGui::SameLine();

  auto prevScene = whichScene;
  if (!objFileConfig.filename.empty()) {
    ImGui::Combo("##whichScene",
        &whichScene,
        "random spheres\0random cylinders\0random cones\0streamlines\0noise volume\0gravity volume\0obj file\0\0");
  } else {
    ImGui::Combo("##whichScene",
        &whichScene,
        "random spheres\0random cylinders\0random cones\0streamlines\0noise volume\0gravity volume\0\0");
  }

  switch (whichScene) {
  case SceneTypes::OBJ_FILE:
    ui_objFileConfig(objFileConfig);
    break;
  case SceneTypes::RANDOM_SPHERES:
    ui_spheresConfig(spheresConfig);
    break;
  case SceneTypes::RANDOM_CYLINDERS:
    ui_cylindersConfig(cylindersConfig);
    break;
  case SceneTypes::RANDOM_CONES:
    ui_conesConfig(conesConfig);
    break;
  case SceneTypes::STREAMLINES:
    ui_curvesConfig(curvesConfig);
    break;
  case SceneTypes::NOISE_VOLUME:
    ui_noiseVolumeConfig(noiseVolumeConfig);
    break;
  case SceneTypes::GRAVITY_VOLUME:
    ui_gravityVolumeConfig(gravityVolumeConfig);
    break;
  default:
    break;
  }

  ImGui::NewLine();

  return ImGui::Button("refresh") || (prevScene != whichScene);
}