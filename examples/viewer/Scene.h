/*
 * Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

// anari
#include <anari/anari_cpp/ext/glm.h>
#include <anari/anari_cpp.hpp>
// std
#include <array>
#include <memory>
#include <variant>

using box3 = std::array<glm::vec3, 2>;
namespace anari {
ANARI_TYPEFOR_SPECIALIZATION(box3, ANARI_FLOAT32_BOX3);
ANARI_TYPEFOR_DEFINITION(box3);
} // namespace anari

inline box3 makeEmptyBounds()
{
  constexpr glm::vec3 vinf(std::numeric_limits<float>::infinity());
  return {glm::vec3(vinf), glm::vec3(-vinf)};
}

enum SceneTypes
{
  RANDOM_SPHERES,
  RANDOM_CYLINDERS,
  RANDOM_CONES,
  NOISE_VOLUME,
  GRAVITY_VOLUME,
  OBJ_FILE
};

struct Config
{
  bool addPlane{true};
};

struct SpheresConfig : public Config
{
  int numSpheres{10000};
  float radius{.025f};
  float opacity{1.f};
};

struct CylindersConfig : public Config
{
  int numCylinders{10};
  float radius{0.025f};
  float opacity{1.f};
  bool useRandomSeed{false};
  bool caps{true};
};

struct ConesConfig : public Config
{
  int numCones{10};
  float opacity{1.f};
  bool useRandomSeed{false};
  bool caps{true};
};

struct VolumeConfig : public Config
{
  int size{64};
  float density{0.1f};
};

struct NoiseVolumeConfig : public VolumeConfig
{
  bool instanceVolume{false};
};

struct GravityVolumeConfig : public VolumeConfig
{
  int size{64};
  int numWells{10};
  bool withVolume{true};
  bool withBodies{true};
};

struct ObjFileConfig : public Config
{
  std::string filename;
};

using SceneConfig = std::variant<SpheresConfig,
    CylindersConfig,
    ConesConfig,
    NoiseVolumeConfig,
    GravityVolumeConfig,
    ObjFileConfig>;

using UICallback = std::function<void()>;
using CleanupCallback = std::function<void()>;

struct Scene
{
  Scene(anari::Device d,
      anari::World w,
      UICallback cb = {},
      CleanupCallback ccb = {});
  ~Scene();

  anari::World world() const;

  void buildUI();

 private:
  anari::Device m_device{nullptr};
  anari::World m_world{nullptr};
  UICallback m_ui;
  CleanupCallback m_cleanup;
};

using ScenePtr = std::unique_ptr<Scene>;

ScenePtr generateScene(anari::Device d, SceneConfig config = {});
