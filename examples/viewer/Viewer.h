/*
 * Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// match3D
#include <match3D/match3D.h>
// anari
#include <anari/anari_cpp.hpp>
// std
#include <array>
#include <cstdio>
#include <string>
#include <vector>
// CUDA
#include <cuda_runtime_api.h>

#include "Orbit.h"
#include "Scene.h"

class Viewer : public match3D::Application
{
 public:
  Viewer(const char *libName, const char *objFileName);
  ~Viewer() override = default;

  // match3D overrides //

  void setup() override;
  void buildUI() override;
  void drawBackground() override;
  void teardown() override;

 private:
  // Internal implementation //

  void updateFrame();
  void updateCamera();
  void updateWorld();
  void updateLights();

  void resetView();
  void resetCameraAZEL();

  void ui_handleInput();
  void ui_updateImage();
  void ui_makeWindow();
  void ui_makeWindow_stats();
  void ui_makeWindow_frame();
  void ui_makeWindow_camera();
  void ui_makeWindow_renderer();
  void ui_makeWindow_lights();

  // Input state //

  glm::vec2 m_previousMouse{-1.f, -1.f};
  bool m_mouseRotating{false};
  bool m_manipulating{false};

  // Scene generation state //

  SpheresConfig m_spheresConfig;
  CylindersConfig m_cylindersConfig;
  ConesConfig m_conesConfig;
  CurvesConfig m_curvesConfig;
  NoiseVolumeConfig m_noiseVolumeConfig;
  GravityVolumeConfig m_gravityVolumeConfig;
  ObjFileConfig m_objFileConfig;
  int m_selectedScene{0};
  int m_lastSceneType{0};
  glm::vec4 m_backgroundTop{0.8f, 0.8f, 0.8f, 1.f};
  glm::vec4 m_backgroundBottom{0.1f, 0.1f, 0.1f, 1.f};
  bool m_backgroundGradient{false};
  int m_pixelSamples{1};
  bool m_checkerboard{false};

  std::unique_ptr<Scene> m_currentScene;

  // ANARI //

  std::string m_libraryName;

  anari::DataType m_format{ANARI_UFIXED8_RGBA_SRGB};

  anari::Library m_library{nullptr};
  anari::Device m_device{nullptr};
  anari::Frame m_frame{nullptr};
  anari::World m_world{nullptr};

  anari::Camera m_perspCamera{nullptr};
  anari::Camera m_orthoCamera{nullptr};

  glm::vec4 m_imageRegion{glm::vec2(0.f), glm::vec2(1.f)};

  std::vector<anari::Renderer> m_renderers;
  float m_ambientIntensity{0.f};
  glm::vec3 m_ambientColor{1.f};
  float m_ambientOcclusionDistance{100.f};
  std::array<anari::Light, 1> m_lights;
  anari::Array1D m_lightsArray{nullptr};

  bool m_useOrthoCamera{false};

  struct
  {
    float directionalIrradiance{1.f};
    float directionalAzimuth{320.f};
    float directionalElevation{320.f};
    glm::vec3 directionalColor{1.f};
  } m_lightConfigs;

  anari::Renderer m_currentRenderer{nullptr};

  bool m_denoise{false};

  Orbit m_arcball;

  // OpenGL + display

  float m_latestFL{1.f};
  float m_minFL{std::numeric_limits<float>::max()};
  float m_maxFL{-std::numeric_limits<float>::max()};
  float m_resolutionScale{1.f};

  bool m_saveNextFrame{false};
  bool m_showDepth{false};

  bool m_haveCUDAInterop{false};
  cudaGraphicsResource_t m_graphicsResource{nullptr};

  GLuint m_framebufferTexture{0};
  GLuint m_framebufferObject{0};
  glm::ivec2 m_windowSize{1920, 1080};
  glm::ivec2 m_windowSizeScaled{1920, 1080};
};