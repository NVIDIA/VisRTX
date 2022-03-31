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

#include "Viewer.h"
// std
#include <cstring>
// stb_image
#include "stb_image_write.h"
// CUDA
#include <cuda_gl_interop.h>

#include "ui_scenes.h"

struct RendererParameter
{
  std::string name;
};

struct RendererState
{
  std::string name{"default"};
  std::vector<RendererParameter> parameters;
};

/* list of renderer options for menu */
static std::vector<RendererState> g_renderers;
static int g_whichRenderer = 0;
extern bool g_verboseOutput;
extern bool g_glInterop;

static RendererState makeRendererState(
    anari::Library library, const char *deviceName, const char *name)
{
  RendererState retval;
  retval.name = name;

  const auto *params =
      anariGetObjectParameters(library, deviceName, name, ANARI_RENDERER);

  if (params == nullptr)
    printf("failed to find parameters for '%s' renderer\n", name);
  else {
    for (int i = 0; params[i].name != nullptr; i++) {
      RendererParameter p;
      p.name = params[i].name;
      retval.parameters.push_back(p);
    }
  }

  return retval;
}

static bool rendererUI_callback(void *, int index, const char **out_text)
{
  *out_text = g_renderers[index].name.c_str();
  return true;
}

static void statusFunc(void *userData,
    ANARIDevice device,
    ANARIObject source,
    ANARIDataType sourceType,
    ANARIStatusSeverity severity,
    ANARIStatusCode code,
    const char *message)
{
  (void)userData;
  if (severity == ANARI_SEVERITY_FATAL_ERROR) {
    fprintf(stderr, "[FATAL][%p] %s\n", source, message);
    std::exit(1);
  } else if (severity == ANARI_SEVERITY_ERROR) {
    fprintf(stderr, "[ERROR][%p] %s\n", source, message);
  } else if (g_verboseOutput) {
    if (severity == ANARI_SEVERITY_WARNING) {
      fprintf(stderr, "[WARN ][%p] %s\n", source, message);
    } else if (severity == ANARI_SEVERITY_PERFORMANCE_WARNING) {
      fprintf(stderr, "[PERF ][%p] %s\n", source, message);
    } else if (severity == ANARI_SEVERITY_INFO) {
      fprintf(stderr, "[INFO ][%p] %s\n", source, message);
    } else if (severity == ANARI_SEVERITY_DEBUG) {
      fprintf(stderr, "[DEBUG][%p] %s\n", source, message);
    }
  }
}

Viewer::Viewer(const char *libName, const char *objFileName)
    : m_libraryName(libName)
{
  m_objFileConfig.filename = objFileName;
  if (!m_objFileConfig.filename.empty())
    m_selectedScene = SceneTypes::OBJ_FILE;
}

///////////////////////////////////////////////////////////////////////////////
// match3D overrides //////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void Viewer::setup()
{
  ImGuiIO &io = ImGui::GetIO();
  io.FontGlobalScale = 1.25f;

  m_library = anari::loadLibrary(m_libraryName.c_str(), statusFunc, nullptr);
  m_device = anari::newDevice(m_library, "default");

  if (!m_device)
    std::exit(1);

  m_haveCUDAInterop = g_glInterop
      && anari::deviceImplements(m_device, "VISRTX_CUDA_OUTPUT_BUFFERS");

  // GL //

  glGenTextures(1, &m_framebufferTexture);
  glBindTexture(GL_TEXTURE_2D, m_framebufferTexture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D,
      0,
      GL_RGBA8,
      m_windowSize.x,
      m_windowSize.y,
      0,
      GL_RGBA,
      GL_UNSIGNED_BYTE,
      0);

  glGenFramebuffers(1, &m_framebufferObject);
  glBindFramebuffer(GL_FRAMEBUFFER, m_framebufferObject);
  glFramebufferTexture2D(GL_FRAMEBUFFER,
      GL_COLOR_ATTACHMENT0,
      GL_TEXTURE_2D,
      m_framebufferTexture,
      0);
  glReadBuffer(GL_COLOR_ATTACHMENT0);

  // ANARI //

  const char **r_subtypes =
      anariGetObjectSubtypes(m_library, "default", ANARI_RENDERER);

  if (r_subtypes != nullptr) {
    for (int i = 0; r_subtypes[i] != nullptr; i++) {
      g_renderers.push_back(
          makeRendererState(m_library, "default", r_subtypes[i]));
    }
  } else
    g_renderers.emplace_back(); // adds 'default' renderer with no parameters

  anari::commit(m_device, m_device);

  m_frame = anari::newObject<anari::Frame>(m_device);
  m_perspCamera = anari::newObject<anari::Camera>(m_device, "perspective");
  m_orthoCamera = anari::newObject<anari::Camera>(m_device, "orthographic");

  for (auto &rstate : g_renderers) {
    m_renderers.push_back(
        anari::newObject<anari::Renderer>(m_device, rstate.name.c_str()));
  }

  m_currentRenderer = m_renderers[0];

  m_lights[0] = anari::newObject<anari::Light>(m_device, "ambient");
  m_lights[1] = anari::newObject<anari::Light>(m_device, "directional");

  updateLights();

  m_lightsArray = anari::newArray1D(m_device, m_lights.data(), m_lights.size());

  updateWorld();
  updateFrame();

  resetCameraAZEL();
  resetView();

  anari::render(m_device, m_frame);
}

void Viewer::buildUI()
{
  bool resized = getWindowSize(m_windowSize.x, m_windowSize.y);
  if (resized) {
    updateCamera();
    updateFrame();

    if (m_haveCUDAInterop) {
      if (m_graphicsResource)
        cudaGraphicsUnregisterResource(m_graphicsResource);
    }

    glViewport(0, 0, m_windowSize.x, m_windowSize.y);

    glBindTexture(GL_TEXTURE_2D, m_framebufferTexture);
    glTexImage2D(GL_TEXTURE_2D,
        0,
        GL_RGBA8,
        m_windowSize.x,
        m_windowSize.y,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        0);

    if (m_haveCUDAInterop) {
      cudaGraphicsGLRegisterImage(&m_graphicsResource,
          m_framebufferTexture,
          GL_TEXTURE_2D,
          cudaGraphicsRegisterFlagsWriteDiscard);
    }
  }

  ui_handleInput();
  ui_updateImage();
  ui_makeWindow();
}

void Viewer::drawBackground()
{
  glBindFramebuffer(GL_READ_FRAMEBUFFER, m_framebufferObject);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

  glClear(GL_COLOR_BUFFER_BIT);
  glBlitFramebuffer(0,
      0,
      m_windowSizeScaled.x,
      m_windowSizeScaled.y,
      0,
      0,
      m_windowSize.x,
      m_windowSize.y,
      GL_COLOR_BUFFER_BIT,
      GL_NEAREST);
}

void Viewer::teardown()
{
  anari::wait(m_device, m_frame);

  anari::release(m_device, m_perspCamera);
  anari::release(m_device, m_orthoCamera);
  for (auto &r : m_renderers)
    anari::release(m_device, r);
  for (auto &l : m_lights)
    anari::release(m_device, l);
  anari::release(m_device, m_lightsArray);
  anari::release(m_device, m_frame);
  anari::release(m_device, m_device);

  anari::unloadLibrary(m_library);

  if (m_graphicsResource)
    cudaGraphicsUnregisterResource(m_graphicsResource);
}

///////////////////////////////////////////////////////////////////////////////
// Internal implementation ////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void Viewer::updateFrame()
{
  m_windowSizeScaled = glm::vec2(m_windowSize) * m_resolutionScale;
  anari::setParameter(
      m_device, m_frame, "size", glm::uvec2(m_windowSizeScaled));
  anari::setParameter(m_device, m_frame, "color", m_format);

  anari::setParameter(m_device, m_frame, "channelColor", true);
  anari::setParameter(m_device, m_frame, "channelAccum", true);
  anari::setParameter(m_device, m_frame, "channelDepth", true);

  anari::setParameter(m_device, m_frame, "world", m_currentScene->world());
  if (m_useOrthoCamera)
    anari::setParameter(m_device, m_frame, "camera", m_orthoCamera);
  else
    anari::setParameter(m_device, m_frame, "camera", m_perspCamera);
  anari::setParameter(m_device, m_frame, "renderer", m_currentRenderer);

  anari::setParameter(
      m_device, m_currentRenderer, "backgroundColor", m_background);
  anari::setParameter(
      m_device, m_currentRenderer, "pixelSamples", m_pixelSamples);
  anari::setParameter(
      m_device, m_currentRenderer, "ambientLight", m_ambientIntensity);

  anari::commit(m_device, m_currentRenderer);
  anari::commit(m_device, m_frame);
}

void Viewer::updateCamera()
{
  anari::setParameter(m_device, m_perspCamera, "position", m_arcball.eye());
  anari::setParameter(m_device, m_perspCamera, "direction", m_arcball.dir());
  anari::setParameter(m_device, m_perspCamera, "up", m_arcball.up());

  anari::setParameter(
      m_device, m_orthoCamera, "position", m_arcball.eye_FixedDistance());
  anari::setParameter(m_device, m_orthoCamera, "direction", m_arcball.dir());
  anari::setParameter(m_device, m_orthoCamera, "up", m_arcball.up());
  anari::setParameter(m_device, m_orthoCamera, "height", m_arcball.distance());

  anari::setParameter(m_device,
      m_perspCamera,
      "aspect",
      m_windowSize.x / float(m_windowSize.y));
  anari::setParameter(m_device,
      m_orthoCamera,
      "aspect",
      m_windowSize.x / float(m_windowSize.y));

  anari::commit(m_device, m_perspCamera);
  anari::commit(m_device, m_orthoCamera);
}

void Viewer::updateWorld()
{
  switch (m_selectedScene) {
  case SceneTypes::OBJ_FILE:
    m_currentScene = generateScene(m_device, m_objFileConfig);
    break;
  case SceneTypes::RANDOM_SPHERES:
    m_currentScene = generateScene(m_device, m_spheresConfig);
    break;
  case SceneTypes::RANDOM_CYLINDERS:
    m_currentScene = generateScene(m_device, m_cylindersConfig);
    break;
  case SceneTypes::RANDOM_CONES:
    m_currentScene = generateScene(m_device, m_conesConfig);
    break;
  case SceneTypes::NOISE_VOLUME:
    m_currentScene = generateScene(m_device, m_noiseVolumeConfig);
    break;
  case SceneTypes::GRAVITY_VOLUME:
  default:
    m_currentScene = generateScene(m_device, m_gravityVolumeConfig);
    break;
  }

  auto world = m_currentScene->world();

  anari::setParameter(m_device, world, "light", m_lightsArray);
  anari::commit(m_device, world);

  anari::setParameter(m_device, m_frame, "world", world);
  anari::commit(m_device, m_frame);

  if (m_selectedScene != m_lastSceneType)
    resetView();
  m_lastSceneType = m_selectedScene;
}

void Viewer::resetView()
{
  box3 bounds;

  auto world = m_currentScene->world();
  anari::getProperty(m_device, world, "bounds", bounds, ANARI_WAIT);

  printf("resetting view using bounds {%f, %f, %f} x {%f, %f, %f}\n",
      bounds[0].x,
      bounds[0].y,
      bounds[0].z,
      bounds[1].x,
      bounds[1].y,
      bounds[1].z);

  auto center = 0.5f * (bounds[0] + bounds[1]);
  auto diag = bounds[1] - bounds[0];

  m_arcball = Orbit(center, 0.25f * glm::length(diag), m_arcball.azel());
  updateCamera();
}

void Viewer::resetCameraAZEL()
{
  m_arcball.setAzel(glm::vec2(0.f, -20.f));
}

void Viewer::updateLights()
{
  auto l = m_lights[0];

  anari::setParameter(
      m_device, l, "intensity", m_lightConfigs.ambientIntensity);
  anari::setParameter(m_device, l, "color", m_lightConfigs.ambientColor);
  anari::setParameter(m_device,
      l,
      "occlusionDistance",
      m_lightConfigs.ambientOcclusionDistance);
  anari::commit(m_device, l);

  l = m_lights[1];

  const float az = glm::radians(m_lightConfigs.directionalAzimuth);
  const float el = glm::radians(m_lightConfigs.directionalElevation);
  glm::vec3 dir;
  dir.x = std::sin(az) * std::cos(el);
  dir.y = std::sin(el);
  dir.z = std::cos(az) * std::cos(el);

  anari::setParameter(m_device, l, "direction", dir);
  anari::setParameter(
      m_device, l, "irradiance", m_lightConfigs.directionalIrradiance);
  anari::setParameter(m_device, l, "color", m_lightConfigs.directionalColor);
  anari::commit(m_device, l);
}

void Viewer::ui_handleInput()
{
  ImGuiIO &io = ImGui::GetIO();

  const bool leftDown = ImGui::IsMouseDown(ImGuiMouseButton_Left);
  const bool rightDown = ImGui::IsMouseDown(ImGuiMouseButton_Right);
  const bool middleDown = ImGui::IsMouseDown(ImGuiMouseButton_Middle);

  const bool anyDown = leftDown || rightDown || middleDown;

  if (!anyDown) {
    m_manipulating = false;
    m_previousMouse = glm::vec2(-1);
  } else if (!io.WantCaptureMouse)
    m_manipulating = true;

  if (m_mouseRotating && !leftDown)
    m_mouseRotating = false;

  if (!m_manipulating)
    return;

  glm::vec2 position;
  std::memcpy(&position, &io.MousePos, sizeof(position));

  const glm::vec2 mouse(position.x, position.y);

  if (anyDown && m_previousMouse != glm::vec2(-1)) {
    const glm::vec2 prev = m_previousMouse;

    const glm::vec2 mouseFrom = prev * 2.f / glm::vec2(m_windowSize);
    const glm::vec2 mouseTo = mouse * 2.f / glm::vec2(m_windowSize);

    const glm::vec2 mouseDelta = mouseFrom - mouseTo;

    if (mouseDelta != glm::vec2(0.f)) {
      if (leftDown) {
        if (!m_mouseRotating) {
          m_arcball.startNewRotation();
          m_mouseRotating = true;
        }

        m_arcball.rotate(mouseDelta);
      } else if (rightDown)
        m_arcball.zoom(mouseDelta.y);
      else if (middleDown)
        m_arcball.pan(mouseDelta);

      updateCamera();
    }
  }

  m_previousMouse = mouse;
}

void Viewer::ui_updateImage()
{
  if (anari::isReady(m_device, m_frame)) {
    float duration = 0.f;
    anari::getProperty(m_device, m_frame, "duration", duration);

    m_latestFL = duration * 1000;
    m_minFL = std::min(m_minFL, m_latestFL);
    m_maxFL = std::max(m_maxFL, m_latestFL);

    if (m_haveCUDAInterop && !m_saveNextFrame && !m_showDepth) {
      const void *fb = anari::map(m_device, m_frame, "colorGPU");
      cudaGraphicsMapResources(1, &m_graphicsResource);
      cudaArray_t array;
      cudaGraphicsSubResourceGetMappedArray(&array, m_graphicsResource, 0, 0);
      cudaMemcpy2DToArray(array,
          0,
          0,
          fb,
          m_renderSize.x * 4,
          m_renderSize.x * 4,
          m_renderSize.y,
          cudaMemcpyDeviceToDevice);
      cudaGraphicsUnmapResources(1, &m_graphicsResource);
      anari::unmap(m_device, m_frame, "colorGPU");
    } else {
      const void *fb =
          anari::map(m_device, m_frame, m_showDepth ? "depth" : "color");

      glBindTexture(GL_TEXTURE_2D, m_framebufferTexture);
      glTexSubImage2D(GL_TEXTURE_2D,
          0,
          0,
          0,
          m_renderSize.x,
          m_renderSize.y,
          m_showDepth ? GL_RED : GL_RGBA,
          m_showDepth ? GL_FLOAT : GL_UNSIGNED_BYTE,
          fb);

      if (m_saveNextFrame) {
        stbi_flip_vertically_on_write(1);
        stbi_write_png("screenshot.png",
            m_renderSize.x,
            m_renderSize.y,
            4,
            fb,
            4 * m_renderSize.x);
        printf("frame saved to 'screenshot.png'\n");
        m_saveNextFrame = false;
      }

      anari::unmap(m_device, m_frame, "color");
    }

    m_renderSize = m_windowSizeScaled;
    anari::render(m_device, m_frame);
  }
}

void Viewer::ui_makeWindow()
{
  ImGuiWindowFlags windowFlags = ImGuiWindowFlags_AlwaysAutoResize
      | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing;

  ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);

  ImGui::Begin("Controls", nullptr, windowFlags);

  if (ImGui::CollapsingHeader("Stats", ImGuiTreeNodeFlags_DefaultOpen))
    ui_makeWindow_stats();

  if (ImGui::CollapsingHeader("Frame"))
    ui_makeWindow_frame();

  if (ImGui::CollapsingHeader("Scene")) {
    ImGui::TextColored(
        ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Scene Generation Parameters:");
    if (ui_scenes(m_spheresConfig,
            m_cylindersConfig,
            m_conesConfig,
            m_noiseVolumeConfig,
            m_gravityVolumeConfig,
            m_objFileConfig,
            m_selectedScene)) {
      updateWorld();
    }
    ImGui::Separator();
    ImGui::TextColored(
        ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Interactive Parameters:");
    m_currentScene->buildUI();
  }

  if (ImGui::CollapsingHeader("Camera"))
    ui_makeWindow_camera();

  if (ImGui::CollapsingHeader("Renderer"))
    ui_makeWindow_renderer();

  if (ImGui::CollapsingHeader("Lights"))
    ui_makeWindow_lights();

  ImGui::End();
}

void Viewer::ui_makeWindow_stats()
{
  ImGui::Text("window: %i x %i", m_windowSize.x, m_windowSize.y);
  int samples = 0;
  anari::getProperty(m_device, m_frame, "numSamples", samples, ANARI_NO_WAIT);
  ImGui::Text("   samples: %i", samples);
  ImGui::Text("   latency: %.2fms", m_latestFL);
  ImGui::Text("     (min): %.2fms", m_minFL);
  ImGui::Text("     (max): %.2fms", m_maxFL);
  ImGuiIO &io = ImGui::GetIO();
  ImGui::Text("        UI: %.2fms", io.DeltaTime * 1000.f);
  if (ImGui::Button("reset min/max")) {
    m_minFL = std::numeric_limits<float>::max();
    m_maxFL = -std::numeric_limits<float>::max();
  }
}

void Viewer::ui_makeWindow_frame()
{
  float scale = m_resolutionScale;
  if (ImGui::RadioButton("100%", m_resolutionScale == 1.f))
    m_resolutionScale = 1.f;
  ImGui::SameLine();
  if (ImGui::RadioButton("75%", m_resolutionScale == 0.75f))
    m_resolutionScale = 0.75f;
  ImGui::SameLine();
  if (ImGui::RadioButton("50%", m_resolutionScale == 0.5f))
    m_resolutionScale = 0.5f;
  ImGui::SameLine();
  if (ImGui::RadioButton("25%", m_resolutionScale == 0.25f))
    m_resolutionScale = 0.25f;

  if (scale != m_resolutionScale)
    updateFrame();

  if (ImGui::Checkbox("denoise", &m_denoise)) {
    anari::setParameter(m_device, m_frame, "denoise", m_denoise);
    anari::commit(m_device, m_frame);
  }

  static bool checkerboard = false;
  if (ImGui::Checkbox("checkerboard", &checkerboard)) {
    anari::setParameter(m_device, m_frame, "checkerboard", checkerboard);
    anari::commit(m_device, m_frame);
  }

  ImGui::Checkbox("show depth", &m_showDepth);

  if (ImGui::Button("take screenshot"))
    m_saveNextFrame = true;
}

void Viewer::ui_makeWindow_camera()
{
  if (ImGui::Checkbox("orthographic", &m_useOrthoCamera))
    updateFrame();

  ImGui::Separator();

  bool regionChanged = false;

  regionChanged |= ImGui::DragFloat4("image region", &m_imageRegion.x, 0.001f);

  if (ImGui::Button("reset region")) {
    regionChanged = true;
    m_imageRegion = glm::vec4(0.f, 0.f, 1.f, 1.f);
  }

  if (regionChanged) {
    anari::setParameter(m_device,
        m_perspCamera,
        "imageRegion",
        ANARI_FLOAT32_BOX2,
        &m_imageRegion);
    anari::setParameter(m_device,
        m_orthoCamera,
        "imageRegion",
        ANARI_FLOAT32_BOX2,
        &m_imageRegion);
    anari::commit(m_device, m_perspCamera);
    anari::commit(m_device, m_orthoCamera);
  }

  ImGui::Separator();

  if (ImGui::Button("reset view")) {
    resetCameraAZEL();
    resetView();
  }
}

void Viewer::ui_makeWindow_renderer()
{
  if (ImGui::Combo("renderer##whichRenderer",
          &g_whichRenderer,
          rendererUI_callback,
          nullptr,
          g_renderers.size())) {
    m_currentRenderer = m_renderers[g_whichRenderer];
    updateFrame();
  }

  if (ImGui::ColorEdit3("background", &m_background.x)) {
    anari::setParameter(
        m_device, m_currentRenderer, "backgroundColor", m_background);
    anari::commit(m_device, m_currentRenderer);
  }

  if (ImGui::SliderInt("pixelSamples", &m_pixelSamples, 1, 256)) {
    anari::setParameter(
        m_device, m_currentRenderer, "pixelSamples", m_pixelSamples);
    anari::commit(m_device, m_currentRenderer);
  }

  ImGui::Text("  parameters:");
  auto &rParams = g_renderers[g_whichRenderer];
  for (auto &rp : rParams.parameters)
    ImGui::Text("    %s", rp.name.c_str());
}

void Viewer::ui_makeWindow_lights()
{
  bool update = false;

  ImGui::Text("ambient:");

  update |= ImGui::DragFloat("intensity##ambient",
      &m_lightConfigs.ambientIntensity,
      0.001f,
      0.f,
      1000.f);

  update |= ImGui::ColorEdit3("color##ambient", &m_lightConfigs.ambientColor.x);

  update |= ImGui::DragFloat("occlusion distance##ambient",
      &m_lightConfigs.ambientOcclusionDistance,
      0.001f,
      0.f,
      100000.f);

  ImGui::Text("directional:");

  update |= ImGui::DragFloat("irradiance##directional",
      &m_lightConfigs.directionalIrradiance,
      0.001f,
      0.f,
      1000.f);

  update |= ImGui::ColorEdit3(
      "color##directional", &m_lightConfigs.directionalColor.x);

  auto maintainUnitCircle = [](float inDegrees) -> float {
    while (inDegrees > 360.f)
      inDegrees -= 360.f;
    while (inDegrees < 0.f)
      inDegrees += 360.f;
    return inDegrees;
  };

  if (ImGui::DragFloat("azimuth", &m_lightConfigs.directionalAzimuth, 0.01f)) {
    update = true;
    m_lightConfigs.directionalAzimuth =
        maintainUnitCircle(m_lightConfigs.directionalAzimuth);
  }

  if (ImGui::DragFloat(
          "elevation", &m_lightConfigs.directionalElevation, 0.01f)) {
    update = true;
    m_lightConfigs.directionalElevation =
        maintainUnitCircle(m_lightConfigs.directionalElevation);
  }

  if (update)
    updateLights();
}
