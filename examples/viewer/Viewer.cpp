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

#include "Viewer.h"
// std
#include <cstring>
#include <glm/geometric.hpp>
// stb_image
#include "Scene.h"
#include "stb_image_write.h"
// CUDA
#include <cuda_gl_interop.h>
// anari
#define ANARI_EXTENSION_UTILITY_IMPL
#include "anari/frontend/anari_extension_utility.h"
// VisRTX
#include "anari/ext/visrtx/visrtx.h"
// glm
#include <glm/ext.hpp>

#include "ui_scenes.h"

#include <filesystem>

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

#if 0
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
#endif

  return retval;
}

static bool rendererUI_callback(void *, int index, const char **out_text)
{
  *out_text = g_renderers[index].name.c_str();
  return true;
}

static void statusFunc(const void * /*userData*/,
    ANARIDevice /*device*/,
    ANARIObject source,
    ANARIDataType /*sourceType*/,
    ANARIStatusSeverity severity,
    ANARIStatusCode /*code*/,
    const char *message)
{
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

  m_library = anari::loadLibrary(m_libraryName.c_str(), statusFunc, nullptr);
  m_device = anari::newDevice(m_library, "default");

  if (!m_device)
    std::exit(1);

  visrtx::Extensions extensions =
      visrtx::getInstanceExtensions(m_device, m_device);
  m_haveCUDAInterop = g_glInterop && extensions.VISRTX_CUDA_OUTPUT_BUFFERS;

  if (extensions.VISRTX_MATERIAL_MDL) {
    auto path = std::filesystem::current_path() / "shaders";
    anari::setParameter(m_device, m_device, "mdlSearchPaths", path.string());
  }
  // ANARI //

  const char **r_subtypes = anariGetObjectSubtypes(m_device, ANARI_RENDERER);

  if (r_subtypes != nullptr) {
    for (int i = 0; r_subtypes[i] != nullptr; i++) {
      g_renderers.push_back(
          makeRendererState(m_library, "default", r_subtypes[i]));
    }
  } else
    g_renderers.emplace_back(); // adds 'default' renderer with no parameters

  anari::commitParameters(m_device, m_device);

  m_frame = anari::newObject<anari::Frame>(m_device);
  m_perspCamera = anari::newObject<anari::Camera>(m_device, "perspective");
  m_orthoCamera = anari::newObject<anari::Camera>(m_device, "orthographic");

  for (auto &rstate : g_renderers) {
    m_renderers.push_back(
        anari::newObject<anari::Renderer>(m_device, rstate.name.c_str()));
  }

  m_currentRenderer = m_renderers[0];

  m_lights[0] = anari::newObject<anari::Light>(m_device, "directional");

  updateLights();

  m_lightsArray = anari::newArray1D(m_device, m_lights.data(), m_lights.size());

  updateWorld();
  updateFrame();

  resetCameraAZEL();
  resetView();
}

///////////////////////////////////////////////////////////////////////////////
// match3D overrides //////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void Viewer::setup()
{
  ImGuiIO &io = ImGui::GetIO();
  io.FontGlobalScale = 1.25f;

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
  anari::setParameter(m_device, m_frame, "channel.color", m_format);
  anari::setParameter(m_device, m_frame, "channel.depth", ANARI_FLOAT32);
  anari::setParameter(m_device, m_frame, "accumulation", true);

  anari::setParameter(m_device, m_frame, "world", m_currentScene->world());
  if (m_useOrthoCamera)
    anari::setParameter(m_device, m_frame, "camera", m_orthoCamera);
  else
    anari::setParameter(m_device, m_frame, "camera", m_perspCamera);
  anari::setParameter(m_device, m_frame, "renderer", m_currentRenderer);

  if (m_backgroundGradient) {
    constexpr int IMAGE_SIZE = 128;
    auto gradientArray =
        anari::newArray2D(m_device, ANARI_FLOAT32_VEC4, 1, IMAGE_SIZE + 1);
    auto *gradientColors = anari::map<glm::vec4>(m_device, gradientArray);
    for (int i = 0; i <= IMAGE_SIZE; i++) {
      gradientColors[i] =
          glm::mix(m_backgroundBottom, m_backgroundTop, i / float(IMAGE_SIZE));
    }
    anari::unmap(m_device, gradientArray);
    anari::setAndReleaseParameter(
        m_device, m_currentRenderer, "background", gradientArray);
  } else {
    anari::setParameter(
        m_device, m_currentRenderer, "background", m_backgroundTop);
  }
  anari::setParameter(
      m_device, m_currentRenderer, "checkerboard", m_checkerboard);
  anari::setParameter(
      m_device, m_currentRenderer, "pixelSamples", m_pixelSamples);
  anari::setParameter(
      m_device, m_currentRenderer, "ambientColor", m_ambientColor);
  anari::setParameter(
      m_device, m_currentRenderer, "ambientRadiance", m_ambientIntensity);
  anari::setParameter(m_device,
      m_currentRenderer,
      "ambientOcclusionDistance",
      m_ambientOcclusionDistance);
  anari::setParameter(m_device, m_currentRenderer, "denoise", m_denoise);

  anari::commitParameters(m_device, m_currentRenderer);
  anari::commitParameters(m_device, m_frame);
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

  anari::commitParameters(m_device, m_perspCamera);
  anari::commitParameters(m_device, m_orthoCamera);
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
  case SceneTypes::STREAMLINES:
    m_currentScene = generateScene(m_device, m_curvesConfig);
    break;
  case SceneTypes::NOISE_VOLUME:
    m_currentScene = generateScene(m_device, m_noiseVolumeConfig);
    break;
#ifdef USE_MDL
  case SceneTypes::MDL_CUBE:
    m_currentScene = generateScene(m_device, m_mdlCubeConfig);
    break;
#endif // defined(USE_MDL)
  case SceneTypes::GRAVITY_VOLUME:
  default:
    m_currentScene = generateScene(m_device, m_gravityVolumeConfig);
    break;
  }

  auto world = m_currentScene->world();

  anari::setParameter(m_device, world, "light", m_lightsArray);
  anari::commitParameters(m_device, world);

  anari::setParameter(m_device, m_frame, "world", world);
  anari::commitParameters(m_device, m_frame);

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
  auto boundingSphereRadius = glm::distance(bounds[0], bounds[1]) / 2.0f;
  auto aspectRatio = float(m_windowSize.y) / m_windowSize.x;
  auto fovy = glm::radians(
      60.f); // default value from the specification. Not changed by the app.
  auto referenceSize = boundingSphereRadius;
  if (aspectRatio > 1.0f) { // Width is the limiting factor here. Compensate for
                            // y and still compute using fovy.
    referenceSize /= aspectRatio;
  }
  float distance = referenceSize / tanf(fovy / 2.0f);
  // Add some slack so that the bounding sphere actually fits in the frustum.
  distance *= 1.1f;

  m_arcball = Orbit(center, distance, m_arcball.azel());
  updateCamera();
}

void Viewer::resetCameraAZEL()
{
  m_arcball.setAzel(glm::vec2(0.f, -20.f));
}

void Viewer::updateLights()
{
  auto l = m_lights[0];

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

  anari::commitParameters(m_device, l);
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
      auto fb = anari::map<void>(m_device, m_frame, "channel.colorGPU");
      cudaGraphicsMapResources(1, &m_graphicsResource);
      cudaArray_t array;
      cudaGraphicsSubResourceGetMappedArray(&array, m_graphicsResource, 0, 0);
      cudaMemcpy2DToArray(array,
          0,
          0,
          fb.data,
          fb.width * 4,
          fb.width * 4,
          fb.height,
          cudaMemcpyDeviceToDevice);
      cudaGraphicsUnmapResources(1, &m_graphicsResource);
      anari::unmap(m_device, m_frame, "channel.colorGPU");
    } else {
      auto fb = anari::map<void>(
          m_device, m_frame, m_showDepth ? "channel.depth" : "channel.color");

      glBindTexture(GL_TEXTURE_2D, m_framebufferTexture);
      glTexSubImage2D(GL_TEXTURE_2D,
          0,
          0,
          0,
          fb.width,
          fb.height,
          fb.pixelType == ANARI_FLOAT32 ? GL_RED : GL_RGBA,
          fb.pixelType == ANARI_FLOAT32 ? GL_FLOAT : GL_UNSIGNED_BYTE,
          fb.data);

      if (!m_showDepth && m_saveNextFrame) {
        stbi_flip_vertically_on_write(1);
        stbi_write_png(
            "screenshot.png", fb.width, fb.height, 4, fb.data, 4 * fb.width);
        printf("frame saved to 'screenshot.png'\n");
        m_saveNextFrame = false;
      }

      anari::unmap(
          m_device, m_frame, m_showDepth ? "channel.depth" : "channel.color");
    }

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
            m_curvesConfig,
            m_noiseVolumeConfig,
            m_gravityVolumeConfig,
            m_objFileConfig,
#ifdef USE_MDL
            m_mdlCubeConfig,
#endif // defined(USE_MDL)
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
    anari::commitParameters(m_device, m_perspCamera);
    anari::commitParameters(m_device, m_orthoCamera);
  }

  ImGui::Separator();

  if (ImGui::Button("fit scene")) {
    resetView();
  }

  if (ImGui::Button("reset view")) {
    resetCameraAZEL();
    resetView();
  }
}

void Viewer::ui_makeWindow_renderer()
{
  bool update = ImGui::Combo("renderer##whichRenderer",
      &g_whichRenderer,
      rendererUI_callback,
      nullptr,
      g_renderers.size());

  if (ImGui::Checkbox("denoise", &m_denoise)) {
    anari::setParameter(m_device, m_currentRenderer, "denoise", m_denoise);
    anari::commitParameters(m_device, m_currentRenderer);
  }

  if (ImGui::Checkbox("checkerboard", &m_checkerboard)) {
    anari::setParameter(
        m_device, m_currentRenderer, "checkerboarding", m_checkerboard);
    anari::commitParameters(m_device, m_currentRenderer);
  }

  update |= ImGui::Checkbox("gradient background", &m_backgroundGradient);

  if (m_backgroundGradient) {
    update |= ImGui::ColorEdit3("backgroundTop", &m_backgroundTop.x);
    update |= ImGui::ColorEdit3("backgroundBottom", &m_backgroundBottom.x);
  } else {
    update |= ImGui::ColorEdit3("background", &m_backgroundTop.x);
  }

  update |= ImGui::SliderInt("pixelSamples", &m_pixelSamples, 1, 256);

  update |= ImGui::DragFloat(
      "ambientRadiance", &m_ambientIntensity, 0.001f, 0.f, 1000.f);

  update |= ImGui::ColorEdit3("ambientColor", &m_ambientColor.x);

  update |= ImGui::DragFloat(
      "occlusion distance", &m_ambientOcclusionDistance, 0.001f, 0.f, 100000.f);

  if (update) {
    m_currentRenderer = m_renderers[g_whichRenderer];
    updateFrame();
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

void Viewer::ui_makeWindow_materials() {}
