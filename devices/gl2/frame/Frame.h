// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "camera/Camera.h"
#include "renderer/Renderer.h"
#include "scene/World.h"
// helium
#include "helium/BaseFrame.h"
// std
#include <vector>

namespace visgl2 {

struct FrameGLState
{
  GLuint colortarget{0};
  GLuint colorbuffer{0};
  GLuint depthtarget{0};
  GLuint depthbuffer{0};
  GLuint fbo{0};
  void *mappedColorPtr{nullptr};
  void *mappedDepthPtr{nullptr};
};

struct Frame : public helium::BaseFrame
{
  Frame(VisGL2DeviceGlobalState *s);
  ~Frame();

  bool isValid() const override;

  VisGL2DeviceGlobalState *deviceState() const;

  bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint32_t flags) override;

  void commitParameters() override;
  void finalize() override;

  void renderFrame() override;

  void *map(std::string_view channel,
      uint32_t *width,
      uint32_t *height,
      ANARIDataType *pixelType) override;
  void unmap(std::string_view channel) override;
  int frameReady(ANARIWaitMask m) override;
  void discard() override;

  bool ready() const;
  void wait() const;

 private:
  template <typename METHOD_T>
  tasking::Future gl_enqueue_method(METHOD_T m);

  void ogl_allocateObjects();
  void ogl_freeObjects();
  void ogl_renderFrame();
  void ogl_mapColorBuffer();
  void ogl_mapDepthBuffer();
  void ogl_unmapColorBuffer();
  void ogl_unmapDepthBuffer();

  //// Data ////

  FrameGLState m_glState;
  tasking::Future m_future;

  uvec2 m_size{0u, 0u};
  anari::DataType m_colorType{ANARI_UNKNOWN};
  anari::DataType m_depthType{ANARI_UNKNOWN};

  helium::IntrusivePtr<Renderer> m_renderer;
  helium::IntrusivePtr<Camera> m_camera;
  helium::IntrusivePtr<World> m_world;

  float m_duration{0.f};
};

// Inlined definitions ////////////////////////////////////////////////////////

template <typename METHOD_T>
inline tasking::Future Frame::gl_enqueue_method(METHOD_T m)
{
  auto &state = *deviceState();
  return state.gl.thread.enqueue(m, this);
}

} // namespace visgl2

VISGL2_ANARI_TYPEFOR_SPECIALIZATION(visgl2::Frame *, ANARI_FRAME);
