// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "Frame.h"
// std
#include <algorithm>
#include <chrono>

namespace visgl2 {

// Helper functions ///////////////////////////////////////////////////////////

static uint32_t cvt_uint32(const float &f)
{
  return static_cast<uint32_t>(255.f * std::clamp(f, 0.f, 1.f));
}

static uint32_t cvt_uint32(const vec4 &v)
{
  return (cvt_uint32(v.x) << 0) | (cvt_uint32(v.y) << 8)
      | (cvt_uint32(v.z) << 16) | (cvt_uint32(v.w) << 24);
}

static uint32_t cvt_uint32_srgb(const vec4 &v)
{
  return cvt_uint32(vec4(
      helium::toneMap(v.x), helium::toneMap(v.y), helium::toneMap(v.z), v.w));
}

static GLenum anari2gl(ANARIDataType format)
{
  switch (format) {
  case ANARI_UFIXED8_RGBA_SRGB:
    return GL_SRGB8_ALPHA8;

  case ANARI_FLOAT32_VEC4:
    return GL_RGBA32F;

  case ANARI_UFIXED8_VEC4:
  default:
    return GL_RGBA8;
  }
}

// Frame definitions //////////////////////////////////////////////////////////

Frame::Frame(VisGL2DeviceGlobalState *s) : helium::BaseFrame(s) {}

Frame::~Frame()
{
  wait();
  gl_enqueue_method(&Frame::ogl_freeObjects);
}

bool Frame::isValid() const
{
  return m_renderer && m_renderer->isValid() && m_camera && m_camera->isValid()
      && m_world && m_world->isValid();
}

VisGL2DeviceGlobalState *Frame::deviceState() const
{
  return (VisGL2DeviceGlobalState *)helium::BaseObject::m_state;
}

void Frame::commitParameters()
{
  m_renderer = getParamObject<Renderer>("renderer");
  m_camera = getParamObject<Camera>("camera");
  m_world = getParamObject<World>("world");
  m_colorType = getParam<anari::DataType>("channel.color", ANARI_UNKNOWN);
  m_depthType = getParam<anari::DataType>("channel.depth", ANARI_UNKNOWN);
  m_size = getParam<uvec2>("size", uvec2(0u, 0u));
}

void Frame::finalize()
{
  if (!m_renderer) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'renderer' on frame");
  }

  if (!m_camera) {
    reportMessage(
        ANARI_SEVERITY_WARNING, "missing required parameter 'camera' on frame");
  }

  if (!m_world) {
    reportMessage(
        ANARI_SEVERITY_WARNING, "missing required parameter 'world' on frame");
  }

  gl_enqueue_method(&Frame::ogl_allocateObjects);
}

bool Frame::getProperty(
    const std::string_view &name, ANARIDataType type, void *ptr, uint32_t flags)
{
  if (type == ANARI_FLOAT32 && name == "duration") {
    helium::writeToVoidP(ptr, m_duration);
    return true;
  }

  return 0;
}

void Frame::renderFrame()
{
  auto start = std::chrono::steady_clock::now();

  wait();

  auto &state = *deviceState();
  state.commitBuffer.flush();

  m_future = gl_enqueue_method(&Frame::ogl_renderFrame);

  auto end = std::chrono::steady_clock::now();
  m_duration = std::chrono::duration<float>(end - start).count();
}

void *Frame::map(std::string_view channel,
    uint32_t *width,
    uint32_t *height,
    ANARIDataType *pixelType)
{
  wait();

  *width = m_size.x;
  *height = m_size.y;
  void *ptr = nullptr;

  if (channel == "channel.color") {
    *pixelType = m_colorType;
    gl_enqueue_method(&Frame::ogl_mapColorBuffer).wait();
    ptr = m_glState.mappedColorPtr;
  } else if (channel == "channel.depth") {
    *pixelType = ANARI_FLOAT32;
    gl_enqueue_method(&Frame::ogl_mapDepthBuffer).wait();
    ptr = m_glState.mappedDepthPtr;
  } else {
    *width = 0;
    *height = 0;
    *pixelType = ANARI_UNKNOWN;
  }

  return ptr;
}

void Frame::unmap(std::string_view channel)
{
  if (channel == "channel.color" && m_glState.mappedColorPtr)
    gl_enqueue_method(&Frame::ogl_unmapColorBuffer);
  else if (channel == "channel.depth" && m_glState.mappedDepthPtr)
    gl_enqueue_method(&Frame::ogl_unmapDepthBuffer);
}

int Frame::frameReady(ANARIWaitMask m)
{
  if (m == ANARI_NO_WAIT)
    return ready();
  else {
    wait();
    return 1;
  }
}

void Frame::discard()
{
  // no-op
}

bool Frame::ready() const
{
  return tasking::isReady(m_future);
}

void Frame::wait() const
{
  tasking::wait(m_future);
}

void Frame::ogl_allocateObjects()
{
  reportMessage(ANARI_SEVERITY_DEBUG, "Frame reallocating GL objects");

  auto &state = *deviceState();
  auto &gl = state.gl.glAPI;

  const uint32_t width = m_size.x;
  const uint32_t height = m_size.y;
  const uint32_t elements = width * height;
  const uint32_t element_size = anari::sizeOf(m_colorType);
  const GLenum format = anari2gl(m_colorType);

  ogl_freeObjects();

  gl.GenBuffers(1, &m_glState.colorbuffer);
  gl.BindBuffer(GL_PIXEL_PACK_BUFFER, m_glState.colorbuffer);
  gl.BufferData(
      GL_PIXEL_PACK_BUFFER, elements * element_size, 0, GL_DYNAMIC_READ);

  gl.GenBuffers(1, &m_glState.depthbuffer);
  gl.BindBuffer(GL_PIXEL_PACK_BUFFER, m_glState.depthbuffer);
  gl.BufferData(
      GL_PIXEL_PACK_BUFFER, elements * sizeof(float), 0, GL_DYNAMIC_READ);

  gl.GenTextures(1, &m_glState.colortarget);
  gl.BindTexture(GL_TEXTURE_2D, m_glState.colortarget);
  gl.TexStorage2D(GL_TEXTURE_2D, 1, format, width, height);
  gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  gl.GenTextures(1, &m_glState.depthtarget);
  gl.BindTexture(GL_TEXTURE_2D, m_glState.depthtarget);
  gl.TexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, width, height);
  gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  gl.GenFramebuffers(1, &m_glState.fbo);
  gl.BindFramebuffer(GL_FRAMEBUFFER, m_glState.fbo);
  gl.FramebufferTexture(
      GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, m_glState.colortarget, 0);
  gl.FramebufferTexture(
      GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, m_glState.depthtarget, 0);
  GLenum bufs[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
  gl.DrawBuffers(2, bufs);
}

void Frame::ogl_freeObjects()
{
  auto &state = *deviceState();
  auto &gl = state.gl.glAPI;

  gl.DeleteBuffers(1, &m_glState.colorbuffer);
  gl.DeleteBuffers(1, &m_glState.depthbuffer);
  gl.DeleteTextures(1, &m_glState.colortarget);
  gl.DeleteTextures(1, &m_glState.depthtarget);
  gl.DeleteFramebuffers(1, &m_glState.fbo);
}

void Frame::ogl_renderFrame()
{
  auto &state = *deviceState();
  auto &gl = state.gl.glAPI;
  auto bgc = m_renderer ? m_renderer->background() : vec4(1.f, 0.f, 1.f, 1.f);

  gl.BindFramebuffer(GL_FRAMEBUFFER, m_glState.fbo);
  gl.Enable(GL_FRAMEBUFFER_SRGB);
  gl.Viewport(0, 0, m_size.x, m_size.y);
  gl.ClearColor(bgc.x, bgc.y, bgc.z, bgc.w);
  gl.Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  gl.BindBuffer(GL_PIXEL_PACK_BUFFER, m_glState.colorbuffer);
  gl.ReadBuffer(GL_COLOR_ATTACHMENT0);
  gl.ReadPixels(0, 0, m_size.x, m_size.y, GL_RGBA, GL_UNSIGNED_BYTE, 0);
}

void Frame::ogl_mapColorBuffer()
{
  auto &state = *deviceState();
  auto &gl = state.gl.glAPI;

  gl.BindBuffer(GL_PIXEL_PACK_BUFFER, m_glState.colorbuffer);
  m_glState.mappedColorPtr = gl.MapBufferRange(
      GL_PIXEL_PACK_BUFFER, 0, m_size.x * m_size.y, GL_MAP_READ_BIT);
}

void Frame::ogl_mapDepthBuffer()
{
  auto &state = *deviceState();
  auto &gl = state.gl.glAPI;

  gl.BindFramebuffer(GL_READ_FRAMEBUFFER, m_glState.fbo);
  gl.BindBuffer(GL_PIXEL_PACK_BUFFER, m_glState.depthbuffer);
  gl.ReadBuffer(GL_COLOR_ATTACHMENT1);
  gl.ReadPixels(0, 0, m_size.x, m_size.y, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
  m_glState.mappedDepthPtr = gl.MapBufferRange(
      GL_PIXEL_PACK_BUFFER, 0, m_size.x * m_size.y, GL_MAP_READ_BIT);
}

void Frame::ogl_unmapColorBuffer()
{
  auto &state = *deviceState();
  auto &gl = state.gl.glAPI;

  gl.BindBuffer(GL_PIXEL_PACK_BUFFER, m_glState.colorbuffer);
  gl.UnmapBuffer(GL_PIXEL_PACK_BUFFER);
  m_glState.mappedColorPtr = nullptr;
}

void Frame::ogl_unmapDepthBuffer()
{
  auto &state = *deviceState();
  auto &gl = state.gl.glAPI;

  gl.BindBuffer(GL_PIXEL_PACK_BUFFER, m_glState.depthbuffer);
  gl.UnmapBuffer(GL_PIXEL_PACK_BUFFER);
  m_glState.mappedDepthPtr = nullptr;
}

} // namespace visgl2

VISGL2_ANARI_TYPEFOR_DEFINITION(visgl2::Frame *);
