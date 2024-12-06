// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Frame.h"
// helium
#include <helium/helium_math.h>
// std
#include <algorithm>
#include <chrono>

namespace tsd_device {

Frame::Frame(DeviceGlobalState *s) : helium::BaseFrame(s) {}

Frame::~Frame()
{
  wait();
}

bool Frame::isValid() const
{
#if 1
  return true;
#else
  return m_renderer && m_renderer->isValid() && m_camera && m_camera->isValid()
      && m_world && m_world->isValid();
#endif
}

DeviceGlobalState *Frame::deviceState() const
{
  return (DeviceGlobalState *)helium::BaseObject::m_state;
}

void Frame::commitParameters()
{
#if 0
  m_renderer = getParamObject<Renderer>("renderer");
  if (!m_renderer) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'renderer' on frame");
  }

  m_camera = getParamObject<Camera>("camera");
  if (!m_camera) {
    reportMessage(
        ANARI_SEVERITY_WARNING, "missing required parameter 'camera' on frame");
  }

  m_world = getParamObject<World>("world");
  if (!m_world) {
    reportMessage(
        ANARI_SEVERITY_WARNING, "missing required parameter 'world' on frame");
  }
#endif

  m_colorType = getParam<anari::DataType>("channel.color", ANARI_UNKNOWN);
  m_depthType = getParam<anari::DataType>("channel.depth", ANARI_UNKNOWN);

  m_size = getParam<uint2>("size", uint2(0u, 0u));
}

void Frame::finalize()
{
  const auto numPixels = m_size.x * m_size.y;

  m_perPixelBytes = 4 * (m_colorType == ANARI_FLOAT32_VEC4 ? 4 : 1);
  m_pixelBuffer.resize(numPixels * m_perPixelBytes);

  m_depthBuffer.resize(m_depthType == ANARI_FLOAT32 ? numPixels : 0);

#if 0
  m_valid = m_renderer && m_renderer->isValid() && m_camera
      && m_camera->isValid() && m_world && m_world->isValid();
#endif
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

  auto *state = deviceState();
  state->commitBuffer.flush();

  // This is where the actual rendering work is done. As a placeholder we just
  // write the background color to every pixel, but this is where a frame
  // rendering operation is started. Note that asynchronous implementations
  // should _start_ rendering here, and rely on ready()/wait() to synchronize
  // with the application (they both are used for anariFrameReady()).
#if 0
  auto bgc = m_renderer ? m_renderer->background() : float4(0.f, 0.f, 0.f, 1.f);
#else
  tsd::float4 bgc(0.f, 0.f, 0.f, 1.f);
#endif
  for (uint32_t y = 0; y < m_size.y; y++)
    for (uint32_t x = 0; x < m_size.x; x++)
      writeSample(x, y, PixelSample(bgc));

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

  if (channel == "channel.color") {
    *pixelType = m_colorType;
    return m_pixelBuffer.data();
  } else if (channel == "channel.depth" && !m_depthBuffer.empty()) {
    *pixelType = ANARI_FLOAT32;
    return m_depthBuffer.data();
  } else {
    *width = 0;
    *height = 0;
    *pixelType = ANARI_UNKNOWN;
    return nullptr;
  }
}

void Frame::unmap(std::string_view channel)
{
  // no-op
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
  // The renderFrame() placeholder implementation isn't asynchronous, so the
  // frame is always ready.
  return true;
}

void Frame::wait() const
{
  // The renderFrame() placeholder implementation isn't asynchronous, so the
  // frame is always ready.

  // no-op
}

void Frame::writeSample(int x, int y, const PixelSample &s)
{
  const auto idx = y * m_size.x + x;
  auto *color = m_pixelBuffer.data() + (idx * m_perPixelBytes);
  switch (m_colorType) {
  case ANARI_UFIXED8_VEC4: {
    auto c = helium::cvt_color_to_uint32(s.color);
    std::memcpy(color, &c, sizeof(c));
    break;
  }
  case ANARI_UFIXED8_RGBA_SRGB: {
    auto c = helium::cvt_color_to_uint32_srgb(s.color);
    std::memcpy(color, &c, sizeof(c));
    break;
  }
  case ANARI_FLOAT32_VEC4: {
    std::memcpy(color, &s.color, sizeof(s.color));
    break;
  }
  default:
    break;
  }
  if (!m_depthBuffer.empty())
    m_depthBuffer[idx] = s.depth;
}

} // namespace tsd_device

TSD_DEVICE_ANARI_TYPEFOR_DEFINITION(tsd_device::Frame *);
