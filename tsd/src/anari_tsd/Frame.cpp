// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Frame.h"
// helium
#include <helium/helium_math.h>
// tsd_io
#include "tsd/io/serialization.hpp"
// std
#include <algorithm>

namespace tsd_device {

Frame::Frame(DeviceGlobalState *s) : helium::BaseFrame(s)
{
  m_anariFrame = anari::newObject<anari::Frame>(s->device);
}

Frame::~Frame()
{
  wait();
  anari::release(deviceState()->device, m_anariFrame);
}

bool Frame::isValid() const
{
  return m_world && m_world->isValid() && m_renderer && m_camera;
}

DeviceGlobalState *Frame::deviceState() const
{
  return (DeviceGlobalState *)helium::BaseObject::m_state;
}

void Frame::commitParameters()
{
  m_world = getParamObject<World>("world");
  m_renderer = getParamObject<TSDObject>("renderer");
  m_camera = getParamObject<TSDObject>("camera");
}

void Frame::finalize()
{
  if (!m_world) {
    reportMessage(
        ANARI_SEVERITY_WARNING, "missing required parameter 'world' on frame");
  }

  if (!m_renderer) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'renderer' on frame");
  }

  if (!m_camera) {
    reportMessage(
        ANARI_SEVERITY_WARNING, "missing required parameter 'camera' on frame");
  }

  if (!isValid())
    return;

  auto *state = deviceState();
  auto d = state->device;
  anari::unsetAllParameters(d, m_anariFrame);
  std::for_each(params_begin(), params_end(), [&](auto &p) {
    if (anari::isObject(p.second.type())) {
      reportMessage(ANARI_SEVERITY_WARNING,
          "skip forwarding object parameter '%s' on ANARIFrame",
          p.first.c_str());
    } else if (p.first == "name" && p.second.type() == ANARI_STRING) {
      anari::setParameter(
          d, m_anariFrame, p.first.c_str(), p.second.getString());
    } else if (p.second.type() != ANARI_UNKNOWN) {
      anari::setParameter(
          d, m_anariFrame, p.first.c_str(), p.second.type(), p.second.data());
    } else {
      reportMessage(ANARI_SEVERITY_WARNING,
          "skip setting parameter '%s' of unknown type on ANARIFrame",
          p.first.c_str());
    }
  });

  auto *ri = m_world->getRenderIndex();

  anari::setParameter(d,
      m_anariFrame,
      "renderer",
      ri->renderer(m_renderer->tsdObject()->index()));
  anari::setParameter(
      d, m_anariFrame, "camera", (anari::Camera)m_camera->anariHandle());
  anari::setParameter(d,
      m_anariFrame,
      "world",
      (anari::World)m_world->getRenderIndex()->world());

  anari::commitParameters(d, m_anariFrame);
}

bool Frame::getProperty(const std::string_view &name,
    ANARIDataType type,
    void *ptr,
    uint64_t size,
    uint32_t flags)
{
  std::string nameStr(name);
  return anariGetProperty(deviceState()->device,
      m_anariFrame,
      nameStr.c_str(),
      type,
      ptr,
      size,
      flags);
}

void Frame::renderFrame()
{
  auto *state = deviceState();
  state->commitBuffer.flush();

  if (m_lastCommitFlushOccured < state->commitBuffer.lastObjectFinalization()) {
    m_lastCommitFlushOccured = state->commitBuffer.lastObjectFinalization();
    const char *filename = "live_capture.tsd";
    reportMessage(ANARI_SEVERITY_INFO, "exporting scene to '%s'", filename);
    tsd::io::save_Scene(state->scene, filename);
  }

  anari::render(state->device, m_anariFrame);
}

void *Frame::map(std::string_view channel,
    uint32_t *width,
    uint32_t *height,
    ANARIDataType *pixelType)
{
  auto *state = deviceState();
  std::string channelStr(channel);
  return (void *)anariMapFrame(state->device,
      m_anariFrame,
      channelStr.c_str(),
      width,
      height,
      pixelType);
}

void Frame::unmap(std::string_view channel)
{
  auto *state = deviceState();
  std::string channelStr(channel);
  anari::unmap(state->device, m_anariFrame, channelStr.c_str());
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
  anari::discard(deviceState()->device, m_anariFrame);
}

bool Frame::ready() const
{
  return anari::isReady(deviceState()->device, m_anariFrame);
}

void Frame::wait() const
{
  anari::wait(deviceState()->device, m_anariFrame);
}

} // namespace tsd_device

TSD_DEVICE_ANARI_TYPEFOR_DEFINITION(tsd_device::Frame *);
