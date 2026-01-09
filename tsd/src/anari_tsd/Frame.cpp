// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Frame.h"
// helium
#include <helium/helium_math.h>
// tsd_io
#include "tsd/io/serialization.hpp"

namespace tsd_device {

Frame::Frame(DeviceGlobalState *s) : helium::BaseFrame(s) {}

Frame::~Frame()
{
  wait();
}

bool Frame::isValid() const
{
  return m_world && m_world->isValid();
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
}

bool Frame::getProperty(const std::string_view &name,
    ANARIDataType type,
    void *ptr,
    uint64_t size,
    uint32_t flags)
{
  return 0;
}

void Frame::renderFrame()
{
  auto *state = deviceState();
  state->commitBuffer.flush();
  reportMessage(
      ANARI_SEVERITY_INFO, "exporting scene to 'tsd_device_export.tsd'");
  tsd::io::save_Scene(state->scene, "tsd_device_export.tsd");
}

void *Frame::map(std::string_view channel,
    uint32_t *width,
    uint32_t *height,
    ANARIDataType *pixelType)
{
  wait();
  *width = 0;
  *height = 0;
  *pixelType = ANARI_UNKNOWN;
  return nullptr;
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
  return true;
}

void Frame::wait() const
{
  // no-op
}

} // namespace tsd_device

TSD_DEVICE_ANARI_TYPEFOR_DEFINITION(tsd_device::Frame *);
