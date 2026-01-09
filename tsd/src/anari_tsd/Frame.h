// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "World.h"
// helium
#include <helium/BaseFrame.h>
// std
#include <vector>

namespace tsd_device {

struct Frame : public helium::BaseFrame
{
  Frame(DeviceGlobalState *s);
  ~Frame();

  bool isValid() const override;

  DeviceGlobalState *deviceState() const;

  bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint64_t size,
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
  helium::IntrusivePtr<World> m_world;
  helium::IntrusivePtr<TSDObject> m_renderer;
  helium::IntrusivePtr<TSDObject> m_camera;

  anari::Frame m_anariFrame{nullptr};
};

} // namespace tsd_device

TSD_DEVICE_ANARI_TYPEFOR_SPECIALIZATION(tsd_device::Frame *, ANARI_FRAME);
