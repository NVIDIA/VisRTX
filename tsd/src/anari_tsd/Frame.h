// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"
// helium
#include <helium/BaseFrame.h>
// std
#include <vector>

namespace tsd_device {

struct PixelSample
{
  tsd::float4 color{0.f, 1.f, 0.f, 1.f};
  float depth{std::numeric_limits<float>::max()};

  PixelSample(float4 c) : color(c) {}
};

struct Frame : public helium::BaseFrame
{
  Frame(DeviceGlobalState *s);
  ~Frame();

  bool isValid() const override;

  DeviceGlobalState *deviceState() const;

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
  void writeSample(int x, int y, const PixelSample &s);

  //// Data ////

  bool m_valid{false};
  int m_perPixelBytes{1};
  uint2 m_size{0u, 0u};

  anari::DataType m_colorType{ANARI_UNKNOWN};
  anari::DataType m_depthType{ANARI_UNKNOWN};

  std::vector<uint8_t> m_pixelBuffer;
  std::vector<float> m_depthBuffer;

  float m_duration{0.f};
};

} // namespace tsd_device

TSD_DEVICE_ANARI_TYPEFOR_SPECIALIZATION(tsd_device::Frame *, ANARI_FRAME);
