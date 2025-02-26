// Copyright 2024-2025 NVIDIA Corporation
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
  void writeSample(int x, int y, const PixelSample &s);

  //// Data ////

  int m_perPixelBytes{1};
  uvec2 m_size{0u, 0u};

  anari::DataType m_colorType{ANARI_UNKNOWN};
  anari::DataType m_depthType{ANARI_UNKNOWN};

  std::vector<uint8_t> m_pixelBuffer;
  std::vector<float> m_depthBuffer;

  helium::IntrusivePtr<Renderer> m_renderer;
  helium::IntrusivePtr<Camera> m_camera;
  helium::IntrusivePtr<World> m_world;

  float m_duration{0.f};
};

} // namespace visgl2

VISGL2_ANARI_TYPEFOR_SPECIALIZATION(visgl2::Frame *, ANARI_FRAME);
