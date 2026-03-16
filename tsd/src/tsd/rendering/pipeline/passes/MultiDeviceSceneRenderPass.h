// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_rendering
#include <tsd/rendering/pipeline/passes/ImagePass.h>
// anari
#include <anari/anari_cpp.hpp>
// std
#include <functional>

namespace tsd::rendering {

struct MultiDeviceSceneRenderPass : public ImagePass
{
  MultiDeviceSceneRenderPass(const std::vector<anari::Device> &devices);
  ~MultiDeviceSceneRenderPass() override;

  size_t numDevices() const;

  void setCamera(size_t i, anari::Camera c);
  void setRenderer(size_t i, anari::Renderer r);
  void setWorld(size_t i, anari::World w);
  void setColorFormat(anari::DataType t);

  // default' true' -- if 'false', then anari::wait() on each pass
  void setRunAsync(bool on);

  anari::Frame getFrame(size_t i = 0) const;

 private:
  void foreach_frame(
      const std::function<void(anari::Device, anari::Frame)> &func) const;

  void updateSize() override;
  void render(ImageBuffers &b, int stageId) override;
  void copyFrameData();
  void composite(ImageBuffers &b, int stageId);
  void cleanup();

  ImageBuffers m_buffers;

  bool m_firstFrame{true};
  bool m_runAsync{true};

  std::vector<anari::Device> m_devices;
  std::vector<anari::Frame> m_frames;
};

} // namespace tsd::rendering
