// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "RenderPass.h"
// anari
#include <anari/anari_cpp.hpp>

namespace tsd::rendering {

struct AnariSceneRenderPass : public RenderPass
{
  AnariSceneRenderPass(anari::Device d);
  ~AnariSceneRenderPass() override;

  void setCamera(anari::Camera c);
  void setRenderer(anari::Renderer r);
  void setWorld(anari::World w);
  void setColorFormat(anari::DataType t);
  void setEnableIDs(bool on);
  void setEnableAlbedo(bool on);
  void setEnableNormals(bool on);

  // default' true', if 'false', then anari::wait() on each pass
  void setRunAsync(bool on);

  // NOTE(jda): these do not increase ref count, no need to release
  anari::Device getDevice() const;
  anari::Frame getFrame() const;
  anari::Camera getCamera() const;

 private:
  void updateSize() override;
  void render(RenderBuffers &b, int stageId) override;
  void copyFrameData();
  void composite(RenderBuffers &b, int stageId);
  void cleanup();

  RenderBuffers m_buffers;

  bool m_firstFrame{true};
  bool m_deviceSupportsCUDAFrames{false};
  bool m_enableIDs{false};
  bool m_enableAlbedo{false};
  bool m_enableNormals{false};
  bool m_runAsync{true};

  anari::Device m_device{nullptr};
  anari::Camera m_camera{nullptr};
  anari::Renderer m_renderer{nullptr};
  anari::World m_world{nullptr};
  anari::Frame m_frame{nullptr};
};

} // namespace tsd::rendering
