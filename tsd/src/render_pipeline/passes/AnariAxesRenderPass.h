// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../RenderPass.h"
// anari
#include <anari/anari_cpp.hpp>

namespace tsd {

struct AnariAxesRenderPass : public RenderPass
{
  AnariAxesRenderPass(anari::Device d);
  ~AnariAxesRenderPass() override;

  void setCamera(anari::Camera c);
  void setRenderer(anari::Renderer r);
  void setWorld(anari::World w);

 private:
  void updateSize() override;
  void render(Buffers &b, int stageId) override;
  void copyFrameData();
  void composite(Buffers &b, int stageId);
  void cleanup();

  Buffers m_buffers;
  bool m_firstFrame{true};

  anari::Device m_device{nullptr};
  anari::Camera m_camera{nullptr};
  anari::Frame m_frame{nullptr};
};

} // namespace tsd
