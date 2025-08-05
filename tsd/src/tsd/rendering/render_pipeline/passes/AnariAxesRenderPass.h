// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "RenderPass.h"
// anari
#include <anari/anari_cpp.hpp>

namespace tsd::rendering {

struct AnariAxesRenderPass : public RenderPass
{
  AnariAxesRenderPass(anari::Device d, const anari::Extensions &e);
  ~AnariAxesRenderPass() override;

  void setView(const tsd::math::float3 &dir, const tsd::math::float3 &up);

 private:
  void setupWorld();
  void updateSize() override;
  void render(Buffers &b, int stageId) override;

  bool m_firstFrame{true};

  anari::Device m_device{nullptr};
  anari::Camera m_camera{nullptr};
  anari::Frame m_frame{nullptr};
};

} // namespace tsd::rendering
