// Copyright 2024-2026 NVIDIA Corporation
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
  bool checkNeededExtensions(const anari::Extensions &e);
  bool isValid() const;
  void setupWorld();
  void updateSize() override;
  void render(RenderBuffers &b, int stageId) override;

  bool m_deviceUsable{true};
  bool m_firstFrame{true};

  anari::Device m_device{nullptr};
  anari::Camera m_camera{nullptr};
  anari::Frame m_frame{nullptr};
};

} // namespace tsd::rendering
