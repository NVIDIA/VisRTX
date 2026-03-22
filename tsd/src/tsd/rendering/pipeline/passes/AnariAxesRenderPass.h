// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ImagePass.h"
// anari
#include <anari/anari_cpp.hpp>

namespace tsd::rendering {

/*
 * ImagePass that renders a small orientation-axes overlay into the corner of
 * the frame using a dedicated ANARI frame; updates when the view direction
 * changes.
 *
 * Example:
 *   auto *pass = pipeline.emplace_back<AnariAxesRenderPass>(device,
 * extensions); pass->setView(manipulator.dir(), manipulator.up());
 */
struct AnariAxesRenderPass : public ImagePass
{
  AnariAxesRenderPass(anari::Device d, const anari::Extensions &e);
  ~AnariAxesRenderPass() override;
  const char *name() const override { return "Axes Overlay"; }

  void setView(const tsd::math::float3 &dir, const tsd::math::float3 &up);

 private:
  bool checkNeededExtensions(const anari::Extensions &e);
  bool isValid() const;
  void setupWorld();
  void updateSize() override;
  void render(ImageBuffers &b, int stageId) override;

  bool m_deviceUsable{true};
  bool m_firstFrame{true};

  anari::Device m_device{nullptr};
  anari::Camera m_camera{nullptr};
  anari::Frame m_frame{nullptr};
};

} // namespace tsd::rendering
