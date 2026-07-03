// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ImagePass.h"
// tsd_core
#include "tsd/core/TSDMath.hpp"

namespace tsd::rendering {

/*
 * Draws the Box Outline (12 edges) of a single world-space AABB into the
 * color buffer through the given camera view. One box per pass instance;
 * hide it via setEnabled(false). Lines are depth-tested against the depth
 * buffer by default and silently fall back to an overlay when no depth
 * buffer is present.
 */
struct BoxOutlineRenderPass : public ImagePass
{
  BoxOutlineRenderPass();
  ~BoxOutlineRenderPass() override;
  const char *name() const override { return "Box Outline"; }

  void setBox(const tsd::math::box3 &box);
  void setPerspectiveView(const tsd::math::float3 &eye,
      const tsd::math::float3 &dir,
      const tsd::math::float3 &up,
      float fovy);
  // 'eye' must lie on the plane where the camera's rays originate (the same
  // eye handed to the ANARI camera, e.g. Manipulator::eye_FixedDistance()) —
  // fragment depth is measured from that plane along 'dir'.
  void setOrthographicView(const tsd::math::float3 &eye,
      const tsd::math::float3 &dir,
      const tsd::math::float3 &up,
      float height);
  void setColor(const tsd::math::float4 &color);
  void setWidth(uint32_t width);
  void setDepthTestEnabled(bool enabled);

 private:
  void render(ImageBuffers &b, int stageId) override;

  enum class ViewKind
  {
    NONE,
    PERSPECTIVE,
    ORTHOGRAPHIC
  };

  tsd::math::box3 m_box{{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}};
  ViewKind m_viewKind{ViewKind::NONE};
  tsd::math::float3 m_eye{0.f, 0.f, 0.f};
  tsd::math::float3 m_dir{0.f, 0.f, -1.f};
  tsd::math::float3 m_up{0.f, 1.f, 0.f};
  float m_fovy{0.f}; // radians, perspective only
  float m_height{0.f}; // world units, orthographic only
  tsd::math::float4 m_color{0.8f, 0.8f, 0.8f, 1.f};
  uint32_t m_width{1};
  bool m_depthTestEnabled{true};
};

} // namespace tsd::rendering
