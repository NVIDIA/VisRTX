// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "BoxOutlineRenderPass.h"
// tsd_algorithms
#include "tsd/algorithms/cpu/boxOutline.hpp"
#ifdef TSD_ALGORITHMS_HAS_CUDA
#include "tsd/algorithms/cuda/boxOutline.hpp"
#endif
// helium
#include <helium/helium_math.h>
// std
#include <cmath>

namespace tsd::rendering {

BoxOutlineRenderPass::BoxOutlineRenderPass() = default;

BoxOutlineRenderPass::~BoxOutlineRenderPass() = default;

void BoxOutlineRenderPass::setBox(const tsd::math::box3 &box)
{
  m_box = box;
}

void BoxOutlineRenderPass::setPerspectiveView(const tsd::math::float3 &eye,
    const tsd::math::float3 &dir,
    const tsd::math::float3 &up,
    float fovy)
{
  m_viewKind = ViewKind::PERSPECTIVE;
  m_eye = eye;
  m_dir = dir;
  m_up = up;
  m_fovy = fovy;
}

void BoxOutlineRenderPass::setOrthographicView(const tsd::math::float3 &eye,
    const tsd::math::float3 &dir,
    const tsd::math::float3 &up,
    float height)
{
  m_viewKind = ViewKind::ORTHOGRAPHIC;
  m_eye = eye;
  m_dir = dir;
  m_up = up;
  m_height = height;
}

void BoxOutlineRenderPass::setColor(const tsd::math::float4 &color)
{
  m_color = color;
}

void BoxOutlineRenderPass::setWidth(uint32_t width)
{
  m_width = width;
}

void BoxOutlineRenderPass::setDepthTestEnabled(bool enabled)
{
  m_depthTestEnabled = enabled;
}

void BoxOutlineRenderPass::render(ImageBuffers &b, int stageId)
{
  if (!b.color || stageId == 0 || m_viewKind == ViewKind::NONE)
    return;

  const auto size = getDimensions();
  if (size.x == 0 || size.y == 0)
    return;

  const float aspect = size.x / float(size.y);

  const auto view =
      linalg::lookat_matrix(m_eye, m_eye + m_dir, m_up);

  // Only the projected xy is consumed downstream (fragment depth comes from
  // the interpolated world position), so near/far need no scene fitting.
  constexpr float near = 0.1f;
  constexpr float far = 1000.f;

  tsd::math::mat4 proj;
  if (m_viewKind == ViewKind::PERSPECTIVE) {
    const float oneOverTanFov = 1.f / std::tan(m_fovy / 2.f);
    proj = tsd::math::mat4{
        {oneOverTanFov / aspect, 0.f, 0.f, 0.f},
        {0.f, oneOverTanFov, 0.f, 0.f},
        {0.f, 0.f, -(far + near) / (far - near), -1.f},
        {0.f, 0.f, -2.f * far * near / (far - near), 0.f},
    };
  } else {
    const float halfHeight = m_height * 0.5f;
    const float halfWidth = halfHeight * aspect;
    proj = tsd::math::mat4{
        {1.f / halfWidth, 0.f, 0.f, 0.f},
        {0.f, 1.f / halfHeight, 0.f, 0.f},
        {0.f, 0.f, -2.f / (far - near), 0.f},
        {0.f, 0.f, -(far + near) / (far - near), 1.f},
    };
  }

  const auto projView = tsd::math::mul(proj, view);
  const auto color = helium::cvt_color_to_uint32(m_color);
  const float *depth = m_depthTestEnabled ? b.depth : nullptr;
  const bool orthographicDepth = m_viewKind == ViewKind::ORTHOGRAPHIC;

#ifdef TSD_ALGORITHMS_HAS_CUDA
  if (b.stream) {
    tsd::algorithms::cuda::boxOutline(b.stream,
        m_box,
        projView,
        m_eye,
        m_dir,
        orthographicDepth,
        depth,
        b.color,
        color,
        m_width,
        size.x,
        size.y);
    return;
  }
#endif
  tsd::algorithms::cpu::boxOutline(m_box,
      projView,
      m_eye,
      m_dir,
      orthographicDepth,
      depth,
      b.color,
      color,
      m_width,
      size.x,
      size.y);
}

} // namespace tsd::rendering
