// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "OutlineRenderPass.h"
// std
#include <algorithm>

#include "detail/parallel_for.h"

namespace tsd::rendering {

// Thrust kernels /////////////////////////////////////////////////////////////

DEVICE_FCN_INLINE uint32_t shadePixel(uint32_t c_in)
{
  auto c_in_f = helium::cvt_color_to_float4(c_in);
  auto c_h = tsd::math::float4(1.f, 0.5f, 0.f, 1.f);
  auto c_out = tsd::math::lerp(c_in_f, c_h, 0.8f);
  return helium::cvt_color_to_uint32(c_out);
};

void computeOutline(RenderBuffers &b, uint32_t outlineId, tsd::math::uint2 size)
{
  detail::parallel_for(
      b.stream, 0u, uint32_t(size.x * size.y), [=] DEVICE_FCN(uint32_t i) {
        uint32_t y = i / size.x;
        uint32_t x = i % size.x;

        int cnt = 0;
        for (unsigned fy = std::max(0u, y - 1);
            fy <= std::min(size.y - 1, y + 1);
            fy++) {
          for (unsigned fx = std::max(0u, x - 1);
              fx <= std::min(size.x - 1, x + 1);
              fx++) {
            size_t fi = fx + size_t(size.x) * fy;
            if (b.objectId[fi] == outlineId)
              cnt++;
          }
        }

        if (cnt > 1 && cnt < 8)
          b.color[i] = shadePixel(b.color[i]);
      });
}

// OutlineRenderPass definitions //////////////////////////////////////////////

OutlineRenderPass::OutlineRenderPass() = default;

OutlineRenderPass::~OutlineRenderPass() = default;

void OutlineRenderPass::setOutlineId(uint32_t id)
{
  m_outlineId = id;
}

void OutlineRenderPass::render(RenderBuffers &b, int stageId)
{
  if (!b.objectId || stageId == 0 || m_outlineId == ~0u)
    return;

  computeOutline(b, m_outlineId, getDimensions());
}

} // namespace tsd::rendering
