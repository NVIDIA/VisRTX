// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "VisualizeAOVPass.h"
// std
#include <algorithm>

#include "detail/parallel_for.h"

namespace tsd::rendering {

// Thrust kernels /////////////////////////////////////////////////////////////

// Same as visrtx gpu_utils.h
DEVICE_FCN tsd::math::float3 makeRandomColor(uint32_t i)
{
  const uint32_t mx = 13 * 17 * 43;
  const uint32_t my = 11 * 29;
  const uint32_t mz = 7 * 23 * 63;
  const uint32_t g = (i * (3 * 5 * 127) + 12312314);

  if (i == ~0u)
    return tsd::math::float3(0.0f);

  return tsd::math::float3((g % mx) * (1.f / (mx - 1)),
      (g % my) * (1.f / (my - 1)),
      (g % mz) * (1.f / (mz - 1)));
}

void computeObjectIdImage(RenderBuffers &b, tsd::math::uint2 size)
{
  detail::parallel_for(
      b.stream, 0u, uint32_t(size.x * size.y), [=] DEVICE_FCN(uint32_t i) {
        uint32_t id = b.objectId ? b.objectId[i] : ~0u;
        auto color = makeRandomColor(id);
        b.color[i] = helium::cvt_color_to_uint32({color, 1.f});
      });
}

void computePrimitiveIdImage(RenderBuffers &b, tsd::math::uint2 size)
{
  detail::parallel_for(
      b.stream, 0u, uint32_t(size.x * size.y), [=] DEVICE_FCN(uint32_t i) {
        uint32_t id = b.primitiveId ? b.primitiveId[i] : ~0u;
        auto color = makeRandomColor(id);
        b.color[i] = helium::cvt_color_to_uint32({color, 1.f});
      });
}

void computeInstanceIdImage(RenderBuffers &b, tsd::math::uint2 size)
{
  detail::parallel_for(
      b.stream, 0u, uint32_t(size.x * size.y), [=] DEVICE_FCN(uint32_t i) {
        uint32_t id = b.instanceId ? b.instanceId[i] : ~0u;
        auto color = makeRandomColor(id);
        b.color[i] = helium::cvt_color_to_uint32({color, 1.f});
      });
}

void computeDepthImage(
    RenderBuffers &b, float minDepth, float maxDepth, tsd::math::uint2 size)
{
  detail::parallel_for(
      b.stream, 0u, uint32_t(size.x * size.y), [=] DEVICE_FCN(uint32_t i) {
        const float depth = b.depth[i];
        const float range = maxDepth - minDepth;
        const float v = range > 0.f
            ? std::clamp((depth - minDepth) / range, 0.f, 1.f)
            : 0.f;
        b.color[i] = helium::cvt_color_to_uint32({tsd::math::float3(v), 1.f});
      });
}

void computeAlbedoImage(RenderBuffers &b, tsd::math::uint2 size)
{
  detail::parallel_for(
      b.stream, 0u, uint32_t(size.x * size.y), [=] DEVICE_FCN(uint32_t i) {
        const auto albedo = b.albedo ? b.albedo[i] : tsd::math::float3(0.f);
        b.color[i] = helium::cvt_color_to_uint32({albedo, 1.f});
      });
}

void computeNormalImage(RenderBuffers &b, tsd::math::uint2 size)
{
  detail::parallel_for(
      b.stream, 0u, uint32_t(size.x * size.y), [=] DEVICE_FCN(uint32_t i) {
        const auto normal = b.normal ? b.normal[i] : tsd::math::float3(0.f);
        // Map normals from [-1,1] to [0,1] for visualization
        const auto visualNormal = (normal + 1.f) * 0.5f;
        b.color[i] = helium::cvt_color_to_uint32({visualNormal, 1.f});
      });
}

void computeEdgesImage(
    RenderBuffers &b, float threshold, bool invert, tsd::math::uint2 size)
{
  detail::parallel_for(
      b.stream, 0u, uint32_t(size.x * size.y), [=] DEVICE_FCN(uint32_t i) {
        uint32_t y = i / size.x;
        uint32_t x = i % size.x;

        // Get center pixel object ID
        uint32_t centerID = b.objectId ? b.objectId[i] : ~0u;

        // Only check for edges if center pixel is not background
        if (centerID == ~0u) {
          b.color[i] =
              helium::cvt_color_to_uint32({tsd::math::float3(0.f), 1.f});
          return;
        }

        // Check if any neighbor has a different object ID (including
        // background)
        bool isEdge = false;

        for (int dy = -1; dy <= 1 && !isEdge; ++dy) {
          for (int dx = -1; dx <= 1 && !isEdge; ++dx) {
            if (dx == 0 && dy == 0)
              continue; // Skip center pixel

            int nx = static_cast<int>(x) + dx;
            int ny = static_cast<int>(y) + dy;

            if (nx >= 0 && nx < static_cast<int>(size.x) && ny >= 0
                && ny < static_cast<int>(size.y)) {
              size_t neighborIdx =
                  static_cast<size_t>(nx) + static_cast<size_t>(ny) * size.x;
              uint32_t neighborID = b.objectId ? b.objectId[neighborIdx] : ~0u;

              // Edge if neighbor has different ID (including background)
              if (centerID != neighborID) {
                isEdge = true;
              }
            }
          }
        }

        float edgeValue = isEdge ? 1.f : 0.f;

        if (invert) {
          edgeValue = 1.f - edgeValue;
        }

        b.color[i] =
            helium::cvt_color_to_uint32({tsd::math::float3(edgeValue), 1.f});
      });
}

// VisualizeAOVPass definitions ///////////////////////////////////////////////

VisualizeAOVPass::VisualizeAOVPass() = default;

VisualizeAOVPass::~VisualizeAOVPass() = default;

void VisualizeAOVPass::setAOVType(AOVType type)
{
  m_aovType = type;
  setEnabled(type != AOVType::NONE);
}

void VisualizeAOVPass::setDepthRange(float minDepth, float maxDepth)
{
  m_minDepth = minDepth;
  m_maxDepth = maxDepth;
}

void VisualizeAOVPass::setEdgeThreshold(float threshold)
{
  m_edgeThreshold = threshold;
}

void VisualizeAOVPass::setEdgeInvert(bool invert)
{
  m_edgeInvert = invert;
}

void VisualizeAOVPass::render(RenderBuffers &b, int stageId)
{
  if (stageId == 0 || m_aovType == AOVType::NONE)
    return;

  const auto size = getDimensions();

  switch (m_aovType) {
  case AOVType::DEPTH:
    computeDepthImage(b, m_minDepth, m_maxDepth, size);
    break;
  case AOVType::ALBEDO:
    computeAlbedoImage(b, size);
    break;
  case AOVType::NORMAL:
    computeNormalImage(b, size);
    break;
  case AOVType::EDGES:
    computeEdgesImage(b, m_edgeThreshold, m_edgeInvert, size);
    break;
  case AOVType::OBJECT_ID:
    computeObjectIdImage(b, size);
    break;
  case AOVType::PRIMITIVE_ID:
    computePrimitiveIdImage(b, size);
    break;
  case AOVType::INSTANCE_ID:
    computeInstanceIdImage(b, size);
    break;
  default:
    break;
  }
}

} // namespace tsd::rendering
