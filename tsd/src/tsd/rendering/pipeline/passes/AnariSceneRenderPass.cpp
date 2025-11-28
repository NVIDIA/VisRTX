// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "AnariSceneRenderPass.h"
#include "tsd/core/Logging.hpp"
// std
#include <algorithm>
#include <cstring>
#include <limits>

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

void compositeFrame(RenderBuffers &b_out,
    const RenderBuffers &b_in,
    tsd::math::uint2 size,
    bool firstPass)
{
  detail::parallel_for(
      b_out.stream, 0u, uint32_t(size.x * size.y), [=] DEVICE_FCN(uint32_t i) {
        const float currentDepth = b_in.depth[i];
        const float incomingDepth = b_out.depth[i];
        if (firstPass || currentDepth < incomingDepth) {
          b_out.depth[i] = currentDepth;
          b_out.color[i] = b_in.color[i];
          if (b_in.objectId)
            b_out.objectId[i] = b_in.objectId[i];
        }
      });
}

// Helper functions ///////////////////////////////////////////////////////////

static bool supportsCUDAFbData(anari::Device d)
{
#ifdef ENABLE_CUDA
  bool supportsCUDA = false;
  auto list = (const char *const *)anariGetObjectInfo(
      d, ANARI_DEVICE, "default", "extension", ANARI_STRING_LIST);

  for (const char *const *i = list; *i != nullptr; ++i) {
    if (std::string(*i) == "ANARI_NV_FRAME_BUFFERS_CUDA") {
      supportsCUDA = true;
      break;
    }
  }

  return supportsCUDA;
#else
  return false;
#endif
}

// AnariSceneRenderPass definitions ///////////////////////////////////////////

AnariSceneRenderPass::AnariSceneRenderPass(anari::Device d) : m_device(d)
{
  anari::retain(d, d);
  m_frame = anari::newObject<anari::Frame>(d);
  anari::setParameter(d, m_frame, "channel.color", ANARI_UFIXED8_RGBA_SRGB);
  anari::setParameter(d, m_frame, "channel.depth", ANARI_FLOAT32);
  anari::setParameter(d, m_frame, "accumulation", true);

  m_deviceSupportsCUDAFrames = supportsCUDAFbData(d);

  if (m_deviceSupportsCUDAFrames)
    tsd::core::logStatus("[RenderPipeline] using CUDA-mapped fb channels");
  else
    tsd::core::logStatus("[RenderPipeline] using host-mapped fb channels");
}

AnariSceneRenderPass::~AnariSceneRenderPass()
{
  cleanup();

  anari::discard(m_device, m_frame);
  anari::wait(m_device, m_frame);

  anari::release(m_device, m_frame);
  anari::release(m_device, m_camera);
  anari::release(m_device, m_renderer);
  anari::release(m_device, m_world);
  anari::release(m_device, m_device);
}

void AnariSceneRenderPass::setCamera(anari::Camera c)
{
  anari::retain(m_device, c);
  anari::setParameter(m_device, m_frame, "camera", c);
  anari::commitParameters(m_device, m_frame);
  anari::release(m_device, m_camera);
  m_camera = c;
}

void AnariSceneRenderPass::setRenderer(anari::Renderer r)
{
  anari::retain(m_device, r);
  anari::setParameter(m_device, m_frame, "renderer", r);
  anari::commitParameters(m_device, m_frame);
  anari::release(m_device, m_renderer);
  m_renderer = r;
}

void AnariSceneRenderPass::setWorld(anari::World w)
{
  anari::retain(m_device, w);
  anari::setParameter(m_device, m_frame, "world", w);
  anari::commitParameters(m_device, m_frame);
  anari::release(m_device, m_world);
  m_world = w;
}

void AnariSceneRenderPass::setColorFormat(anari::DataType t)
{
  anari::setParameter(m_device, m_frame, "channel.color", t);
  anari::commitParameters(m_device, m_frame);
}

void AnariSceneRenderPass::setEnableIDs(bool on)
{
  if (on == m_enableIDs)
    return;

  m_enableIDs = on;

  if (on) {
    tsd::core::logInfo("[RenderPipeline] enabling objectId frame channel");

    anari::discard(m_device, m_frame);
    anari::wait(m_device, m_frame);

    anari::setParameter(m_device, m_frame, "channel.objectId", ANARI_UINT32);
    anari::commitParameters(m_device, m_frame);

    anari::render(m_device, m_frame);
    anari::wait(m_device, m_frame);
  } else {
    tsd::core::logInfo("[RenderPipeline] disabling objectId frame channel");
    anari::unsetParameter(m_device, m_frame, "channel.objectId");
    anari::commitParameters(m_device, m_frame);

    auto size = getDimensions();
    const size_t totalSize = size_t(size.x) * size_t(size.y);
    std::fill(m_buffers.objectId, m_buffers.objectId + totalSize, ~0u);
  }
}

void AnariSceneRenderPass::setEnableAlbedo(bool on)
{
  if (on == m_enableAlbedo)
    return;

  m_enableAlbedo = on;

  if (on) {
    tsd::core::logInfo("[RenderPipeline] enabling albedo frame channel");

    anari::discard(m_device, m_frame);
    anari::wait(m_device, m_frame);

    anari::setParameter(m_device, m_frame, "channel.albedo", ANARI_FLOAT32_VEC3);
    anari::commitParameters(m_device, m_frame);

    anari::render(m_device, m_frame);
    anari::wait(m_device, m_frame);
  } else {
    tsd::core::logInfo("[RenderPipeline] disabling albedo frame channel");
    anari::unsetParameter(m_device, m_frame, "channel.albedo");
    anari::commitParameters(m_device, m_frame);
  }
}

void AnariSceneRenderPass::setEnableNormals(bool on)
{
  if (on == m_enableNormals)
    return;

  m_enableNormals = on;

  if (on) {
    tsd::core::logInfo("[RenderPipeline] enabling normal frame channel");

    anari::discard(m_device, m_frame);
    anari::wait(m_device, m_frame);

    anari::setParameter(m_device, m_frame, "channel.normal", ANARI_FLOAT32_VEC3);
    anari::commitParameters(m_device, m_frame);

    anari::render(m_device, m_frame);
    anari::wait(m_device, m_frame);
  } else {
    tsd::core::logInfo("[RenderPipeline] disabling normal frame channel");
    anari::unsetParameter(m_device, m_frame, "channel.normal");
    anari::commitParameters(m_device, m_frame);
  }
}

void AnariSceneRenderPass::setRunAsync(bool on)
{
  m_runAsync = on;
}

anari::Device AnariSceneRenderPass::getDevice() const
{
  return m_device;
}

anari::Frame AnariSceneRenderPass::getFrame() const
{
  return m_frame;
}

anari::Camera AnariSceneRenderPass::getCamera() const
{
  return m_camera;
}

void AnariSceneRenderPass::updateSize()
{
  cleanup();
  auto size = getDimensions();
  anari::setParameter(m_device, m_frame, "size", size);
  anari::commitParameters(m_device, m_frame);

  const size_t totalSize = size_t(size.x) * size_t(size.y);
  m_buffers.color = detail::allocate<uint32_t>(totalSize);
  m_buffers.depth = detail::allocate<float>(totalSize);
  m_buffers.objectId = detail::allocate<uint32_t>(totalSize);
  m_buffers.albedo = detail::allocate<tsd::math::float3>(totalSize);
  m_buffers.normal = detail::allocate<tsd::math::float3>(totalSize);
}

void AnariSceneRenderPass::render(RenderBuffers &b, int stageId)
{
  m_buffers.stream = b.stream;

  if (m_firstFrame)
    anari::render(m_device, m_frame);

  if (m_firstFrame || !m_runAsync) {
    anari::wait(m_device, m_frame);
    m_firstFrame = false;
  }

  if (anari::isReady(m_device, m_frame)) {
    copyFrameData();
    anari::render(m_device, m_frame);
  }

  composite(b, stageId);
}

void AnariSceneRenderPass::copyFrameData()
{
  const char *colorChannel =
      m_deviceSupportsCUDAFrames ? "channel.colorCUDA" : "channel.color";
  const char *depthChannel =
      m_deviceSupportsCUDAFrames ? "channel.depthCUDA" : "channel.depth";
  const char *idChannel =
      m_deviceSupportsCUDAFrames ? "channel.objectIdCUDA" : "channel.objectId";
  const char *albedoChannel =
      m_deviceSupportsCUDAFrames ? "channel.albedoCUDA" : "channel.albedo";
  const char *normalChannel =
      m_deviceSupportsCUDAFrames ? "channel.normalCUDA" : "channel.normal";

  auto color = anari::map<void>(m_device, m_frame, colorChannel);
  auto depth = anari::map<float>(m_device, m_frame, depthChannel);

  const tsd::math::uint2 size(getDimensions());
  const size_t totalSize = size.x * size.y;
  if (totalSize > 0 && size.x == color.width && size.y == color.height) {
    if (color.pixelType == ANARI_FLOAT32_VEC4) {
      detail::convertFloatColorBuffer_(m_buffers.stream,
          (const float *)color.data,
          (uint8_t *)m_buffers.color,
          totalSize * 4);
    } else
      detail::copy(m_buffers.color, (uint32_t *)color.data, totalSize);

    detail::copy(m_buffers.depth, depth.data, totalSize);
    if (m_enableIDs) {
      auto objectId = anari::map<uint32_t>(m_device, m_frame, idChannel);
      if (objectId.data)
        detail::copy(m_buffers.objectId, objectId.data, totalSize);
    }
    if (m_enableAlbedo) {
      auto albedo = anari::map<tsd::math::float3>(m_device, m_frame, albedoChannel);
      if (albedo.data)
        detail::copy(m_buffers.albedo, albedo.data, totalSize);
    }
    if (m_enableNormals) {
      auto normal = anari::map<tsd::math::float3>(m_device, m_frame, normalChannel);
      if (normal.data)
        detail::copy(m_buffers.normal, normal.data, totalSize);
    }
  }

  anari::unmap(m_device, m_frame, colorChannel);
  anari::unmap(m_device, m_frame, depthChannel);
  if (m_enableIDs)
    anari::unmap(m_device, m_frame, idChannel);
  if (m_enableAlbedo)
    anari::unmap(m_device, m_frame, albedoChannel);
  if (m_enableNormals)
    anari::unmap(m_device, m_frame, normalChannel);
}

void AnariSceneRenderPass::composite(RenderBuffers &b, int stageId)
{
  const bool firstPass = stageId == 0;
  const tsd::math::uint2 size(getDimensions());
  const size_t totalSize = size.x * size.y;

  if (firstPass) {
    detail::copy(b.color, m_buffers.color, totalSize);
    detail::copy(b.depth, m_buffers.depth, totalSize);
    detail::copy(b.objectId, m_buffers.objectId, totalSize);
    if (m_enableAlbedo)
      detail::copy(b.albedo, m_buffers.albedo, totalSize);
    if (m_enableNormals)
      detail::copy(b.normal, m_buffers.normal, totalSize);
  } else {
    compositeFrame(b, m_buffers, size, firstPass);
  }
}

void AnariSceneRenderPass::cleanup()
{
  detail::free(m_buffers.color);
  detail::free(m_buffers.depth);
  detail::free(m_buffers.objectId);
  detail::free(m_buffers.albedo);
  detail::free(m_buffers.normal);
}

} // namespace tsd::rendering
