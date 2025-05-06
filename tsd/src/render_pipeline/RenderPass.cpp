// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "RenderPass.h"
#include "tsd/core/Logging.hpp"
// std
#include <algorithm>
#include <cstring>
#include <limits>

#include "detail/parallel_for.h"
#include "detail/parallel_transform.h"

#ifdef ENABLE_CUDA
// cuda
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#endif

namespace tsd {

// Thrust kernels /////////////////////////////////////////////////////////////

void convertFloatColorBuffer(const float *v, uint8_t *out, size_t totalSize)
{
  detail::parallel_transform(v, v + totalSize, out, [] DEVICE_FCN(float v) {
    return uint8_t(v * 255);
  });
}

DEVICE_FCN_INLINE uint32_t shadePixel(uint32_t c_in)
{
  auto c_in_f = helium::cvt_color_to_float4(c_in);
  auto c_h = tsd::float4(1.f, 0.5f, 0.f, 1.f);
  auto c_out = tsd::math::lerp(c_in_f, c_h, 0.8f);
  return helium::cvt_color_to_uint32(c_out);
};

void compositeFrame(RenderPass::Buffers &b_out,
    const RenderPass::Buffers &b_in,
    tsd::uint2 size,
    bool firstPass)
{
  detail::parallel_for(
      0u, uint32_t(size.x * size.y), [=] DEVICE_FCN(uint32_t i) {
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

void computeOutline(RenderPass::Buffers &b, uint32_t outlineId, tsd::uint2 size)
{
  detail::parallel_for(
      0u, uint32_t(size.x * size.y), [=] DEVICE_FCN(uint32_t i) {
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

void computeDepthImage(RenderPass::Buffers &b, float maxDepth, tsd::uint2 size)
{
  detail::parallel_for(
      0u, uint32_t(size.x * size.y), [=] DEVICE_FCN(uint32_t i) {
        const float depth = b.depth[i];
        const float v = std::clamp(depth / maxDepth, 0.f, 1.f);
        b.color[i] = helium::cvt_color_to_uint32({tsd::float3(v), 1.f});
      });
}

///////////////////////////////////////////////////////////////////////////////
// RenderPass /////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

RenderPass::RenderPass() = default;

RenderPass::~RenderPass() = default;

void RenderPass::setEnabled(bool enabled)
{
  m_enabled = enabled;
}

bool RenderPass::isEnabled() const
{
  return m_enabled;
}

void RenderPass::updateSize()
{
  // no-up
}

tsd::uint2 RenderPass::getDimensions() const
{
  return m_size;
}

void RenderPass::setDimensions(uint32_t width, uint32_t height)
{
  if (m_size.x == width && m_size.y == height)
    return;

  m_size.x = width;
  m_size.y = height;

  updateSize();
}

///////////////////////////////////////////////////////////////////////////////
// AnariRenderPass ////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

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

AnariRenderPass::AnariRenderPass(anari::Device d) : m_device(d)
{
  anari::retain(d, d);
  m_frame = anari::newObject<anari::Frame>(d);
  anari::setParameter(d, m_frame, "channel.color", ANARI_UFIXED8_RGBA_SRGB);
  anari::setParameter(d, m_frame, "channel.depth", ANARI_FLOAT32);
  anari::setParameter(d, m_frame, "accumulation", true);

  m_deviceSupportsCUDAFrames = supportsCUDAFbData(d);

  if (m_deviceSupportsCUDAFrames)
    tsd::logStatus("[render_pipeline] using CUDA-mapped fb channels");
  else
    tsd::logStatus("[render_pipeline] using host-mapped fb channels");
}

AnariRenderPass::~AnariRenderPass()
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

void AnariRenderPass::setCamera(anari::Camera c)
{
  anari::retain(m_device, c);
  anari::setParameter(m_device, m_frame, "camera", c);
  anari::commitParameters(m_device, m_frame);
  anari::release(m_device, m_camera);
  m_camera = c;
}

void AnariRenderPass::setRenderer(anari::Renderer r)
{
  anari::retain(m_device, r);
  anari::setParameter(m_device, m_frame, "renderer", r);
  anari::commitParameters(m_device, m_frame);
  anari::release(m_device, m_renderer);
  m_renderer = r;
}

void AnariRenderPass::setWorld(anari::World w)
{
  anari::retain(m_device, w);
  anari::setParameter(m_device, m_frame, "world", w);
  anari::commitParameters(m_device, m_frame);
  anari::release(m_device, m_world);
  m_world = w;
}

void AnariRenderPass::setColorFormat(anari::DataType t)
{
  anari::setParameter(m_device, m_frame, "channel.color", t);
  anari::commitParameters(m_device, m_frame);
}

void AnariRenderPass::setEnableIDs(bool on)
{
  if (on == m_enableIDs)
    return;

  m_enableIDs = on;

  if (on) {
    tsd::logInfo("[render_pipeline] enabling objectId frame channel");

    anari::discard(m_device, m_frame);
    anari::wait(m_device, m_frame);

    anari::setParameter(m_device, m_frame, "channel.objectId", ANARI_UINT32);
    anari::commitParameters(m_device, m_frame);

    anari::render(m_device, m_frame);
    anari::wait(m_device, m_frame);
  } else {
    tsd::logInfo("[render_pipeline] disabling objectId frame channel");
    anari::unsetParameter(m_device, m_frame, "channel.objectId");

    auto size = getDimensions();
    const size_t totalSize = size_t(size.x) * size_t(size.y);
#if ENABLE_CUDA
    thrust::fill(m_buffers.objectId, m_buffers.objectId + totalSize, ~0u);
#else
    std::fill(m_buffers.objectId, m_buffers.objectId + totalSize, ~0u);
#endif
    anari::commitParameters(m_device, m_frame);
  }
}

anari::Frame AnariRenderPass::getFrame() const
{
  return m_frame;
}

void AnariRenderPass::updateSize()
{
  cleanup();
  auto size = getDimensions();
  anari::setParameter(m_device, m_frame, "size", size);
  anari::commitParameters(m_device, m_frame);

  const size_t totalSize = size_t(size.x) * size_t(size.y);
  m_buffers.color = detail::allocate<uint32_t>(totalSize);
  m_buffers.depth = detail::allocate<float>(totalSize);
  m_buffers.objectId = detail::allocate<uint32_t>(totalSize);
#if ENABLE_CUDA
  thrust::fill(m_buffers.color, m_buffers.color + totalSize, 0u);
  thrust::fill(m_buffers.depth,
      m_buffers.depth + totalSize,
      std::numeric_limits<float>::infinity());
  thrust::fill(m_buffers.objectId, m_buffers.objectId + totalSize, ~0u);
#else
  std::fill(m_buffers.color, m_buffers.color + totalSize, 0u);
  std::fill(m_buffers.depth,
      m_buffers.depth + totalSize,
      std::numeric_limits<float>::infinity());
  std::fill(m_buffers.objectId, m_buffers.objectId + totalSize, ~0u);
#endif
}

void AnariRenderPass::render(Buffers &b, int stageId)
{
  if (m_firstFrame) {
    anari::render(m_device, m_frame);
    m_firstFrame = false;
  }

  if (anari::isReady(m_device, m_frame)) {
    copyFrameData();
    anari::render(m_device, m_frame);
  }

  composite(b, stageId);
}

void AnariRenderPass::copyFrameData()
{
  const char *colorChannel =
      m_deviceSupportsCUDAFrames ? "channel.colorCUDA" : "channel.color";
  const char *depthChannel =
      m_deviceSupportsCUDAFrames ? "channel.depthCUDA" : "channel.depth";
  const char *idChannel =
      m_deviceSupportsCUDAFrames ? "channel.objectIdCUDA" : "channel.objectId";

  auto color = anari::map<void>(m_device, m_frame, colorChannel);
  auto depth = anari::map<float>(m_device, m_frame, depthChannel);

  const tsd::uint2 size(getDimensions());
  const size_t totalSize = size.x * size.y;
  if (totalSize > 0 && size.x == color.width && size.y == color.height) {
    if (color.pixelType == ANARI_FLOAT32_VEC4) {
      convertFloatColorBuffer(
          (const float *)color.data, (uint8_t *)m_buffers.color, totalSize * 4);
    } else
      detail::copy(m_buffers.color, (uint32_t *)color.data, totalSize);

    detail::copy(m_buffers.depth, depth.data, totalSize);
    if (m_enableIDs) {
      auto objectId = anari::map<uint32_t>(m_device, m_frame, idChannel);
      if (objectId.data)
        detail::copy(m_buffers.objectId, objectId.data, totalSize);
    }
  }

  anari::unmap(m_device, m_frame, colorChannel);
  anari::unmap(m_device, m_frame, depthChannel);
  if (m_enableIDs)
    anari::unmap(m_device, m_frame, idChannel);
}

void AnariRenderPass::composite(Buffers &b, int stageId)
{
  const bool firstPass = stageId == 0;
  const tsd::uint2 size(getDimensions());
  const size_t totalSize = size.x * size.y;

  if (firstPass) {
    detail::copy(b.color, m_buffers.color, totalSize);
    detail::copy(b.depth, m_buffers.depth, totalSize);
    detail::copy(b.objectId, m_buffers.objectId, totalSize);
  } else {
    compositeFrame(b, m_buffers, size, firstPass);
  }
}

void AnariRenderPass::cleanup()
{
  detail::free(m_buffers.color);
  detail::free(m_buffers.depth);
  detail::free(m_buffers.objectId);
}

///////////////////////////////////////////////////////////////////////////////
// PickPass ///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

PickPass::PickPass() = default;

PickPass::~PickPass() = default;

void PickPass::setPickOperation(PickOpFunc &&f)
{
  m_op = std::move(f);
}

void PickPass::render(Buffers &b, int /*stageId*/)
{
  if (m_op)
    m_op(b);
}

///////////////////////////////////////////////////////////////////////////////
// VisualizeDepthPass //////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

VisualizeDepthPass::VisualizeDepthPass() = default;

VisualizeDepthPass::~VisualizeDepthPass() = default;

void VisualizeDepthPass::setMaxDepth(float d)
{
  m_maxDepth = d;
}

void VisualizeDepthPass::render(Buffers &b, int stageId)
{
  if (stageId == 0)
    return;

  computeDepthImage(b, m_maxDepth, getDimensions());
}

///////////////////////////////////////////////////////////////////////////////
// OutlineRenderPass //////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

OutlineRenderPass::OutlineRenderPass() = default;

OutlineRenderPass::~OutlineRenderPass() = default;

void OutlineRenderPass::setOutlineId(uint32_t id)
{
  m_outlineId = id;
}

void OutlineRenderPass::render(Buffers &b, int stageId)
{
  if (!b.objectId || stageId == 0 || m_outlineId == ~0u)
    return;

  computeOutline(b, m_outlineId, getDimensions());
}

#ifdef ENABLE_SDL
///////////////////////////////////////////////////////////////////////////////
// CopyToSDLTexturePass ///////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

CopyToSDLTexturePass::CopyToSDLTexturePass(SDL_Renderer *renderer)
    : m_renderer(renderer)
{}

CopyToSDLTexturePass::~CopyToSDLTexturePass()
{
  SDL_DestroyTexture(m_texture);
}

SDL_Texture *CopyToSDLTexturePass::getTexture() const
{
  return m_texture;
}

void CopyToSDLTexturePass::render(Buffers &b, int /*stageId*/)
{
  const auto size = getDimensions();
  SDL_UpdateTexture(
      m_texture, nullptr, b.color, getDimensions().x * sizeof(b.color[0]));
}

void CopyToSDLTexturePass::updateSize()
{
  if (m_texture)
    SDL_DestroyTexture(m_texture);
  auto newSize = getDimensions();
  m_texture = SDL_CreateTexture(m_renderer,
      SDL_PIXELFORMAT_RGBA32,
      SDL_TEXTUREACCESS_STREAMING,
      newSize.x,
      newSize.y);
}
#endif

///////////////////////////////////////////////////////////////////////////////
// Utility functions //////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

namespace detail {

void *allocate_(size_t numBytes)
{
#ifdef ENABLE_CUDA
  void *ptr = nullptr;
  cudaMallocManaged(&ptr, numBytes);
  return ptr;
#else
  return std::malloc(numBytes);
#endif
}

void free_(void *ptr)
{
#ifdef ENABLE_CUDA
  cudaFree(ptr);
#else
  std::free(ptr);
#endif
}

void memcpy_(void *dst, const void *src, size_t numBytes)
{
#ifdef ENABLE_CUDA
  cudaMemcpy(dst, src, numBytes, cudaMemcpyDefault);
#else
  std::memcpy(dst, src, numBytes);
#endif
}

} // namespace detail

} // namespace tsd
