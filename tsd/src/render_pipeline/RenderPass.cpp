// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "RenderPass.h"
#include "tsd/core/Logging.hpp"
// std
#include <algorithm>
#include <cstring>
#include <limits>

#include "detail/parallel_for.h"

#ifdef ENABLE_CUDA
// cuda
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#endif

namespace tsd {

// Thrust kernels /////////////////////////////////////////////////////////////

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

void computeOutline(
    RenderPass::Buffers &b, uint32_t outlineId, tsd::uint2 size)
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

void computeDepthImage(
    RenderPass::Buffers &b, float maxDepth, tsd::uint2 size)
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

AnariRenderPass::AnariRenderPass(anari::Device d) : m_device(d)
{
  anari::retain(d, d);
  m_frame = anari::newObject<anari::Frame>(d);
  anari::setParameter(d, m_frame, "channel.color", ANARI_UFIXED8_RGBA_SRGB);
  anari::setParameter(d, m_frame, "channel.depth", ANARI_FLOAT32);
  anari::setParameter(d, m_frame, "channel.objectId", ANARI_UINT32);
  anari::setParameter(d, m_frame, "accumulation", true);

#ifdef ENABLE_CUDA
  anari::getProperty(d, d, "visrtx", m_deviceSupportsCUDAFrames);
#endif
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
  anari::release(m_device, m_camera);
  anari::setParameter(m_device, m_frame, "camera", c);
  anari::commitParameters(m_device, m_frame);
  m_camera = c;
  anari::retain(m_device, m_camera);
}

void AnariRenderPass::setRenderer(anari::Renderer r)
{
  anari::release(m_device, m_renderer);
  anari::setParameter(m_device, m_frame, "renderer", r);
  anari::commitParameters(m_device, m_frame);
  m_renderer = r;
  anari::retain(m_device, m_renderer);
}

void AnariRenderPass::setWorld(anari::World w)
{
  anari::release(m_device, m_world);
  anari::setParameter(m_device, m_frame, "world", w);
  anari::commitParameters(m_device, m_frame);
  m_world = w;
  anari::retain(m_device, m_world);
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
  auto color = anari::map<uint32_t>(m_device,
      m_frame,
      m_deviceSupportsCUDAFrames ? "channel.colorGPU" : "channel.color");
  auto depth = anari::map<float>(m_device,
      m_frame,
      m_deviceSupportsCUDAFrames ? "channel.depthGPU" : "channel.depth");
  auto objectId = anari::map<uint32_t>(m_device,
      m_frame,
      m_deviceSupportsCUDAFrames ? "channel.objectIdGPU" : "channel.objectId");

  const tsd::uint2 size(getDimensions());
  if (size.x == color.width && size.y == color.height) {
    const size_t totalSize = size.x * size.y;

    detail::copy(m_buffers.color, color.data, totalSize);
    detail::copy(m_buffers.depth, depth.data, totalSize);
    if (objectId.data)
      detail::copy(m_buffers.objectId, objectId.data, totalSize);
  }

  anari::unmap(m_device, m_frame, "channel.color");
  anari::unmap(m_device, m_frame, "channel.depth");
  anari::unmap(m_device, m_frame, "channel.objectId");
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

#ifdef ENABLE_OPENGL
///////////////////////////////////////////////////////////////////////////////
// CopyToGLImagePass //////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

struct CopyToGLImagePass::CopyToGLImagePassImpl
{
  GLuint texture{0};
  bool glInteropAvailable{false};
#ifdef ENABLE_CUDA
  cudaGraphicsResource_t graphicsResource{nullptr};
#endif
};

CopyToGLImagePass::CopyToGLImagePass()
{
  m_impl = new CopyToGLImagePassImpl;
  glGenTextures(1, &m_impl->texture);
  glBindTexture(GL_TEXTURE_2D, m_impl->texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  m_impl->glInteropAvailable = checkGLInterop();
}

CopyToGLImagePass::~CopyToGLImagePass()
{
#ifdef ENABLE_CUDA
  if (m_impl->graphicsResource)
    cudaGraphicsUnregisterResource(m_impl->graphicsResource);
#endif
  glDeleteTextures(1, &m_impl->texture);
  delete m_impl;
  m_impl = nullptr;
}

GLuint CopyToGLImagePass::getGLTexture() const
{
  return m_impl->texture;
}

bool CopyToGLImagePass::checkGLInterop()
{
#ifdef ENABLE_CUDA
  unsigned int numDevices = 0;
  int cudaDevices[8]; // Assuming max 8 devices for simplicity

  cudaError_t err =
      cudaGLGetDevices(&numDevices, cudaDevices, 8, cudaGLDeviceListAll);
  if (err != cudaSuccess) {
    tsd::logWarning("[render_pipeline] failed to get CUDA GL devices");
    return false;
  }

  if (numDevices > 0) {
    int currentDevice = 0;
    cudaGetDevice(&currentDevice);
    for (unsigned int i = 0; i < numDevices; ++i) {
      if (currentDevice == cudaDevices[i]) {
        tsd::logStatus("[render_pipeline] using CUDA-GL interop");
        return true;
      }
    }
  }
#endif

  tsd::logWarning("[render_pipeline] unable to use CUDA-GL interop");
  return false;
}

void CopyToGLImagePass::render(Buffers &b, int /*stageId*/)
{
  const auto size = getDimensions();
#ifdef ENABLE_CUDA
  if (m_impl->glInteropAvailable) {
    cudaGraphicsMapResources(1, &m_impl->graphicsResource);
    cudaArray_t array;
    cudaGraphicsSubResourceGetMappedArray(
        &array, m_impl->graphicsResource, 0, 0);
    cudaMemcpy2DToArray(array,
        0,
        0,
        b.color,
        size.x * 4,
        size.x * 4,
        size.y,
        cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &m_impl->graphicsResource);
  } else {
#endif
    glBindTexture(GL_TEXTURE_2D, m_impl->texture);
    glTexSubImage2D(GL_TEXTURE_2D,
        0,
        0,
        0,
        size.x,
        size.y,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        b.color);
#ifdef ENABLE_CUDA
  }
#endif
}

void CopyToGLImagePass::updateSize()
{
#ifdef ENABLE_CUDA
  if (m_impl->graphicsResource)
    cudaGraphicsUnregisterResource(m_impl->graphicsResource);
#endif

  auto newSize = getDimensions();
  glViewport(0, 0, newSize.x, newSize.y);
  glBindTexture(GL_TEXTURE_2D, m_impl->texture);
  glTexImage2D(GL_TEXTURE_2D,
      0,
      GL_RGBA8,
      newSize.x,
      newSize.y,
      0,
      GL_RGBA,
      GL_UNSIGNED_BYTE,
      0);

#ifdef ENABLE_CUDA
  cudaGraphicsGLRegisterImage(&m_impl->graphicsResource,
      m_impl->texture,
      GL_TEXTURE_2D,
      cudaGraphicsRegisterFlagsWriteDiscard);
#endif
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
