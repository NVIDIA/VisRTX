/*
 * Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "RenderPass.h"
// std
#include <limits>
// cuda
#include <cuda_gl_interop.h>
// thrust
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
// anari
#include <anari/anari_cpp/ext/glm.h>

#define DEVICE_FCN __forceinline__ __device__

namespace renderpipeline {

// Helper GPU functions ///////////////////////////////////////////////////////

template <bool SRGB = true>
DEVICE_FCN float toneMap(float v)
{
  if constexpr (SRGB)
    return std::pow(v, 1.f / 2.2f);
  else
    return v;
}

DEVICE_FCN glm::vec4 cvt_color_to_float4(uint32_t rgba)
{
  const float a = ((rgba >> 24) & 0xff) / 255.f;
  const float b = ((rgba >> 16) & 0xff) / 255.f;
  const float g = ((rgba >> 8) & 0xff) / 255.f;
  const float r = ((rgba >> 0) & 0xff) / 255.f;
  return glm::vec4(r, g, b, a);
}

DEVICE_FCN uint32_t cvt_color_to_uint32(const float &f)
{
  return static_cast<uint32_t>(255.f * std::clamp(f, 0.f, 1.f));
}

DEVICE_FCN uint32_t cvt_color_to_uint32(const glm::vec4 &v)
{
  return (cvt_color_to_uint32(v.x) << 0) | (cvt_color_to_uint32(v.y) << 8)
      | (cvt_color_to_uint32(v.z) << 16) | (cvt_color_to_uint32(v.w) << 24);
}

DEVICE_FCN uint32_t cvt_color_to_uint32_srgb(const glm::vec4 &v)
{
  return cvt_color_to_uint32(
      glm::vec4(toneMap(v.x), toneMap(v.y), toneMap(v.z), v.w));
}

DEVICE_FCN uint32_t shadePixel(uint32_t c_in)
{
  auto c_in_f = cvt_color_to_float4(c_in);
  auto c_h = glm::vec4(1.f, 0.5f, 0.f, 1.f);
  auto c_out = glm::mix(c_in_f, c_h, 0.8f);
  return cvt_color_to_uint32(c_out);
};

// Thrust kernels /////////////////////////////////////////////////////////////

static void thrustCompositeFrame(RenderPass::CUDABuffers &b_out,
    const RenderPass::CUDABuffers &b_in,
    glm::uvec2 size,
    bool firstPass)
{
  thrust::for_each(thrust::device,
      thrust::make_counting_iterator(0u),
      thrust::make_counting_iterator(uint32_t(size.x * size.y)),
      [=] __device__(uint32_t i) {
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

static void thrustComputeOutline(
    RenderPass::CUDABuffers &b, uint32_t outlineId, glm::uvec2 size)
{
  thrust::for_each(thrust::device,
      thrust::make_counting_iterator(0u),
      thrust::make_counting_iterator(uint32_t(size.x * size.y)),
      [=] __device__(uint32_t i) {
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

///////////////////////////////////////////////////////////////////////////////
// RenderPass /////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

RenderPass::RenderPass() = default;

RenderPass::~RenderPass() = default;

void RenderPass::setDimensions(uint32_t width, uint32_t height)
{
  if (m_size.x == width && m_size.y == height)
    return;

  m_size.x = width;
  m_size.y = height;

  updateSize();
}

void RenderPass::updateSize()
{
  // no-up
}

glm::uvec2 RenderPass::getDimensions() const
{
  return m_size;
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

  anari::getProperty(d, d, "visrtx", m_deviceSupportsCUDAFrames);
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
  cudaMallocManaged((void **)&m_buffers.color, totalSize * sizeof(uint32_t));
  cudaMallocManaged((void **)&m_buffers.depth, totalSize * sizeof(float));
  cudaMallocManaged((void **)&m_buffers.objectId, totalSize * sizeof(uint32_t));

  thrust::fill(m_buffers.color, m_buffers.color + totalSize, 0u);
  thrust::fill(m_buffers.depth,
      m_buffers.depth + totalSize,
      std::numeric_limits<float>::infinity());
  thrust::fill(m_buffers.objectId, m_buffers.objectId + totalSize, ~0u);
}

void AnariRenderPass::render(CUDABuffers &b, int stageId)
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

  const glm::uvec2 size(getDimensions());
  if (size.x == color.width && size.y == color.height) {
    const size_t totalSize = size.x * size.y;

    cudaMemcpy(m_buffers.color,
        color.data,
        totalSize * sizeof(uint32_t),
        cudaMemcpyDefault);
    cudaMemcpy(m_buffers.depth,
        depth.data,
        totalSize * sizeof(float),
        cudaMemcpyDefault);
    if (objectId.data) {
      cudaMemcpy(m_buffers.objectId,
          objectId.data,
          totalSize * sizeof(uint32_t),
          cudaMemcpyDefault);
    }
  }

  anari::unmap(m_device, m_frame, "channel.color");
  anari::unmap(m_device, m_frame, "channel.depth");
  anari::unmap(m_device, m_frame, "channel.objectId");
}

void AnariRenderPass::composite(CUDABuffers &b, int stageId)
{
  const bool firstPass = stageId == 0;
  const glm::uvec2 size(getDimensions());
  const size_t totalSize = size.x * size.y;

  if (firstPass) {
    cudaMemcpy(b.color,
        m_buffers.color,
        totalSize * sizeof(uint32_t),
        cudaMemcpyDefault);
    cudaMemcpy(
        b.depth, m_buffers.depth, totalSize * sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(b.objectId,
        m_buffers.objectId,
        totalSize * sizeof(uint32_t),
        cudaMemcpyDefault);
  } else {
    thrustCompositeFrame(b, m_buffers, size, firstPass);
  }
}

void AnariRenderPass::cleanup()
{
  cudaFree(m_buffers.color);
  cudaFree(m_buffers.depth);
  cudaFree(m_buffers.objectId);
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

void OutlineRenderPass::render(CUDABuffers &b, int stageId)
{
  if (!b.objectId || stageId == 0 || m_outlineId == ~0u)
    return;

  thrustComputeOutline(b, m_outlineId, getDimensions());
}

///////////////////////////////////////////////////////////////////////////////
// CopyToGLImagePass //////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

CopyToGLImagePass::CopyToGLImagePass()
{
  glGenTextures(1, &m_texture);
  glBindTexture(GL_TEXTURE_2D, m_texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

CopyToGLImagePass::~CopyToGLImagePass()
{
  if (m_graphicsResource)
    cudaGraphicsUnregisterResource(m_graphicsResource);
  glDeleteTextures(1, &m_texture);
}

GLuint CopyToGLImagePass::getGLTexture() const
{
  return m_texture;
}

void CopyToGLImagePass::render(CUDABuffers &b, int /*stageId*/)
{
  cudaGraphicsMapResources(1, &m_graphicsResource);
  cudaArray_t array;
  cudaGraphicsSubResourceGetMappedArray(&array, m_graphicsResource, 0, 0);
  const auto size = getDimensions();
  cudaMemcpy2DToArray(array,
      0,
      0,
      b.color,
      size.x * 4,
      size.x * 4,
      size.y,
      cudaMemcpyDeviceToDevice);
  cudaGraphicsUnmapResources(1, &m_graphicsResource);
}

void CopyToGLImagePass::updateSize()
{
  if (m_graphicsResource)
    cudaGraphicsUnregisterResource(m_graphicsResource);

  auto newSize = getDimensions();
  glViewport(0, 0, newSize.x, newSize.y);
  glBindTexture(GL_TEXTURE_2D, m_texture);
  glTexImage2D(GL_TEXTURE_2D,
      0,
      GL_RGBA8,
      newSize.x,
      newSize.y,
      0,
      GL_RGBA,
      GL_UNSIGNED_BYTE,
      0);

  cudaGraphicsGLRegisterImage(&m_graphicsResource,
      m_texture,
      GL_TEXTURE_2D,
      cudaGraphicsRegisterFlagsWriteDiscard);
}

} // namespace renderpipeline
