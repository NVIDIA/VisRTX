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

#pragma once

// glm
#include <glm/glm.hpp>
// std
#include <vector>
// anari
#include <anari/anari_cpp.hpp>
// OpenGL
#include <GL/gl.h>
// CUDA
#include <cuda_runtime_api.h>

namespace renderpipeline {

struct RenderPass
{
  struct CUDABuffers
  {
    uint32_t *color{nullptr};
    float *depth{nullptr};
    uint32_t *objectId{nullptr};
  };

  RenderPass();
  virtual ~RenderPass();

  glm::uvec2 getDimensions() const;

 protected:
  virtual void render(CUDABuffers &b, int stageId) = 0;
  virtual void updateSize();

 private:
  void setDimensions(uint32_t width, uint32_t height);

  glm::uvec2 m_size{0, 0};

  friend struct RenderPipeline;
};

// RenderPass subtypes ////////////////////////////////////////////////////////

struct AnariRenderPass : public RenderPass
{
  AnariRenderPass(anari::Device d);
  ~AnariRenderPass() override;

  void setCamera(anari::Camera c);
  void setRenderer(anari::Renderer r);
  void setWorld(anari::World w);

  anari::Frame getFrame() const;

 private:
  void updateSize() override;
  void render(CUDABuffers &b, int stageId) override;
  void copyFrameData();
  void composite(CUDABuffers &b, int stageId);
  void cleanup();

  CUDABuffers m_buffers;

  bool m_firstFrame{true};
  bool m_deviceSupportsCUDAFrames{false};

  anari::Device m_device{nullptr};
  anari::Frame m_frame{nullptr};
  anari::Camera m_camera{nullptr};
  anari::Renderer m_renderer{nullptr};
  anari::World m_world{nullptr};
};

struct OutlineRenderPass : public RenderPass
{
  OutlineRenderPass();
  ~OutlineRenderPass() override;

  void setOutlineId(uint32_t id);

 private:
  void render(CUDABuffers &b, int stageId) override;

  uint32_t m_outlineId{~0u};
};

struct CopyToGLImagePass : public RenderPass
{
  CopyToGLImagePass();
  ~CopyToGLImagePass() override;

  GLuint getGLTexture() const;

 private:
  void render(CUDABuffers &b, int stageId) override;
  void updateSize() override;

  GLuint m_texture{0};
  cudaGraphicsResource_t m_graphicsResource{nullptr};
};

} // namespace renderpipeline
