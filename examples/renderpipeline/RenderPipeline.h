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

#include "RenderPass.h"
// std
#include <memory>

namespace renderpipeline {

struct RenderPipeline final
{
  RenderPipeline();
  RenderPipeline(int width, int height);
  ~RenderPipeline();

  void setDimensions(uint32_t width, uint32_t height);
  void render();

  const uint32_t *getColorBuffer() const;

  size_t size() const;
  bool empty() const;
  void clear();

  template <typename T, typename... Args>
  T *emplace_back(Args &&...args);

 private:
  void cleanup();

  std::vector<std::unique_ptr<RenderPass>> m_passes;
  RenderPass::CUDABuffers m_buffers;
  glm::uvec2 m_size{0, 0};
};

// Inlined definitions ////////////////////////////////////////////////////////

template <typename T, typename... Args>
inline T *RenderPipeline::emplace_back(Args &&...args)
{
  auto *p = new T(std::forward<Args>(args)...);
  if (m_size.x != 0 && m_size.y != 0)
    p->setDimensions(m_size.x, m_size.y);
  m_passes.emplace_back(p);
  return p;
}

} // namespace renderpipeline
