/*
 * Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "RenderPipeline.h"
// std
#include <chrono>
#include <cstring>
#include <limits>
// cuda
#include <cuda_runtime_api.h>

namespace renderpipeline {

RenderPipeline::RenderPipeline() = default;

RenderPipeline::RenderPipeline(int width, int height)
{
  setDimensions(width, height);
}

RenderPipeline::~RenderPipeline()
{
  void cleanup();
}

void RenderPipeline::setDimensions(uint32_t width, uint32_t height)
{
  if (m_size.x == width && m_size.y == height)
    return;
  m_size.x = width;
  m_size.y = height;
  cleanup();
  const size_t totalSize = size_t(width) * size_t(height);
  cudaMallocManaged((void **)&m_buffers.color, totalSize * sizeof(uint32_t));
  cudaMallocManaged((void **)&m_buffers.depth, totalSize * sizeof(float));
  cudaMallocManaged((void **)&m_buffers.objectId, totalSize * sizeof(uint32_t));
  for (auto &p : m_passes)
    p->setDimensions(width, height);
}

void RenderPipeline::render()
{
  int stageId = 0;
#define PRINT_LATENCIES 0
  for (auto &p : m_passes) {
#if PRINT_LATENCIES
    auto start = std::chrono::steady_clock::now();
#endif
    p->render(m_buffers, stageId++);
#if PRINT_LATENCIES
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration<float>(end - start).count();
    printf("[%i] %fms\t", stageId - 1, time * 1000);
#endif
  }
#if PRINT_LATENCIES
  printf("\n");
#endif
}

const uint32_t *RenderPipeline::getColorBuffer() const
{
  return m_buffers.color;
}

size_t RenderPipeline::size() const
{
  return m_passes.size();
}

bool RenderPipeline::empty() const
{
  return m_passes.empty();
}

void RenderPipeline::clear()
{
  m_passes.clear();
}

void RenderPipeline::cleanup()
{
  cudaFree(m_buffers.color);
  cudaFree(m_buffers.depth);
  cudaFree(m_buffers.objectId);
}

} // namespace renderpipeline
