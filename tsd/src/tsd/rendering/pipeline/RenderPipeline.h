// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "passes/AnariAxesRenderPass.h"
#include "passes/AnariSceneRenderPass.h"
#include "passes/ClearBuffersPass.h"
#if ENABLE_SDL
#include "passes/CopyToSDLTexturePass.h"
#endif
#include "passes/CopyFromColorBufferPass.hpp"
#include "passes/CopyToColorBufferPass.hpp"
#include "passes/MultiDeviceSceneRenderPass.h"
#include "passes/OutlineRenderPass.h"
#include "passes/PickPass.h"
#include "passes/SaveToFilePass.h"
#include "passes/VisualizeAOVPass.h"
// std
#include <memory>

namespace tsd::rendering {

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
  RenderBuffers m_buffers;
  tsd::math::uint2 m_size{0, 0};
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

} // namespace tsd::rendering
