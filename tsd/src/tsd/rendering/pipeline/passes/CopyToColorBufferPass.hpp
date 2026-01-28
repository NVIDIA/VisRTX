// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_rendering
#include "tsd/rendering/pipeline/passes/RenderPass.h"

namespace tsd::rendering {

struct CopyToColorBufferPass : public RenderPass
{
  CopyToColorBufferPass();
  ~CopyToColorBufferPass() override;

  void setExternalBuffer(std::vector<uint8_t> &buffer);

 private:
  void render(RenderBuffers &b, int stageId) override;

  std::vector<uint8_t> *m_externalBuffer{nullptr};
};

} // namespace tsd::rendering
