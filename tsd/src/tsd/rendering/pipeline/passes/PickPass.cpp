// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "PickPass.h"

namespace tsd::rendering {

PickPass::PickPass() = default;

PickPass::~PickPass() = default;

void PickPass::setPickOperation(PickOpFunc &&f)
{
  m_op = std::move(f);
}

void PickPass::render(RenderBuffers &b, int /*stageId*/)
{
  if (m_op)
    m_op(b);
}

} // namespace tsd::rendering
