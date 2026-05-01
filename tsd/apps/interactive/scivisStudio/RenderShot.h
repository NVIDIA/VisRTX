// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ProjectContext.h"

#include <functional>

namespace tsd::scivis_studio {

struct RenderShotProgress
{
  std::function<bool(int frame, int totalFrames)> onFrame;
};

bool renderActiveShotToFrames(
    ProjectContext &projectContext, RenderShotProgress *progress = nullptr);

} // namespace tsd::scivis_studio
