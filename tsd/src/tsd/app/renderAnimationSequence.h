// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// std
#include <functional>
#include <string>

namespace tsd::app {

struct Context;

// Callback type for per-frame actions; return false to abort sequence
using RenderSequenceCallback =
    std::function<bool(int frameIndex, int numFrames)>;

void renderAnimationSequence(Context &ctx,
    const std::string &outputDir,
    const std::string &filePrefix,
    RenderSequenceCallback preFrameCallback = {});

} // namespace tsd::app
