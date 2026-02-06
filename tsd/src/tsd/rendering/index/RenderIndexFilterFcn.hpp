// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_core
#include "tsd/core/scene/Object.hpp"
// std
#include <functional>

namespace tsd::rendering {

using RenderIndexFilterFcn = std::function<bool(const tsd::core::Object *)>;

} // namespace tsd::rendering
