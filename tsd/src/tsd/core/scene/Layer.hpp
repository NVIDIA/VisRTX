// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/scene/LayerNodeData.hpp"

namespace tsd::core {

using Layer = Forest<LayerNodeData>;
using LayerVisitor = Layer::Visitor;

} // namespace tsd::core
