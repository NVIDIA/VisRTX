// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// anari
#include <anari/anari_cpp.hpp>
// std
#include <cstddef>
#include <functional>

namespace tsd::scene {

using IndexRemapper = std::function<size_t(anari::DataType, size_t)>;
using DefragCallback = std::function<void(const IndexRemapper &)>;

} // namespace tsd::scene
