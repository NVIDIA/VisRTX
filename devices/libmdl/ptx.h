// Copyright 2024 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nonstd/span.hpp>

#include <vector>

namespace visrtx::libmdl {

std::vector<char> stitchPTXs(
    nonstd::span<const nonstd::span<const char>> ptxBlobs);

} // namespace visrtx::libmdl
