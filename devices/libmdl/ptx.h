// Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <nonstd/span.hpp>

#include <vector>

namespace visrtx::libmdl {

std::vector<char> stitchPTXs(
    nonstd::span<const nonstd::span<const char>> ptxBlobs);

} // namespace visrtx::libmdl
