// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <vector>

namespace obj2header::TestOrb_floor {

// clang-format off
static std::vector<float> vertex_position = {
    -2.500000f, 0.000000f, 2.500000f,
    2.500000f, 0.000000f, 2.500000f,
    2.500000f, 0.000000f, -2.500000f,
    -2.500000f, 0.000000f, 2.500000f,
    2.500000f, 0.000000f, -2.500000f,
    -2.500000f, 0.000000f, -2.500000f,
}; // vertex_position

static std::vector<float> vertex_normal = {
    0.000000f, 1.000000f, 0.000000f,
    0.000000f, 1.000000f, 0.000000f,
    0.000000f, 1.000000f, 0.000000f,
    0.000000f, 1.000000f, 0.000000f,
    0.000000f, 1.000000f, 0.000000f,
    0.000000f, 1.000000f, 0.000000f,
}; // vertex_normal

static std::vector<float> vertex_uv = {
    0.000000f, 0.000000f,
    0.000000f, 1.000000f,
    1.000000f, 1.000000f,
    0.000000f, 0.000000f,
    1.000000f, 1.000000f,
    1.000000f, 0.000000f,
}; // vertex_uv

// clang-format on

} // namespace obj2header::TestOrb_floor
