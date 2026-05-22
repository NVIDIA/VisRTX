// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace visrtx::libmdl {

// MDL backend codegen sizing. These match the shadingState.h declarations of
// MDLShadingState::textureCoords / textureTangentsU / textureTangentsV
// (sized by kNumTextureSpaces) and MDLShadingState::textureResults (sized by
// kNumTextureResults). The MDL backend is told these values at PTX-generation
// time via the `num_texture_spaces` / `num_texture_results` options; the
// generated PTX then indexes the corresponding arrays in MDLShadingState. The
// two sides MUST agree, so keep them defined in one place.
//
// kNumTextureSpaces sized to MDL's default upper bound: any material whose
// source references `state::texture_coordinate(i)` with i in [0, 4) generates
// PTX that indexes textureCoords[i]. Shrinking this below 4 would let such a
// material read past the array — there is no asset-loader check that rejects
// multi-UV materials, so the codegen budget must cover them.
constexpr int kNumTextureSpaces = 4;
constexpr int kNumTextureResults = 8;

} // namespace visrtx::libmdl
