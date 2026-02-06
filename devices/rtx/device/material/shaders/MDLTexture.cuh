/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "gpu/gpu_decl.h"
#include "gpu/shadingState.h"

#include <mi/base/types.h>
#include <mi/neuraylib/target_code_types.h>
#include <texture_types.h>
#include <vector_functions.h>
#include <vector_types.h>

namespace visrtx {

// 2D Lookup

VISRTX_CALLABLE void tex_lookup_float4_2d(float (&result)[4],
    TextureHandler *textureHandler,
    mi::Uint32 textureIdx,
    float const coord[2],
    mi::neuraylib::Tex_wrap_mode wrap_u,
    mi::neuraylib::Tex_wrap_mode wrap_v,
    float const crop_u[2],
    float const crop_v[2],
    float frame);

VISRTX_CALLABLE void tex_lookup_float3_2d(float (&result)[3],
    TextureHandler *textureHandler,
    mi::Uint32 textureIdx,
    float const coord[2],
    mi::neuraylib::Tex_wrap_mode wrap_u,
    mi::neuraylib::Tex_wrap_mode wrap_v,
    float const crop_u[2],
    float const crop_v[2],
    float frame);

VISRTX_CALLABLE void tex_texel_float4_2d(float (&result)[4],
    TextureHandler *textureHandler,
    mi::Uint32 textureIdx,
    int const coord[2],
    int const uv_tile[2],
    float frame);

// 3D lookup

VISRTX_CALLABLE void tex_lookup_float4_3d(float (&result)[4],
    TextureHandler *textureHandler,
    mi::Uint32 textureIdx,
    float const coord[3],
    mi::neuraylib::Tex_wrap_mode wrap_u,
    mi::neuraylib::Tex_wrap_mode wrap_v,
    mi::neuraylib::Tex_wrap_mode wrap_w,
    float const crop_u[2],
    float const crop_v[2],
    float const crop_w[2],
    float frame);

VISRTX_CALLABLE void tex_lookup_float3_3d(float (&result)[3],
    TextureHandler *textureHandler,
    mi::Uint32 textureIdx,
    float const coord[3],
    mi::neuraylib::Tex_wrap_mode wrap_u,
    mi::neuraylib::Tex_wrap_mode wrap_v,
    mi::neuraylib::Tex_wrap_mode wrap_w,
    float const crop_u[2],
    float const crop_v[2],
    float const crop_w[2],
    float frame);

// Cubemap lookup

VISRTX_CALLABLE void tex_lookup_float4_cube(float (&result)[4],
    TextureHandler *textureHandler,
    mi::Uint32 textureIdx,
    float const coord[3]);

VISRTX_CALLABLE void tex_lookup_float3_cube(float (&result)[3],
    TextureHandler *textureHandler,
    mi::Uint32 textureIdx,
    float const coord[3],
    float frame);

// Misc textures

VISRTX_CALLABLE bool tex_texture_isvalid(
    TextureHandler *textureHandler, mi::Uint32 texture_idx);

VISRTX_CALLABLE void tex_resolution_2d(int (&result)[2],
    TextureHandler *textureHandler,
    mi::Uint32 textureIdx,
    int const uv_tile[2],
    float frame);

VISRTX_CALLABLE void tex_resolution_3d(int (&result)[3],
    TextureHandler *textureHandler,
    mi::Uint32 textureIdx,
    float frame);

// Scene data lookup

VISRTX_CALLABLE bool scene_data_isvalid(TextureHandler const *self_base,
    ShadingStateMaterial *state,
    unsigned scene_data_id);

VISRTX_CALLABLE void scene_data_lookup_float4(float result[4],
    TextureHandler const *self_base,
    ShadingStateMaterial *state,
    unsigned scene_data_id,
    float const default_value[4],
    bool uniform_lookup);

VISRTX_CALLABLE void scene_data_lookup_float3(float result[3],
    TextureHandler const *self_base,
    ShadingStateMaterial *state,
    unsigned scene_data_id,
    float const default_value[3],
    bool uniform_lookup);

VISRTX_CALLABLE void scene_data_lookup_color(float result[3],
    TextureHandler const *self_base,
    ShadingStateMaterial *state,
    unsigned scene_data_id,
    float const default_value[3],
    bool uniform_lookup);

VISRTX_CALLABLE void scene_data_lookup_float2(float result[2],
    TextureHandler const *self_base,
    ShadingStateMaterial *state,
    unsigned scene_data_id,
    float const default_value[2],
    bool uniform_lookup);

VISRTX_CALLABLE float scene_data_lookup_float(TextureHandler const *self_base,
    ShadingStateMaterial *state,
    unsigned scene_data_id,
    float const default_value,
    bool uniform_lookup);

VISRTX_CALLABLE void scene_data_lookup_int4(int result[4],
    TextureHandler const *self_base,
    ShadingStateMaterial *state,
    unsigned scene_data_id,
    int const default_value[4],
    bool uniform_lookup);

VISRTX_CALLABLE void scene_data_lookup_int3(int result[3],
    TextureHandler const *self_base,
    ShadingStateMaterial *state,
    unsigned scene_data_id,
    int const default_value[3],
    bool uniform_lookup);

VISRTX_CALLABLE void scene_data_lookup_int2(int result[2],
    TextureHandler const *self_base,
    ShadingStateMaterial *state,
    unsigned scene_data_id,
    int const default_value[2],
    bool uniform_lookup);

VISRTX_CALLABLE int scene_data_lookup_int(TextureHandler const *self_base,
    ShadingStateMaterial *state,
    unsigned scene_data_id,
    int default_value,
    bool uniform_lookup);

VISRTX_CALLABLE void scene_data_lookup_float4x4(float result[16],
    TextureHandler const *self_base,
    ShadingStateMaterial *state,
    unsigned scene_data_id,
    float const default_value[16],
    bool uniform_lookup);
} // namespace visrtx
