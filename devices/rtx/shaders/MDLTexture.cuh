#pragma once

#include <mi/base/types.h>
#include <mi/neuraylib/target_code_types.h>
#include <texture_types.h>
#include <vector_types.h>
#include "gpu/gpu_decl.h"
#include "gpu/gpu_objects.h"

namespace visrtx {

// See
// https://raytracing-docs.nvidia.com/mdl/api/mi_neuray_example_execution_native.html
//  and
//  https://raytracing-docs.nvidia.com/mdl/api/mi_neuray_example_execution_ptx.html
struct TextureHandler : mi::neuraylib::Texture_handler_base
{
  const FrameGPUData *fd;
  const ScreenSample *ss;
  DeviceObjectIndex samplers[32];
  uint numSamplers;
};

using ShadingStateMaterial = mi::neuraylib::Shading_state_material;
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
