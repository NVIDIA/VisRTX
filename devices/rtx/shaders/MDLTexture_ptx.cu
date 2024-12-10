#include "MDLTexture.cuh"
#include "gpu/evalMaterial.h"
#include "gpu/gpu_decl.h"
#include "gpu/gpu_objects.h"
#include "mi/neuraylib/target_code_types.h"

#include <texture_types.h>
#include <vector_functions.h>
#include <vector_types.h>

#include <mi/mdl/mdl_stdlib_types.h>

VISRTX_DEVICE bool handleWrapping(float &coord,
    float invDim,
    mi::neuraylib::Tex_wrap_mode wrapMode,
    const float cropVals[2])
{
  if (wrapMode == mi::neuraylib::TEX_WRAP_REPEAT)
    coord = coord - floorf(coord);
  else {
    if ((wrapMode) == mi::neuraylib::TEX_WRAP_CLIP
        && (coord < 0.0f || coord >= 1.0f)) {
      return false;
    } else if ((wrapMode) == mi::neuraylib::TEX_WRAP_MIRRORED_REPEAT) {
      float floored_val = floorf(coord);
      if ((int(floored_val) & 1) != 0)
        coord = 1.0f - (coord - floored_val);
      else
        coord = coord - floored_val;
    }
    float inv_hdim = 0.5f * (invDim);
    coord = fminf(fmaxf(coord, inv_hdim), 1.f - inv_hdim);
  }
  coord = coord * ((cropVals)[1] - (cropVals)[0]) + (cropVals)[0];

  return true;
}

#ifdef USE_SMOOTHERSTEP_FILTER
// Modify texture coordinates to get better texture filtering,
// see http://www.iquilezles.org/www/articles/texture/texture.htm
#define APPLY_SMOOTHERSTEP_FILTER()                                            \
  do {                                                                         \
    u = u * tex.size.x + 0.5f;                                                 \
    v = v * tex.size.y + 0.5f;                                                 \
                                                                               \
    float u_i = floorf(u), v_i = floorf(v);                                    \
    float u_f = u - u_i;                                                       \
    float v_f = v - v_i;                                                       \
    u_f = u_f * u_f * u_f * (u_f * (u_f * 6.f - 15.f) + 10.f);                 \
    v_f = v_f * v_f * v_f * (v_f * (v_f * 6.f - 15.f) + 10.f);                 \
    u = u_i + u_f;                                                             \
    v = v_i + v_f;                                                             \
                                                                               \
    u = (u - 0.5f) * tex.inv_size.x;                                           \
    v = (v - 0.5f) * tex.inv_size.y;                                           \
  } while (0)
#else
#define APPLY_SMOOTHERSTEP_FILTER()
#endif

namespace visrtx {

// Helpers

using Size = unsigned long long;

template <typename T, Size SIZE, Size... Indices>
VISRTX_DEVICE void _storeResult(
    T (&res)[SIZE], std::integer_sequence<Size, Indices...> indices, T v)
{
  ((res[Indices] = v), ...);
}

template <typename T, Size SIZE>
VISRTX_DEVICE void storeResult(T (&res)[SIZE], T v)
{
  _storeResult(res, std::make_integer_sequence<Size, SIZE>{}, v);
}

template <typename T, Size SIZE, typename Vec>
VISRTX_DEVICE void storeResult(T (&res)[SIZE], Vec vec)
{
  res[0] = vec.x;
  if constexpr (SIZE > 1)
    res[1] = vec.y;
  if constexpr (SIZE > 2)
    res[2] = vec.z;
  if constexpr (SIZE > 3)
    res[3] = vec.w;
}

template <typename T, Size SIZE, Size... Indices, typename... Vs>
VISRTX_DEVICE void _storeResult(
    T (&res)[SIZE], std::integer_sequence<Size, Indices...> indices, Vs &&...vs)
{
  ((res[Indices] = vs), ...);
}

template <typename T, Size SIZE, typename... Vs>
VISRTX_DEVICE void storeResult(T (&res)[SIZE], Vs &&...vs)
{
  static_assert(
      SIZE == sizeof...(Vs), "Not the right number of values to unpack");
  _storeResult(
      res, std::make_integer_sequence<Size, SIZE>{}, std::forward<Vs>(vs)...);
}

// 2D lookup

VISRTX_CALLABLE void tex_lookup_float4_2d(float (&result)[4],
    TextureHandler *textureHandler,
    mi::Uint32 textureIdx,
    float const coord[2],
    mi::neuraylib::Tex_wrap_mode wrapU,
    mi::neuraylib::Tex_wrap_mode wrapV,
    float const cropU[2],
    float const cropV[2],
    float frame)
{
  vec4 sample(0.0f);
  if (tex_texture_isvalid(textureHandler, textureIdx)) {
    auto samplerData = getSamplerData(
        *textureHandler->fd, textureHandler->samplers[textureIdx - 1]);
    auto invSize = samplerData.image2D.invSize;
    float2 coords = {coord[0], coord[1]};

    if (handleWrapping(coords.x, invSize.x, wrapU, cropU)
        && handleWrapping(coords.y, invSize.y, wrapV, cropV))
      sample = evaluateImageTextureSampler(
          samplerData, vec4(coords.x, coords.y, 0.0f, 0.0f));
  }
  storeResult(result, sample);
}

VISRTX_CALLABLE void tex_lookup_float3_2d(float (&result)[3],
    TextureHandler *textureHandler,
    mi::Uint32 textureIdx,
    float const coord[2],
    mi::neuraylib::Tex_wrap_mode wrapU,
    mi::neuraylib::Tex_wrap_mode wrapV,
    float const cropU[2],
    float const cropV[2],
    float frame)
{
  vec3 sample(0.0f);
  if (tex_texture_isvalid(textureHandler, textureIdx)) {
    auto samplerData = getSamplerData(
        *textureHandler->fd, textureHandler->samplers[textureIdx - 1]);
    auto invSize = samplerData.image2D.invSize;
    float2 coords = {coord[0], coord[1]};

    if (handleWrapping(coords.x, invSize.x, wrapU, cropU)
        && handleWrapping(coords.y, invSize.y, wrapV, cropV))
      sample = evaluateImageTextureSampler(
          samplerData, vec4(coords.x, coords.y, 0.0f, 0.0f));
  }

  storeResult(result, sample);
}

VISRTX_CALLABLE void tex_texel_float4_2d(float (&result)[4],
    TextureHandler *textureHandler,
    mi::Uint32 textureIdx,
    int const coord[2],
    int const uvTile[2],
    float frame)
{
  vec4 sample(0.0f);
  if (tex_texture_isvalid(textureHandler, textureIdx)) {
    auto samplerData = getSamplerData(
        *textureHandler->fd, textureHandler->samplers[textureIdx - 1]);
    sample = evaluateImageTexelSampler(
        samplerData, vec4(coord[0], coord[1], 0.0f, 0.0f));
  }

  storeResult(result, sample);
}

// 3D lookup

VISRTX_CALLABLE void tex_lookup_float4_3d(float (&result)[4],
    TextureHandler *textureHandler,
    mi::Uint32 textureIdx,
    float const coord[3],
    mi::neuraylib::Tex_wrap_mode wrapU,
    mi::neuraylib::Tex_wrap_mode wrapV,
    mi::neuraylib::Tex_wrap_mode wrapW,
    float const cropU[2],
    float const cropV[2],
    float const cropW[2],
    float frame)
{
  vec4 sample(0.0f);
  if (tex_texture_isvalid(textureHandler, textureIdx)) {
    auto samplerData = getSamplerData(
        *textureHandler->fd, textureHandler->samplers[textureIdx - 1]);
    auto invSize = samplerData.image3D.invSize;
    float3 coords = {coord[0], coord[1], coord[2]};

    if (handleWrapping(coords.x, invSize.x, wrapU, cropU)
        && handleWrapping(coords.y, invSize.y, wrapV, cropV)
        && handleWrapping(coords.z, invSize.z, wrapW, cropW))
      sample = evaluateImageTextureSampler(
          samplerData, vec4(coords.x, coords.y, coords.z, 0.0f));
  }

  storeResult(result, sample.x, sample.y, sample.z, sample.w);
}

VISRTX_CALLABLE void tex_lookup_float3_3d(float (&result)[3],
    TextureHandler *textureHandler,
    mi::Uint32 textureIdx,
    float const coord[3],
    mi::neuraylib::Tex_wrap_mode wrapU,
    mi::neuraylib::Tex_wrap_mode wrapV,
    mi::neuraylib::Tex_wrap_mode wrapW,
    float const cropU[2],
    float const cropV[2],
    float const cropW[2],
    float frame)
{
  vec3 sample(0.0f);
  if (tex_texture_isvalid(textureHandler, textureIdx)) {
    auto samplerData = getSamplerData(
        *textureHandler->fd, textureHandler->samplers[textureIdx - 1]);
    auto invSize = samplerData.image3D.invSize;
    float3 coords = {coord[0], coord[1], coord[2]};

    if (handleWrapping(coords.x, invSize.x, wrapU, cropU)
        && handleWrapping(coords.y, invSize.y, wrapV, cropV)
        && handleWrapping(coords.z, invSize.z, wrapW, cropW))
      sample = evaluateImageTextureSampler(
          samplerData, vec4(coords.x, coords.y, coords.z, 0.0f));
  }

  storeResult(result, sample.x, sample.y, sample.z);
}

VISRTX_CALLABLE void tex_texel_float4_3d(float (&result)[4],
    TextureHandler *textureHandler,
    mi::Uint32 textureIdx,
    int const coord[3],
    float frame)
{
  vec4 sample(0.0f);
  if (tex_texture_isvalid(textureHandler, textureIdx)) {
    auto samplerData = getSamplerData(
        *textureHandler->fd, textureHandler->samplers[textureIdx - 1]);
    sample = evaluateImageTexelSampler(
        samplerData, vec4(coord[0], coord[1], coord[2], 0.0f));
  }

  storeResult(result, sample);
}

// Misc texture calls

VISRTX_CALLABLE bool tex_texture_isvalid(
    TextureHandler *textureHandler, mi::Uint32 texture_idx)
{
  return texture_idx != 0 && texture_idx <= textureHandler->numSamplers
      && textureHandler->samplers[texture_idx - 1] != DeviceObjectIndex(~0);
}

VISRTX_CALLABLE void tex_resolution_2d(int (&result)[2],
    TextureHandler *textureHandler,
    mi::Uint32 textureIdx,
    int const uv_tile[2],
    float frame)
{
  ivec2 v(0);

  if (tex_texture_isvalid(textureHandler, textureIdx)) {
    auto samplerData = getSamplerData(
        *textureHandler->fd, textureHandler->samplers[textureIdx - 1]);
    v = samplerData.image2D.size;
  }

  storeResult(result, v);
}

VISRTX_CALLABLE void tex_resolution_3d(int (&result)[3],
    TextureHandler *textureHandler,
    mi::Uint32 textureIdx,
    float frame)
{
  ivec3 v(0);

  if (tex_texture_isvalid(textureHandler, textureIdx)) {
    auto samplerData = getSamplerData(
        *textureHandler->fd, textureHandler->samplers[textureIdx - 1]);
    v = samplerData.image3D.size;
  }

  storeResult(result, v);
}

// Scene data lookup

VISRTX_DEVICE bool scene_data_isvalid(TextureHandler const *self_base,
    ShadingStateMaterial *state,
    unsigned scene_data_id)
{
  return false;
}

VISRTX_CALLABLE void scene_data_lookup_float4(float result[4],
    TextureHandler const *self_base,
    ShadingStateMaterial *state,
    unsigned scene_data_id,
    float const default_value[4],
    bool uniform_lookup)
{
  // just return default value
  result[0] = default_value[0];
  result[1] = default_value[1];
  result[2] = default_value[2];
  result[3] = default_value[3];
}

VISRTX_CALLABLE void scene_data_lookup_float3(float result[3],
    TextureHandler const *self_base,
    ShadingStateMaterial *state,
    unsigned scene_data_id,
    float const default_value[3],
    bool uniform_lookup)
{
  // just return default value
  result[0] = default_value[0];
  result[1] = default_value[1];
  result[2] = default_value[2];
}

VISRTX_CALLABLE void scene_data_lookup_color(float result[3],
    TextureHandler const *self_base,
    ShadingStateMaterial *state,
    unsigned scene_data_id,
    float const default_value[3],
    bool uniform_lookup)
{
  // just return default value
  result[0] = default_value[0];
  result[1] = default_value[1];
  result[2] = default_value[2];
}

VISRTX_CALLABLE void scene_data_lookup_float2(float result[2],
    TextureHandler const *self_base,
    ShadingStateMaterial *state,
    unsigned scene_data_id,
    float const default_value[2],
    bool uniform_lookup)
{
  // just return default value
  result[0] = default_value[0];
  result[1] = default_value[1];
}

VISRTX_CALLABLE float scene_data_lookup_float(TextureHandler const *self_base,
    ShadingStateMaterial *state,
    unsigned scene_data_id,
    float const default_value,
    bool uniform_lookup)
{
  // just return default value
  return default_value;
}

VISRTX_CALLABLE void scene_data_lookup_int4(int result[4],
    TextureHandler const *self_base,
    ShadingStateMaterial *state,
    unsigned scene_data_id,
    int const default_value[4],
    bool uniform_lookup)
{
  // just return default value
  result[0] = default_value[0];
  result[1] = default_value[1];
  result[2] = default_value[2];
  result[3] = default_value[3];
}

VISRTX_CALLABLE void scene_data_lookup_int3(int result[3],
    TextureHandler const *self_base,
    ShadingStateMaterial *state,
    unsigned scene_data_id,
    int const default_value[3],
    bool uniform_lookup)
{
  // just return default value
  result[0] = default_value[0];
  result[1] = default_value[1];
  result[2] = default_value[2];
}

VISRTX_CALLABLE void scene_data_lookup_int2(int result[2],
    TextureHandler const *self_base,
    ShadingStateMaterial *state,
    unsigned scene_data_id,
    int const default_value[2],
    bool uniform_lookup)
{
  // just return default value
  result[0] = default_value[0];
  result[1] = default_value[1];
}

VISRTX_CALLABLE int scene_data_lookup_int(TextureHandler const *self_base,
    ShadingStateMaterial *state,
    unsigned scene_data_id,
    int default_value,
    bool uniform_lookup)
{
  // just return default value
  return default_value;
}

VISRTX_CALLABLE void scene_data_lookup_float4x4(float result[16],
    TextureHandler const *self_base,
    ShadingStateMaterial *state,
    unsigned scene_data_id,
    float const default_value[16],
    bool uniform_lookup)
{
  // just return default value
  for (int i = 0; i < 16; ++i)
    result[i] = default_value[i];
}

} // namespace visrtx
