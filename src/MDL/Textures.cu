/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <optix_world.h>

using namespace optix;


// Array of texture sampler IDs for the material expression associated with this OptiX program.
rtDeclareVariable(int, texture_sampler_ids, , );
typedef rtBufferId<int> BufferInt;


// The wrap mode determines the texture lookup behavior if a lookup coordinate
// is exceeding the normalized half-open texture space range of [0, 1).
enum Tex_wrap_mode {
    wrap_clamp           = 0,  // clamps the lookup coordinate to the range
    wrap_repeat          = 1,  // takes the fractional part of the lookup coordinate
                               // effectively repeating the texture along this axis
    wrap_mirrored_repeat = 2,  // like wrap_repeat but takes one minus the fractional
                               // part every other interval to mirror every second
                               // instance of the texture
    wrap_clip            = 3   // makes the texture lookup return zero for texture
                               // coordinates outside of the range
};


// Applies wrapping and cropping to the given coordinate.
// Note: This macro returns if wrap mode is clip and the coordinate is out of range.
#define WRAP_AND_CROP_OR_RETURN_BLACK(val, wrap_mode, crop_vals, store_res_func)               \
    do {                                                                                       \
        if ( (wrap_mode) == wrap_repeat && (crop_vals)[0] == 0.0f && (crop_vals)[1] == 1.0f )  \
        {                                                                                      \
            /* Do nothing, use OptiX texture sampler default behaviour */                      \
        }                                                                                      \
        else                                                                                   \
        {                                                                                      \
            if ( (wrap_mode) == wrap_clamp )                                                   \
            {                                                                                  \
                if ( val < 0.0f ) val = 0.0f;                                                  \
                else if ( val > 1.0f ) val = 1.0f;                                             \
            }                                                                                  \
            else if ( (wrap_mode) == wrap_clip && (val < 0.0f || val > 1.0f) )                 \
            {                                                                                  \
                store_res_func(result, 0.0f);                                                  \
                return;                                                                        \
            }                                                                                  \
            else if ( (wrap_mode) == wrap_mirrored_repeat )                                    \
            {                                                                                  \
                int int_val = int(floorf(val));                                                \
                if ( (int_val & 1) != 0 )                                                      \
                    val = 1.0f - (val - float(int_val));                                       \
                else                                                                           \
                    val = val - float(int_val);                                                \
            }                                                                                  \
            else   /* wrap_repeat */                                                           \
                val = val - floorf(val);                                                       \
            val = val * ((crop_vals)[1] - (crop_vals)[0]) + (crop_vals)[0];                    \
        }                                                                                      \
    } while(0)


// Unused structure which is part of texture functions API.
struct Core_tex_handler;


// Stores a float4 in a float[4] array.
__device__ void store_result4(float res[4], const float4 &v)
{
    res[0] = v.x;
    res[1] = v.y;
    res[2] = v.z;
    res[3] = v.w;
}

// Stores a float in all elements of a float[4] array.
__device__ void store_result4(float res[4], const float v)
{
    res[0] = res[1] = res[2] = res[3] = v;
}

// Stores the given float values in a float[4] array.
__device__ void store_result4(
    float res[4], const float v0, const float v1, const float v2, const float v3)
{
    res[0] = v0;
    res[1] = v1;
    res[2] = v2;
    res[3] = v3;
}

// Stores a float4 in a float[3] array, ignoring v.w.
__device__ void store_result3(float res[3], const float4 &v)
{
    res[0] = v.x;
    res[1] = v.y;
    res[2] = v.z;
}

// Stores a float in all elements of a float[3] array.
__device__ void store_result3(float res[3], const float v)
{
    res[0] = res[1] = res[2] = v;
}

// Stores the given float values in a float[3] array.
__device__ void store_result3(float res[3], const float v0, const float v1, const float v2)
{
    res[0] = v0;
    res[1] = v1;
    res[2] = v2;
}


// Implementation of tex::lookup_float4 for a texture_2d texture.
RT_CALLABLE_PROGRAM void tex_lookup_float4_2d(
    float                  result[4],
    Core_tex_handler const *self,
    unsigned               texture_idx,
    float const            coord[2],
    Tex_wrap_mode          wrap_u,
    Tex_wrap_mode          wrap_v,
    float const            crop_u[2],
    float const            crop_v[2])
{
    if ( texture_idx == 0 ) {
        // invalid texture returns zero
        store_result4(result, 0.0f);
        return;
    }

    float u = coord[0], v = coord[1];
    WRAP_AND_CROP_OR_RETURN_BLACK(u, wrap_u, crop_u, store_result4);
    WRAP_AND_CROP_OR_RETURN_BLACK(v, wrap_v, crop_v, store_result4);

    const BufferInt ids(texture_sampler_ids);
    store_result4(result, rtTex2D<float4>(ids[texture_idx - 1], u, v));
}

// Implementation of tex::lookup_float3 for a texture_2d texture.
RT_CALLABLE_PROGRAM void tex_lookup_float3_2d(
    float                  result[3],
    Core_tex_handler const *self,
    unsigned               texture_idx,
    float const            coord[2],
    Tex_wrap_mode          wrap_u,
    Tex_wrap_mode          wrap_v,
    float const            crop_u[2],
    float const            crop_v[2])
{
    if ( texture_idx == 0 ) {
        // invalid texture returns zero
        store_result3(result, 0.0f);
        return;
    }

    float u = coord[0], v = coord[1];
    WRAP_AND_CROP_OR_RETURN_BLACK(u, wrap_u, crop_u, store_result3);
    WRAP_AND_CROP_OR_RETURN_BLACK(v, wrap_v, crop_v, store_result3);

    const BufferInt ids(texture_sampler_ids);
    store_result3(result, rtTex2D<float4>(ids[texture_idx - 1], u, v));
}

// Implementation of tex::texel_float4 for a texture_2d texture.
// Note: uvtile textures are not supported
RT_CALLABLE_PROGRAM void tex_texel_float4_2d(
    float                  result[4],
    Core_tex_handler const *self,
    unsigned               texture_idx,
    int const              coord[2],
    int const              /*uv_tile*/[2])
{
    if ( texture_idx == 0 ) {
        // invalid texture returns zero
        store_result4(result, 0.0f);
        return;
    }

    const BufferInt ids(texture_sampler_ids);
    store_result4(result, rtTex2DFetch<float4>(ids[texture_idx - 1],
        coord[0], coord[1]));
}

// Implementation of tex::lookup_float4 for a texture_3d texture.
RT_CALLABLE_PROGRAM void tex_lookup_float4_3d(
    float                  result[4],
    Core_tex_handler const *self,
    unsigned               texture_idx,
    float const            coord[3],
    Tex_wrap_mode          wrap_u,
    Tex_wrap_mode          wrap_v,
    Tex_wrap_mode          wrap_w,
    float const            crop_u[2],
    float const            crop_v[2],
    float const            crop_w[2])
{
    if ( texture_idx == 0 ) {
        // invalid texture returns zero
        store_result4(result, 0.0f);
        return;
    }

    float u = coord[0], v = coord[1], w = coord[2];
    WRAP_AND_CROP_OR_RETURN_BLACK(u, wrap_u, crop_u, store_result4);
    WRAP_AND_CROP_OR_RETURN_BLACK(v, wrap_v, crop_v, store_result4);
    WRAP_AND_CROP_OR_RETURN_BLACK(w, wrap_w, crop_w, store_result4);

    const BufferInt ids(texture_sampler_ids);
    store_result4(result, rtTex3D<float4>(ids[texture_idx - 1], u, v, w));
}

// Implementation of tex::lookup_float3 for a texture_3d texture.
RT_CALLABLE_PROGRAM void tex_lookup_float3_3d(
    float                  result[3],
    Core_tex_handler const *self,
    unsigned               texture_idx,
    float const            coord[3],
    Tex_wrap_mode          wrap_u,
    Tex_wrap_mode          wrap_v,
    Tex_wrap_mode          wrap_w,
    float const            crop_u[2],
    float const            crop_v[2],
    float const            crop_w[2])
{
    if ( texture_idx == 0 ) {
        // invalid texture returns zero
        store_result3(result, 0.0f);
        return;
    }

    float u = coord[0], v = coord[1], w = coord[2];
    WRAP_AND_CROP_OR_RETURN_BLACK(u, wrap_u, crop_u, store_result3);
    WRAP_AND_CROP_OR_RETURN_BLACK(v, wrap_v, crop_v, store_result3);
    WRAP_AND_CROP_OR_RETURN_BLACK(w, wrap_w, crop_w, store_result3);

    const BufferInt ids(texture_sampler_ids);
    store_result3(result, rtTex3D<float4>(ids[texture_idx - 1], u, v, w));
}

// Implementation of tex::texel_float4 for a texture_3d texture.
RT_CALLABLE_PROGRAM void tex_texel_float4_3d(
    float                  result[4],
    Core_tex_handler const *self,
    unsigned               texture_idx,
    int const              coord[3])
{
    if ( texture_idx == 0 ) {
        // invalid texture returns zero
        store_result4(result, 0.0f);
        return;
    }

    const BufferInt ids(texture_sampler_ids);
    store_result4(result, rtTex3DFetch<float4>(ids[texture_idx - 1],
        coord[0], coord[1], coord[2]));
}

// Implementation of tex::lookup_float3 for a texture_cube texture.
RT_CALLABLE_PROGRAM void tex_lookup_float3_cube(
    float                  result[3],
    Core_tex_handler const *self,
    unsigned               texture_idx,
    float const            coord[3])
{
    if ( texture_idx == 0 ) {
        // invalid texture returns zero
        store_result3(result, 0.0f);
        return;
    }

    const BufferInt ids(texture_sampler_ids);
    store_result3(result, rtTexCubemap<float4>(ids[texture_idx - 1],
        coord[0], coord[1], coord[2]));
}


// Implementation of resolution_2d function needed by generated code.
// Note: uvtile textures are not supported in this example implementation
RT_CALLABLE_PROGRAM void tex_resolution_2d(
    int                    result[2],
    Core_tex_handler const *self,
    unsigned               texture_idx,
    int const              /*uv_tile*/[2])
{
    if ( texture_idx == 0 ) {
        // invalid texture returns zero
        result[0] = 0;
        result[1] = 0;
        return;
    }

    const BufferInt ids(texture_sampler_ids);
    uint3 size = rtTexSize(ids[texture_idx - 1]);

    result[0] = size.x;
    result[1] = size.y;
}
