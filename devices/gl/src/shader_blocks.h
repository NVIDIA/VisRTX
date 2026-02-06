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

#include "VisGLString.h"

namespace visgl {

#define GLOBAL_SSBO_OFFSET 4
#define GLOBAL_TEX_OFFSET 1
#define GLOBAL_TRANSFORM_OFFSET 0

#define GEOMETRY_MAX_RESOURCES 16
#define MATERIAL_MAX_RESOURCES 16
#define SURFACE_MAX_RESOURCES (GEOMETRY_MAX_RESOURCES + MATERIAL_MAX_RESOURCES)
#define GEOMETRY_RESOURCE(X) X
#define MATERIAL_RESOURCE(X) (X + GEOMETRY_MAX_RESOURCES)

#define ATTRIBUTE_COLOR 0
#define ATTRIBUTE_ATTRIBUTE0 1
#define ATTRIBUTE_ATTRIBUTE1 2
#define ATTRIBUTE_ATTRIBUTE2 3
#define ATTRIBUTE_ATTRIBUTE3 4
#define ATTRIBUTE_WORLD_POSITION 5
#define ATTRIBUTE_WORLD_NORMAL 6
#define ATTRIBUTE_OBJECT_POSITION 7
#define ATTRIBUTE_OBJECT_NORMAL 8
#define ATTRIBUTE_PRIMITIVE_ID 9
#define ATTRIBUTE_COUNT 10

#define ATTRIBUTE_FLAG_USED 1u
#define ATTRIBUTE_FLAG_SAMPLED 2u
#define ATTRIBUTE_FLAG_TRANSPARENT 4u

static int attribIndex(int value)
{
  switch (value) {
  case STRING_ENUM_attribute0: return ATTRIBUTE_ATTRIBUTE0;
  case STRING_ENUM_attribute1: return ATTRIBUTE_ATTRIBUTE1;
  case STRING_ENUM_attribute2: return ATTRIBUTE_ATTRIBUTE2;
  case STRING_ENUM_attribute3: return ATTRIBUTE_ATTRIBUTE3;
  case STRING_ENUM_color: return ATTRIBUTE_COLOR;
  case STRING_ENUM_objectNormal: return ATTRIBUTE_OBJECT_NORMAL;
  case STRING_ENUM_objectPosition: return ATTRIBUTE_OBJECT_POSITION;
  case STRING_ENUM_primitiveId: return ATTRIBUTE_PRIMITIVE_ID;
  case STRING_ENUM_worldNormal: return ATTRIBUTE_WORLD_NORMAL;
  case STRING_ENUM_worldPosition: return ATTRIBUTE_WORLD_POSITION;
  default: return ATTRIBUTE_COUNT;
  }
}

#define LIGHT_TYPE_POINT 1
#define LIGHT_TYPE_DIRECTIONAL 2
#define LIGHT_TYPE_SPOT 3

// clang-format off
#define UNPACK_LIGHT(I)                                                        \
"    uvec4 indices = lightIndices[" I "];\n"                                   \
"    vec4 c = lights[indices.x];\n"                                            \
"    mat4 t = transforms[indices.y];\n"                                        \
"    vec4 x = t*lights[indices.x+1u];\n"                                       \
"    vec3 light_color = c.xyz * c.w;\n"                                        \
"    vec3 direction = x.xyz - worldPosition.xyz*x.w;\n"                        \
"    float attenuation = 1.0/dot(direction, direction);\n"                     \
"    if(indices.w == 3u) {\n"                                                  \
"      vec4 cone = lights[indices.x+2u];\n"                                    \
"      attenuation *= step(cone.w, dot(normalize(direction), cone.xyz));\n"    \
"    }\n"
// clang-format on

static const char *version_320_es = "#version 320 es\n";

static const char *version_430 = "#version 430\n";

static const char *shader_preamble = R"GLSL(
precision highp float;
precision highp int;

layout(std140, binding = 0) uniform WorldBlock {
  uint cameraIdx;
  uint ambientIdx;
  uint lightCount;
  uint occlusionMode;
  uint frame_width;
  uint frame_height;
  uint samples;
  uint pad5;
  uvec4 lightIndices[254];
};

layout(location = 0) uniform uvec4 instanceIndices;

layout(std430, binding = 0) buffer TransformBlock {
  mat4 transforms[];
};

layout(std430, binding = 1) buffer LightBlock {
  vec4 lights[];
};

layout(std430, binding = 2) buffer MaterialBlock {
  vec4 materials[];
};
)GLSL";

static const char *occlusion_declaration = R"GLSL(
layout(std430, binding = 3) coherent restrict buffer OcclusionBlock {
  float occlusion[];
};
)GLSL";

static const char *clear_occlusion_source = R"GLSL(
precision highp float;
precision highp int;

layout(local_size_x=128) in;
layout(location = 0) uniform int N;

void main() {
  int threads = int(gl_NumWorkGroups.x*gl_WorkGroupSize.x);
  for(int i = int(gl_GlobalInvocationID);i<N;i+=threads) {
    occlusion[i] = 0.0;
  }
}
)GLSL";

struct ShadowProjection
{
  std::array<float, 20> matrix;
};

struct ShadowData
{
  float samples;
  float count;
  float pad2;
  float pad3;
  ShadowProjection projections[12];
};

static const char *shadow_block_declaration = R"GLSL(
struct ShadowProjection {
  mat4 matrix;
  vec4 meta;
};

layout(std140, binding = 1) uniform ShadowBlock {
  vec4 meta;
  ShadowProjection shadowProjection[12];
};
)GLSL";

static const char *shadow_map_declaration = R"GLSL(
struct ShadowProjection {
  mat4 matrix;
  vec4 meta;
};

layout(std140, binding = 1) uniform ShadowBlock {
  vec4 meta;
  ShadowProjection shadowProjection[12];
};

layout(binding = 0) uniform highp sampler2DArrayShadow shadowSampler;

float sampleShadow(vec4 worldPosition, vec3 geometryNormal, uint i) {
  if(i>=12u) {
    return 1.0;
  }
  mat4 projection = shadowProjection[i].matrix;
  float texelsize = dot(shadowProjection[i].meta.xyz, worldPosition.xyz) + shadowProjection[i].meta.w;

  vec4 shadow = projection*(worldPosition + vec4(4.0*texelsize*geometryNormal, 0.0));
  shadow.xyz = 0.5*shadow.xyz + vec3(0.5);
  shadow.z -= 0.0001;
  return texture(shadowSampler, vec4(shadow.xy, i, shadow.z));
}

float sampleShadowBias(vec4 worldPosition, float bias, uint i) {
  if(i>=12u) {
    return 1.0;
  }
  mat4 projection = shadowProjection[i].matrix;

  vec4 shadow = projection*worldPosition;
  shadow.xyz = 0.5*shadow.xyz + vec3(0.5);
  shadow.z -= bias;
  return texture(shadowSampler, vec4(shadow.xy, i, shadow.z));
}

vec3 sampleShadowDir(uint i) {
  if(i>=12u) {
    return vec3(0.0);
  }
  mat4 projection = shadowProjection[i].matrix;
  return vec3(projection[0][2], projection[1][2], projection[2][2]);
}
)GLSL";

static const char *empty_fragment_shader = R"GLSL(
void main() { }
)GLSL";

static const char *shader_conversions = R"GLSL(
const float PI = 3.1415926535897932384626433832795;
const float uintDivisor = 1.0/4294967295.0;
const float intDivisor = 1.0/2147483647.0;
float sqr(float x) { return x*x; }
float linear(float x) {
  float a = x*0.0773993808;
  float b = pow((x+0.055)*0.94786729857, 2.4);
  return mix(a, b, x<=0.04045);
}
vec2 linear(vec2 x) {
  vec2 a = x*0.0773993808;
  vec2 b = pow((x+0.055)*0.94786729857, vec2(2.4));
  return mix(a, b, lessThanEqual(x, vec2(0.04045)));
}
vec3 linear(vec3 x) {
  vec3 a = x*0.0773993808;
  vec3 b = pow((x+0.055)*0.94786729857, vec3(2.4));
  return mix(a, b, lessThanEqual(x, vec3(0.04045)));
}
)GLSL";

// generic snippets
static const char *semicolon = ";\n";

static const char *ssboArraySamplePrimitiveID[] = {
    "sampleArray0(primitiveId);\n",
    "sampleArray1(primitiveId);\n",
    "sampleArray2(primitiveId);\n",
    "sampleArray3(primitiveId);\n",
    "sampleArray4(primitiveId);\n",
    "sampleArray5(primitiveId);\n",
    "sampleArray6(primitiveId);\n",
    "sampleArray7(primitiveId);\n",
    "sampleArray8(primitiveId);\n",
    "sampleArray9(primitiveId);\n",
    "sampleArray10(primitiveId);\n",
    "sampleArray11(primitiveId);\n",
    "sampleArray12(primitiveId);\n",
    "sampleArray13(primitiveId);\n",
    "sampleArray14(primitiveId);\n",
    "sampleArray15(primitiveId);\n",
};

static const char *ssboArraySampleFun[] = {
    "sampleArray0(",
    "sampleArray1(",
    "sampleArray2(",
    "sampleArray3(",
    "sampleArray4(",
    "sampleArray5(",
    "sampleArray6(",
    "sampleArray7(",
    "sampleArray8(",
    "sampleArray9(",
    "sampleArray10(",
    "sampleArray11(",
    "sampleArray12(",
    "sampleArray13(",
    "sampleArray14(",
    "sampleArray15(",
};

static const char *ssboArrayName[] = {
    "ssboArray0",
    "ssboArray1",
    "ssboArray2",
    "ssboArray3",
    "ssboArray4",
    "ssboArray5",
    "ssboArray6",
    "ssboArray7",
    "ssboArray8",
    "ssboArray9",
    "ssboArray10",
    "ssboArray11",
    "ssboArray12",
    "ssboArray13",
    "ssboArray14",
    "ssboArray15",
};

#define ARRAY_SAMPLE_STRING2(I) ARRAY_SAMPLE_STRING(I)

#define ARRAY_SAMPLE_SWITCH                                                    \
  switch (i) {                                                                 \
  case 0: return ARRAY_SAMPLE_STRING2(0);                                      \
  case 1: return ARRAY_SAMPLE_STRING2(1);                                      \
  case 2: return ARRAY_SAMPLE_STRING2(2);                                      \
  case 3: return ARRAY_SAMPLE_STRING2(3);                                      \
  case 4: return ARRAY_SAMPLE_STRING2(4);                                      \
  case 5: return ARRAY_SAMPLE_STRING2(5);                                      \
  case 6: return ARRAY_SAMPLE_STRING2(6);                                      \
  case 7: return ARRAY_SAMPLE_STRING2(7);                                      \
  case 8: return ARRAY_SAMPLE_STRING2(8);                                      \
  case 9: return ARRAY_SAMPLE_STRING2(9);                                      \
  case 10: return ARRAY_SAMPLE_STRING2(10);                                    \
  case 11: return ARRAY_SAMPLE_STRING2(11);                                    \
  case 12: return ARRAY_SAMPLE_STRING2(12);                                    \
  case 13: return ARRAY_SAMPLE_STRING2(13);                                    \
  case 14: return ARRAY_SAMPLE_STRING2(14);                                    \
  case 15: return ARRAY_SAMPLE_STRING2(15);                                    \
  default: return "";                                                          \
  }

// clang-format off
static inline const char* glsl_sample_array(ANARIDataType t, int i) {
  switch(t) {
    case ANARI_UFIXED8_R_SRGB:
#define ARRAY_SAMPLE_STRING(I)\
"layout(std430, binding = " #I ") buffer ssboBlock" #I " { uint ssboArray" #I "[]; };\n"\
"vec4 sampleArray" #I "(uint index) {\n"\
"  vec4 a = unpackUnorm4x8(ssboArray" #I "[index>>2u]);\n"\
"  return vec4(linear(a[index&3u]), 0.0, 0.0, 1.0);\n"\
"}\n"
ARRAY_SAMPLE_SWITCH
#undef ARRAY_SAMPLE_STRING

    case ANARI_UFIXED8:
#define ARRAY_SAMPLE_STRING(I)\
"layout(std430, binding = " #I ") buffer ssboBlock" #I " { uint ssboArray" #I "[]; };\n"\
"vec4 sampleArray" #I "(uint index) {\n"\
"  vec4 a = unpackUnorm4x8(ssboArray" #I "[index>>2u]);\n"\
"  return vec4(a[index&3u], 0.0, 0.0, 1.0);\n"\
"}\n"
ARRAY_SAMPLE_SWITCH
#undef ARRAY_SAMPLE_STRING

    case ANARI_UFIXED8_RA_SRGB:
#define ARRAY_SAMPLE_STRING(I)\
"layout(std430, binding = " #I ") buffer ssboBlock" #I " { uint ssboArray" #I "[]; };\n"\
"vec4 sampleArray" #I "(uint index) {\n"\
"  vec4 a = unpackUnorm4x8(ssboArray" #I "[index>>1u]);\n"\
"  vec2 b = (index&1u)==0u ? a.xy : a.zw;\n"\
"  return vec4(linear(b), 0.0, 1.0);\n"\
"}\n"
ARRAY_SAMPLE_SWITCH
#undef ARRAY_SAMPLE_STRING

    case ANARI_UFIXED8_VEC2:
#define ARRAY_SAMPLE_STRING(I)\
"layout(std430, binding = " #I ") buffer ssboBlock" #I " { uint ssboArray" #I "[]; };\n"\
"vec4 sampleArray" #I "(uint index) {\n"\
"  vec4 a = unpackUnorm4x8(ssboArray" #I "[index>>1u]);\n"\
"  return vec4((index&1u)==0u ? a.xy : a.zw, 0.0, 1.0);\n"\
"}\n"
ARRAY_SAMPLE_SWITCH
#undef ARRAY_SAMPLE_STRING


    case ANARI_UFIXED8_RGB_SRGB:
    case ANARI_UFIXED8_RGBA_SRGB:
#define ARRAY_SAMPLE_STRING(I)\
"layout(std430, binding = " #I ") buffer ssboBlock" #I " {  uint ssboArray" #I "[]; };\n"\
"vec4 sampleArray" #I "(uint index) {\n"\
"  vec4 a = unpackUnorm4x8(ssboArray" #I "[index]);\n"\
"  return vec4(linear(a.xyz), a.w);\n"\
"}\n"
ARRAY_SAMPLE_SWITCH
#undef ARRAY_SAMPLE_STRING

    case ANARI_UFIXED8_VEC3:
    case ANARI_UFIXED8_VEC4:
#define ARRAY_SAMPLE_STRING(I)\
"layout(std430, binding = " #I ") buffer ssboBlock" #I " {  uint ssboArray" #I "[]; };\n"\
"vec4 sampleArray" #I "(uint index) {\n"\
"  return unpackUnorm4x8(ssboArray" #I "[index]);\n"\
"}\n"
ARRAY_SAMPLE_SWITCH
#undef ARRAY_SAMPLE_STRING

    case ANARI_UFIXED16:
#define ARRAY_SAMPLE_STRING(I)\
"layout(std430, binding = " #I ") buffer ssboBlock" #I " { uint ssboArray" #I "[]; };\n"\
"vec4 sampleArray" #I "(uint index) {\n"\
"  vec2 a = unpackUnorm2x16(ssboArray" #I "[index>>1u]);\n"\
"  return vec4(a[index&1u], 0.0, 0.0, 1.0);\n"\
"}\n"
ARRAY_SAMPLE_SWITCH
#undef ARRAY_SAMPLE_STRING

    case ANARI_UFIXED16_VEC2:
#define ARRAY_SAMPLE_STRING(I)\
"layout(std430, binding = " #I ") buffer ssboBlock" #I " {  uint ssboArray" #I "[]; };\n"\
"vec4 sampleArray" #I "(uint index) {\n"\
"  return vec4(unpackUnorm2x16(ssboArray" #I "[index]), 0.0, 1.0);\n"\
"}\n"
ARRAY_SAMPLE_SWITCH
#undef ARRAY_SAMPLE_STRING

    case ANARI_UFIXED16_VEC3:
    case ANARI_UFIXED16_VEC4:
#define ARRAY_SAMPLE_STRING(I)\
"layout(std430, binding = " #I ") buffer ssboBlock" #I " {  uvec2 ssboArray" #I "[]; };\n"\
"vec4 sampleArray" #I "(uint index) {\n"\
"  return vec4(unpackUnorm2x16(ssboArray" #I "[index].x), unpackUnorm2x16(ssboArray" #I "[index].y));\n"\
"}\n"
ARRAY_SAMPLE_SWITCH
#undef ARRAY_SAMPLE_STRING

    case ANARI_UFIXED32:
#define ARRAY_SAMPLE_STRING(I)\
"layout(std430, binding = " #I ") buffer ssboBlock" #I " {  uint ssboArray" #I "[]; };\n"\
"vec4 sampleArray" #I "(uint index) {\n"\
"  return vec4(float(ssboArray" #I "[index])*uintDivisor, 0, 0, 1);\n"\
"}\n"
ARRAY_SAMPLE_SWITCH
#undef ARRAY_SAMPLE_STRING

    case ANARI_UFIXED32_VEC2:
#define ARRAY_SAMPLE_STRING(I)\
"layout(std430, binding = " #I ") buffer ssboBlock" #I " {  uvec2 ssboArray" #I "[]; };\n"\
"vec4 sampleArray" #I "(uint index) {\n"\
"  return vec4(vec2(ssboArray" #I "[index])*uintDivisor, 0, 1);\n"\
"}\n"
ARRAY_SAMPLE_SWITCH
#undef ARRAY_SAMPLE_STRING

    case ANARI_UFIXED32_VEC3:
#define ARRAY_SAMPLE_STRING(I)\
"layout(std430, binding = " #I ") buffer ssboBlock" #I " {  uint ssboArray" #I "[]; };\n"\
"vec4 sampleArray" #I "(uint index) {\n"\
"  return vec4(vec3(ssboArray" #I "[3u*index], ssboArray" #I "[3u*index+1u], ssboArray" #I "[3u*index+2u])*uintDivisor, 1);\n"\
"}\n"
ARRAY_SAMPLE_SWITCH
#undef ARRAY_SAMPLE_STRING

    case ANARI_UFIXED32_VEC4:
#define ARRAY_SAMPLE_STRING(I)\
"layout(std430, binding = " #I ") buffer ssboBlock" #I " {  uvec4 ssboArray" #I "[]; };\n"\
"vec4 sampleArray" #I "(uint index) {\n"\
"  return vec4(ssboArray" #I "[index])*uintDivisor;\n"\
"}\n"
ARRAY_SAMPLE_SWITCH
#undef ARRAY_SAMPLE_STRING


    case ANARI_UINT32:
#define ARRAY_SAMPLE_STRING(I)\
"layout(std430, binding = " #I ") buffer ssboBlock" #I " {  uint ssboArray" #I "[]; };\n"\
"vec4 sampleArray" #I "(uint index) {\n"\
"  return vec4(float(ssboArray" #I "[index]), 0, 0, 1);\n"\
"}\n"
ARRAY_SAMPLE_SWITCH
#undef ARRAY_SAMPLE_STRING

    case ANARI_UINT32_VEC2:
#define ARRAY_SAMPLE_STRING(I)\
"layout(std430, binding = " #I ") buffer ssboBlock" #I " {  uvec2 ssboArray" #I "[]; };\n"\
"vec4 sampleArray" #I "(uint index) {\n"\
"  return vec4(vec2(ssboArray" #I "[index]), 0, 1);\n"\
"}\n"
ARRAY_SAMPLE_SWITCH
#undef ARRAY_SAMPLE_STRING

    case ANARI_UINT32_VEC3:
#define ARRAY_SAMPLE_STRING(I)\
"layout(std430, binding = " #I ") buffer ssboBlock" #I " {  uint ssboArray" #I "[]; };\n"\
"vec4 sampleArray" #I "(uint index) {\n"\
"  return vec4(ssboArray" #I "[3u*index], ssboArray" #I "[3u*index+1u], ssboArray" #I "[3u*index+2u], 1);\n"\
"}\n"
ARRAY_SAMPLE_SWITCH
#undef ARRAY_SAMPLE_STRING

    case ANARI_UINT32_VEC4:
#define ARRAY_SAMPLE_STRING(I)\
"layout(std430, binding = " #I ") buffer ssboBlock" #I " {  uvec4 ssboArray" #I "[]; };\n"\
"vec4 sampleArray" #I "(uint index) {\n"\
"  return vec4(ssboArray" #I "[index]);\n"\
"}\n"
ARRAY_SAMPLE_SWITCH
#undef ARRAY_SAMPLE_STRING



    case ANARI_FLOAT32:
#define ARRAY_SAMPLE_STRING(I)\
"layout(std430, binding = " #I ") buffer ssboBlock" #I " {  float ssboArray" #I "[]; };\n"\
"vec4 sampleArray" #I "(uint index) {\n"\
"  return vec4(ssboArray" #I "[index], 0, 0, 1);\n"\
"}\n"
ARRAY_SAMPLE_SWITCH
#undef ARRAY_SAMPLE_STRING

    case ANARI_FLOAT32_VEC2:
#define ARRAY_SAMPLE_STRING(I)\
"layout(std430, binding = " #I ") buffer ssboBlock" #I " {  vec2 ssboArray" #I "[]; };\n"\
"vec4 sampleArray" #I "(uint index) {\n"\
"  return vec4(ssboArray" #I "[index], 0, 1);\n"\
"}\n"
ARRAY_SAMPLE_SWITCH
#undef ARRAY_SAMPLE_STRING

    case ANARI_FLOAT32_VEC3:
#define ARRAY_SAMPLE_STRING(I)\
"layout(std430, binding = " #I ") buffer ssboBlock" #I " {  float ssboArray" #I "[]; };\n"\
"vec4 sampleArray" #I "(uint index) {\n"\
"  return vec4(ssboArray" #I "[3u*index], ssboArray" #I "[3u*index+1u], ssboArray" #I "[3u*index+2u], 1);\n"\
"}\n"
ARRAY_SAMPLE_SWITCH
#undef ARRAY_SAMPLE_STRING

    case ANARI_FLOAT32_VEC4:
#define ARRAY_SAMPLE_STRING(I)\
"layout(std430, binding = " #I ") buffer ssboBlock" #I " {  vec4 ssboArray" #I "[]; };\n"\
"vec4 sampleArray" #I "(uint index) {\n"\
"  return ssboArray" #I "[index];\n"\
"}\n"
ARRAY_SAMPLE_SWITCH
#undef ARRAY_SAMPLE_STRING

    default: return "";
  }
}
//clang-format on

}