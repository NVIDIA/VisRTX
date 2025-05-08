/*
 * Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gpu/gpu_math.h"

// optix
#include <optix.h>
// curand
#include <curand_kernel.h>
// anari
#include <anari/anari_cpp.hpp>
#include <glm/ext/vector_uint2.hpp>
#include <glm/ext/matrix_float3x4.hpp>
// nanovdb
#include <nanovdb/NanoVDB.h>

#define DECLARE_FRAME_DATA(n)                                                  \
  extern "C" {                                                                 \
  __constant__ FrameGPUData n;                                                 \
  }

namespace visrtx {

using RandState = curandStatePhilox4_32_10_t;
using DeviceObjectIndex = int32_t;

///////////////////////////////////////////////////////////////////////////////
// Objects ////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// Cameras //

enum class CameraType
{
  PERSPECTIVE,
  ORTHOGRAPHIC,
  UNKNOWN
};

struct PerspectiveCameraGPUData
{
  vec3 dir_du;
  vec3 dir_dv;
  vec3 dir_00;
  float scaledAperture;
  float aspect;
};

struct OrthographicCameraGPUData
{
  vec3 pos_du;
  vec3 pos_dv;
  vec3 pos_00;
};

struct CameraGPUData
{
  CameraType type{CameraType::UNKNOWN};
  vec4 region;
  vec3 pos;
  vec3 dir;
  vec3 up;
  union
  {
    PerspectiveCameraGPUData perspective;
    OrthographicCameraGPUData orthographic;
  };
};

// Geometry //

enum class GeometryType
{
  TRIANGLE,
  QUAD,
  CYLINDER,
  CURVE,
  CONE,
  SPHERE,
  UNKNOWN
};

struct AttributeData
{
  ANARIDataType type;
  int numChannels;
  const void *data;
};

using AttributeDataSet = AttributeData[5]; // attribute0-3 + color
using AttributeDataSetUniform = vec4[5]; // attribute0-3 + color

struct TriangleGeometryData
{
  const uvec3 *indices;
  const vec3 *vertices;
  AttributeDataSet vertexAttr;
  AttributeDataSet vertexAttrFV;
  const vec3 *vertexNormals;
  const vec3 *vertexNormalsFV;
  bool cullBackfaces;
};

struct QuadGeometryData
{
  const uvec3 *indices;
  const vec3 *vertices;
  AttributeDataSet vertexAttr;
  const vec3 *vertexNormals;
  bool cullBackfaces;
};

struct CylinderGeometryData
{
  const uvec2 *indices;
  const vec3 *vertices;
  AttributeDataSet vertexAttr;
  const float *radii;
  float radius;
  bool caps;
};

struct ConeGeometryData
{
  const uvec2 *indices;
  const vec3 *vertices;
  const float *radii;
  AttributeDataSet vertexAttr;
};

struct CurveGeometryData
{
  const uint32_t *indices;
  const vec3 *vertices;
  AttributeDataSet vertexAttr;
  const float *radii;
};

struct SphereGeometryData
{
  const uint32_t *indices;
  const vec3 *centers;
  AttributeDataSet vertexAttr;
  const float *radii;
  float radius;
};

struct GeometryGPUData
{
  GeometryType type{GeometryType::UNKNOWN};
  AttributeDataSet attr;
  AttributeDataSetUniform attrUniform;
  const uint32_t *primitiveId;
  union
  {
    TriangleGeometryData tri{};
    QuadGeometryData quad;
    CylinderGeometryData cylinder;
    CurveGeometryData curve;
    ConeGeometryData cone;
    SphereGeometryData sphere;
  };
};

// Samplers //

enum class SamplerType
{
  TEXTURE1D,
  TEXTURE2D,
  TEXTURE3D,
  PRIMITIVE,
  TRANSFORM,
  UNKNOWN
};

struct Image1DData
{
  cudaTextureObject_t texobj;
  cudaTextureObject_t texelTexobj;
  uint32_t size;
  float invSize;
};

struct Image2DData
{
  cudaTextureObject_t texobj;
  cudaTextureObject_t texelTexobj;
  uvec2 size;
  vec2 invSize;
};

struct Image3DData
{
  cudaTextureObject_t texobj;
  cudaTextureObject_t texelTexobj;
  uvec3 size;
  vec3 invSize;
};

struct PrimIDSamplerData
{
  AttributeData attr;
  uint32_t offset;
};

struct SamplerGPUData
{
  SamplerType type{SamplerType::UNKNOWN};
  int attribute{-1};
  mat4 inTransform;
  vec4 inOffset;
  mat4 outTransform;
  vec4 outOffset;
  union
  {
    Image1DData image1D;
    Image2DData image2D;
    Image3DData image3D;
    PrimIDSamplerData primitive;
  };
};

// Material //

enum class MaterialParameterType
{
  VALUE,
  SAMPLER,
  ATTRIB_COLOR,
  ATTRIB_0,
  ATTRIB_1,
  ATTRIB_2,
  ATTRIB_3,
  WORLD_POSITION,
  WORLD_NORMAL,
  OBJECT_POSITION,
  OBJECT_NORMAL,
  UNKNOWN
};

struct MaterialParameter
{
  MaterialParameterType type{MaterialParameterType::UNKNOWN};
  union
  {
    vec4 value;
    DeviceObjectIndex sampler;
  };

  MaterialParameter() = default;
  MaterialParameter(vec4 v)
  {
    type = MaterialParameterType::VALUE;
    value = v;
  }
};

enum AlphaMode
{
  OPAQUE = 0,
  BLEND,
  MASK
};

constexpr int MV_BASE_COLOR = 0;
constexpr int MV_OPACITY = 1;
constexpr int MV_METALLIC = 2;
constexpr int MV_ROUGHNESS = 3;

enum class MaterialType
{
  UNKNOWN = -1,
  MATTE = 0, // Akin to callable id
  PHYSICALLYBASED,
  MDL,
};

struct MaterialGPUData
{
  struct Matte
  {
    MaterialParameter color;
    MaterialParameter opacity;
    float cutoff;
    AlphaMode alphaMode;
  };

  struct PhysicallyBased
  {
    MaterialParameter baseColor;
    MaterialParameter opacity;
    float cutoff;
    AlphaMode alphaMode;
    MaterialParameter metallic;
    MaterialParameter roughness;
    float ior;
  };

  struct MDL
  {
    const char *argBlock;
    uint32_t implementationIndex;
    uint32_t numSamplers;
    DeviceObjectIndex
        samplers[32]; // Should be sized according to MDL's execution context
                      // configuration. See MDLCompiler.cpp.
  };

  MaterialType materialType;

  union
  {
    Matte matte;
    PhysicallyBased physicallyBased;
    MDL mdl;
  };

  MaterialGPUData()
  {
    materialType = MaterialType::UNKNOWN;
    matte = {};
    physicallyBased = {};
    mdl = {};
  }
};

struct MaterialValues
{
  MaterialType materialType;
  vec3 baseColor;
  float opacity;
  float metallic;
  float roughness;
  float ior;
};

// Surface //

struct SurfaceGPUData
{
  DeviceObjectIndex material;
  DeviceObjectIndex geometry;
  uint32_t id;
};

// Spatial Fields //

enum class SpatialFieldType
{
  STRUCTURED_REGULAR,
  NANOVDB_REGULAR,
  UNKNOWN
};

struct UniformGridData
{
  ivec3 dims;
  box3 worldBounds;
  box1 *valueRanges; // min/max ranges
  float *maxOpacities; // used for adaptive sampling/space skipping
};

struct StructuredRegularData
{
  cudaTextureObject_t texObj;
  vec3 origin;
  vec3 spacing;
  vec3 invSpacing;
};

struct NVdbRegularData
{
  vec3 origin;
  vec3 voxelSize;
  nanovdb::GridType gridType;
  const void *gridData;
};

struct SpatialFieldGPUData
{
  SpatialFieldType type{SpatialFieldType::UNKNOWN};
  union
  {
    StructuredRegularData structuredRegular;
    NVdbRegularData nvdbRegular;
  } data;
  UniformGridData grid;
};

// Volume //

enum class VolumeType
{
  TF1D,
  UNKNOWN
};

struct TF1DVolumeGPUData
{
  DeviceObjectIndex field;
  cudaTextureObject_t tfTex{};
  box1 valueRange;
  float unitDistance;
  vec3 uniformColor;
  float uniformOpacity;
};

struct VolumeGPUData
{
  VolumeType type{VolumeType::TF1D};
  union
  {
    TF1DVolumeGPUData tf1d{};
  } data;
  float stepSize;
  box3 bounds;
  uint32_t id;
};

// Lights //

enum class LightType
{
  DIRECTIONAL,
  POINT,
  SPOT,
  HDRI,
  UNKNOWN
};

struct DirectionalLightGPUData
{
  vec3 direction;
  float irradiance;
};

struct PointLightGPUData
{
  vec3 position;
  float intensity;
};

struct SpotLightGPUData
{
  vec3 position;
  vec3 direction;
  float cosOuterAngle;
  float cosInnerAngle;
  float intensity;
};

struct HDRILightGPUData
{
  mat3 xfm;
  cudaTextureObject_t radiance;
  float scale;
  bool visible;
};

struct LightGPUData
{
  LightType type{LightType::UNKNOWN};
  vec3 color;
  union
  {
    DirectionalLightGPUData distant;
    PointLightGPUData point;
    SpotLightGPUData spot;
    HDRILightGPUData hdri;
  };
};

// Instance //

struct InstanceSurfaceGPUData
{
  const DeviceObjectIndex *surfaces;
  AttributeDataSet attrUniformArray;
  AttributeDataSetUniform attrUniform;
  bool attrUniformArrayPresent[5];
  bool attrUniformPresent[5];
  uint32_t id;
  uint32_t localArrayId; // offset inside an instance with a transform arrays
};

struct InstanceVolumeGPUData
{
  const DeviceObjectIndex *volumes;
  uint32_t id;
};

struct InstanceLightGPUData
{
  const DeviceObjectIndex *indices;
  size_t numLights;
  mat4 xfm;
};

// World //

struct WorldGPUData
{
  const InstanceSurfaceGPUData *surfaceInstances;
  size_t numSurfaceInstances;
  OptixTraversableHandle surfacesTraversable;

  const InstanceVolumeGPUData *volumeInstances;
  size_t numVolumeInstances;
  OptixTraversableHandle volumesTraversable;

  const InstanceLightGPUData *lightInstances;
  size_t numLightInstances;

  DeviceObjectIndex hdri;
};

// Renderer //

struct DebugRendererGPUData
{
  int method;
};

struct AORendererGPUData
{
  int aoSamples;
};

struct DPTRendererGPUData
{
  int maxDepth;
};

struct DirectLightRendererGPUData
{
  float lightFalloff;
  int aoSamples;
  vec3 aoColor;
  float aoIntensity;
};

union RendererParametersGPUData
{
  DebugRendererGPUData debug;
  AORendererGPUData ao;
  DPTRendererGPUData dpt;
  DirectLightRendererGPUData directLight;
};

enum class BackgroundMode
{
  COLOR,
  IMAGE
};

union RendererBackgroundGPUData
{
  glm::vec4 color;
  cudaTextureObject_t texobj;
};

struct RendererGPUData
{
  RendererParametersGPUData params;
  BackgroundMode backgroundMode;
  RendererBackgroundGPUData background;
  glm::vec3 ambientColor;
  int numIterations;
  float ambientIntensity;
  float inverseVolumeSamplingRate;
  float occlusionDistance;
  bool cullTriangleBF;
};

// Frame //

enum class FrameFormat
{
  FLOAT,
  UINT,
  SRGB,
  UNKNOWN
};

struct FrameBuffers
{
  glm::vec4 *colorAccumulation;
  glm::vec4 *outColorVec4;
  uint32_t *outColorUint;
  float *depth;
  uint32_t *primID;
  uint32_t *objID;
  uint32_t *instID;
  glm::vec3 *albedo;
  glm::vec3 *normal;
};

struct FramebufferGPUData
{
  FrameBuffers buffers;
  int frameID;
  int checkerboardID;
  float invFrameID;
  FrameFormat format;
  glm::uvec2 size;
  glm::vec2 invSize;
};

struct FrameGPUData
{
  FramebufferGPUData fb;
  RendererGPUData renderer;
  WorldGPUData world;
  CameraGPUData *camera;

  // Objects //

  struct ObjectRegistry
  {
    const SamplerGPUData *samplers;
    const GeometryGPUData *geometries;
    const MaterialGPUData *materials;
    const SurfaceGPUData *surfaces;
    const LightGPUData *lights;
    const SpatialFieldGPUData *fields;
    const VolumeGPUData *volumes;
  } registry;
};

///////////////////////////////////////////////////////////////////////////////
// Misc types /////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

struct ScreenSample
{
  glm::uvec2 pixel;
  glm::vec2 screen;
  mutable RandState rs;
  const FrameGPUData *frameData;
};

} // namespace visrtx
