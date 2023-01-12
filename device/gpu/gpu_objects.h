/*
 * Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "anari/anari_enums.h"

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

struct AttributePtr
{
  ANARIDataType type;
  int numChannels;
  void *data;
};

struct TriangleGeometryData
{
  const uvec3 *indices;

  const vec3 *vertices;
  AttributePtr vertexAttr[5]; // attribute0-3 + color
  const vec3 *vertexNormals;

  const uvec3 *vertexNormalIndices;
  const uvec3 *vertexAttrIndices[5];
};

struct QuadGeometryData
{
  const uvec3 *indices;

  const vec3 *vertices;
  AttributePtr vertexAttr[5]; // attribute0-3 + color
  const vec3 *vertexNormals;

  const uvec3 *vertexNormalIndices;
  const uvec3 *vertexAttrIndices[5];
};

struct CylinderGeometryData
{
  const uvec2 *indices;
  const vec3 *vertices;
  AttributePtr vertexAttr[5]; // attribute0-3 + color
  const float *radii;
  float radius;
  bool caps;
};

struct CurveGeometryData
{
  const uint32_t *indices;
  const vec3 *vertices;
  AttributePtr vertexAttr[5]; // attribute0-3 + color
  const float *radii;
};

struct ConeGeometryData
{
  const uvec3 *indices; // actually triangles
  const vec3 *vertices;
  AttributePtr vertexAttr[5]; // attribute0-3 + color
  uint8_t trianglesPerCone;
};

struct SphereGeometryData
{
  const uint32_t *indices;
  vec3 *centers;
  AttributePtr vertexAttr[5]; // attribute0-3 + color
  const float *radii;
  float radius;
};

struct GeometryGPUData
{
  GeometryType type{GeometryType::UNKNOWN};
  AttributePtr attr[5]; // attribute0-3 + color
  const uint32 *primID{nullptr};
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
  PRIMITIVE,
  COLOR_MAP,
  UNKNOWN
};

struct Image1DData
{
  cudaTextureObject_t texobj;
};

struct Image2DData
{
  cudaTextureObject_t texobj;
};

struct PrimIDSamplerData
{
  AttributePtr attr;
};

struct ColorMapGPUData
{
  cudaTextureObject_t tfTex{};
  box1 valueRange{};
};

struct SamplerGPUData
{
  SamplerType type{SamplerType::UNKNOWN};
  int attribute{-1};
  union
  {
    ColorMapGPUData colormap{};
    Image1DData image1D;
    Image2DData image2D;
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

template <typename T>
struct MaterialParameter
{
  MaterialParameterType type{MaterialParameterType::UNKNOWN};
  union
  {
    T value;
    DeviceObjectIndex sampler;
  };

  MaterialParameter() = default;
  MaterialParameter(T v)
  {
    type = MaterialParameterType::VALUE;
    value = v;
  }
};

struct MaterialGPUData
{
  MaterialParameter<vec3> baseColor{vec3(1.f)};
  MaterialParameter<float> metalness{1.f};
  MaterialParameter<vec3> emissive{vec3(0.f)};
  MaterialParameter<float> roughness{1.f};
  MaterialParameter<float> transmissiveness{0.f};
  MaterialParameter<float> opacity{1.f};
};

struct MaterialValues
{
  vec3 baseColor;
  float metalness;
  vec3 emissive;
  float roughness;
  float transmissiveness;
  float opacity;
};

// Surface //

struct SurfaceGPUData
{
  DeviceObjectIndex material;
  DeviceObjectIndex geometry;
};

// Spatial Fields //

enum class SpatialFieldType
{
  STRUCTURED_REGULAR,
  UNKNOWN
};

struct StructuredRegularData
{
  cudaTextureObject_t texObj{};
  vec3 origin;
  vec3 invSpacing;
};

struct UniformGridData
{
  ivec3 dims;
  box3 worldBounds;
  box1 *valueRanges; // min/max ranges
  float *maxOpacities; // used for adaptive sampling/space skipping
};

struct SpatialFieldGPUData
{
  SpatialFieldType type{SpatialFieldType::UNKNOWN};
  union
  {
    StructuredRegularData structuredRegular{};
  } data;
  UniformGridData grid;
};

// Volume //

enum class VolumeType
{
  SCIVIS,
  UNKNOWN
};

struct ScivisVolumeGPUData
{
  DeviceObjectIndex field;
  cudaTextureObject_t tfTex{};
  box1 valueRange;
  float densityScale;
};

struct VolumeGPUData
{
  VolumeType type{VolumeType::SCIVIS};
  union
  {
    ScivisVolumeGPUData scivis{};
  } data;
  float stepSize;
  box3 bounds;
};

// Lights //

enum class LightType
{
  AMBIENT,
  DIRECTIONAL,
  POINT,
  UNKNOWN
};

struct AmbientLightGPUData
{
  float intensity;
  float distance;
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

struct LightGPUData
{
  LightType type{LightType::UNKNOWN};
  vec3 color;
  union
  {
    AmbientLightGPUData ambient;
    DirectionalLightGPUData distant;
    PointLightGPUData point;
  };
};

// Instance //

struct InstanceSurfaceGPUData
{
  const DeviceObjectIndex *surfaces;
};

struct InstanceVolumeGPUData
{
  const DeviceObjectIndex *volumes;
};

struct InstanceLightGPUData
{
  const DeviceObjectIndex *indices;
  size_t numLights;
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

struct SciVisRendererGPUData
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
  SciVisRendererGPUData scivis;
};

struct RendererGPUData
{
  RendererParametersGPUData params;
  glm::vec4 bgColor;
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
  uint3 launchIdx;
  glm::uvec2 pixel;
  RandState rs;
  const FrameGPUData *frameData;
};

struct Ray
{
  vec3 org;
  vec3 dir;
  box1 t{0.f, std::numeric_limits<float>::max()};
};

struct SurfaceHit
{
  bool foundHit;
  float t;
  vec3 hitpoint;
  vec3 Ng;
  vec3 Ns;
  vec3 uvw;
  uint32_t primID;
  float epsilon;
  const GeometryGPUData *geometry{nullptr};
  const MaterialGPUData *material{nullptr};
};

struct VolumeHit
{
  bool foundHit;
  Ray localRay;
  uint32_t volID{~0u};
  uint32_t instID{~0u};
  const VolumeGPUData *volumeData{nullptr};
};

using Hit = SurfaceHit;

} // namespace visrtx
