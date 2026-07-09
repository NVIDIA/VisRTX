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

#include "gpu/gpu_math.h"
#include "gpu/sbt.h"

// optix
#include <optix.h>
// cuda runtime — cudaTextureObject_t and friends
#include <cuda_runtime.h>
// PCG RNG, see gpu/pcg.h
#include "gpu/pcg.h"
// anari
#include <anari/anari_cpp.hpp>
#include <glm/ext/matrix_float3x4.hpp>
#include <glm/ext/vector_uint2.hpp>
// nanovdb
#include <nanovdb/NanoVDB.h>

// cuda half precision
#ifdef VISRTX_USE_NEURAL
#include <cuda_fp16.h>
#endif

#define DECLARE_FRAME_DATA(n)                                                  \
  extern "C" {                                                                 \
  __constant__ FrameGPUData n;                                                 \
  }

namespace visrtx {

using RandState = PCGState;
using DeviceObjectIndex = int32_t;

enum class MaterialAttribute : uint8_t
{
  ATTRIB_0,
  ATTRIB_1,
  ATTRIB_2,
  ATTRIB_3,
  COLOR,
  OBJECT_POSITION,
  OBJECT_NORMAL,
  WORLD_POSITION,
  WORLD_NORMAL,
  UNKNOWN
};

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
  CameraType type;
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
  SDF,
  NEURAL,
  ISOSURFACE,
  UNKNOWN
};

enum class SDFType : uint8_t
{
  SPHERE = 0,
  PILL = 1,
  CONE_PILL = 2,
  CONE_PILL_SIGMOID = 3,
  CONE = 4,
  TORUS = 5,
  CUT_SPHERE = 6,
  VESICA = 7,
  ELLIPSOID = 8
};

// Byte-exact layout shared between C++ host and CUDA device code.
// sizeof(SDFPrimitive) == 72, verified by static_assert in SDF.cpp.
struct SDFPrimitive
{
  uint64_t userData{0};
  vec3 userParams{0.f}; // x=displacement amplitude, y=frequency, z=unused
  vec3 p0{0.f};
  vec3 p1{0.f};
  float r0{-1.f};
  float r1{-1.f};
  uint32_t _pad{0};
  uint64_t neighboursIndex{0};
  uint8_t numNeighbours{0};
  uint8_t type{0};
  uint8_t _pad2[6]{};
};

struct SDFGeometryData
{
  const SDFPrimitive *geometries;
  const uint64_t *neighbours;
  uint32_t numGeometries;
  float epsilon;
  uint32_t nbMarchIterations;
  float blendFactor;
  float blendLerpFactor;
  float omega;
  float distanceFromCamera;
  float noiseFactor; // [0,1]: 0=no noise, 1=max organic surface noise
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
  const vec4 *vertexTangents;
  const vec4 *vertexTangentsFV;
  bool cullBackfaces;
  // Geometry Light sampling: normalized cumulative object-space area CDF over
  // primitives (null unless this geometry backs an Emissive Surface).
  const float *primAreaCdf;
  uint32_t numPrimitives;
  float totalArea; // object-space
};

struct QuadGeometryData
{
  const uvec3 *indices;
  const vec3 *vertices;
  AttributeDataSet vertexAttr;
  const vec3 *vertexNormals;
  bool cullBackfaces;
};

// Cap enablement is a per-endpoint bitmask (bit0 = first/p0 end,
// bit1 = second/p1 end; see CapBit in intersectPrimitives.h). vertexCaps, when
// non-null, overrides defaultCapFlags per endpoint: element!=0 enables that
// endpoint's cap (spec vertex.cap: 0=no cap, 1=flat).
struct CylinderGeometryData
{
  const uvec2 *indices;
  const vec3 *vertices;
  AttributeDataSet vertexAttr;
  const float *radii;
  float radius;
  uint8_t defaultCapFlags;
  const uint8_t *vertexCaps;
};

struct ConeGeometryData
{
  const uvec2 *indices;
  const vec3 *vertices;
  const float *radii;
  AttributeDataSet vertexAttr;
  uint8_t defaultCapFlags;
  const uint8_t *vertexCaps;
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

// Value-only bisection steps used to refine an isosurface crossing within one
// march step. The hit is localized to the final bracket, stepSize/2^iters; the
// secondary-ray offset (populateSurfaceHit) is derived from this so AO/shadow
// rays clear the marched surface instead of self-occluding (acne).
constexpr int kIsosurfaceBisectionIters = 6;

struct IsosurfaceGeometryData
{
  DeviceObjectIndex field; // index into frameData.registry.fields
  const float *isovalues; // device array, length numIsovalues
  uint32_t numIsovalues; // <= 128 (hitKind carrier limit)
  const box3 *brickBounds; // per-GAS-primitive coarse brick box, object space
  float stepSize; // march step, captured from field->stepSize()
};

#ifdef VISRTX_USE_NEURAL

constexpr uint32_t NEURAL_NB_MAX_LAYERS = 5;
constexpr uint32_t NEURAL_LAYER_SIZE = 128;
struct NeuralGeometryData
{
  __half *weights[NEURAL_NB_MAX_LAYERS]; // Array of weight matrices
  __half *biases[NEURAL_NB_MAX_LAYERS]; // Array of bias vectors
  uint32_t nb_layers;
  vec3 boundMin;
  vec3 boundMax;
  float threshold;
};
#endif

struct GeometryGPUData
{
  GeometryType type{GeometryType::UNKNOWN};
  AttributeDataSet attr;
  AttributeDataSetUniform attrUniform;
  const uint32_t *primitiveId;
  // Object-space coordinate magnitude the analytic intersectors' arithmetic
  // runs at (max |AABB corner| over the geometry's primitives); 0 when unused.
  // Floors hit.epsilon so secondary-ray offsets clear the solve's fp noise
  // band even where the hitpoint's own coordinates are small (populateHit.h).
  float epsilonScale{0.f};
  union
  {
    TriangleGeometryData tri{};
    QuadGeometryData quad;
    CylinderGeometryData cylinder;
    CurveGeometryData curve;
    ConeGeometryData cone;
    SphereGeometryData sphere;
    SDFGeometryData sdf;
    IsosurfaceGeometryData isosurface;
#ifdef VISRTX_USE_NEURAL
    NeuralGeometryData neural;
#endif
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
  MaterialAttribute attribute{MaterialAttribute::UNKNOWN};
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

enum class MaterialParameterType : uint8_t
{
  VALUE,
  ATTRIBUTE,
  SAMPLER,
  UNKNOWN
};

struct MaterialParameter
{
  MaterialParameterType type{MaterialParameterType::UNKNOWN};
  union
  {
    vec4 value;
    MaterialAttribute attribute;
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
    DeviceObjectIndex normalSampler;
    MaterialParameter emissive;
    MaterialParameter transmission;

    float ior;

    // KHR_materials_* extensions
    DeviceObjectIndex occlusionSampler;
    MaterialParameter specular;
    MaterialParameter specularColor;
    uint32_t useSpecular;
    MaterialParameter clearcoat;
    MaterialParameter clearcoatRoughness;
    DeviceObjectIndex clearcoatNormalSampler;
    MaterialParameter thickness;
    float attenuationDistance;
    vec3 attenuationColor;
    MaterialParameter sheenColor;
    MaterialParameter sheenRoughness;
    MaterialParameter iridescence;
    float iridescenceIor;
    MaterialParameter iridescenceThickness;
  };

  struct MDL
  {
    const char *argBlock;
    uint32_t numSamplers;
    // Should be sized according to MDL's execution context
    // configuration. See MDLCompiler.cpp.
    DeviceObjectIndex samplers[32];
  };

  uint32_t callableBaseIndex{~0u};

  // Static shadow attenuator: OPAQUE alphaMode, opacity + alpha channels
  // constant 1.0, transmission constant 0.0, no opacity/transmission
  // sampler. Shadow anyhit short-circuits via optixTerminateRay.
  bool isFullyOpaque{false};

  // Emission is a nonzero constant (not attribute/sampler bound). Gates the
  // hit-side Geometry Light MIS weight without a per-surface lookup; kept in
  // sync by the material's own commit, so it is never stale.
  bool emissionIsConstant{false};

  union MaterialData
  {
    Matte matte;
    PhysicallyBased physicallyBased;
    MDL mdl;
  } materialData = {};

  MaterialGPUData() = default;
};

// Surface //

struct SurfaceGPUData
{
  DeviceObjectIndex material;
  DeviceObjectIndex geometry;
  uint32_t id;
};

// Spatial Fields //

struct UniformGridData
{
  ivec3 dims;
  box3 objectBounds;
  box1 *valueRanges; // min/max ranges
  float2 *opacityBounds; // Per-cell α bounds (.x = min, .y = max) over the TF
                         // lookup.
};

enum class SpatialFieldFilter : uint8_t
{
  Nearest = 0,
  Linear = 1
};

struct StructuredRegularData
{
  cudaTextureObject_t texObj;
  vec3 origin;
  vec3 invSpacing;
  ivec3 dims; // voxel (texel) counts — drives the isosurface voxel-DDA
  bool cellCentered;
  SpatialFieldFilter filter;

  StructuredRegularData() = default;
};

struct NVdbRegularData
{
  nanovdb::GridType gridType;
  const void *gridData;
  bool cellCentered;
  SpatialFieldFilter filter;
  // Host-precomputed 1 / (2 · voxelSize). Keeps Vec3d off the device init.
  vec3 invTwoVoxelSize;

  NVdbRegularData() = default;
};

struct StructuredRectilinearData
{
  cudaTextureObject_t texObj;
  vec3 dims;
  bool cellCentered;
  cudaTextureObject_t axisLUT[3]; // object -> index (sampling)
  cudaTextureObject_t invAxisLUT[3]; // index -> object (isosurface voxel-DDA)
  vec3 axisBoundsMin;
  vec3 axisBoundsMax;
  SpatialFieldFilter filter;

  StructuredRectilinearData() = default;
};

struct NVdbRectilinearData
{
  nanovdb::GridType gridType;
  const void *gridData;
  bool cellCentered;
  SpatialFieldFilter filter;
  cudaTextureObject_t
      axisLUT[3]; // normalized uniform index -> rectilinear index
  cudaTextureObject_t invAxisLUT[3]; // inverse (isosurface voxel-DDA)
  vec3 invAvgVoxelSize;

  NVdbRectilinearData() = default;
};

struct CustomFieldData
{
  CustomFieldData() = default;
  uint32_t subType;

  // Generic storage for field-specific data
  // External projects can use this to store
  // their custom field parameters and reinterpret_cast as needed
  // Aligned to 8 bytes to support cudaTextureObject_t and other 64-bit types
  alignas(8) uint8_t fieldData[256];
};

struct SpatialFieldGPUData
{
  SbtCallableEntryPoints samplerCallableIndex{SbtCallableEntryPoints::Invalid};
  union
  {
    StructuredRegularData structuredRegular;
    NVdbRegularData nvdbRegular;
    StructuredRectilinearData structuredRectilinear;
    NVdbRectilinearData nvdbRectilinear;
    CustomFieldData custom;
  } data;
  UniformGridData grid;
  box3 roi;
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
  float oneOverUnitDistance;
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
  SPHERE,
  RECT,
  SPOT,
  RING,
  HDRI,
  GEOMETRY,
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

struct SphereLightGPUData
{
  vec3 position;
  float intensity;
  float radius;
  float oneOverArea;
};

struct RectLightGPUData
{
  vec3 position;
  vec3 edge1;
  vec3 edge2;
  float intensity;
  struct
  {
    unsigned int front : 1;
    unsigned int back : 1;
  } side;
  float oneOverArea;
};

struct SpotLightGPUData
{
  vec3 position;
  vec3 direction;
  float cosOuterAngle;
  float cosInnerAngle;
  float intensity;
};

struct RingLightGPUData
{
  vec3 position;
  vec3 direction;
  float cosOuterAngle;
  float cosInnerAngle;
  float radius;
  float innerRadius;
  float intensity;
  float oneOverArea;
};

struct HDRILightGPUData
{
  mat3 xfm;
  uvec2 size;
  cudaTextureObject_t radiance;
  const float *marginalCDF;
  const float *conditionalCDF;
  float scale;
  bool visible;
  float pdfWeight;
#ifdef VISRTX_ENABLE_HDRI_SAMPLING_DEBUG
  uint32_t *samples; // pixelmap of sample counts
#endif
};

// A light synthesized from an Emissive Surface. References the surface's
// geometry (for the per-primitive area CDF and vertices) and carries the baked
// constant radiance; the object-space total area sizes its Pick Power.
struct GeometryLightGPUData
{
  DeviceObjectIndex geometryIndex; // index into registry.geometries[]
  vec3 radiance; // baked constant emitted radiance (Stage 1)
  float area; // object-space total area
};

struct LightGPUData
{
  LightType type{LightType::UNKNOWN};
  vec3 color;
  union
  {
    DirectionalLightGPUData distant;
    PointLightGPUData point;
    SphereLightGPUData sphere;
    RectLightGPUData rect;
    SpotLightGPUData spot;
    RingLightGPUData ring;
    HDRILightGPUData hdri;
    GeometryLightGPUData geometry;
  };
};

// Instance //

struct InstanceSurfaceGPUData
{
  // Pre-computed at instance commit (World.cpp's buildInstance*GPUData).
  mat3x4 objectToWorld;
  mat3x4 worldToObject;

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
  mat3x4 objectToWorld;
  mat3x4 worldToObject;

  const DeviceObjectIndex *volumes;
  uint32_t id;
};

struct InstanceLightGPUData
{
  DeviceObjectIndex lightIndex; // Index into registry.lights[]
  mat4 xfm; // Transform for this light instance
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

  const InstanceLightGPUData *hdriLightInstances;
  size_t numHdriLightInstances;

  // Power-proportional Light Pick (built in World::buildInstanceLightGPUData).
  // Normalized cumulative Pick Power over lightInstances (length
  // numLightInstances, last entry == 1); pick a slot with inverseSampleCDF. The
  // slot's discrete pick probability is lightPickCdf[i]-lightPickCdf[i-1].
  const float *lightPickCdf;
  float totalLightPower; // sum of un-normalized instance Pick Powers
  float hdriPower; // subset sum over HDRI instances (env-MIS pick probability)
  float sceneRadius; // bounding-sphere radius, for the ambient term's power
};

// Renderer //

struct DebugRendererGPUData
{
  int method;
};

struct FastRendererGPUData
{
  int aoSamples;
  float aoBlend;
};

struct QualityRendererGPUData
{
  int maxRayDepth;
  int maxTransparencyDepth;
};

struct InteractiveRendererGPUData
{
  float lightFalloff;
  int aoSamples;
  vec3 aoColor;
  float aoIntensity;
  float inverseVolumeSamplingRateShadows;
};

union RendererParametersGPUData
{
  DebugRendererGPUData debug;
  FastRendererGPUData fast;
  QualityRendererGPUData quality;
  InteractiveRendererGPUData interactive;
};

enum class BackgroundMode
{
  COLOR,
  IMAGE
};

// Per-sample firefly suppression strategy applied during accumulation.
enum class FireflyFilterMode
{
  NONE, // accumulate raw radiance (unbiased)
  TONEMAP, // reversible Reinhard round-trip (legacy; dims highlights)
  CLAMP, // per-pixel Welford luminance clamp (energy-preserving)
  TRIM // adaptive upper-trimmed mean (consistent; near-unbiased at high spp)
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
  bool premultiplyBackground;
  FireflyFilterMode
      fireflyFilterMode; // per-sample outlier suppression strategy
  float fireflyFilterSigma; // CLAMP/TRIM: k in threshold = mean + k*stddev
  int fireflyFilterWarmup; // CLAMP mode: samples before the Welford cap engages
  int fireflyFilterTrim; // TRIM mode: count of brightest samples
                         // tracked/trimmed
  glm::vec4 cutPlane; // cutting plane (nx,ny,nz,d); disabled when all zero (GPU
                      // default)
};

// Frame //

enum class FrameFormat
{
  FLOAT,
  UINT,
  SRGB,
  UNKNOWN
};

// Per-pixel running Welford statistics for firefly suppression, tracked per RGB
// channel so a single-channel (chromatic) outlier is caught even when its
// luminance is unremarkable. `n` is the shared sample count — kept here because
// checkerboarding makes frameID a poor proxy for "how many samples this pixel
// has seen". CLAMP uses all three channels; TRIM uses only the luminance
// Welford in channel x and reads n as its sample divisor.
struct PixelLumStats
{
  glm::vec3 mean; // per-channel running mean
  glm::vec3 m2; // per-channel sum of squared deltas
  float n; // sample count (shared across channels and with TRIM)
};

struct FrameBuffers
{
  glm::vec4 *colorAccumulation;
  PixelLumStats *lumStats;
  // TRIM mode: the `trim` brightest samples seen per pixel, laid out
  // [pixel*trim + slot] as (rgb in xyz, luminance in w; w < 0 marks an empty
  // slot). At resolve the trimmed mean removes the outliers among these from
  // the running colorAccumulation sum, so it needs only O(trim) memory per
  // pixel. The sample count lives in lumStats->n.
  glm::vec4 *trimTopK;
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
  glm::uvec2 size;
  glm::vec2 invSize;
};

struct FrameGPUData
{
  FramebufferGPUData fb;
  RendererGPUData renderer;
  WorldGPUData world;
  CameraGPUData camera;

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
  // Adaptive shadow ratio-tracking knob. Set by the raygen before each
  // shadow trace to (max pre-attenuation contribution) / RR_BASE, capped
  // at 1.0. Smaller values raise the RR threshold inside
  // applyShadowRussianRoulette so dim-contribution shadow rays terminate
  // sooner. Default 1.0 = full-precision RR (current behaviour).
  float shadowContribWeight;
};

} // namespace visrtx
