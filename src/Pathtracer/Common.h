/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

 // Note: Always order struct fields by CUDA alignment restrictions (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#vector-types)
 // float4, float2, float3/float/int, short, byte   (buffer/program/sampler ids are integers)
 // Note: Pad struct to 16 bytes if used in array


// Tests for pathtracer correctness (enabling any of these limits the number of bounces!)
//#define TEST_DIRECT_ONLY  // Brute force pathtracing, illumination only from rays hitting light geometry (light evaluation only)
//#define TEST_NEE_ONLY     // Next event estimation only (light sampling only)
//#define TEST_MIS          // Weighted combination of direct + NEE using multiple importance sampling

//#define PRINT_PIXEL_X 700
//#define PRINT_PIXEL_Y 540


// Convenience 
#ifdef TEST_NEE_ONLY
#undef TEST_DIRECT_ONLY
#endif
#ifdef TEST_MIS
#undef TEST_DIRECT_ONLY
#undef TEST_NEE_ONLY
#endif


#define RT_FUNCTION __forceinline__ __device__
#define RT_USE_TEMPLATED_RTCALLABLEPROGRAM 1

typedef unsigned int uint;
typedef unsigned char uint8_t;
typedef unsigned int uint32_t;


#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <mi/neuraylib/target_code_types.h>

#include "../MDL/Config.h"
#include "Random.h"

#define OPTIX_VERSION_MAJOR (OPTIX_VERSION / 10000)
#define OPTIX_VERSION_MINOR ((OPTIX_VERSION % 10000) / 100)
#define OPTIX_VERSION_MICRO (OPTIX_VERSION % 100)


#define RADIANCE_RAY_TYPE   0
#define OCCLUSION_RAY_TYPE  1
#define PICK_RAY_TYPE       2

#define PERSPECTIVE_CAMERA   0
#define ORTHOGRAPHIC_CAMERA  1

#define PDF_DIRAC 0.0f // "symbolic" constant for the PDF of a Dirac distribution

typedef uint32_t MaterialId;
const MaterialId MATERIAL_NULL = 0;

const uint32_t BASIC_MATERIAL_BIT = (1u << 31);
const uint32_t MDL_MATERIAL_BIT = (1u << 30);
const uint32_t MATERIAL_PRIORITY_MASK = (255u << 22);
const uint32_t MATERIAL_INDEX_MASK = ~(BASIC_MATERIAL_BIT | MDL_MATERIAL_BIT | MATERIAL_PRIORITY_MASK);


struct BasicMaterialParameters
{
    optix::float3 diffuseColor;
    int diffuseTexture;

    optix::float3 specularColor;
    int specularTexture;
    
    float specularExponent;
    int specularExponentTexture;

    float opacity;
    int opacityTexture;
    
    optix::float3 transparencyFilterColor;

    int bumpMapTexture;

    optix::float3 emissiveColor;
    int emissiveTexture;
    float luminosity;
};


#ifdef __CUDACC__
typedef rtCallableProgramId<mi::neuraylib::Bsdf_init_function> MDL_BSDF_Init_Func;
typedef rtCallableProgramId<mi::neuraylib::Bsdf_sample_function> MDL_BSDF_Sample_Func;
typedef rtCallableProgramId<mi::neuraylib::Bsdf_evaluate_function> MDL_BSDF_Evaluate_Func;
typedef rtCallableProgramId<mi::neuraylib::Bsdf_pdf_function> MDL_BSDF_PDF_Func;
typedef rtCallableProgramId<mi::neuraylib::Material_expr_function> MDL_Expression_Func;
#else
typedef int MDL_BSDF_Init_Func;
typedef int MDL_BSDF_Sample_Func;
typedef int MDL_BSDF_Evaluate_Func;
typedef int MDL_BSDF_PDF_Func;
typedef int MDL_Expression_Func;
#endif

struct MDLMaterialParameters
{
    MDL_BSDF_Init_Func init;
    MDL_BSDF_Sample_Func sample;
    MDL_BSDF_Evaluate_Func evaluate;
    // MDL_BSDF_PDF_Func pdf; // currently not used    
    MDL_Expression_Func opacity;
	MDL_Expression_Func thinwalled;
	MDL_Expression_Func ior;
	MDL_Expression_Func absorption;

    int hasArgBlock;
    char argBlock[MDL_ARGUMENT_BLOCK_SIZE];

    // = 7 * 4 + 372 = 400 bytes 
};


#ifdef __CUDACC__
typedef rtBufferId<float> BufferFloat;
typedef rtBufferId<optix::float2> BufferFloat2;
typedef rtBufferId<optix::float4> BufferFloat4;
typedef rtBufferId<int2> BufferInt2;
typedef rtBufferId<uint> BufferUint;
typedef rtBufferId<uint4> BufferUint4;
typedef rtBufferId<optix::uchar4> BufferUchar4;
typedef rtBufferId<MaterialId> BufferMaterialId;

typedef rtBufferId<optix::float4, 2> Buffer2DFloat4;
typedef rtBufferId<optix::uchar4, 2> Buffer2DUchar4;
typedef rtBufferId<float, 2> Buffer2DFloat;

#else
typedef int BufferFloat;
typedef int BufferFloat2;
typedef int BufferFloat4;
typedef int BufferInt2;
typedef int BufferUint;
typedef int BufferUint4;
typedef int BufferUchar4;
typedef int BufferMaterialId;

typedef int Buffer2DFloat4;
typedef int Buffer2DUchar4;
typedef int Buffer2DFloat;

typedef int rtObject;
#endif


#pragma pack(push, 1)
struct LaunchParameters 
{
	float2 imageBegin;
	float2 imageSize; // imageEnd - imageBegin

    int width;
    int height;

    Buffer2DFloat4 accumulationBuffer;
    Buffer2DFloat4 frameBuffer;
    Buffer2DUchar4 ucharFrameBuffer;
    Buffer2DFloat depthBuffer;

    int cameraType;
    float3 pos;
    float3 U;
    float3 V;
    float3 W;	
    float focalDistance;
    float apertureRadius;
    float orthoWidth;
    float orthoHeight;

    int frameNumber;
    int useAIDenoiser;

    int writeFrameBuffer;
    int writeUcharFrameBuffer;
    float clipMin;
    float clipMax;
    float clipDiv;

    float occlusionEpsilon;
    float alphaCutoff;
    int numBouncesMin;
    int numBouncesMax;
    int writeBackground;
    float fireflyClampingDirect;
    float fireflyClampingIndirect;    
    int sampleAllLights;
    int numLightsDirect;
    int numLightsMiss;

    int toneMapping;
    float3 colorBalance;
    float invGamma;
    float invWhitePoint;
    float burnHighlights;
    float crushBlacks;
    float saturation;
};
#pragma pack(pop)



struct PathtracePRD
{
    optix::float4 color;

    optix::float2 texCoord;    

    optix::float3 normal;
    optix::float3 geometricNormal;
    MaterialId material;
    float tHit;
    float animationFactor;

    optix::float3 lightEdf;
    float lightPdf;
    float lastLightPdfFactor; // = 1 / numLightsActiveForNEE
    float lastPdf; // last material BSDF sample pdf    

    optix::float3 radiance;
    optix::float3 alpha;    
    int depth;
    int numCutoutOpacityHits;
    RandState* randState;

    bool light;
	bool frontFacing;
};

struct OcclusionPRD
{
    optix::float3 occlusion;
    RandState* randState;
};

//struct MaterialStackElement
//{
//	MaterialId material; // default: MATERIAL_NULL (air)
//	optix::float3 ior; // default: 1
//
//	bool topmost; // default: true
//	bool oddParity; // default: true
//};
//
//const uint32_t MATERIAL_STACK_SIZE = 8;

struct VolumeStackElement
{
	MaterialId material;
	optix::float3 ior;
	optix::float3 absorptionCoefficient;
};

const uint32_t VOLUME_MAX_STACK_SIZE = 4;





// Helper functions.
#ifdef __CUDACC__
RT_FUNCTION optix::float3 logf(const optix::float3& v)
{
    return optix::make_float3(::logf(v.x), ::logf(v.y), ::logf(v.z));
}

RT_FUNCTION optix::float2 floorf(const optix::float2& v)
{
    return optix::make_float2(::floorf(v.x), ::floorf(v.y));
}

RT_FUNCTION optix::float3 floorf(const optix::float3& v)
{
    return optix::make_float3(::floorf(v.x), ::floorf(v.y), ::floorf(v.z));
}

RT_FUNCTION optix::float3 ceilf(const optix::float3& v)
{
    return optix::make_float3(::ceilf(v.x), ::ceilf(v.y), ::ceilf(v.z));
}

RT_FUNCTION optix::float3 powf(const optix::float3& v, const float e)
{
    return optix::make_float3(::powf(v.x, e), ::powf(v.y, e), ::powf(v.z, e));
}

RT_FUNCTION optix::float4 powf(const optix::float4& v, const float e)
{
    return optix::make_float4(::powf(v.x, e), ::powf(v.y, e), ::powf(v.z, e), ::powf(v.w, e));
}


RT_FUNCTION optix::float2 fminf(const optix::float2& v, const float m)
{
    return optix::make_float2(::fminf(v.x, m), ::fminf(v.y, m));
}
RT_FUNCTION optix::float3 fminf(const optix::float3& v, const float m)
{
    return optix::make_float3(::fminf(v.x, m), ::fminf(v.y, m), ::fminf(v.z, m));
}
RT_FUNCTION optix::float4 fminf(const optix::float4& v, const float m)
{
    return optix::make_float4(::fminf(v.x, m), ::fminf(v.y, m), ::fminf(v.z, m), ::fminf(v.w, m));
}

RT_FUNCTION optix::float2 fminf(const float m, const optix::float2& v)
{
    return optix::make_float2(::fminf(m, v.x), ::fminf(m, v.y));
}
RT_FUNCTION optix::float3 fminf(const float m, const optix::float3& v)
{
    return optix::make_float3(::fminf(m, v.x), ::fminf(m, v.y), ::fminf(m, v.z));
}
RT_FUNCTION optix::float4 fminf(const float m, const optix::float4& v)
{
    return optix::make_float4(::fminf(m, v.x), ::fminf(m, v.y), ::fminf(m, v.z), ::fminf(m, v.w));
}


RT_FUNCTION optix::float2 fmaxf(const optix::float2& v, const float m)
{
    return optix::make_float2(::fmaxf(v.x, m), ::fmaxf(v.y, m));
}
RT_FUNCTION optix::float3 fmaxf(const optix::float3& v, const float m)
{
    return optix::make_float3(::fmaxf(v.x, m), ::fmaxf(v.y, m), ::fmaxf(v.z, m));
}
RT_FUNCTION optix::float4 fmaxf(const optix::float4& v, const float m)
{
    return optix::make_float4(::fmaxf(v.x, m), ::fmaxf(v.y, m), ::fmaxf(v.z, m), ::fmaxf(v.w, m));
}

RT_FUNCTION optix::float2 fmaxf(const float m, const optix::float2& v)
{
    return optix::make_float2(::fmaxf(m, v.x), ::fmaxf(m, v.y));
}
RT_FUNCTION optix::float3 fmaxf(const float m, const optix::float3& v)
{
    return optix::make_float3(::fmaxf(m, v.x), ::fmaxf(m, v.y), ::fmaxf(m, v.z));
}
RT_FUNCTION optix::float4 fmaxf(const float m, const optix::float4& v)
{
    return optix::make_float4(::fmaxf(m, v.x), ::fmaxf(m, v.y), ::fmaxf(m, v.z), ::fmaxf(m, v.w));
}


RT_FUNCTION bool isNull(const optix::float3& v)
{
    return (v.x == 0.0f && v.y == 0.0f && v.z == 0.0f);
}

RT_FUNCTION bool isNotNull(const optix::float3& v)
{
    return (v.x != 0.0f || v.y != 0.0f || v.z != 0.0f);
}


RT_FUNCTION optix::float3 clampRadiance(float clamp, int depth, const optix::float3& radiance)
{
    return (depth > 0 && clamp > 0.0f) ? fminf(radiance, optix::make_float3(clamp)) : radiance;
}


RT_FUNCTION optix::float3 max(const optix::float3& a, const optix::float3& b)
{
    return optix::make_float3(::fmaxf(a.x, b.x), ::fmaxf(a.y, b.y), ::fmaxf(a.z, b.z));
}
#endif


RT_FUNCTION bool operator!=(const optix::float3& a, const optix::float3& b)
{
    return (a.x != b.x) || (a.y != b.y) || (a.z != b.z);
}