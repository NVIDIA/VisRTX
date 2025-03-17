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

#include "gpu/gpu_objects.h"
#include "gpu/gpu_util.h"
#include "gpu/sampleLight.h"
#include "shaders/MDLShader.cuh"
#include "shaders/MatteShader.cuh"
#include "shaders/PhysicallyBasedShader.cuh"

#include "utility/AnariTypeHelpers.h"

#include <texture_indirect_functions.h>

namespace visrtx {

VISRTX_DEVICE bool isPopulated(const AttributeData &ap)
{
  return ap.numChannels > 0;
}

VISRTX_DEVICE const SamplerGPUData &getSamplerData(
    const FrameGPUData &frameData, DeviceObjectIndex idx)
{
  return frameData.registry.samplers[idx];
}

template <typename T>
VISRTX_DEVICE const T *typedOffset(const void *mem, uint32_t offset)
{
  return ((const T *)mem) + offset;
}

template <typename ELEMENT_T>
VISRTX_DEVICE vec4 getAttributeValue_ufixed(
    const AttributeData &attr, uint32_t offset)
{
  constexpr float m = float(static_cast<ELEMENT_T>(0xFFFFFFFF));
  vec4 retval(0.f, 0.f, 0.f, 1.f);
  switch (attr.numChannels) {
  case 4:
    retval.w =
        *typedOffset<ELEMENT_T>(attr.data, attr.numChannels * offset + 3) / m;
  case 3:
    retval.z =
        *typedOffset<ELEMENT_T>(attr.data, attr.numChannels * offset + 2) / m;
  case 2:
    retval.y =
        *typedOffset<ELEMENT_T>(attr.data, attr.numChannels * offset + 1) / m;
  case 1:
    retval.x =
        *typedOffset<ELEMENT_T>(attr.data, attr.numChannels * offset + 0) / m;
  default:
    break;
  }

  return retval;
}

VISRTX_DEVICE vec4 getAttributeValue_f32(
    const AttributeData &attr, uint32_t offset)
{
  switch (attr.numChannels) {
  case 1:
    return vec4(*typedOffset<float>(attr.data, offset), 0.f, 0.f, 1.f);
  case 2:
    return vec4(*typedOffset<vec2>(attr.data, offset), 0.f, 1.f);
  case 3:
    return vec4(*typedOffset<vec3>(attr.data, offset), 1.f);
  case 4:
    return *typedOffset<vec4>(attr.data, offset);
  default:
    break;
  }

  return vec4(0.f, 0.f, 0.f, 1.f);
}

VISRTX_DEVICE vec4 getAttributeValue(
    const AttributeData &attr, uint32_t offset, const vec4 &uniformFallback)
{
  if (attr.data == nullptr || offset == 0xFFFFFFFF)
    return uniformFallback;

  if (isFloat32(attr.type))
    return getAttributeValue_f32(attr, offset);
  else if (isFixed8(attr.type))
    return getAttributeValue_ufixed<uint8_t>(attr, offset);
  else if (isSrgb8(attr.type))
    return convertLinearToSRGB(getAttributeValue_ufixed<uint8_t>(attr, offset));
  else if (isFixed16(attr.type))
    return getAttributeValue_ufixed<uint16_t>(attr, offset);
  else if (isFixed32(attr.type))
    return getAttributeValue_ufixed<uint32_t>(attr, offset);

  return uniformFallback;
}

VISRTX_DEVICE uint32_t decodeSphereAttributeIndices(
    const GeometryGPUData &ggd, const SurfaceHit &hit)
{
  if (ggd.sphere.indices != nullptr)
    return ggd.sphere.indices[hit.primID];
  else
    return hit.primID;
}

VISRTX_DEVICE uvec3 decodeTriangleAttributeIndices(
    const GeometryGPUData &ggd, uint32_t attributeID, const SurfaceHit &hit)
{
  if (ggd.tri.indices != nullptr)
    return ggd.tri.indices[hit.primID];
  else
    return 3 * hit.primID + uvec3(0, 1, 2);
}

VISRTX_DEVICE uvec4 decodeQuadAttributeIndices(
    const GeometryGPUData &ggd, uint32_t attributeID, uint32_t _primID)
{
  auto primID = _primID & ~0x1;
  auto i0 = ggd.quad.indices[primID + 0];
  auto i1 = ggd.quad.indices[primID + 1];
  return uvec4(i0.x, i0.y, i0.z, i1.x); // 0, 1, 3, 2
}

VISRTX_DEVICE uvec2 decodeCylinderAttributeIndices(
    const GeometryGPUData &ggd, uint32_t attributeID, const SurfaceHit &hit)
{
  if (ggd.cylinder.indices != nullptr)
    return ggd.cylinder.indices[hit.primID];
  else
    return 2 * hit.primID + uvec2(0, 1);
}

VISRTX_DEVICE uvec2 decodeConeAttributeIndices(
    const GeometryGPUData &ggd, uint32_t attributeID, const SurfaceHit &hit)
{
  if (ggd.cone.indices != nullptr)
    return ggd.cone.indices[hit.primID];
  else
    return 2 * hit.primID + uvec2(0, 1);
}

VISRTX_DEVICE uint32_t decodeCurveAttributeIndices(
    const GeometryGPUData &ggd, uint32_t attributeID, const SurfaceHit &hit)
{
  return ggd.curve.indices[hit.primID];
}

VISRTX_DEVICE vec4 readAttributeValue(
    uint32_t attributeID, const SurfaceHit &hit)
{
  const auto &isd = *hit.instance;
  const auto &ggd = *hit.geometry;
  const vec4 &uf = ggd.attrUniform[attributeID];

  // First check per-vertex attributes
  if (ggd.type == GeometryType::TRIANGLE) {
    const auto &ap = ggd.tri.vertexAttr[attributeID];
    const auto &apFV = ggd.tri.vertexAttrFV[attributeID];
    if (isPopulated(apFV)) {
      const uvec3 idx = uvec3(0, 1, 2) + (hit.primID * 3);
      const vec3 b = hit.uvw;
      return b.x * getAttributeValue(apFV, idx.x, uf)
          + b.y * getAttributeValue(apFV, idx.y, uf)
          + b.z * getAttributeValue(apFV, idx.z, uf);
    } else if (isPopulated(ap)) {
      const uvec3 idx = decodeTriangleAttributeIndices(ggd, attributeID, hit);
      const vec3 b = hit.uvw;
      return b.x * getAttributeValue(ap, idx.x, uf)
          + b.y * getAttributeValue(ap, idx.y, uf)
          + b.z * getAttributeValue(ap, idx.z, uf);
    }
  } else if (ggd.type == GeometryType::QUAD) {
    const auto &ap = ggd.quad.vertexAttr[attributeID];
    if (isPopulated(ap)) {
      const uvec4 idx =
          decodeQuadAttributeIndices(ggd, attributeID, hit.primID);
      const vec3 b = hit.uvw;
      const auto v0 = getAttributeValue(ap, idx.x, uf);
      const auto v1 = getAttributeValue(ap, idx.y, uf);
      const auto v2 = getAttributeValue(ap, idx.w, uf);
      const auto v3 = getAttributeValue(ap, idx.z, uf);
      const float u = hit.primID & 0x1 ? b.y : 1.f - b.y;
      const float v = hit.primID & 0x1 ? b.z : 1.f - b.z;
      const auto l0 = v1 + (v0 - v1) * u;
      const auto l1 = v2 + (v3 - v2) * u;
      return l1 + (l0 - l1) * v;
    }
  } else if (ggd.type == GeometryType::CYLINDER) {
    const auto &ap = ggd.cylinder.vertexAttr[attributeID];
    if (isPopulated(ap)) {
      const uvec2 idx = decodeCylinderAttributeIndices(ggd, attributeID, hit);
      const vec3 b = hit.uvw;
      return b.z * getAttributeValue(ap, idx.x, uf)
          + b.y * getAttributeValue(ap, idx.y, uf);
    }
  } else if (ggd.type == GeometryType::CONE) {
    const auto &ap = ggd.cone.vertexAttr[attributeID];
    if (isPopulated(ap)) {
      const uvec2 idx = decodeConeAttributeIndices(ggd, attributeID, hit);
      const vec3 b = hit.uvw;
      return b.z * getAttributeValue(ap, idx.x, uf)
          + b.y * getAttributeValue(ap, idx.y, uf);
    }
  } else if (ggd.type == GeometryType::CURVE) {
    const auto &ap = ggd.curve.vertexAttr[attributeID];
    if (isPopulated(ap)) {
      const uint32_t idx = decodeCurveAttributeIndices(ggd, attributeID, hit);
      const vec3 b = hit.uvw;
      return b.z * getAttributeValue(ap, idx, uf)
          + b.y * getAttributeValue(ap, idx + 1, uf);
    }
  } else if (ggd.type == GeometryType::SPHERE) {
    const auto &ap = ggd.sphere.vertexAttr[attributeID];
    const uint32_t idx = decodeSphereAttributeIndices(ggd, hit);
    if (isPopulated(ap))
      return getAttributeValue(ap, idx, uf);
  }

  // Else check for  per-primitive attributes
  if (const auto &ap = ggd.attr[attributeID]; isPopulated(ap)) {
    if (ggd.type == GeometryType::QUAD)
      return getAttributeValue(ap, hit.primID / 2, uf);
    else
      return getAttributeValue(ap, hit.primID, uf);
  }

  // Eventually process instance values if any
  if (isd.attrUniformArrayPresent[attributeID]) {
    return getAttributeValue(
        isd.attrUniformArray[attributeID], isd.localArrayId, uf);
  }

  if (isd.attrUniformPresent[attributeID])
    return isd.attrUniform[attributeID];

  return uf;
}

VISRTX_DEVICE vec4 evaluateImageTextureSampler(
    const SamplerGPUData &sampler, vec4 at)
{
  vec4 retval{0.f, 0.f, 0.f, 1.f};
  const vec4 tc = sampler.inTransform * at + sampler.inOffset;
  switch (sampler.type) {
  case SamplerType::TEXTURE1D: {
    retval = make_vec4(tex1D<::float4>(sampler.image1D.texobj, tc.x));
    break;
  }
  case SamplerType::COMPRESSED_TEXTURE2D: {
    retval = make_vec4(
        tex2D<::float4>(sampler.compressedImage2D.texobj, tc.x, tc.y));
    break;
  }
  case SamplerType::TEXTURE2D: {
    retval = make_vec4(tex2D<::float4>(sampler.image2D.texobj, tc.x, tc.y));
    break;
  }
  case SamplerType::TEXTURE3D: {
    retval =
        make_vec4(tex3D<::float4>(sampler.image3D.texobj, tc.x, tc.y, tc.z));
    break;
  }
  default:
    break;
  }
  return sampler.outTransform * retval + sampler.outOffset;
}

VISRTX_DEVICE vec4 evaluateImageTexelSampler(
    const SamplerGPUData &sampler, ivec4 at)
{
  vec4 retval{0.f, 0.f, 0.f, 1.f};
  const vec4 tc = sampler.inTransform * at + sampler.inOffset;
  switch (sampler.type) {
  case SamplerType::TEXTURE1D: {
    retval = make_vec4(tex1D<::float4>(sampler.image1D.texelTexobj, tc.x));
    break;
  }
  case SamplerType::COMPRESSED_TEXTURE2D: {
    retval = make_vec4(
        tex2D<::float4>(sampler.compressedImage2D.texelTexobj, tc.x, tc.y));
    break;
  }
  case SamplerType::TEXTURE2D: {
    retval =
        make_vec4(tex2D<::float4>(sampler.image2D.texelTexobj, tc.x, tc.y));
    break;
  }
  case SamplerType::TEXTURE3D: {
    retval = make_vec4(
        tex3D<::float4>(sampler.image3D.texelTexobj, tc.x, tc.y, tc.z));
    break;
  }
  default:
    break;
  }
  return sampler.outTransform * retval + sampler.outOffset;
}

VISRTX_DEVICE vec4 evaluateSampler(
    const FrameGPUData &fd, const DeviceObjectIndex _s, const SurfaceHit &hit)
{
  vec4 retval{0.f, 0.f, 0.f, 1.f};
  const auto &sampler = getSamplerData(fd, _s);
  const vec4 tc =
      sampler.inTransform * readAttributeValue(sampler.attribute, hit)
      + sampler.inOffset;
  switch (sampler.type) {
  case SamplerType::TEXTURE1D: {
    retval = make_vec4(tex1D<::float4>(sampler.image1D.texobj, tc.x));
    break;
  }
  case SamplerType::COMPRESSED_TEXTURE2D: {
    retval = make_vec4(
        tex2D<::float4>(sampler.compressedImage2D.texobj, tc.x, tc.y));

    break;
  }
  case SamplerType::TEXTURE2D: {
    retval = make_vec4(tex2D<::float4>(sampler.image2D.texobj, tc.x, tc.y));
    break;
  }
  case SamplerType::TEXTURE3D: {
    retval =
        make_vec4(tex3D<::float4>(sampler.image3D.texobj, tc.x, tc.y, tc.z));
    break;
  }
  case SamplerType::PRIMITIVE: {
    retval = getAttributeValue(
        sampler.primitive.attr, hit.primID + sampler.primitive.offset, retval);
    break;
  }
  case SamplerType::TRANSFORM: {
    retval = tc;
    break;
  }
  default:
    break;
  }
  return sampler.outTransform * retval + sampler.outOffset;
}

VISRTX_DEVICE vec4 getMaterialParameter(
    const FrameGPUData &fd, const MaterialParameter &mp, const SurfaceHit &hit)
{
  switch (mp.type) {
  case MaterialParameterType::VALUE:
    return mp.value;
  case MaterialParameterType::SAMPLER:
    return evaluateSampler(fd, mp.sampler, hit);
  case MaterialParameterType::ATTRIB_0:
    return readAttributeValue(0, hit);
  case MaterialParameterType::ATTRIB_1:
    return readAttributeValue(1, hit);
  case MaterialParameterType::ATTRIB_2:
    return readAttributeValue(2, hit);
  case MaterialParameterType::ATTRIB_3:
    return readAttributeValue(3, hit);
  case MaterialParameterType::ATTRIB_COLOR:
    return readAttributeValue(4, hit);
  case MaterialParameterType::WORLD_POSITION:
    return vec4(hit.hitpoint, 1.f);
  case MaterialParameterType::WORLD_NORMAL:
    return vec4(hit.Ns, 1.f);
  /////////////////////////////////////////////////////////////////////////////
  // NOTE: these are in world space - need to quantify best performing option
  case MaterialParameterType::OBJECT_POSITION:
    return vec4(hit.hitpoint, 1.f);
  case MaterialParameterType::OBJECT_NORMAL:
    return vec4(hit.Ns, 1.f);
  /////////////////////////////////////////////////////////////////////////////
  default:
    break;
  }

  return vec4{};
}

VISRTX_DEVICE float adjustedMaterialOpacity(
    float opacityIn, AlphaMode mode, float cutoff)
{
  if (mode == AlphaMode::OPAQUE)
    return 1.f;
  else {
    if (mode == AlphaMode::BLEND)
      return opacityIn;
    else
      return opacityIn < cutoff ? 0.f : 1.f;
  }
}

VISRTX_DEVICE float adjustedMaterialOpacity(
    float opacityIn, const MaterialGPUData::Matte &md)
{
  return adjustedMaterialOpacity(opacityIn, md.alphaMode, md.cutoff);
}

VISRTX_DEVICE float adjustedMaterialOpacity(
    float opacityIn, const MaterialGPUData::PhysicallyBased &md)
{
  return adjustedMaterialOpacity(opacityIn, md.alphaMode, md.cutoff);
}

VISRTX_DEVICE float adjustedMaterialOpacity(
    float opacityIn, const MaterialGPUData &md)
{
  switch (md.materialType) {
  case MaterialType::MATTE: {
    return adjustedMaterialOpacity(opacityIn, md.matte);
    break;
  }
  case MaterialType::PHYSICALLYBASED: {
    return adjustedMaterialOpacity(opacityIn, md.physicallyBased);
    break;
  }
  default:
    return 1.0f;
  }
}

VISRTX_DEVICE MaterialValues getMaterialValues(const FrameGPUData &fd,
    const MaterialGPUData::Matte &md,
    const SurfaceHit &hit)
{
  vec4 color = getMaterialParameter(fd, md.color, hit);
  float opacity = getMaterialParameter(fd, md.opacity, hit).x;

  return {
      MaterialType::MATTE,
      vec3(color),
      adjustedMaterialOpacity(color.w * opacity, md),
      0.0f,
      0.5f,
      1.0f,
  };
}

VISRTX_DEVICE MaterialValues getMaterialValues(const FrameGPUData &fd,
    const MaterialGPUData::PhysicallyBased &md,
    const SurfaceHit &hit)
{
  vec4 color = getMaterialParameter(fd, md.baseColor, hit);
  float opacity = getMaterialParameter(fd, md.opacity, hit).x;

  return {
      MaterialType::PHYSICALLYBASED,
      vec3(color),
      adjustedMaterialOpacity(color.w * opacity, md),
      getMaterialParameter(fd, md.metallic, hit).x,
      getMaterialParameter(fd, md.roughness, hit).x,
      md.ior,
  };
}

VISRTX_DEVICE MaterialValues getMaterialValues(
    const FrameGPUData &fd, const MaterialGPUData &md, const SurfaceHit &hit)
{
  switch (md.materialType) {
  case MaterialType::MATTE:
    return getMaterialValues(fd, md.matte, hit);
  case MaterialType::PHYSICALLYBASED:
    return getMaterialValues(fd, md.physicallyBased, hit);
  default:
    return {
        MaterialType::MATTE,
        vec3(0.8f, 0.8f, 0.8f),
        1.0f,
        0.0f,
        0.5f,
        1.0f,
    };
  }
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

VISRTX_DEVICE vec4 evalMaterial(const FrameGPUData &fd,
    const ScreenSample &ss,
    const MaterialGPUData &md,
    const SurfaceHit &hit,
    const Ray &ray,
    const LightSample &ls)
{
  switch (md.materialType) {
  case MaterialType::MATTE: {
    return shadeMatteSurface(fd, md.matte, ray, hit, ls);
  }
  case MaterialType::PHYSICALLYBASED: {
    return shadePhysicallyBasedSurface(fd, md.physicallyBased, ray, hit, ls);
  }
  case MaterialType::MDL: {
    return shadeMDLSurface(fd, ss, md.mdl, ray, hit, ls);
  }
  default:
    return vec4(0.8f, 0.8f, 0.8f, 1.0f);
  }
}

} // namespace visrtx
