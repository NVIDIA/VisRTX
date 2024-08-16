/*
 * Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gpu/gpu_util.h"
#include "utility/AnariTypeHelpers.h"

namespace visrtx {

RT_FUNCTION bool isPopulated(const AttributeData &ap)
{
  return ap.numChannels > 0;
}

RT_FUNCTION const SamplerGPUData &getSamplerData(
    const FrameGPUData &frameData, DeviceObjectIndex idx)
{
  return frameData.registry.samplers[idx];
}

template <typename T>
RT_FUNCTION const T *typedOffset(const void *mem, uint32_t offset)
{
  return ((const T *)mem) + offset;
}

template <typename ELEMENT_T>
RT_FUNCTION vec4 getAttributeValue_ufixed(
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

RT_FUNCTION vec4 getAttributeValue_f32(
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

RT_FUNCTION vec4 getAttributeValue(
    const AttributeData &attr, uint32_t offset, const vec4 &uniformFallback)
{
  if (attr.data == nullptr || offset == 0xFFFFFFFF)
    return uniformFallback;

  if (isFloat(attr.type))
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

RT_FUNCTION uint32_t decodeSphereAttributeIndices(
    const GeometryGPUData &ggd, const SurfaceHit &hit)
{
  if (ggd.sphere.indices != nullptr)
    return ggd.sphere.indices[hit.primID];
  else
    return hit.primID;
}

RT_FUNCTION uvec3 decodeTriangleAttributeIndices(
    const GeometryGPUData &ggd, uint32_t attributeID, const SurfaceHit &hit)
{
  if (ggd.tri.indices != nullptr)
    return ggd.tri.indices[hit.primID];
  else
    return 3 * hit.primID + uvec3(0, 1, 2);
}

RT_FUNCTION uvec4 decodeQuadAttributeIndices(
    const GeometryGPUData &ggd, uint32_t attributeID, uint32_t _primID)
{
  auto primID = _primID & ~0x1;
  auto i0 = ggd.quad.indices[primID + 0];
  auto i1 = ggd.quad.indices[primID + 1];
  return uvec4(i0.x, i0.y, i0.z, i1.x); // 0, 1, 3, 2
}

RT_FUNCTION uvec2 decodeCylinderAttributeIndices(
    const GeometryGPUData &ggd, uint32_t attributeID, const SurfaceHit &hit)
{
  if (ggd.cylinder.indices != nullptr)
    return ggd.cylinder.indices[hit.primID];
  else
    return 2 * hit.primID + uvec2(0, 1);
}

RT_FUNCTION uvec2 decodeConeAttributeIndices(
    const GeometryGPUData &ggd, uint32_t attributeID, const SurfaceHit &hit)
{
  if (ggd.cone.indices != nullptr)
    return ggd.cone.indices[hit.primID];
  else
    return 2 * hit.primID + uvec2(0, 1);
}

RT_FUNCTION uint32_t decodeCurveAttributeIndices(
    const GeometryGPUData &ggd, uint32_t attributeID, const SurfaceHit &hit)
{
  return ggd.curve.indices[hit.primID];
}

RT_FUNCTION vec4 readAttributeValue(uint32_t attributeID, const SurfaceHit &hit)
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

RT_FUNCTION vec4 evaluateSampler(
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
  case SamplerType::TEXTURE2D: {
    retval = make_vec4(tex2D<::float4>(sampler.image2D.texobj, tc.x, tc.y));
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

RT_FUNCTION vec4 getMaterialParameter(
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
    return vec4(hit.Ng, 1.f);
  /////////////////////////////////////////////////////////////////////////////
  // NOTE: these are in world space - need to quantify best performing option
  case MaterialParameterType::OBJECT_POSITION:
    return vec4(hit.hitpoint, 1.f);
  case MaterialParameterType::OBJECT_NORMAL:
    return vec4(hit.Ng, 1.f);
  /////////////////////////////////////////////////////////////////////////////
  default:
    break;
  }

  return vec4{};
}

RT_FUNCTION float adjustedMaterialOpacity(
    float opacityIn, const MaterialGPUData &md)
{
  if (md.mode == AlphaMode::OPAQUE)
    return 1.f;
  else {
    if (md.mode == AlphaMode::BLEND)
      return opacityIn;
    else
      return opacityIn < md.cutoff ? 0.f : 1.f;
  }
}

RT_FUNCTION MaterialValues getMaterialValues(
    const FrameGPUData &fd, const MaterialGPUData &md, const SurfaceHit &hit)
{
  MaterialValues retval;
  retval.isPBR = md.isPBR;

  // NOTE(jda): We _need_ to use a loop here to get parameters, otherwise
  //            compile times go through the roof. We actually don't want this
  //            unrolled, which always happens when MaterialGPUData::values[]
  //            are each their own named data members of that struct. This is
  //            weird, but it's a less terrible development experience this way.
  vec4 values[4];
  for (int i = 0; i < 4; i++)
    values[i] = getMaterialParameter(fd, md.values[i], hit);

  // baseColor
  retval.baseColor = vec3(values[MV_BASE_COLOR]);
  // opacity
  retval.opacity = adjustedMaterialOpacity(
      values[MV_OPACITY].x * values[MV_BASE_COLOR].w, md);
  // mettalic
  retval.metallic = values[MV_METALLIC].x;
  // roughness
  retval.roughness = values[MV_ROUGHNESS].x;
  // ior
  retval.ior = md.ior;

  return retval;
}

RT_FUNCTION vec4 evalMaterial(const FrameGPUData &fd,
    const MaterialGPUData &md,
    const SurfaceHit &hit,
    const vec3 &viewDir,
    const vec3 &lightDir,
    const vec3 &lightIntensity)
{
  const auto matValues = getMaterialValues(fd, md, hit);

  if (!matValues.isPBR)
    return {matValues.baseColor * lightIntensity, matValues.opacity};

  const vec3 H = normalize(lightDir + viewDir);
  const float NdotH = dot(hit.Ns, H);
  const float NdotL = dot(hit.Ns, lightDir);
  const float NdotV = dot(hit.Ns, viewDir);
  const float VdotH = dot(viewDir, H);
  const float LdotH = dot(lightDir, H);

  // Alpha
  const float alpha = pow2(matValues.roughness) * matValues.opacity;

  // Fresnel
  const vec3 f0 =
      glm::mix(vec3(pow2((1.f - matValues.ior) / (1.f + matValues.ior))),
          matValues.baseColor,
          matValues.metallic);
  const vec3 F = f0 + (vec3(1.f) - f0) * pow5(1.f - fabsf(VdotH));

  // Metallic materials don't reflect diffusely:
  const vec3 diffuseColor =
      glm::mix(matValues.baseColor, vec3(0.f), matValues.metallic);

  const vec3 diffuseBRDF =
      (vec3(1.f) - F) * float(M_1_PI) * diffuseColor * fmaxf(0.f, NdotL);

  // GGX microfacet distribution
  const float D = (alpha * alpha * heaviside(NdotH))
      / (float(M_PI) * pow2(NdotH * NdotH * (alpha * alpha - 1.f) + 1.f));

  // Masking-shadowing term
  const float G =
      ((2.f * fabsf(NdotL) * heaviside(LdotH))
          / (fabsf(NdotL)
              + sqrtf(alpha * alpha + (1.f - alpha * alpha) * NdotL * NdotL)))
      * ((2.f * fabsf(NdotV) * heaviside(VdotH))
          / (fabsf(NdotV)
              + sqrtf(alpha * alpha + (1.f - alpha * alpha) * NdotV * NdotV)));

  const float denom = 4.f * fabsf(NdotV) * fabsf(NdotL);
  const vec3 specularBRDF = denom != 0.f ? (F * D * G) / denom : vec3(0.f);

  return {(diffuseBRDF + specularBRDF) * lightIntensity, matValues.opacity};
}

} // namespace visrtx
