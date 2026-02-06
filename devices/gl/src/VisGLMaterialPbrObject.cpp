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

#include "VisGLSpecializations.h"
#include "shader_blocks.h"
#include "MaterialMacros.h"

namespace visgl {

Object<MaterialPhysicallyBased>::Object(ANARIDevice d, ANARIObject handle)
    : DefaultObject(d, handle)
{
  material_index = thisDevice->materials.allocate(11);

  commit();
}

// clang-format off
const char *pbr_material_eval_block = R"GLSL(
  vec4 lighting = vec4(0.0, 0.0, 0.0, 1.0);

  vec4 scalars = materials[instanceIndices.y];
  float ior = scalars.x;

  vec4 V0 = transforms[cameraIdx+6u][1];
  vec3 V = -normalize(worldPosition.xyz*V0.w - V0.xyz);
  vec3 N = worldNormal.xyz;

  float c_ior = sqr((ior - 1.0) / (ior + 1.0));

  vec3 c_diff = baseColor.xyz*(1.0-metallic.x);
  vec3 f0 = mix(vec3(c_ior), baseColor.xyz, metallic.x);
  float alpha = roughness.x*roughness.x;
  float alpha2 = sqr(alpha);

  float NdotV = dot(N,V);

  for(uint i=0u;i<lightCount;++i) {
)GLSL"
UNPACK_LIGHT("i")
R"GLSL(
    float shadow = sampleShadow(worldPosition, geometryNormal, lightIndices[i].z);

    vec3 L = normalize(direction);
    vec3 H = normalize(L+V);

    float k = 1.0 - abs(dot(V, H));

    vec3 F = f0 + (vec3(1.0)-f0)*k*k*k*k*k;

    float NdotH = dot(N,H);
    float NdotL = dot(N,L);

    float HdotL = dot(H,L);
    float HdotV = dot(H,V);

    float D = alpha*alpha*step(0.0, NdotH)/(PI*sqr(sqr(NdotH)*(alpha2-1.0)+1.0));

    float G =
      2.0*NdotL*step(0.0, HdotL)/(NdotL + sqrt(alpha2 + sqr(1.0-alpha)*sqr(NdotL)))*
      2.0*NdotV*step(0.0, HdotV)/(NdotV + sqrt(alpha2 + sqr(1.0-alpha)*sqr(NdotV)));

    vec3 f_diffuse = (vec3(1.0) - F)*(1.0/PI)*c_diff;
    vec3 f_specular = F*D*G/(4.0*abs(NdotV)*abs(NdotL));

    lighting.xyz += max(vec3(0.0), shadow*attenuation*light_color*(f_diffuse + f_specular)*max(0.0, NdotL));
  }

  lighting.xyz += fragmentOcclusion*lights[ambientIdx].xyz;

  FragColor = vec4(lighting.xyz*baseColor.xyz, opacity.x);
  FragColor.w *= coverage;
}
)GLSL";
// clang-format on

#define BASE_COLOR_SAMPLER MATERIAL_RESOURCE(0)
#define OPACITY_SAMPLER MATERIAL_RESOURCE(1)
#define METALLIC_SAMPLER MATERIAL_RESOURCE(2)
#define ROUGHNESS_SAMPLER MATERIAL_RESOURCE(3)
#define EMISSIVE_SAMPLER MATERIAL_RESOURCE(4)
#define OCCLUSION_SAMPLER MATERIAL_RESOURCE(5)
#define SPECULAR_SAMPLER MATERIAL_RESOURCE(6)
#define SPECULAR_COLOR_SAMPLER MATERIAL_RESOURCE(7)
#define CLEARCOAT_SAMPLER MATERIAL_RESOURCE(8)
#define CLOEARCOAT_ROUGHNESS_SAMPLER MATERIAL_RESOURCE(9)

void Object<MaterialPhysicallyBased>::commit()
{
  DefaultObject::commit();

  std::array<float, 4> scalars{1.5f, 0.0f, 0.0f, 0.0f};
  current.ior.get(ANARI_FLOAT32, scalars.data());
  thisDevice->materials.set(material_index, scalars);

  MATERIAL_COMMIT_ATTRIBUTE(baseColor, ANARI_FLOAT32_VEC3, 1)
  MATERIAL_COMMIT_ATTRIBUTE(opacity, ANARI_FLOAT32, 2)
  MATERIAL_COMMIT_ATTRIBUTE(metallic, ANARI_FLOAT32, 3)
  MATERIAL_COMMIT_ATTRIBUTE(roughness, ANARI_FLOAT32, 4)
  MATERIAL_COMMIT_ATTRIBUTE(emissive, ANARI_FLOAT32_VEC3, 5)
  MATERIAL_COMMIT_ATTRIBUTE(occlusion, ANARI_FLOAT32, 6)
  MATERIAL_COMMIT_ATTRIBUTE(specular, ANARI_FLOAT32, 7)
  MATERIAL_COMMIT_ATTRIBUTE(specularColor, ANARI_FLOAT32_VEC3, 8)
  MATERIAL_COMMIT_ATTRIBUTE(clearcoat, ANARI_FLOAT32, 9)
  MATERIAL_COMMIT_ATTRIBUTE(clearcoatRoughness, ANARI_FLOAT32, 10)
}

uint32_t Object<MaterialPhysicallyBased>::index()
{
  return material_index;
}

void Object<MaterialPhysicallyBased>::allocateResources(SurfaceObjectBase *surf)
{
  ALLOCATE_SAMPLERS(baseColor, BASE_COLOR_SAMPLER)
  ALLOCATE_SAMPLERS(opacity, OPACITY_SAMPLER)
  ALLOCATE_SAMPLERS(metallic, METALLIC_SAMPLER)
  ALLOCATE_SAMPLERS(roughness, ROUGHNESS_SAMPLER)
  ALLOCATE_SAMPLERS(emissive, EMISSIVE_SAMPLER)
  ALLOCATE_SAMPLERS(occlusion, OCCLUSION_SAMPLER)
  ALLOCATE_SAMPLERS(specular, SPECULAR_SAMPLER)
  ALLOCATE_SAMPLERS(specularColor, SPECULAR_COLOR_SAMPLER)
  ALLOCATE_SAMPLERS(clearcoat, CLEARCOAT_SAMPLER)
  ALLOCATE_SAMPLERS(clearcoatRoughness, CLOEARCOAT_ROUGHNESS_SAMPLER)
}

void Object<MaterialPhysicallyBased>::drawCommand(
    SurfaceObjectBase *surf, DrawCommand &command)
{
  MATERIAL_DRAW_COMMAND(baseColor, BASE_COLOR_SAMPLER)
  MATERIAL_DRAW_COMMAND(opacity, OPACITY_SAMPLER)
  MATERIAL_DRAW_COMMAND(metallic, METALLIC_SAMPLER)
  MATERIAL_DRAW_COMMAND(roughness, ROUGHNESS_SAMPLER)
  MATERIAL_DRAW_COMMAND(emissive, EMISSIVE_SAMPLER)
  MATERIAL_DRAW_COMMAND(occlusion, OCCLUSION_SAMPLER)
  MATERIAL_DRAW_COMMAND(specular, SPECULAR_SAMPLER)
  MATERIAL_DRAW_COMMAND(specularColor, SPECULAR_COLOR_SAMPLER)
  MATERIAL_DRAW_COMMAND(clearcoat, CLEARCOAT_SAMPLER)
  MATERIAL_DRAW_COMMAND(clearcoatRoughness, CLOEARCOAT_ROUGHNESS_SAMPLER)
}

void Object<MaterialPhysicallyBased>::fragmentShaderDeclarations(
    SurfaceObjectBase *surf, AppendableShader &shader)
{
  MATERIAL_FRAG_DECL(baseColor, BASE_COLOR_SAMPLER)
  MATERIAL_FRAG_DECL(opacity, OPACITY_SAMPLER)
  MATERIAL_FRAG_DECL(metallic, METALLIC_SAMPLER)
  MATERIAL_FRAG_DECL(roughness, ROUGHNESS_SAMPLER)
  MATERIAL_FRAG_DECL(emissive, EMISSIVE_SAMPLER)
  MATERIAL_FRAG_DECL(occlusion, OCCLUSION_SAMPLER)
  MATERIAL_FRAG_DECL(specular, SPECULAR_SAMPLER)
  MATERIAL_FRAG_DECL(specularColor, SPECULAR_COLOR_SAMPLER)
  MATERIAL_FRAG_DECL(clearcoat, CLEARCOAT_SAMPLER)
  MATERIAL_FRAG_DECL(clearcoatRoughness, CLOEARCOAT_ROUGHNESS_SAMPLER)

  shader.append(shadow_map_declaration);
}

void Object<MaterialPhysicallyBased>::fragmentShaderMain(
    SurfaceObjectBase *surf, AppendableShader &shader)
{
  MATERIAL_FRAG_SAMPLE(
      "baseColor", baseColor, ANARI_FLOAT32_VEC3, 1, BASE_COLOR_SAMPLER)
  MATERIAL_FRAG_SAMPLE("opacity", opacity, ANARI_FLOAT32, 2, OPACITY_SAMPLER)

  if(current.alphaMode.getStringEnum() == STRING_ENUM_opaque) {
    shader.append("  opacity.x = 1.0;\n");
  } else if(current.alphaMode.getStringEnum() == STRING_ENUM_blend) {
    shader.append("  opacity.x *= baseColor.w;\n");
  } else if(current.alphaMode.getStringEnum() == STRING_ENUM_mask) {
    shader.append("  opacity.x = step(opacity.x*baseColor.w, 0.5);\n");
  }

  MATERIAL_FRAG_SAMPLE("metallic", metallic, ANARI_FLOAT32, 3, METALLIC_SAMPLER)
  MATERIAL_FRAG_SAMPLE(
      "roughness", roughness, ANARI_FLOAT32, 4, ROUGHNESS_SAMPLER)
  MATERIAL_FRAG_SAMPLE(
      "emissive", emissive, ANARI_FLOAT32_VEC3, 5, EMISSIVE_SAMPLER)
  MATERIAL_FRAG_SAMPLE(
      "occlusion", occlusion, ANARI_FLOAT32, 6, OCCLUSION_SAMPLER)
  MATERIAL_FRAG_SAMPLE("specular", specular, ANARI_FLOAT32, 7, SPECULAR_SAMPLER)
  MATERIAL_FRAG_SAMPLE("specularColor",
      specularColor,
      ANARI_FLOAT32_VEC3,
      8,
      SPECULAR_COLOR_SAMPLER)
  MATERIAL_FRAG_SAMPLE(
      "clearcoat", clearcoat, ANARI_FLOAT32, 9, CLEARCOAT_SAMPLER)
  MATERIAL_FRAG_SAMPLE("clearcoatRoughness",
      clearcoatRoughness,
      ANARI_FLOAT32,
      10,
      CLOEARCOAT_ROUGHNESS_SAMPLER)

  shader.append(pbr_material_eval_block);
}

void Object<MaterialPhysicallyBased>::fragmentShaderShadowDeclarations(
    SurfaceObjectBase *surf, AppendableShader &shader)
{}

void Object<MaterialPhysicallyBased>::fragmentShaderShadowMain(
    SurfaceObjectBase *surf, AppendableShader &shader)
{}

} // namespace visgl
