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

Object<MaterialMatte>::Object(ANARIDevice d, ANARIObject handle)
    : DefaultObject(d, handle)
{
  material_index = thisDevice->materials.allocate(2);

  commit();
}

const char *matte_baseColor = "  vec4 baseColor = ";

const char *matte_uniformColor = "materials[instanceIndices.y]\n";

const char *matte_opacity = "  vec4 opacity = ";

const char *matte_uniformOpacity = "materials[instanceIndices.y+1u]\n";

// clang-format off
const char *matte_material_eval_block = R"GLSL(
  vec4 lighting = vec4(0.0, 0.0, 0.0, 1.0);

  baseColor.w *= opacity.x;

  for(uint i=0u;i<lightCount;++i) {
)GLSL"
UNPACK_LIGHT("i")
R"GLSL(
    float shadow = sampleShadow(worldPosition, geometryNormal, lightIndices[i].z);
    lighting.xyz += shadow*attenuation*light_color*max(0.0, dot(normalize(direction), worldNormal.xyz));
  }

  //FragColor = vec4(vec3(fragmentOcclusion), 1.0); return;

  //lighting.xyz += 0.1*fragmentOcclusion;
  //FragColor = lighting;

  lighting.xyz += fragmentOcclusion*lights[ambientIdx].xyz;

  FragColor = baseColor*lighting;
  FragColor.w *= coverage;
}
)GLSL";
// clang-format on

#define COLOR_SAMPLER MATERIAL_RESOURCE(0)
#define OPACITY_SAMPLER MATERIAL_RESOURCE(1)

void Object<MaterialMatte>::commit()
{
  DefaultObject::commit();

  MATERIAL_COMMIT_ATTRIBUTE(color, ANARI_FLOAT32_VEC3, 0)
  MATERIAL_COMMIT_ATTRIBUTE(opacity, ANARI_FLOAT32, 1)
}

uint32_t Object<MaterialMatte>::index()
{
  return material_index;
}

void Object<MaterialMatte>::allocateResources(SurfaceObjectBase *surf)
{
  ALLOCATE_SAMPLERS(color, COLOR_SAMPLER)
  ALLOCATE_SAMPLERS(opacity, OPACITY_SAMPLER)
}

void Object<MaterialMatte>::drawCommand(
    SurfaceObjectBase *surf, DrawCommand &command)
{
  MATERIAL_DRAW_COMMAND(color, COLOR_SAMPLER)
  MATERIAL_DRAW_COMMAND(opacity, OPACITY_SAMPLER)
}

void Object<MaterialMatte>::fragmentShaderDeclarations(
    SurfaceObjectBase *surf, AppendableShader &shader)
{
  MATERIAL_FRAG_DECL(color, COLOR_SAMPLER)
  MATERIAL_FRAG_DECL(opacity, OPACITY_SAMPLER)

  shader.append(shadow_map_declaration);
}
void Object<MaterialMatte>::fragmentShaderMain(
    SurfaceObjectBase *surf, AppendableShader &shader)
{
  MATERIAL_FRAG_SAMPLE("baseColor", color, ANARI_FLOAT32_VEC3, 0, COLOR_SAMPLER)

  if(current.alphaMode.getStringEnum() == STRING_ENUM_opaque) {
    shader.append("  baseColor.w = 1.0;\n");
  }

  MATERIAL_FRAG_SAMPLE("opacity", opacity, ANARI_FLOAT32, 1, OPACITY_SAMPLER)

  shader.append(matte_material_eval_block);
}

void Object<MaterialMatte>::fragmentShaderShadowDeclarations(
    SurfaceObjectBase *surf, AppendableShader &shader)
{}

void Object<MaterialMatte>::fragmentShaderShadowMain(
    SurfaceObjectBase *surf, AppendableShader &shader)
{}

} // namespace visgl
