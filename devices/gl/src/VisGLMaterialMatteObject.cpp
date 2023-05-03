// Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "VisGLSpecializations.h"
#include "shader_blocks.h"
#include "MaterialMacros.h"

namespace visgl{

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

const char *matte_material_eval_block = R"GLSL(
  vec4 lighting = vec4(0.0, 0.0, 0.0, 1.0);

  baseColor.w *= opacity.x;

  for(uint i=0u;i<lightCount;++i) {
    vec4 c = lights[lightIndices[i].x];
    mat4 t = transforms[lightIndices[i].y];
    vec4 x = t*lights[lightIndices[i].x+1u];
    float shadow = sampleShadow(worldPosition, geometryNormal, lightIndices[i].z);

    vec3 light_color = c.xyz * c.w;
    vec3 direction = x.xyz - worldPosition.xyz*x.w;
    float attenuation = 1.0/dot(direction, direction);
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

#define COLOR_SAMPLER MATERIAL_RESOURCE(0)
#define OPACITY_SAMPLER MATERIAL_RESOURCE(1)


void Object<MaterialMatte>::commit()
{
  DefaultObject::commit();

  MATERIAL_COMMIT_ATTRIBUTE(color, ANARI_FLOAT32_VEC3, 0)
  MATERIAL_COMMIT_ATTRIBUTE(opacity, ANARI_FLOAT32, 1)
}

uint32_t Object<MaterialMatte>::index() {
  return material_index;
}

void Object<MaterialMatte>::allocateResources(SurfaceObjectBase *surf)
{
  ALLOCATE_SAMPLERS(color, COLOR_SAMPLER)
  ALLOCATE_SAMPLERS(opacity, OPACITY_SAMPLER)
}

void Object<MaterialMatte>::drawCommand(SurfaceObjectBase *surf, DrawCommand &command)
{
  MATERIAL_DRAW_COMMAND(color, COLOR_SAMPLER)
  MATERIAL_DRAW_COMMAND(opacity, OPACITY_SAMPLER)
}

void Object<MaterialMatte>::fragmentShaderDeclarations(SurfaceObjectBase *surf, AppendableShader &shader)
{
  MATERIAL_FRAG_DECL(color, COLOR_SAMPLER)
  MATERIAL_FRAG_DECL(opacity, OPACITY_SAMPLER)

  shader.append(shadow_map_declaration);
}
void Object<MaterialMatte>::fragmentShaderMain(SurfaceObjectBase *surf, AppendableShader &shader)
{
  MATERIAL_FRAG_SAMPLE("baseColor", color, ANARI_FLOAT32_VEC3, 0, COLOR_SAMPLER)
  MATERIAL_FRAG_SAMPLE("opacity", opacity, ANARI_FLOAT32, 1, OPACITY_SAMPLER)

  shader.append(matte_material_eval_block);
}

void Object<MaterialMatte>::fragmentShaderShadowDeclarations(SurfaceObjectBase *surf, AppendableShader &shader)
{
}

void Object<MaterialMatte>::fragmentShaderShadowMain(SurfaceObjectBase *surf, AppendableShader &shader)
{
}

} //namespace visgl

