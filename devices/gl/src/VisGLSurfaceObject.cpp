// Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "VisGLSpecializations.h"
#include "AppendableShader.h"
#include "shader_blocks.h"
#include "shader_compile_segmented.h"

#include <cstdio>

namespace visgl{

Object<Surface>::Object(ANARIDevice d, ANARIObject handle)
    : DefaultObject(d, handle)
{

}

void Object<Surface>::commit()
{
  DefaultObject::commit();
  geometry = acquire<GeometryObjectBase*>(current.geometry);
  material = acquire<MaterialObjectBase*>(current.material);
}

void Object<Surface>::allocateTexture(int slot, GLenum target, GLuint texture, GLuint sampler)
{
  if(slot >= SURFACE_MAX_RESOURCES) {
    return;
  }

  resources[slot].type = TEX;
  resources[slot].index = texCount++;
  resources[slot].tex.target = target;
  resources[slot].tex.texture = texture;
  resources[slot].tex.sampler = sampler;
}

void Object<Surface>::allocateStorageBuffer(int slot, GLuint buffer)
{
  if(slot >= SURFACE_MAX_RESOURCES) {
    return;
  }

  resources[slot].type = SSBO;
  resources[slot].index = ssboCount++;
  resources[slot].ssbo.buffer = buffer;
}

void Object<Surface>::allocateTransform(int slot)
{
  if(slot >= SURFACE_MAX_RESOURCES) {
    return;
  }

  resources[slot].type = TRANSFORM;
  resources[slot].index = transformCount++;
}

int Object<Surface>::resourceIndex(int slot)
{
  if(slot < SURFACE_MAX_RESOURCES) {
    return resources[slot].index;
  } else {
    return -1;
  }
}

void Object<Surface>::addAttributeFlags(int attrib, uint32_t flags) {
  if(0<=attrib && attrib<ATTRIBUTE_COUNT) {
    attributeFlags[attrib] |= flags;
  }
}

uint32_t Object<Surface>::getAttributeFlags(int attrib) {
  if(0<=attrib && attrib<ATTRIBUTE_COUNT) {
    return attributeFlags[attrib];
  } else {
    return 0u;
  }
}


static void surface_compile_shader(ObjectRef<Surface> surfaceObj,
  StaticAppendableShader<SHADER_SEGMENTS> vs, StaticAppendableShader<SHADER_SEGMENTS> fs,
  StaticAppendableShader<SHADER_SEGMENTS> vs_shadow,
  StaticAppendableShader<SHADER_SEGMENTS> gs_shadow,
  StaticAppendableShader<SHADER_SEGMENTS> fs_shadow,
  StaticAppendableShader<SHADER_SEGMENTS> vs_occlusion
) {
  if(surfaceObj->shader == 0) {
    surfaceObj->shader = surfaceObj->thisDevice->shaders.get(vs, fs);
    surfaceObj->shadow_shader = surfaceObj->thisDevice->shaders.get(vs_shadow, gs_shadow, fs_shadow);

    const char *version = surfaceObj->thisDevice->gl.VERSION_4_3 ? version_430 : version_320_es;
    
    StaticAppendableShader<SHADER_SEGMENTS> empty;
    empty.append(version);
    empty.append(empty_fragment_shader);
    
    surfaceObj->occlusion_shader = surfaceObj->thisDevice->shaders.get(vs_occlusion, empty);
  }
}

void Object<Surface>::update()
{
  DefaultObject::update();
  
  bool rebuild = false;
  if(material) {
    rebuild |= material_epoch < material->objectEpoch();
  }

  if(geometry) {
    rebuild |= geometry_epoch < geometry->objectEpoch();
  }

  if(shader == 0 || rebuild) {
    if(geometry && material) {
      const char *version = thisDevice->gl.VERSION_4_3 ? version_430 : version_320_es;

      // reset
      ssboCount = GLOBAL_SSBO_OFFSET;
      texCount = GLOBAL_TEX_OFFSET;
      transformCount = GLOBAL_TRANSFORM_OFFSET;
      std::fill(attributeFlags.begin(), attributeFlags.end(), 0u);

      geometry->allocateResources(this);
      material->allocateResources(this);

      StaticAppendableShader<SHADER_SEGMENTS> vs;
      vs.append(version);
      vs.append(shader_preamble);
      geometry->vertexShader(this, vs);
       
      StaticAppendableShader<SHADER_SEGMENTS> fs;
      fs.append(version);
      fs.append(shader_preamble);
      fs.append(shader_conversions);
      material->fragmentShaderDeclarations(this, fs);
      geometry->fragmentShaderMain(this, fs);
      material->fragmentShaderMain(this, fs);


      StaticAppendableShader<SHADER_SEGMENTS> vs_shadow;
      vs_shadow.append(version);
      vs_shadow.append(shader_preamble);
      geometry->vertexShaderShadow(this, vs_shadow);

      StaticAppendableShader<SHADER_SEGMENTS> gs_shadow;
      gs_shadow.append(version);
      gs_shadow.append(shader_preamble);
      geometry->geometryShaderShadow(this, gs_shadow);
       
      StaticAppendableShader<SHADER_SEGMENTS> fs_shadow;
      fs_shadow.append(version);
      fs_shadow.append(shader_preamble);
      fs_shadow.append(shader_conversions);
      material->fragmentShaderShadowDeclarations(this, fs_shadow);
      geometry->fragmentShaderShadowMain(this, fs_shadow);
      material->fragmentShaderShadowMain(this, fs_shadow);


      StaticAppendableShader<SHADER_SEGMENTS> vs_occlusion;
      vs_occlusion.append(version);
      vs_occlusion.append(shader_preamble);
      geometry->vertexShaderOcclusion(this, vs_occlusion);

      thisDevice->queue.enqueue(surface_compile_shader, this, vs, fs, vs_shadow, gs_shadow, fs_shadow, vs_occlusion).wait();


      material_epoch = material->objectEpoch();
      geometry_epoch = geometry->objectEpoch();
    }
  }
}

void Object<Surface>::drawCommand(DrawCommand &command)
{
  geometry->drawCommand(this, command);
  material->drawCommand(this, command);
  command.shader = shader;
  command.shadow_shader = shadow_shader;
  command.occlusion_resolve_shader = occlusion_shader;
}


} //namespace visgl

