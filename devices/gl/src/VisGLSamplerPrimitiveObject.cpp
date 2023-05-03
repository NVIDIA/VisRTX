// Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "VisGLSpecializations.h"
#include "shader_blocks.h"
#include "anari2gl_types.h"

namespace visgl{

Object<SamplerPrimitive>::Object(ANARIDevice d, ANARIObject handle)
    : DefaultObject(d, handle)
{
}

void Object<SamplerPrimitive>::commit()
{
  DefaultObject::commit();
  array = acquire<DataArray1D*>(current.array);
}

void Object<SamplerPrimitive>::update()
{
  DefaultObject::update();
  uint64_t offset = 0;
  current.offset.get(ANARI_UINT64, &offset);

  meta[0] = offset;
}

void Object<SamplerPrimitive>::allocateResources(SurfaceObjectBase *surf, int slot)
{
  if(array) {
    surf->allocateStorageBuffer(slot, array->getBuffer());
  }
  surf->addAttributeFlags(ATTRIBUTE_PRIMITIVE_ID, ATTRIBUTE_FLAG_USED);
}

void Object<SamplerPrimitive>::drawCommand(int index, DrawCommand &command)
{
  if(array) {
    // these sevens are magic numbers for now
    array->drawCommand(index, command);
  }
}

void Object<SamplerPrimitive>::declare(int index, AppendableShader &shader)
{
  if(array) {
    array->declare(index, shader);
  }
}

void Object<SamplerPrimitive>::sample(int index, AppendableShader &shader, const char *meta)
{
  if(array) {
    array->sample(index, shader);
    shader.append("primitiveId+floatBitsToUint(");
    shader.append(meta);
    shader.append(".x));\n");
  } else {
    shader.append("vec4(1,0,1,1);\n");
  }
}

std::array<uint32_t, 4> Object<SamplerPrimitive>::metadata() {
  return meta;
}

Object<SamplerPrimitive>::~Object()
{
}

} //namespace visgl

