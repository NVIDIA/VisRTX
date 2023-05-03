// Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "VisGLSpecializations.h"
#include "anari/type_utility.h"
#include "math_util.h"
#include <math.h>

#include <cstdlib>
#include <cstring>

namespace visgl{


Object<RendererDefault>::Object(ANARIDevice d, ANARIObject handle)
    : DefaultObject(d, handle)
{
  ambient_index = thisDevice->lights.allocate(1);
  commit();
}

void Object<RendererDefault>::commit()
{
  DefaultObject::commit();

  current.ambientColor.get(ANARI_FLOAT32_VEC3, ambient.data());
  current.ambientRadiance.get(ANARI_FLOAT32, ambient.data()+3);
  ambient[0] *= ambient[3];
  ambient[1] *= ambient[3];
  ambient[2] *= ambient[3];
  ambient[3] = 1.0f;
  dirty = true;
}

void Object<RendererDefault>::update()
{
  DefaultObject::update();
  if(dirty) {
    thisDevice->lights.set(ambient_index+0, ambient);
    dirty = false;
  }
}

uint32_t Object<RendererDefault>::index() {
  return ambient_index;
}


} //namespace visgl

