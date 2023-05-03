// Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "VisGLSpecializations.h"
#include "anari/type_utility.h"
#include "math_util.h"
#include <math.h>

#include <cstdlib>
#include <cstring>

namespace visgl{


Object<LightDirectional>::Object(ANARIDevice d, ANARIObject handle)
    : DefaultObject(d, handle)
{
  light_index = thisDevice->lights.allocate(2);

  commit();
}

void Object<LightDirectional>::commit()
{
  DefaultObject::commit();

  current.color.get(ANARI_FLOAT32_VEC3, color.data());
  current.irradiance.get(ANARI_FLOAT32, color.data()+3);
  current.direction.get(ANARI_FLOAT32_VEC3, direction.data());
  direction[0] = -direction[0];
  direction[1] = -direction[1];
  direction[2] = -direction[2];
  direction[3] = 0;
  dirty = true;
}

void Object<LightDirectional>::update()
{
  DefaultObject::update();
  if(dirty) {
    thisDevice->lights.set(light_index+0, color);
    thisDevice->lights.set(light_index+1, direction);
    dirty = false;
  }
}

uint32_t Object<LightDirectional>::index() {
  return light_index;
}


} //namespace visgl

