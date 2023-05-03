// Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "VisGLSpecializations.h"
#include "anari/type_utility.h"
#include "math_util.h"
#include <math.h>

#include <cstdlib>
#include <cstring>

namespace visgl{


Object<LightPoint>::Object(ANARIDevice d, ANARIObject handle)
    : DefaultObject(d, handle)
{
  light_index = thisDevice->lights.allocate(2);

  commit();
}

void Object<LightPoint>::commit()
{
  DefaultObject::commit();

  current.color.get(ANARI_FLOAT32_VEC3, color.data());
  current.intensity.get(ANARI_FLOAT32, color.data()+3);
  current.position.get(ANARI_FLOAT32_VEC3, position.data());
  position[3] = 1;
  dirty = true;
}

void Object<LightPoint>::update()
{
  DefaultObject::update();
  if(dirty) {
    thisDevice->lights.set(light_index+0, color);
    thisDevice->lights.set(light_index+1, position);
    dirty = false;
  }
}

uint32_t Object<LightPoint>::index() {
  return light_index;
}

} //namespace visgl

