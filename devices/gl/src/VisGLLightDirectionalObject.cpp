/*
 * Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

