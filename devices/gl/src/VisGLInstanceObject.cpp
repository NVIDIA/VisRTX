// Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "VisGLSpecializations.h"
#include "math_util.h"

namespace visgl{

Object<Instance>::Object(ANARIDevice d, ANARIObject handle)
    : DefaultObject(d, handle)
{
  transform_index = thisDevice->transforms.allocate(3);

  commit();
}

void Object<Instance>::commit()
{
  DefaultObject::commit();

  current.transform.get(ANARI_FLOAT32_MAT4, instanceTransform.data());

  setInverse(inverseTransform.data(), instanceTransform.data());
  setNormalTransform(normalTransform.data(), instanceTransform.data());

  dirty = true;
}

void Object<Instance>::update()
{
  DefaultObject::update();
  if(dirty) {
    thisDevice->transforms.set(transform_index, instanceTransform);
    thisDevice->transforms.set(transform_index+1, inverseTransform);
    thisDevice->transforms.set(transform_index+2, normalTransform);
    dirty = false;
  }
}

const std::array<float, 16>& Object<Instance>::transform() {
  return instanceTransform;
}

uint32_t Object<Instance>::index() {
  return transform_index;
}

} //namespace visgl

