// Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause


#pragma once

#include "VisGLDevice.h"

namespace visgl{


template <>
class Object<Instance> : public DefaultObject<Instance, InstanceObjectBase>
{
  std::array<float, 16> instanceTransform;
  std::array<float, 16> inverseTransform;
  std::array<float, 16> normalTransform;
  size_t transform_index;
  bool dirty = true;
public:

  Object(ANARIDevice d, ANARIObject handle);

  void commit() override;
  void update() override;
  const std::array<float, 16>& transform() override;
  uint32_t index() override;

};

} //namespace visgl

