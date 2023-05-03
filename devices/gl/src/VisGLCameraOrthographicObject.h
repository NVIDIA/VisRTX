// Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause


#pragma once

#include "VisGLDevice.h"

#include <array>

namespace visgl{


template <>
class Object<CameraOrthographic> : public DefaultObject<CameraOrthographic, CameraObjectBase>
{
  float position[3];
  float direction[3];
  float up[3];
  float transform[16];
  float region[4];
  float aspect;
  float height;
  
  void calculateMatrices(float near, float far);
public:

  Object(ANARIDevice d, ANARIObject handle);

  void updateAt(size_t index, float *bounds) const override;

  void commit() override;
};

} //namespace visgl

