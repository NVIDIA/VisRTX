// Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause


#pragma once

#include "VisGLDevice.h"

namespace visgl{


template <>
class Object<Group> : public DefaultObject<Group>
{
public:
  Object(ANARIDevice d, ANARIObject handle);

  void commit() override;
};

} //namespace visgl

