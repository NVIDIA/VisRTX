// Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "VisGLSpecializations.h"

namespace visgl{

Object<Group>::Object(ANARIDevice d, ANARIObject handle)
    : DefaultObject(d, handle)
{

}

void Object<Group>::commit()
{
  DefaultObject::commit();
}

} //namespace visgl

