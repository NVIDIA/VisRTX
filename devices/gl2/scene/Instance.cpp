// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "Instance.h"

namespace visgl2 {

Instance::Instance(VisGL2DeviceGlobalState *s) : Object(ANARI_INSTANCE, s) {}

void Instance::commitParameters()
{
  m_id = getParam<uint32_t>("id", ~0u);
  m_xfm = getParam<mat4>("transform", mat4(1));
  m_group = getParamObject<Group>("group");
  if (!m_group)
    reportMessage(ANARI_SEVERITY_WARNING, "missing 'group' on ANARIInstance");
}

uint32_t Instance::id() const
{
  return m_id;
}

const mat4 &Instance::xfm() const
{
  return m_xfm;
}

bool Instance::xfmIsIdentity() const
{
  return xfm() == mat4(1);
}

const Group *Instance::group() const
{
  return m_group.ptr;
}

Group *Instance::group()
{
  return m_group.ptr;
}

bool Instance::isValid() const
{
  return m_group;
}

} // namespace visgl2

VISGL2_ANARI_TYPEFOR_DEFINITION(visgl2::Instance *);
