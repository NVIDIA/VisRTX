// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "Group.h"

namespace visgl2 {

struct Instance : public Object
{
  Instance(VisGL2DeviceGlobalState *s);
  ~Instance() override = default;

  void commitParameters() override;

  uint32_t id() const;

  const mat4 &xfm() const;
  bool xfmIsIdentity() const;

  const Group *group() const;
  Group *group();

  bool isValid() const override;

 private:
  uint32_t m_id{~0u};
  mat4 m_xfm;
  helium::IntrusivePtr<Group> m_group;
};

} // namespace visgl2

VISGL2_ANARI_TYPEFOR_SPECIALIZATION(visgl2::Instance *, ANARI_INSTANCE);
