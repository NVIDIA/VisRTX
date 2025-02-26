// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "Object.h"

namespace visgl2 {

// Inherit from this, add your functionality, and add it to 'createInstance()'
struct Volume : public Object
{
  Volume(VisGL2DeviceGlobalState *d);
  virtual ~Volume() = default;
  static Volume *createInstance(
      std::string_view subtype, VisGL2DeviceGlobalState *d);

  void commitParameters() override;

  uint32_t id() const;

 private:
  uint32_t m_id{~0u};
};

// Inlined definitions ////////////////////////////////////////////////////////

inline uint32_t Volume::id() const
{
  return m_id;
}

} // namespace visgl2

VISGL2_ANARI_TYPEFOR_SPECIALIZATION(visgl2::Volume *, ANARI_VOLUME);
