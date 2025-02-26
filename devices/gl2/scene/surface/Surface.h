// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "geometry/Geometry.h"
#include "material/Material.h"

namespace visgl2 {

// Inherit from this, add your functionality, and add it to 'createInstance()'
struct Surface : public Object
{
  Surface(VisGL2DeviceGlobalState *s);
  ~Surface() override = default;

  void commitParameters() override;

  uint32_t id() const;
  const Geometry *geometry() const;
  const Material *material() const;

  bool isValid() const override;

 private:
  uint32_t m_id{~0u};
  helium::IntrusivePtr<Geometry> m_geometry;
  helium::IntrusivePtr<Material> m_material;
};

} // namespace visgl2

VISGL2_ANARI_TYPEFOR_SPECIALIZATION(visgl2::Surface *, ANARI_SURFACE);
