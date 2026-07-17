// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/scene/Object.hpp"

namespace tsd::scene {

/*
 * ANARI Material object that controls the shading model and surface appearance
 * through its parameter map and a device-specific subtype token.
 *
 * Example:
 *   auto mat = scene.createObject<Material>(tokens::material::physicallyBased);
 *   mat->setParameter("baseColor", float3{0.8f, 0.2f, 0.2f});
 */
struct Material : public Object
{
  DECLARE_OBJECT_DEFAULT_LIFETIME(Material);

  Material(Token subtype = tokens::unknown);
  virtual ~Material() = default;

  ObjectPoolRef<Material> self() const;

  anari::Object makeANARIObject(anari::Device d) const override;
};

using MaterialRef = ObjectPoolRef<Material>;

// Configure a `materialx` Material as a default Autodesk standard_surface:
// authors an inline instantiation document (the nodedef comes from the
// device-resolved MaterialX distribution) and exposes a curated set of its
// inputs as editable parameters (clean MaterialX names; the device remaps
// them).
void applyMaterialXStandardSurfacePreset(Material &m);

namespace tokens::material {

extern const Token matte;
extern const Token physicallyBased;
extern const Token physicallyBasedMDL;
extern const Token mdl;
extern const Token materialx;

} // namespace tokens::material

} // namespace tsd::scene
