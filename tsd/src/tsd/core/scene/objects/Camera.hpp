// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/scene/Object.hpp"

namespace tsd::core {

struct Camera : public Object
{
  static constexpr anari::DataType ANARI_TYPE = ANARI_CAMERA;

  DECLARE_OBJECT_DEFAULT_LIFETIME(Camera);

  Camera(Token subtype = tokens::unknown);
  virtual ~Camera() override = default;

  ObjectPoolRef<Camera> self() const;

  anari::Object makeANARIObject(anari::Device d) const override;
};

using CameraRef = ObjectPoolRef<Camera>;

namespace tokens::camera {

extern const Token perspective;
extern const Token orthographic;
extern const Token omnidirectional;

} // namespace tokens::camera

} // namespace tsd::core
