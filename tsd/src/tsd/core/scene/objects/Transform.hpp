// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anari/anari_cpp/Traits.h>
#include "tsd/core/TSDTypes.hpp"
#include "tsd/core/scene/objects/Array.hpp"

namespace tsd::core {

namespace tokens::transform {

extern const Token transform;
extern const Token id;
extern const Token color;
extern const Token attribute0;
extern const Token attribute1;
extern const Token attribute2;
extern const Token attribute3;

} // namespace tokens::transform

struct Transform : public Object
{
  using InstanceParameterMap = FlatMap<Token, Any>;

  DECLARE_OBJECT_DEFAULT_LIFETIME(Transform);

  Transform(Token subtype = tokens::transform::transform);
  virtual ~Transform() = default;

  void setTransform(const math::mat3 &srt);
  void setTransform(const math::mat4 &m);
  math::mat4 getTransform() const;

  void setTransformArray(ArrayRef transforms);
  Array *getTransformArray() const;

  Any getTransformAsAny() const;

  ObjectPoolRef<Transform> self() const;

  anari::Object makeANARIObject(anari::Device d) const override;

  InstanceParameterMap getInstanceParameterMap() const;
};

using TransformRef = ObjectPoolRef<Transform>;

} // namespace tsd::core

namespace anari {
ANARI_TYPEFOR_SPECIALIZATION(tsd::core::Transform, tsd::core::TSD_TRANSFORM);
} // namespace anari
