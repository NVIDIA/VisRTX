// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/scene/objects/Transform.hpp"
#include "tsd/core/scene/Scene.hpp"

namespace tsd::core {

Transform::Transform(Token subtype) : Object(TSD_TRANSFORM, subtype)
{
  addParameter(tokens::transform::transform).setValue(math::IDENTITY_MAT4);
}

void Transform::setTransform(const math::mat4 &m)
{
  setParameter(tokens::transform::transform, m);
}

math::mat4 Transform::getTransform() const
{
  auto *p = parameter(tokens::transform::transform);
  if (!p || p->value().type() != ANARI_FLOAT32_MAT4)
    return math::IDENTITY_MAT4;
  return p->value().get<math::mat4>();
}

void Transform::setTransformArray(ArrayRef transforms)
{
  if (!transforms.valid())
    return;
  setParameterObject(tokens::transform::transform, *transforms);
}

Array *Transform::getTransformArray() const
{
  return parameterValueAsObject<Array>(tokens::transform::transform);
}

Any Transform::getTransformAsAny() const
{
  auto *p = parameter(tokens::transform::transform);
  if (!p)
    return Any(math::IDENTITY_MAT4);
  return p->value();
}

void Transform::setTransform(const math::mat3 &srt)
{
  auto &sc = srt[0];
  auto &azelrot = srt[1];
  auto &tl = srt[2];

  auto rot = math::IDENTITY_MAT4;
  rot = math::mul(rot,
      math::rotation_matrix(math::rotation_quat(
          math::float3(0.f, 1.f, 0.f), math::radians(azelrot.x))));
  rot = math::mul(rot,
      math::rotation_matrix(math::rotation_quat(
          math::float3(1.f, 0.f, 0.f), math::radians(azelrot.y))));
  rot = math::mul(rot,
      math::rotation_matrix(math::rotation_quat(
          math::float3(0.f, 0.f, 1.f), math::radians(azelrot.z))));

  auto m = math::mul(
      math::translation_matrix(tl), math::mul(rot, math::scaling_matrix(sc)));

  setTransform(m);
}

ObjectPoolRef<Transform> Transform::self() const
{
  return scene() ? scene()->getObject<Transform>(index())
                 : ObjectPoolRef<Transform>{};
}

anari::Object Transform::makeANARIObject(anari::Device d) const
{
  return {};
}

Transform::InstanceParameterMap Transform::getInstanceParameterMap() const
{
  InstanceParameterMap res;

  // Go with a fixed set of possible parameters.
  if (auto p = parameter(tokens::transform::id)) {
    res[tokens::transform::id] = p->value();
  }
  if (auto p = parameter(tokens::transform::color)) {
    res[tokens::transform::color] = p->value();
  }
  if (auto p = parameter(tokens::transform::attribute0)) {
    res[tokens::transform::attribute0] = p->value();
  }
  if (auto p = parameter(tokens::transform::attribute1)) {
    res[tokens::transform::attribute1] = p->value();
  }
  if (auto p = parameter(tokens::transform::attribute2)) {
    res[tokens::transform::attribute2] = p->value();
  }
  if (auto p = parameter(tokens::transform::attribute3)) {
    res[tokens::transform::attribute3] = p->value();
  }

  return res;
}

} // namespace tsd::core

namespace tsd::core::tokens::transform {

const Token transform = "transform";
const Token id = "id";
const Token color = "color";
const Token attribute0 = "attribute0";
const Token attribute1 = "attribute1";
const Token attribute2 = "attribute2";
const Token attribute3 = "attribute3";

} // namespace tsd::core::tokens::transform
