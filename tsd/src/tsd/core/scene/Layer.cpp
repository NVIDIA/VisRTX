// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/scene/Layer.hpp"
#include "tsd/core/scene/Context.hpp"

namespace tsd::core {

LayerNodeData::LayerNodeData(Any v, const char *n) : value(v), name(n)
{
  if (!isObject())
    defaultValue = v;
}

bool LayerNodeData::hasDefault() const
{
  return defaultValue;
}

bool LayerNodeData::isObject() const
{
  return anari::isObject(value.type());
}

bool LayerNodeData::isTransform() const
{
  return value.type() == ANARI_FLOAT32_MAT4;
}

bool LayerNodeData::isEmpty() const
{
  return value;
}

Object *LayerNodeData::getObject(Context *ctx) const
{
  return ctx->getObject(value);
}

math::mat4 LayerNodeData::getTransform() const
{
  return isTransform() ? value.getAs<math::mat4>() : math::IDENTITY_MAT4;
}

} // namespace tsd::core
