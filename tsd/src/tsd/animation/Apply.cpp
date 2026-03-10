// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Animation.hpp"
#include "tsd/scene/Scene.hpp"

namespace tsd::animation {

void applyResults(EvaluationResult &result, tsd::scene::Scene &scene)
{
  for (auto &sub : result.parameters) {
    if (!sub.target)
      continue;
    if (anari::isObject(sub.value.type())) {
      auto *obj = scene.getObject(sub.value);
      if (obj)
        sub.target->setParameterObject(sub.paramName, *obj);
    } else {
      sub.target->setParameter(
          sub.paramName, sub.value.type(), sub.value.data());
    }
  }

  for (auto &sub : result.transforms) {
    if (!sub.target)
      continue;
    (*sub.target)->setAsTransform(sub.transform);
    scene.signalLayerTransformChanged(sub.target->value().layer());
  }
}

} // namespace tsd::animation
