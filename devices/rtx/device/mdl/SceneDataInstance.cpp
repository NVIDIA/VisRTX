// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "SceneDataInstance.h"

#include <algorithm>
#include <utility>

namespace visrtx::mdl {

SceneDataInstance::SceneDataInstance(
    const libmdl::SceneDataDescriptor &descriptor)
    : m_descriptor(descriptor), m_bindings(descriptor.m_namesById.size())
{}

nonstd::span<const MDLSceneDataBinding> SceneDataInstance::data() const
{
  return m_bindings;
}

bool SceneDataInstance::update(nonstd::span<const NamedBinding> bindings)
{
  std::vector<MDLSceneDataBinding> next(m_descriptor.m_namesById.size());
  for (const auto &binding : bindings) {
    const auto id = m_descriptor.find(binding.name);
    if (!id)
      continue;

    auto &attribute = next[*id].attribute;
    switch (binding.attribute) {
    case MaterialAttribute::ATTRIB_0:
    case MaterialAttribute::ATTRIB_1:
    case MaterialAttribute::ATTRIB_2:
    case MaterialAttribute::ATTRIB_3:
    case MaterialAttribute::COLOR:
      attribute = binding.attribute;
      break;
    default:
      attribute = MaterialAttribute::UNKNOWN;
      break;
    }
  }

  if (std::equal(next.begin(),
          next.end(),
          m_bindings.begin(),
          [](const auto &a, const auto &b) {
            return a.attribute == b.attribute;
          }))
    return false;

  m_bindings = std::move(next);
  return true;
}

} // namespace visrtx::mdl
