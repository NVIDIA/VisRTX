// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "SceneDataDescriptor.h"

#include <mi/neuraylib/imdl_backend.h>

#include <limits>
#include <stdexcept>

namespace visrtx::libmdl {

SceneDataDescriptor::SceneDataDescriptor(
    const mi::neuraylib::ITarget_code &targetCode)
{
  const auto count = targetCode.get_string_constant_count();
  if (count > std::numeric_limits<Id>::max())
    throw std::runtime_error("MDL string table exceeds scene-data ID capacity");

  m_namesById.resize(count);
  for (Id id = 0; id < count; ++id) {
    if (const char *name = targetCode.get_string_constant(id))
      m_namesById[id] = name;
  }
}

std::optional<SceneDataDescriptor::Id> SceneDataDescriptor::find(
    std::string_view name) const
{
  if (name.empty())
    return {};
  for (std::size_t id = 0; id < m_namesById.size(); ++id) {
    if (m_namesById[id] == name)
      return static_cast<Id>(id);
  }
  return {};
}

} // namespace visrtx::libmdl
