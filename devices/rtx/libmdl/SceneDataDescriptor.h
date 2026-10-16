// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mi::neuraylib {
class ITarget_code;
}

namespace visrtx::libmdl {

struct SceneDataDescriptor
{
  using Id = std::uint32_t;

  SceneDataDescriptor() = default;
  explicit SceneDataDescriptor(const mi::neuraylib::ITarget_code &targetCode);

  SceneDataDescriptor(const SceneDataDescriptor &) = default;
  SceneDataDescriptor &operator=(const SceneDataDescriptor &) = default;
  SceneDataDescriptor(SceneDataDescriptor &&) = default;
  SceneDataDescriptor &operator=(SceneDataDescriptor &&) = default;

  std::optional<Id> find(std::string_view name) const;

  // Own the target-code strings at their original IDs, including empty slots.
  // Not every string is necessarily used by a scene-data lookup.
  std::vector<std::string> m_namesById;
};

} // namespace visrtx::libmdl
