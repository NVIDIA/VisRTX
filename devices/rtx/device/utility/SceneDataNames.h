// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace visrtx {

/*
 * Device-local, exact string identity. Tokens never move or get reused during
 * the device lifetime; no SDK string IDs or worker-owned state enter this map.
 */
class SceneDataNames
{
 public:
  uint32_t intern(std::string_view name);

 private:
  std::mutex m_mutex;
  std::unordered_map<std::string, uint32_t> m_tokens;
};

// Inlined definitions ////////////////////////////////////////////////////////

inline uint32_t SceneDataNames::intern(std::string_view name)
{
  if (name.empty())
    return 0;
  const std::lock_guard<std::mutex> lock(m_mutex);
  const auto found = m_tokens.find(std::string(name));
  if (found != m_tokens.end())
    return found->second;
  if (m_tokens.size() >= std::numeric_limits<uint32_t>::max())
    throw std::runtime_error("MDL scene-data name token capacity exhausted");
  const auto token = static_cast<uint32_t>(m_tokens.size() + 1);
  m_tokens.emplace(name, token);
  return token;
}

} // namespace visrtx
