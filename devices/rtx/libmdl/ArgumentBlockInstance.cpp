// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "ArgumentBlockInstance.h"

#include "ArgumentBlockDescriptor.h"

#include <mi/base/enums.h>
#include <mi/base/handle.h>
#include <mi/base/ilogger.h>
#include <mi/base/interface_implement.h>
#include <mi/base/types.h>
#include <mi/neuraylib/icanvas.h>
#include <mi/neuraylib/icompiled_material.h>
#include <mi/neuraylib/iimage.h>
#include <mi/neuraylib/iimage_api.h>
#include <mi/neuraylib/imdl_backend.h>
#include <mi/neuraylib/itexture.h>
#include <mi/neuraylib/itile.h>
#include <mi/neuraylib/itransaction.h>
#include <mi/neuraylib/itype.h>
#include <mi/neuraylib/ivalue.h>

#include <fmt/format.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

using namespace std::string_view_literals;

namespace visrtx::libmdl {

ArgumentBlockInstance::ArgumentBlockInstance(
    const ArgumentBlockDescriptor &argumentBlockDescriptor, Core *core)
    : m_argumentBlockDescriptor(argumentBlockDescriptor),
      m_argumentBlock(
          argumentBlockDescriptor.m_argumentBlock.is_valid_interface()
              ? argumentBlockDescriptor.m_argumentBlock->clone()
              : nullptr),
      m_core(core)
{
}

template<typename T>
auto ArgumentBlockInstance::_setValue(std::string_view name, T value) -> void
{
  assert(m_argumentBlock.is_valid_interface() && m_argumentBlock->get_data());
  auto it = m_argumentBlockDescriptor.m_nameToArgbBlockLayout.find(std::string(name));
  if (it == cend(m_argumentBlockDescriptor.m_nameToArgbBlockLayout)) return;

  auto data = reinterpret_cast<T*>(m_argumentBlock->get_data() + it->second.offset);
  *data = value;
}

template<typename T, std::size_t S>
auto ArgumentBlockInstance::_setValue(std::string_view name, const T (&value)[S]) -> void
{
  assert(m_argumentBlock.is_valid_interface() && m_argumentBlock->get_data());
  auto it = m_argumentBlockDescriptor.m_nameToArgbBlockLayout.find(std::string(name));
  if (it == cend(m_argumentBlockDescriptor.m_nameToArgbBlockLayout)) return;

  auto data = reinterpret_cast<T*>(m_argumentBlock->get_data() + it->second.offset);
  for (auto i = 0; i < S; ++i)
    data[i] = value[i];
}

auto ArgumentBlockInstance::setValue(std::string_view name, bool value) -> void {
  _setValue(name, value);
}

auto ArgumentBlockInstance::setValue(std::string_view name, int value) -> void {
  _setValue(name, value);
}

auto ArgumentBlockInstance::setValue(std::string_view name, const int (&value)[2]) -> void {
  _setValue(name, value);
}

auto ArgumentBlockInstance::setValue(std::string_view name, const int (&value)[3]) -> void {
  _setValue(name, value);
}

auto ArgumentBlockInstance::setValue(std::string_view name, const int (&value)[4]) -> void {
  _setValue(name, value);
}

auto ArgumentBlockInstance::setValue(std::string_view name, float value) -> void {
  _setValue(name, value);
}

auto ArgumentBlockInstance::setValue(std::string_view name, const float (&value)[2]) -> void {
  _setValue(name, value);
}

auto ArgumentBlockInstance::setValue(std::string_view name, const float (&value)[3]) -> void {
  _setValue(name, value);
}

auto ArgumentBlockInstance::setValue(std::string_view name, const float (&value)[4]) -> void {
  _setValue(name, value);
}

auto ArgumentBlockInstance::reset(std::string_view name) -> void
{
  auto argumentLayoutIt = m_argumentBlockDescriptor.m_nameToArgbBlockLayout.find(std::string(name));
  if (argumentLayoutIt == cend(m_argumentBlockDescriptor.m_nameToArgbBlockLayout)) return;

  std::memcpy(m_argumentBlock->get_data() + argumentLayoutIt->second.offset,
      m_argumentBlockDescriptor.m_argumentBlock->get_data() + argumentLayoutIt->second.offset,
      argumentLayoutIt->second.size);
}

auto ArgumentBlockInstance::getFloat3Value(std::string_view name) const
    -> std::optional<std::array<float, 3>>
{
  if (!m_argumentBlock.is_valid_interface())
    return std::nullopt;

  const auto &args = m_argumentBlockDescriptor.m_arguments;
  auto argIt = std::find_if(cbegin(args), cend(args), [&](const auto &arg) {
    return arg.name == name;
  });
  if (argIt == cend(args))
    return std::nullopt;
  if (argIt->type != ArgumentType::Color && argIt->type != ArgumentType::Float3
      && argIt->type != ArgumentType::Float)
    return std::nullopt;

  auto layoutIt = m_argumentBlockDescriptor.m_nameToArgbBlockLayout.find(
      std::string(name));
  if (layoutIt == cend(m_argumentBlockDescriptor.m_nameToArgbBlockLayout))
    return std::nullopt;

  // Type and layout come from the same descriptor so they cannot disagree,
  // but this is the only READ of the block — bound the access anyway.
  const std::size_t needed =
      (argIt->type == ArgumentType::Float ? 1 : 3) * sizeof(float);
  if (layoutIt->second.size < needed)
    return std::nullopt;

  const auto *data = reinterpret_cast<const float *>(
      m_argumentBlock->get_data() + layoutIt->second.offset);
  if (argIt->type == ArgumentType::Float)
    return std::array<float, 3>{data[0], data[0], data[0]};
  return std::array<float, 3>{data[0], data[1], data[2]};
}

nonstd::span<const std::byte> ArgumentBlockInstance::getArgumentBlockData()
    const
{
  if (m_argumentBlock.is_valid_interface()) {
    return {reinterpret_cast<const std::byte *>(m_argumentBlock->get_data()),
        m_argumentBlock->get_size()};
  } else {
    return {static_cast<const std::byte*>(nullptr), 0};
  }
}

} // namespace visrtx::libmdl
