// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "ArgumentBlockDescriptor.h"

#include "Core.h"

#include <mi/base/handle.h>
#include <mi/base/ilogger.h>
#include <mi/neuraylib/argument_editor.h>
#include <mi/neuraylib/imdl_backend.h>
#include <mi/neuraylib/imdl_factory.h>
#include <mi/neuraylib/itransaction.h>
#include <mi/neuraylib/ivalue.h>

#include <algorithm>
#include <nonstd/span.hpp>

#include <cstdint>
#include <string_view>
#include <unordered_map>

namespace visrtx::libmdl {

class ArgumentBlockInstance
{
 public:
  ArgumentBlockInstance() = default;

  ArgumentBlockInstance(
      const ArgumentBlockDescriptor &argumentBlockDescriptor, Core *core);

  using ArgumentType = ArgumentBlockDescriptor::ArgumentType;
  using Argument = ArgumentBlockDescriptor::Argument;

  // Returns the list of supported arguments and their types.
  nonstd::span<const Argument> enumerateArguments() const
  {
    return m_argumentBlockDescriptor.m_arguments;
  }

  bool hasArgument(std::string_view name) const
  {
    return m_argumentBlockDescriptor.m_nameToArgbBlockLayout.count(
               std::string(name))
        > 0;
  }

  void setValue(std::string_view name, bool value);
  void setValue(std::string_view name, int value);
  void setValue(std::string_view name, const int (&value)[2]);
  void setValue(std::string_view name, const int (&value)[3]);
  void setValue(std::string_view name, const int (&value)[4]);
  void setValue(std::string_view name, float value);
  void setValue(std::string_view name, const float (&value)[2]);
  void setValue(std::string_view name, const float (&value)[3]);
  void setValue(std::string_view name, const float (&value)[4]);

  void reset(std::string_view name);

  enum class ColorSpace
  {
    Auto,
    Raw,
    sRGB,
  };

  // Return the up to date argument block content.
  nonstd::span<const std::byte> getArgumentBlockData() const;

  // Resources as processed by our MDL's ResourceCallback implementation.
  template <typename T>
  struct HandleHasher
  {
    std::size_t operator()(const mi::base::Handle<T> &v) const noexcept
    {
      return std::hash<T *>()(v.get());
    }
  };
  using ResourceMapping =
      std::unordered_map<mi::base::Handle<const mi::neuraylib::IValue_resource>,
          std::uint32_t,
          HandleHasher<const mi::neuraylib::IValue_resource>>;
  using ResourceDescriptors =
      std::unordered_map<mi::base::Handle<const mi::neuraylib::IValue_resource>,
          TextureDescriptor,
          HandleHasher<const mi::neuraylib::IValue_resource>>;

 private:
  ArgumentBlockDescriptor m_argumentBlockDescriptor = {};
  mi::base::Handle<mi::neuraylib::ITarget_argument_block> m_argumentBlock = {};
  Core *m_core = {};

  // Some helper functions for setValue
  template <typename T>
  void _setValue(std::string_view name, T value);
  template <typename T, std::size_t S>
  void _setValue(std::string_view name, const T (&value)[S]);
};

} // namespace visrtx::libmdl
