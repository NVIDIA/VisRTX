// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "Core.h"
#include "types.h"


#include <mi/base/handle.h>
#include <mi/neuraylib/argument_editor.h>
#include <mi/neuraylib/icompiled_material.h>
#include <mi/neuraylib/imdl_backend.h>
#include <mi/neuraylib/imdl_factory.h>
#include <mi/neuraylib/itransaction.h>
#include <mi/neuraylib/ivalue.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace visrtx::libmdl {

struct ArgumentBlockDescriptor
{
  enum class ArgumentType
  {
    Bool,
    Int,
    Int2,
    Int3,
    Int4,
    Float,
    Float2,
    Float3,
    Float4,
    Color,
    Texture,
  };

  struct Argument
  {
    std::string name;
    ArgumentType type;
  };

  struct ArgumentLayout
  {
    std::size_t offset;
    std::size_t size;
  };

  ArgumentBlockDescriptor() = default;

  ArgumentBlockDescriptor(libmdl::Core *core,
      const mi::neuraylib::ICompiled_material *compiledMaterial,
      const mi::neuraylib::ITarget_code *targetCode,
      const std::vector<TextureDescriptor> &textureDescs,
      std::uint32_t argumentBlockIndex = 0);

  mi::base::Handle<const mi::neuraylib::ITarget_code> m_targetCode;
  mi::base::Handle<const mi::neuraylib::ITarget_value_layout>
      m_argumentBlockLayout;
  mi::base::Handle<const mi::neuraylib::ITarget_argument_block> m_argumentBlock;

  std::vector<Argument> m_arguments;
  std::vector<TextureDescriptor> m_defaultAndBodyTextureDescriptors;
  std::unordered_map<std::string, ArgumentLayout> m_nameToArgbBlockLayout;
};

} // namespace visrtx::libmdl
