// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "ArgumentBlockDescriptor.h"

#include "Core.h"
#include "types.h"

#include <mi/base/enums.h>
#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>
#include <mi/neuraylib/icompiled_material.h>
#include <mi/neuraylib/imdl_backend.h>
#include <mi/neuraylib/itransaction.h>
#include <mi/neuraylib/itype.h>
#include <mi/neuraylib/ivalue.h>

#include <fmt/base.h>

#include <cstdint>
#include <cstdio>
#include <unordered_map>
#include <vector>

namespace visrtx::libmdl {

ArgumentBlockDescriptor::ArgumentBlockDescriptor(libmdl::Core *core,
    const mi::neuraylib::ICompiled_material *compiledMaterial,
    const mi::neuraylib::ITarget_code *targetCode,
    const std::vector<TextureDescriptor> &defaultAndBodyTextureDescs,
    std::uint32_t argumentBlockIndex)
    : m_targetCode(targetCode, mi::base::DUP_INTERFACE),
      m_argumentBlockLayout(
          targetCode->get_argument_block_layout(argumentBlockIndex),
          mi::base::DUP_INTERFACE),
      m_argumentBlock(targetCode->get_argument_block(argumentBlockIndex)),
      m_defaultAndBodyTextureDescriptors(defaultAndBodyTextureDescs)
{
  for (auto i = 0ull, num_params = compiledMaterial->get_parameter_count();
      i < num_params;
      ++i) {
    auto name = compiledMaterial->get_parameter_name(i);
    if (!name)
      continue;

    auto argument = make_handle(compiledMaterial->get_argument(i));
    switch (argument->get_kind()) {
    case mi::neuraylib::IValue::VK_BOOL: {
      m_arguments.push_back({name, ArgumentType::Bool});
      break;
    }
    case mi::neuraylib::IValue::VK_INT: {
      m_arguments.push_back({name, ArgumentType::Int});
      break;
    }
    case mi::neuraylib::IValue::VK_FLOAT: {
      m_arguments.push_back({name, ArgumentType::Float});
      break;
    }
    case mi::neuraylib::IValue::VK_VECTOR: {
      auto ivector =
          make_handle(argument->get_interface<mi::neuraylib::IValue_vector>());
      auto size = ivector->get_size();
      auto type = make_handle(ivector->get_type());
      auto element_type = make_handle(type->get_element_type());
      switch (size) {
      case 2: {
        switch (element_type->get_kind()) {
        case mi::neuraylib::IType::TK_INT: {
          m_arguments.push_back({name, ArgumentType::Int2});
          break;
        }
        case mi::neuraylib::IType::TK_FLOAT: {
          m_arguments.push_back({name, ArgumentType::Float2});
          break;
        }
        default: {
          core->logMessage(mi::base::MESSAGE_SEVERITY_WARNING,
              "Unsupport vector type {}, ignoring",
              int(element_type->get_kind()));
          break;
        }
        }
        break;
      }
      case 3: {
        switch (element_type->get_kind()) {
        case mi::neuraylib::IType::TK_INT: {
          m_arguments.push_back({name, ArgumentType::Int3});
          break;
        }
        case mi::neuraylib::IType::TK_FLOAT: {
          m_arguments.push_back({name, ArgumentType::Float3});
          break;
        }
        default: {
          core->logMessage(mi::base::MESSAGE_SEVERITY_WARNING,
              "Unsupport vector type {}, ignoring",
              int(element_type->get_kind()));
          break;
        }
        }
        break;
      }
      case 4: {
        switch (element_type->get_kind()) {
        case mi::neuraylib::IType::TK_INT: {
          m_arguments.push_back({name, ArgumentType::Int4});
          break;
        }
        case mi::neuraylib::IType::TK_FLOAT: {
          m_arguments.push_back({name, ArgumentType::Float4});
          break;
        }
        default: {
          core->logMessage(mi::base::MESSAGE_SEVERITY_WARNING,
              "Unsupport vector type {}, ignoring",
              int(element_type->get_kind()));
          break;
        }
        }
        break;
      }
      default: {
        core->logMessage(mi::base::MESSAGE_SEVERITY_WARNING,
            "Unsupport vector size {}, ignoring",
            size);
        break;
      }
      }
      break;
    }

    case mi::neuraylib::IValue::VK_COLOR: {
      m_arguments.push_back({name, ArgumentType::Color});
      break;
    }
    case mi::neuraylib::IValue::VK_TEXTURE: {
      m_arguments.push_back({name, ArgumentType::Texture});
      break;
    }
    case mi::neuraylib::IValue::VK_ENUM: {
      m_arguments.push_back({name, ArgumentType::Int});
      break;
    }
    default:
      continue;
    }

    mi::neuraylib::IValue::Kind kind2;
    mi::Size param_size;
    mi::Size offset = m_argumentBlockLayout->get_layout(
        kind2, param_size, m_argumentBlockLayout->get_nested_state(i));
    m_nameToArgbBlockLayout[name] = { offset, param_size };
  }
}

} // namespace visrtx::libmdl
