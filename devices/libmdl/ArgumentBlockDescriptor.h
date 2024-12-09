#pragma once

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
  ArgumentBlockDescriptor() = default;

  ArgumentBlockDescriptor(
      const mi::neuraylib::ICompiled_material *compiledMaterial,
      const mi::neuraylib::ITarget_code *targetCode,
      const std::vector<std::string> &defaultAndBodyTextureNames,
      std::uint32_t argumentBlockIndex = 0);

  enum class ArgumentType
  {
    Bool,
    Int,
    Float,
    Color,
    Texture,
  };

  struct Argument
  {
    std::string name;
    ArgumentType type;
  };

  mi::base::Handle<const mi::neuraylib::ITarget_code> m_targetCode;
  mi::base::Handle<const mi::neuraylib::ITarget_value_layout>
      m_argumentBlockLayout;
  mi::base::Handle<const mi::neuraylib::ITarget_argument_block> m_argumentBlock;

  std::vector<Argument> m_arguments;
  std::vector<std::string> m_defaultAndBodyTextureNames;
  std::unordered_map<std::string, mi::neuraylib::Target_value_layout_state>
      m_nameToLayout;
};

} // namespace visrtx::libmdl
