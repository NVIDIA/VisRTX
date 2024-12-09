#include "ArgumentBlockDescriptor.h"

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>
#include <mi/neuraylib/icompiled_material.h>
#include <mi/neuraylib/imdl_backend.h>
#include <mi/neuraylib/itransaction.h>
#include <mi/neuraylib/itype.h>
#include <mi/neuraylib/ivalue.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace visrtx::libmdl {

ArgumentBlockDescriptor::ArgumentBlockDescriptor(
    const mi::neuraylib::ICompiled_material *compiledMaterial,
    const mi::neuraylib::ITarget_code *targetCode,
    const std::vector<std::string> &defaultAndBodyTextureNames,
    std::uint32_t argumentBlockIndex)
    : m_targetCode(targetCode, mi::base::DUP_INTERFACE),
      m_argumentBlockLayout(
          targetCode->get_argument_block_layout(argumentBlockIndex),
          mi::base::DUP_INTERFACE),
      m_argumentBlock(targetCode->get_argument_block(argumentBlockIndex)),
      m_defaultAndBodyTextureNames(defaultAndBodyTextureNames)
{
  for (auto i = 0ull, num_params = compiledMaterial->get_parameter_count();
      i < num_params;
      ++i) {
    auto name = compiledMaterial->get_parameter_name(i);
    if (!name)
      continue;

    auto argument = mi::base::make_handle(compiledMaterial->get_argument(i));
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
    case mi::neuraylib::IValue::VK_COLOR: {
      m_arguments.push_back({name, ArgumentType::Color});
      break;
    }
    case mi::neuraylib::IValue::VK_TEXTURE: {
      m_arguments.push_back({name, ArgumentType::Texture});
      break;
    }
    default:
      continue;
    }
    m_nameToLayout[name] = m_argumentBlockLayout->get_nested_state(i);
  }
}

} // namespace visrtx::libmdl
