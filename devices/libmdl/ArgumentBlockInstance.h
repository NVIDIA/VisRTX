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
      const ArgumentBlockDescriptor *argumentBlockDescriptor, Core *core);

  using ArgumentType = ArgumentBlockDescriptor::ArgumentType;
  using Argument = ArgumentBlockDescriptor::Argument;

  // Returns the list of supported arguments and their types.
  nonstd::span<const Argument> enumerateArguments() const
  {
    return m_argumentBlockDescriptor->m_arguments;
  }

  // Generic setter for given value.
  void setValue(std::string_view name,
      const mi::neuraylib::IValue *value,
      mi::neuraylib::ITransaction *transaction);

  // Shorthand setters creating resources in the database before setting the
  // matching parameter.
  void setValue(std::string_view name,
      bool value,
      mi::neuraylib::ITransaction *transaction,
      mi::neuraylib::IMdl_factory *factory);
  void setValue(std::string_view name,
      float value,
      mi::neuraylib::ITransaction *transaction,
      mi::neuraylib::IMdl_factory *factory);
  void setValue(std::string_view name,
      int value,
      mi::neuraylib::ITransaction *transaction,
      mi::neuraylib::IMdl_factory *factory);
  void setColorValue(std::string_view name,
      const float (&value)[3],
      mi::neuraylib::ITransaction *transaction,
      mi::neuraylib::IMdl_factory *factory);
  void setTextureValue(std::string_view name,
      std::string_view pathName,
      mi::neuraylib::ITransaction *transaction,
      mi::neuraylib::IMdl_factory *factory);

  // Return the up to date argument block content.
  nonstd::span<const std::byte> getArgumentBlockData() const;

  // Reset resources prior to setValue calls
  void resetResources();
  // Returns the database names of the resources as they match the setValue
  // calls.
  std::vector<std::string> getTextureResourceNames() const;

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

 private:
  const ArgumentBlockDescriptor *m_argumentBlockDescriptor = {};
  mi::base::Handle<mi::neuraylib::ITarget_argument_block> m_argumentBlock = {};
  ResourceMapping m_textureResourceMapping = {};
  Core *m_core = {};
};

} // namespace visrtx::libmdl
