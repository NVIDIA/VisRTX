#include "ArgumentBlockInstance.h"

#include "ArgumentBlockDescriptor.h"
#include "fmt/base.h"

#include <mi/base/enums.h>
#include <mi/base/handle.h>
#include <mi/base/ilogger.h>
#include <mi/base/interface_implement.h>
#include <mi/neuraylib/icompiled_material.h>
#include <mi/neuraylib/iimage.h>
#include <mi/neuraylib/imdl_backend.h>
#include <mi/neuraylib/itexture.h>
#include <mi/neuraylib/itransaction.h>
#include <mi/neuraylib/itype.h>
#include <mi/neuraylib/ivalue.h>

#include <fmt/format.h>

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>

using mi::base::make_handle;

namespace visrtx::libmdl {

class ResourceCallback : public mi::base::Interface_implement<
                             mi::neuraylib::ITarget_resource_callback>
{
 public:
  ResourceCallback(const mi::neuraylib::ITarget_code *targetCode,
      mi::neuraylib::ITransaction *transaction,
      ArgumentBlockInstance::ResourceMapping &resourceMapping)
      : m_targetCode(targetCode),
        m_transaction(transaction),
        m_resourceMapping(resourceMapping),
        m_textureCounter(targetCode->get_texture_count())
  {}

  mi::Uint32 get_resource_index(
      const mi::neuraylib::IValue_resource *resource) override
  {
    auto refdResource = mi::base::make_handle_dup(resource);

    if (auto it = m_resourceMapping.find(refdResource);
        it != cend(m_resourceMapping)) {
      return it->second;
    }

    if (auto index =
            m_targetCode->get_known_resource_index(m_transaction, resource)) {
      m_resourceMapping[refdResource] = index;
      return index;
    }

    switch (resource->get_type()->get_kind()) {
    case mi::neuraylib::IType_resource::TK_TEXTURE: {
      auto index = m_textureCounter++;
      m_resourceMapping[refdResource] = index;
      return index;
    }
    default:
      break;
    }

    return 0;
  }

  mi::Uint32 get_string_index(mi::neuraylib::IValue_string const *s) override
  {
    char const *str_val = s->get_value();
    if (str_val == nullptr)
      return 0u;

    for (mi::Size i = 0, n = m_targetCode->get_string_constant_count(); i < n;
        ++i) {
      if (strcmp(m_targetCode->get_string_constant(i), str_val) == 0)
        return mi::Uint32(i);
    }

    // string not known by code
    return 0u;
  }

 private:
  const mi::neuraylib::ITarget_code *m_targetCode;
  mi::neuraylib::ITransaction *m_transaction;
  ArgumentBlockInstance::ResourceMapping &m_resourceMapping;
  std::uint32_t m_textureCounter = 0;
};

ArgumentBlockInstance::ArgumentBlockInstance(
    const ArgumentBlockDescriptor *argumentBlockDescriptor, Core *core)
    : m_argumentBlockDescriptor(argumentBlockDescriptor),
      m_argumentBlock(
          argumentBlockDescriptor->m_argumentBlock.is_valid_interface()
              ? argumentBlockDescriptor->m_argumentBlock->clone()
              : nullptr),
      m_core(core)
{}

auto ArgumentBlockInstance::setValue(std::string_view name,
    bool value,
    mi::neuraylib::ITransaction *transaction,
    mi::neuraylib::IMdl_factory *factory) -> void
{
  auto vf = make_handle(factory->create_value_factory(transaction));

  auto v = make_handle(vf->create_bool(value));
  setValue(name, v.get(), transaction);
}

void ArgumentBlockInstance::setValue(std::string_view name,
    float value,
    mi::neuraylib::ITransaction *transaction,
    mi::neuraylib::IMdl_factory *factory)
{
  auto vf = make_handle(factory->create_value_factory(transaction));

  auto v = make_handle(vf->create_float(value));
  setValue(name, v.get(), transaction);
}

void ArgumentBlockInstance::setValue(std::string_view name,
    int value,
    mi::neuraylib::ITransaction *transaction,
    mi::neuraylib::IMdl_factory *factory)
{
  auto vf = make_handle(factory->create_value_factory(transaction));

  auto v = make_handle(vf->create_int(value));
  setValue(name, v.get(), transaction);
}

void ArgumentBlockInstance::setColorValue(std::string_view name,
    const float (&value)[3],
    mi::neuraylib::ITransaction *transaction,
    mi::neuraylib::IMdl_factory *factory)
{
  auto vf = make_handle(factory->create_value_factory(transaction));

  auto v = make_handle(vf->create_color(value[0], value[1], value[2]));
  setValue(name, v.get(), transaction);
}

void ArgumentBlockInstance::setTextureValue(std::string_view name,
    std::string_view filePath,
    mi::neuraylib::ITransaction *transaction,
    mi::neuraylib::IMdl_factory *factory)
{
  auto textureDbName = std::string(filePath);
  if (auto texture = mi::base::make_handle(
          transaction->access<mi::neuraylib::ITexture>(textureDbName.c_str()));
      !texture.is_valid_interface()) {
    // Texture has never been loaded. Load it now.
    auto newTexture = mi::base::make_handle(
        transaction->create<mi::neuraylib::ITexture>("Texture"));
    auto image = mi::base::make_handle(
        transaction->create<mi::neuraylib::IImage>("Image"));
    fmt::println(stderr, "  Setting texture to path {}", filePath);
    if (image->reset_file(std::string(filePath).c_str()) != 0) {
      if (m_core)
        m_core->logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
            fmt::format("Cannot load texture `{}`", filePath).c_str());
      return;
    }

    auto textureDbName = std::string(filePath);
    auto imageDbName = textureDbName + "_image";
    transaction->store(image.get(), imageDbName.c_str());
    newTexture->set_image(imageDbName.c_str());
    // texture->set_gamma(gamma);
    transaction->store(newTexture.get(), textureDbName.c_str());
  }

  auto vf = make_handle(factory->create_value_factory(transaction));
  auto tf = vf->get_type_factory();

  auto t = make_handle(tf->create_texture(mi::neuraylib::IType_texture::TS_2D));
  auto v = make_handle(vf->create_texture(t.get(), textureDbName.c_str()));

  setValue(name, v.get(), transaction);
}

nonstd::span<const std::byte> ArgumentBlockInstance::getArgumentBlockData()
    const
{
  if (m_argumentBlock.is_valid_interface()) {
    return {reinterpret_cast<const std::byte *>(m_argumentBlock->get_data()),
        m_argumentBlock->get_size()};
  } else {
    return {nullptr, 0};
  }
}

void ArgumentBlockInstance::resetResources()
{
  m_textureResourceMapping.clear();
}

void ArgumentBlockInstance::setValue(std::string_view name,
    const mi::neuraylib::IValue *value,
    mi::neuraylib::ITransaction *transaction)
{
  if (auto it =
          m_argumentBlockDescriptor->m_nameToLayout.find(std::string(name));
      it != cend(m_argumentBlockDescriptor->m_nameToLayout)) {
    auto res = m_argumentBlockDescriptor->m_argumentBlockLayout->set_value(
        m_argumentBlock->get_data(),
        value,
        make_handle(
            new ResourceCallback(m_argumentBlockDescriptor->m_targetCode.get(),
                transaction,
                m_textureResourceMapping))
            .get(),
        it->second);

    if (m_core) {
      switch (res) {
      case 0: // OK
        break;
      case -1:
        m_core->logMessage(mi::base::MESSAGE_SEVERITY_WARNING,
            "resources",
            fmt::format(
                "Cannot set parameter {}: Invalid parameters, block or value is a NULL pointer",
                name)
                .c_str());
        break;
      case -2:
        m_core->logMessage(mi::base::MESSAGE_SEVERITY_WARNING,
            "resources",
            fmt::format("Cannot set parameter {}: Invalid state provided", name)
                .c_str());
        break;
      case -3:
        m_core->logMessage(mi::base::MESSAGE_SEVERITY_WARNING,
            "resources",
            fmt::format(
                "Cannot set parameter {}: Value kind does not match expected kind",
                name)
                .c_str());
        break;
      case -4:
        m_core->logMessage(mi::base::MESSAGE_SEVERITY_WARNING,
            "resources",
            fmt::format(
                "Cannot set parameter {}: Size of compound value does not match expected size",
                name)
                .c_str());
        break;
      case -5:
        m_core->logMessage(mi::base::MESSAGE_SEVERITY_WARNING,
            "resources",
            fmt::format("Cannot set parameter {}: Unsupported value type", name)
                .c_str());
        break;
      }
    }
  } else {
    if (m_core)
      m_core->logMessage(mi::base::MESSAGE_SEVERITY_WARNING,
          "resources",
          fmt::format(
              "Cannot set parameter: Cannot find material instance parameter named {}",
              name)
              .c_str());
  }
}

std::vector<std::string> ArgumentBlockInstance::getTextureResourceNames() const
{
  std::vector<std::string> resources =
      m_argumentBlockDescriptor->m_defaultAndBodyTextureNames;
  auto maxIdxIt = std::max_element(cbegin(m_textureResourceMapping),
      cend(m_textureResourceMapping),
      [](const auto &a, const auto &b) { return a.second < b.second; });
  if (maxIdxIt != cend(m_textureResourceMapping))
    resources.resize(
        std::max(resources.size(), std::size_t(maxIdxIt->second + 1)));
  for (auto &&[res, idx] : m_textureResourceMapping) {
    resources[idx] = res->get_value();
  }

  return resources;
};

} // namespace visrtx::libmdl
