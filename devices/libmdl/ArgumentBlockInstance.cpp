#include "ArgumentBlockInstance.h"

#include "ArgumentBlockDescriptor.h"
#include "fmt/base.h"

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
#include <type_traits>
#include <vector>

using mi::base::make_handle;
using namespace std::string_view_literals;

namespace visrtx::libmdl {

class ResourceCallback : public mi::base::Interface_implement<
                             mi::neuraylib::ITarget_resource_callback>
{
 public:
  ResourceCallback(const mi::neuraylib::ITarget_code *targetCode,
      mi::neuraylib::ITransaction *transaction,
      ArgumentBlockInstance::ResourceMapping &resourceMapping,
      std::uint32_t &textureCounter)
      : m_targetCode(targetCode),
        m_transaction(transaction),
        m_resourceMapping(resourceMapping),
        m_textureCounter(textureCounter)
  {}

  mi::Uint32 get_resource_index(
      const mi::neuraylib::IValue_resource *resource) override
  {
    auto refdResource = make_handle_dup(resource);

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
  std::uint32_t &m_textureCounter;
};

ArgumentBlockInstance::ArgumentBlockInstance(
    const ArgumentBlockDescriptor &argumentBlockDescriptor, Core *core)
    : m_argumentBlockDescriptor(argumentBlockDescriptor),
      m_argumentBlock(
          argumentBlockDescriptor.m_argumentBlock.is_valid_interface()
              ? argumentBlockDescriptor.m_argumentBlock->clone()
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

template <typename T, std::size_t S>
void ArgumentBlockInstance::_setValue(std::string_view name,
    const T (&value)[S],
    mi::neuraylib::ITransaction *transaction,
    mi::neuraylib::IMdl_factory *factory)
{
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, int>);

  auto vf = make_handle(factory->create_value_factory(transaction));
  auto tf = vf->get_type_factory();
  auto valuetype = make_handle(std::is_same_v<T, float>
          ? static_cast<const mi::neuraylib::IType_atomic *>(tf->create_float())
          : tf->create_int());
  auto vectortype = make_handle(tf->create_vector(valuetype.get(), 2));

  auto v = make_handle(vf->create_vector(vectortype.get()));
  for (auto i = 0; i < S; ++i) {
    v->set_value(i, make_handle(vf->create_float(value[i])).get());
  }

  setValue(name, v.get(), transaction);
}

void ArgumentBlockInstance::setValue(std::string_view name,
    const int (&value)[2],
    mi::neuraylib::ITransaction *transaction,
    mi::neuraylib::IMdl_factory *factory)
{
  _setValue<int, 2>(name, value, transaction, factory);
}

void ArgumentBlockInstance::setValue(std::string_view name,
    const int (&value)[3],
    mi::neuraylib::ITransaction *transaction,
    mi::neuraylib::IMdl_factory *factory)
{
  _setValue<int, 3>(name, value, transaction, factory);
}

void ArgumentBlockInstance::setValue(std::string_view name,
    const int (&value)[4],
    mi::neuraylib::ITransaction *transaction,
    mi::neuraylib::IMdl_factory *factory)
{
  _setValue<int, 4>(name, value, transaction, factory);
}

void ArgumentBlockInstance::setValue(std::string_view name,
    const float (&value)[2],
    mi::neuraylib::ITransaction *transaction,
    mi::neuraylib::IMdl_factory *factory)
{
  _setValue<float, 2>(name, value, transaction, factory);
}

void ArgumentBlockInstance::setValue(std::string_view name,
    const float (&value)[3],
    mi::neuraylib::ITransaction *transaction,
    mi::neuraylib::IMdl_factory *factory)
{
  _setValue<float, 3>(name, value, transaction, factory);
}

void ArgumentBlockInstance::setValue(std::string_view name,
    const float (&value)[4],
    mi::neuraylib::ITransaction *transaction,
    mi::neuraylib::IMdl_factory *factory)
{
  _setValue<float, 4>(name, value, transaction, factory);
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

void ArgumentBlockInstance::loadTextureToDb(std::string_view filePath,
    ColorSpace colorspace,
    mi::neuraylib::ITransaction *transaction)
{
  auto textureDbName = std::string(filePath);

  auto imageApi = mi::base::make_handle(
      m_core->getINeuray()->get_api_component<mi::neuraylib::IImage_api>());

  auto url = std::string(filePath);

  auto image = make_handle(transaction->create<mi::neuraylib::IImage>("Image"));
  if (image->reset_file(url.c_str()) != 0) {
    if (m_core)
      m_core->logMessage(
          mi::base::MESSAGE_SEVERITY_ERROR, "Cannot load texture `{}`", url);
    return;
  }

  if (image->is_uvtile() || image->is_animated()) {
    m_core->logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
        "Unsupported uvtile and/or animated textures.");
    return;
  }

  auto imageType = image->get_type(0, 0);
  auto imageDbName = fmt::format("{}_image", textureDbName);
  transaction->store(image.get(), imageDbName.c_str());

  auto texture =
      make_handle(transaction->create<mi::neuraylib::ITexture>("Texture"));
  texture->set_image(imageDbName.c_str());

  switch (colorspace) {
  case ColorSpace::sRGB: {
    // FIXME: Is this enough?
    texture->set_gamma(2.2);
    break;
  }
  case ColorSpace::Raw: {
    // FIXME: Is this enough? Will the data be linearized if this input image is
    // not?
    texture->set_gamma(1.0);
    break;
  }
  case ColorSpace::Auto: {
    // That's more or less the heuristic used by USD to guess the colorspace.
    // FIXME: To be complete and correct, we'need to get the actual gamma from
    // the file/canvas/image and see how to act on the texture.

    if (imageType == "Rgb"sv || imageType == "Rgba"sv) {
      texture->set_gamma(2.2);
    } else {
      texture->set_gamma(1.0);
    }
    break;
  }
  }

  transaction->store(texture.get(), textureDbName.c_str());
}

bool ArgumentBlockInstance::loadTextureToDb(
    const libmdl::ArgumentBlockDescriptor::TextureDescriptor &textureDesc,
    mi::neuraylib::ITransaction *transaction)
{
  // Texture has never been loaded. Load its related image and create it now.
  // FIXME: How to handle non image types and compressed textures???
  // Bsdf data are 3d textures
  // Compressed textures should be a 1D array IArray to be stored in the
  // database.
  switch (textureDesc.shape) {
  case libmdl::ArgumentBlockDescriptor::TextureDescriptor::Shape::TwoD:
  case libmdl::ArgumentBlockDescriptor::TextureDescriptor::Shape::ThreeD: {
    auto textureDbName = textureDesc.url;
    auto image =
        make_handle(transaction->create<mi::neuraylib::IImage>("Image"));
    if (image->reset_file(textureDesc.url.c_str()) != 0) {
      if (m_core)
        m_core->logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
            "Cannot load texture `{}`",
            textureDesc.url);
      return {};
    }

    auto imageDbName = fmt::format("{}_image", textureDbName);
    transaction->store(image.get(), imageDbName.c_str());

    auto newTexture =
        make_handle(transaction->create<mi::neuraylib::ITexture>("Texture"));
    newTexture->set_image(imageDbName.c_str());

    switch (textureDesc.colorSpace) {
    case libmdl::ArgumentBlockDescriptor::TextureDescriptor::ColorSpace::sRGB: {
      // FIXME: Is this enough?
      newTexture->set_gamma(2.2);
      break;
    }
    case libmdl::ArgumentBlockDescriptor::TextureDescriptor::ColorSpace::
        Linear: {
      // FIXME: Is this enough? Will the data be linearized if this input image
      // is not?
      newTexture->set_gamma(1.0);
      break;
    }
    }

    transaction->store(newTexture.get(), textureDbName.c_str());
    break;
  }
  case libmdl::ArgumentBlockDescriptor::TextureDescriptor::Shape::Cube: {
    m_core->logMessage(mi::base::MESSAGE_SEVERITY_ERROR, "Cubemap not handled");
    return false;
    break;
  }
  case libmdl::ArgumentBlockDescriptor::TextureDescriptor::Shape::BsdfData: {
    auto textureDbName = textureDesc.url;
    auto imageApi = mi::base::make_handle(
        m_core->getINeuray()->get_api_component<mi::neuraylib::IImage_api>());

    if (textureDesc.colorSpace
        != libmdl::ArgumentBlockDescriptor::TextureDescriptor::ColorSpace::
            Linear) {
      m_core->logMessage(mi::base::MESSAGE_SEVERITY_WARNING,
          "Bsdf data textures must be in linear color space");
    }
    auto canvas = imageApi->create_canvas(
    textureDesc.bsdf.pixelFormat ?  textureDesc.bsdf.pixelFormat : "Float32",
         textureDesc.bsdf.dims[0],
        textureDesc.bsdf.dims[1],
        textureDesc.bsdf.dims[2]);
        
    std::uint64_t colorByteSize = sizeof(float);

    if (textureDesc.bsdf.pixelFormat == "Sint8"sv) {
      colorByteSize = sizeof(mi::Sint8);
    } else if (textureDesc.bsdf.pixelFormat == "Sint32"sv) {
      colorByteSize = sizeof(mi::Sint32);
    } else if (textureDesc.bsdf.pixelFormat == "Float32"sv) {
      colorByteSize = sizeof(mi::Float32);
    } else if (textureDesc.bsdf.pixelFormat == "Float32<2>"sv) {
      colorByteSize = sizeof(mi::Float32_2);
    } else if (textureDesc.bsdf.pixelFormat == "Float32<3>"sv) {
      colorByteSize = sizeof(mi::Float32_3);
    } else if (textureDesc.bsdf.pixelFormat == "Float32<4>"sv) {
      colorByteSize = sizeof(mi::Float32_4);
    } else if (textureDesc.bsdf.pixelFormat == "Rgb"sv) {
      colorByteSize = sizeof(mi::Uint8) * 3;
    } else if (textureDesc.bsdf.pixelFormat == "Rgba"sv) {
      colorByteSize = sizeof(mi::Uint8) * 4;
    } else if (textureDesc.bsdf.pixelFormat == "Rgbe"sv) {
      colorByteSize = sizeof(mi::Uint8) * 4;
    } else if (textureDesc.bsdf.pixelFormat == "Rgbea"sv) {
      colorByteSize = sizeof(mi::Uint8) * 5;
    } else if (textureDesc.bsdf.pixelFormat == "Rgb_16"sv) {
      colorByteSize = sizeof(mi::Uint16) * 4;
    } else if (textureDesc.bsdf.pixelFormat == "Rgba_16"sv) {
      colorByteSize = sizeof(mi::Uint16) * 4;
    } else if (textureDesc.bsdf.pixelFormat == "Rgb_fp"sv) {
      colorByteSize = sizeof(mi::Float32) * 3;
    } else if (textureDesc.bsdf.pixelFormat == "Color"sv) {
      colorByteSize = sizeof(mi::Color);
    }
    

    std::memcpy(canvas->get_tile()->get_data(),
        textureDesc.bsdf.data,
        textureDesc.bsdf.dims[0] * textureDesc.bsdf.dims[1]
            * textureDesc.bsdf.dims[2] * colorByteSize);

    auto image =
        make_handle(transaction->create<mi::neuraylib::IImage>("Image"));
    image->set_from_canvas(canvas);

    auto imageDbName = fmt::format("{}_image", textureDbName);
    transaction->store(image.get(), imageDbName.c_str());

    auto newTexture =
        make_handle(transaction->create<mi::neuraylib::ITexture>("Texture"));
    newTexture->set_image(imageDbName.c_str());
    break;
  }
  case libmdl::ArgumentBlockDescriptor::TextureDescriptor::Shape::PTex:
  case libmdl::ArgumentBlockDescriptor::TextureDescriptor::Shape::Unknown: {
    m_core->logMessage(
        mi::base::MESSAGE_SEVERITY_ERROR, "Unsupported texture shape");
    return false;
  }
  }

  return true;
}

void ArgumentBlockInstance::setTextureValue(std::string_view name,
    std::string_view filePath,
    ColorSpace colorspace,
    mi::neuraylib::ITransaction *transaction,
    mi::neuraylib::IMdl_factory *factory)
{
  auto textureDbName = std::string(filePath);
  auto texture = make_handle(
      transaction->access<mi::neuraylib::ITexture>(textureDbName.c_str()));
  if (!texture.is_valid_interface()) {
    loadTextureToDb(filePath, colorspace, transaction);
    texture = make_handle(
        transaction->access<mi::neuraylib::ITexture>(textureDbName.c_str()));
  }

  if (!texture.is_valid_interface()) {
    m_core->logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
        "Failed to find texture `{}` in the database",
        filePath);
    return;
  }

  auto imageDbName = texture->get_image();

  auto image =
      make_handle(transaction->access<mi::neuraylib::IImage>(imageDbName));
  auto textureType = image->resolution_z(0, 0, 0) == 1
      ? mi::neuraylib::IType_texture::TS_2D
      : mi::neuraylib::IType_texture::TS_3D;

  auto vf = make_handle(factory->create_value_factory(transaction));
  auto tf = vf->get_type_factory();

  auto t = make_handle(tf->create_texture(textureType));
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
  m_textureCounter =
      m_argumentBlockDescriptor.m_targetCode->get_texture_count();
}

void ArgumentBlockInstance::setValue(std::string_view name,
    const mi::neuraylib::IValue *value,
    mi::neuraylib::ITransaction *transaction)
{
  if (auto it =
          m_argumentBlockDescriptor.m_nameToLayout.find(std::string(name));
      it != cend(m_argumentBlockDescriptor.m_nameToLayout)) {
    auto res = m_argumentBlockDescriptor.m_argumentBlockLayout->set_value(
        m_argumentBlock->get_data(),
        value,
        make_handle(
            new ResourceCallback(m_argumentBlockDescriptor.m_targetCode.get(),
                transaction,
                m_textureResourceMapping,
                m_textureCounter))
            .get(),
        it->second);

    if (m_core) {
      switch (res) {
      case 0: // OK
        break;
      case -1:
        m_core->logMessage(mi::base::MESSAGE_SEVERITY_WARNING,

            "Cannot set parameter {}: Invalid parameters, block or value is a NULL pointer",
            name);
        break;
      case -2:
        m_core->logMessage(mi::base::MESSAGE_SEVERITY_WARNING,
            "resources",
            "Cannot set parameter {}: Invalid state provided",
            name);
        break;
      case -3:
        m_core->logMessage(mi::base::MESSAGE_SEVERITY_WARNING,
            "resources",

            "Cannot set parameter {}: Value kind does not match expected kind",
            name);
        break;
      case -4:
        m_core->logMessage(mi::base::MESSAGE_SEVERITY_WARNING,
            "Cannot set parameter {}: Size of compound value does not match expected size",
            name);
        break;
      case -5:
        m_core->logMessage(mi::base::MESSAGE_SEVERITY_WARNING,
            "Cannot set parameter {}: Unsupported value type",
            name);
        break;
      }
    }
  } else {
    if (m_core)
      m_core->logMessage(mi::base::MESSAGE_SEVERITY_WARNING,
          "Cannot set parameter: Cannot find material instance parameter named {}",
          name);
  }
}

void ArgumentBlockInstance::finalizeResourceCreation(
    mi::neuraylib::ITransaction *transaction)
{
  // Starts from default and body textures
  m_textureResourceNames.clear();
  for (const auto &textureDesc :
      m_argumentBlockDescriptor.m_defaultAndBodyTextureDescriptors) {
    m_textureResourceNames.push_back(textureDesc.url);
  }

  // Then update them with actually assigned onces.
  if (!m_textureResourceMapping.empty()) {
    auto maxIdxIt = std::max_element(cbegin(m_textureResourceMapping),
        cend(m_textureResourceMapping),
        [](const auto &a, const auto &b) { return a.second < b.second; });
    if (maxIdxIt != cend(m_textureResourceMapping)) {
      m_textureResourceNames.resize(std::max(
          m_textureResourceNames.size(), std::size_t(maxIdxIt->second)));

      for (auto &&[res, idx] : m_textureResourceMapping) {
        m_textureResourceNames[idx - 1] = res->get_value();
      }
    }
  }

  // Default resources might not be loaded yet. Make sure they are.
  for (const auto &desc :
      m_argumentBlockDescriptor.m_defaultAndBodyTextureDescriptors) {
    if (auto res = make_handle(
            transaction->access<mi::neuraylib::ITexture>(desc.url.c_str()));
        !res.is_valid_interface()) {
      loadTextureToDb(desc, transaction);
    }
  }
}

} // namespace visrtx::libmdl
