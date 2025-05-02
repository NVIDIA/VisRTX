#include "SamplerRegistry.h"

#include "array/Array2D.h"
#include "array/Array3D.h"
#include "optix_visrtx.h"
#include "scene/surface/material/sampler/Image2D.h"

#include <anari/frontend/anari_enums.h>
#include <mi/base/enums.h>
#include <mi/base/handle.h>
#include <mi/neuraylib/icanvas.h>
#include <mi/neuraylib/iimage.h>
#include <mi/neuraylib/iimage_api.h>
#include <mi/neuraylib/imdl_backend.h>
#include <mi/neuraylib/itexture.h>
#include <mi/neuraylib/itile.h>
#include <mi/neuraylib/itransaction.h>

#include <fmt/format.h>

#include <limits>
#include <string>
#include <string_view>

using namespace std::string_view_literals;

namespace visrtx::mdl {

SamplerRegistry::SamplerRegistry(
    libmdl::Core *core, DeviceGlobalState *deviceState)
    : m_core(core), m_deviceState(deviceState)
{}

SamplerRegistry::~SamplerRegistry()
{
  if (!m_dbToSampler.empty()) {
    m_core->logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
        "SamplerRegistry is not empty on destruction");
  }
}

Sampler *SamplerRegistry::createSamplerFromDb(
    const std::string &textureDbName, mi::neuraylib::ITransaction *transaction)
{
  // Get access to the texture data by the texture database name from the target
  // code.
  auto texture = mi::base::make_handle(
      transaction->access<mi::neuraylib::ITexture>(textureDbName.c_str()));

  if (!texture.is_valid_interface()) {
    m_core->logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
        "Texture {} not found is the database",
        textureDbName);
    return {};
  }

  // Access image and canvas via the texture object
  auto image = mi::base::make_handle(
      transaction->access<mi::neuraylib::IImage>(texture->get_image()));
  auto canvas = mi::base::make_handle(image->get_canvas(0, 0, 0));
  mi::Uint32 tex_width = canvas->get_resolution_x();
  mi::Uint32 tex_height = canvas->get_resolution_y();
  mi::Uint32 tex_layers = canvas->get_layers_size();
  std::string image_type = image->get_type(0, 0);

  if (image->is_uvtile() || image->is_animated()) {
    m_core->logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
        "Unsupported uvtile and/or animated textures.");
    return {};
  }

  auto imageApi = mi::base::make_handle(
      m_core->getINeuray()->get_api_component<mi::neuraylib::IImage_api>());

  bool isSrgb = texture->get_gamma() != 1.0f;

  image_type = canvas->get_type();

  auto data = canvas->get_tile(0)->get_data();
  bool dataNeedRelease = false;

  ANARIDataType dataType;
  if (image_type == "Sint8"sv) {
    auto unsignedData = new unsigned char[tex_width * tex_height * tex_layers];
    auto signedData = static_cast<const char *>(data);
    for (size_t i = 0; i < tex_width * tex_height * tex_layers; ++i) {
      // first cast to unsigned where overflow behavior is defined
      unsignedData[i] = static_cast<unsigned char>(signedData[i])
          - static_cast<unsigned char>(std::numeric_limits<char>::lowest());
    }

    dataType = isSrgb ? ANARI_UFIXED8_R_SRGB : ANARI_UFIXED8;
    data = unsignedData;
    dataNeedRelease = true;
  } else if (image_type == "Sint32"sv) {
    auto unsignedData = new unsigned int[tex_width * tex_height * tex_layers];
    auto signedData = static_cast<const int *>(data);
    for (size_t i = 0; i < tex_width * tex_height * tex_layers; ++i) {
      // first cast to unsigned where overflow behavior is defined
      unsignedData[i] = static_cast<unsigned int>(signedData[i])
          - static_cast<unsigned int>(std::numeric_limits<int>::lowest());
    }

    dataType = ANARI_UFIXED32;
    data = unsignedData;
    dataNeedRelease = true;
  } else if (image_type == "Float32"sv) {
    dataType = ANARI_FLOAT32;
  } else if (image_type == "Float32<2>"sv) {
    dataType = ANARI_FLOAT32_VEC2;
  } else if (image_type == "Float32<3>"sv) {
    dataType = ANARI_FLOAT32_VEC3;
  } else if (image_type == "Float32<4>"sv || image_type == "Color"sv) {
    dataType = ANARI_FLOAT32_VEC4;
  } else if (image_type == "Rgb"sv) {
    dataType = isSrgb ? ANARI_UFIXED8_RGB_SRGB : ANARI_UFIXED8_VEC3;
  } else if (image_type == "Rgba"sv) {
    dataType = isSrgb ? ANARI_UFIXED8_RGBA_SRGB : ANARI_UFIXED8_VEC4;
  } else if (image_type == "Rgb_16"sv) {
    dataType = ANARI_UFIXED16_VEC3;
  } else if (image_type == "Rgba_16"sv) {
    dataType = ANARI_UFIXED16_VEC4;
  } else if (image_type == "Rgb_fp"sv) {
    dataType = ANARI_FLOAT32_VEC3;
  } else if (image_type == "Color") {
    dataType = ANARI_FLOAT32_VEC3;
  } else { // rgbe, rgbea,
    m_core->logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
        "Unsupported image type '{}'",
        image_type);
    return {};
  }

  auto textureShape = image->resolution_z(0, 0, 0) == 1
      ? mi::neuraylib::ITarget_code::Texture_shape_2d
      : mi::neuraylib::ITarget_code::Texture_shape_3d;

  Sampler *sampler = {};

  switch (textureShape) {
  case mi::neuraylib::ITarget_code::Texture_shape_2d: {
    Array2DMemoryDescriptor desc = {
        {
            data,
            nullptr,
            nullptr,
            dataType,
        },
        tex_width,
        tex_height,
    };

    auto array2d = new Array2D(m_deviceState, desc);
    array2d->commitParameters();
    array2d->uploadArrayData();
    auto image2d = new Image2D(m_deviceState);
    image2d->setParam("image", array2d);
    image2d->commitParameters();
    image2d->upload();
    sampler = image2d;
    array2d->refDec();
    break;
  }
  case mi::neuraylib::ITarget_code::Texture_shape_3d: {
    Array3DMemoryDescriptor desc = {
        {
            data,
            nullptr,
            nullptr,
            dataType,
        },
        tex_width,
        tex_height,
        tex_layers,
    };

    auto array3d = new Array3D(m_deviceState, desc);
    array3d->commitParameters();
    array3d->uploadArrayData();
    auto image3d = new Image2D(m_deviceState);
    image3d->setParam("image", array3d);
    image3d->commitParameters();
    image3d->upload();
    sampler = image3d;
    array3d->refDec();

    if (dataNeedRelease) {
      if (dataType == ANARI_UFIXED8) {
        delete[] static_cast<const unsigned char *>(data);
      } else if (dataType == ANARI_UFIXED32) {
        delete[] static_cast<const unsigned int *>(data);
      }
    }

    break;
  }

  default: {
    m_core->logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
        "Unsupported sampler type {}",
        int(textureShape));
    break;
  }
  }

  if (sampler) {
    sampler->refInc();
    m_dbToSampler.insert({textureDbName, sampler});
  } else {
    m_core->logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
        "Unable to create sampler for texture db name `{}`",
        textureDbName);
  }

  return sampler;
}

Sampler *SamplerRegistry::acquireSampler(
    const std::string &textureDbName, mi::neuraylib::ITransaction *transaction)
{
  {
    if (auto it = m_dbToSampler.find(textureDbName);
        it != std::end(m_dbToSampler)) {
      it->second->refInc();
      return it->second;
    }
  }

  return createSamplerFromDb(textureDbName, transaction);
}

bool SamplerRegistry::releaseSampler(const Sampler *sampler)
{
  if (auto it = std::find_if(std::begin(m_dbToSampler),
          std::end(m_dbToSampler),
          [sampler](const auto &p) { return p.second == sampler; });
      it != std::end(m_dbToSampler)) {
    it->second->refDec();
    if (it->second->useCount() == 1) {
      it->second->refDec();
      m_dbToSampler.erase(it);
      return true;
    }
  } else {
    m_core->logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
        "Removing an unknown sampler {}\n",
        fmt::ptr(sampler));
  }

  return false;
}

} // namespace visrtx::mdl