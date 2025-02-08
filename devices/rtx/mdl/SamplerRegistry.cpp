#include "SamplerRegistry.h"

#include "array/Array2D.h"
#include "array/Array3D.h"
#include "optix_visrtx.h"
#include "scene/surface/material/sampler/Image2D.h"

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

#include <string>
#include <string_view>

using namespace std::string_view_literals;

namespace visrtx::mdl {

SamplerRegistry::SamplerRegistry(
    libmdl::Core *core, DeviceGlobalState *deviceState)
    : m_core(core), m_deviceState(deviceState)
{}

SamplerRegistry::~SamplerRegistry() {}

Sampler *SamplerRegistry::acquireSampler(std::string_view dbName,
    mi::neuraylib::ITarget_code::Texture_shape textureShape,
    mi::neuraylib::ITransaction *transaction)
{
  auto textureDbName = std::string(dbName);
  if (auto it = m_dbToSampler.find(std::string(textureDbName));
      it != end(m_dbToSampler)) {
    it->second->refInc();
    return it->second;
  }

  // Get access to the texture data by the texture database name from the target
  // code.
  auto texture = mi::base::make_handle(
      transaction->access<mi::neuraylib::ITexture>(textureDbName.c_str()));

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
        "texture",
        "Unsupported uvtile and/or animated textures.");
    return {};
  }

  auto imageApi = mi::base::make_handle(
      m_core->getINeuray()->get_api_component<mi::neuraylib::IImage_api>());

  // Convert to linear color space if necessary
  if (texture->get_effective_gamma(0, 0) != 1.0f) {
    // Copy/convert to float4 canvas and adjust gamma from "effective gamma"
    // to 1.
    mi::base::Handle<mi::neuraylib::ICanvas> gamma_canvas(
        imageApi->convert(canvas.get(), "Color"));
    gamma_canvas->set_gamma(texture->get_effective_gamma(0, 0));
    imageApi->adjust_gamma(gamma_canvas.get(), 1.0f);
    canvas = gamma_canvas;
  }

  image_type = canvas->get_type();

  ANARIDataType dataType;
  if (image_type == "Sint8"sv) {
    dataType = ANARI_FIXED8;
  } else if (image_type == "Sint32"sv) {
    dataType = ANARI_FIXED32;
  } else if (image_type == "Float32"sv) {
    dataType = ANARI_FLOAT32;
  } else if (image_type == "Float32<2>"sv) {
    dataType = ANARI_FLOAT32_VEC2;
  } else if (image_type == "Float32<3>"sv) {
    dataType = ANARI_FLOAT32_VEC3;
  } else if (image_type == "Float32<4>"sv || image_type == "Color"sv) {
    dataType = ANARI_FLOAT32_VEC4;
  } else if (image_type == "Rgb"sv) {
    dataType = ANARI_UFIXED8_VEC3;
  } else if (image_type == "Rgba"sv) {
    dataType = ANARI_UFIXED8_VEC4;
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
        "texture",
        fmt::format("Unsupported image type '{}'", image_type).c_str());
    return {};
  }

  Sampler *sampler = {};

  switch (textureShape) {
  case mi::neuraylib::ITarget_code::Texture_shape_2d: {
    Array2DMemoryDescriptor desc = {
        {
            canvas->get_tile(0)->get_data(),
            nullptr,
            nullptr,
            dataType,
        },
        tex_width,
        tex_height,
    };

    auto array2d = new Array2D(m_deviceState, desc);
    array2d->commitParameters();
    auto image2d = new Image2D(m_deviceState);
    image2d->setParam("image", array2d);
    image2d->commitParameters();
    sampler = image2d;
    break;
  }
  case mi::neuraylib::ITarget_code::Texture_shape_3d:
  case mi::neuraylib::ITarget_code::Texture_shape_bsdf_data: {
    Array3DMemoryDescriptor desc = {
        {
            canvas->get_tile(0)->get_data(),
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
    auto image3d = new Image2D(m_deviceState);
    image3d->setParam("image", array3d);
    image3d->commitParameters();
    sampler = image3d;
    break;
  }

  default: {
    m_core->logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
        "sampler",
        fmt::format("Unsupported sampler type {}", int(textureShape)).c_str());
    break;
  }
  }

  if (sampler) {
    m_dbToSampler.insert({textureDbName, sampler});
  }

  return sampler;
}

bool SamplerRegistry::releaseSampler(const Sampler *sampler)
{
  if (auto it = std::find_if(begin(m_dbToSampler),
          end(m_dbToSampler),
          [sampler](const auto &p) { return p.second == sampler; });
      it != end(m_dbToSampler)) {
    if (it->second->useCount() == 1) {
      it->second->refDec();
      m_dbToSampler.erase(it);
      return true;
    }
  } else {
    fprintf(stderr, "Removing an unknown sampler %p\n", sampler);
  }

  return false;
}

} // namespace visrtx::mdl