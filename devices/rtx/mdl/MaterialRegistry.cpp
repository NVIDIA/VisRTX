#include "MaterialRegistry.h"

#include "ptx.h"

#include "libmdl/ArgumentBlockDescriptor.h"
#include "libmdl/ArgumentBlockInstance.h"
#include "libmdl/TimeStamp.h"
#include "libmdl/ptx.h"

#include <mi/base/enums.h>
#include <mi/base/handle.h>
#include <mi/base/ilogger.h>
#include <mi/base/types.h>
#include <mi/neuraylib/icompiled_material.h>
#include <mi/neuraylib/iimage.h>
#include <mi/neuraylib/imdl_backend.h>
#include <mi/neuraylib/imdl_configuration.h>
#include <mi/neuraylib/itexture.h>
#include <mi/neuraylib/version.h>

#include <glm/fwd.hpp>

#include <nonstd/scope.hpp>

#include <fmt/base.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

using namespace std::string_literals;

namespace visrtx::mdl {

MaterialRegistry::MaterialRegistry(libmdl::Core *core)
    : m_core(core),
      m_scope(m_core->createScope("VisRTXMaterialResgistryScope"s
          + std::to_string(std::uintptr_t(this))))
{
  std::string_view defaultModule =
#include "mdl/visrtx_default.mdl.inc"
      ;
  m_core->addBuiltinModule("visrtx::default", defaultModule);
}

MaterialRegistry::~MaterialRegistry()
{
  m_core->removeScope(m_scope.get());
}

std::tuple<libmdl::Uuid, libmdl::ArgumentBlockDescriptor>
MaterialRegistry::acquireMaterial(
    std::string_view moduleName, std::string_view materialName)
{
  using mi::base::make_handle;

  // First check if the material has already been compiled.
  auto fullMaterialName = fmt::format("{}::{}", moduleName, materialName);
  m_core->logMessage(mi::base::MESSAGE_SEVERITY_INFO,
      "Acquiring material {}",
      fullMaterialName);

  if (auto uuidIt = m_materialNameToUuid.find(fullMaterialName);
      uuidIt != cend(m_materialNameToUuid)) {
    if (auto indexIt = m_uuidToIndex.find(std::get<0>(uuidIt->second));
        indexIt != cend(m_uuidToIndex)) {
      m_core->logMessage(mi::base::MESSAGE_SEVERITY_DEBUG,
          "Reusing compiled material {}",
          fullMaterialName);
      m_targetCodes[indexIt->second].refCount++;
      return uuidIt->second;
    }
  }

  // If not, try and get a compiled version of it.
  auto transaction = make_handle(m_core->createTransaction(m_scope.get()));
  bool doCommit = false;
  auto finalizeTransaction =
      nonstd::make_scope_exit([transaction, &doCommit]() {
        if (doCommit) {
          transaction->commit();
        } else {
          transaction->abort();
        }
      });

  auto module = make_handle(m_core->loadModule(moduleName, transaction.get()));
  if (!module.is_valid_interface()) {
    m_core->logMessage(
        mi::base::MESSAGE_SEVERITY_ERROR, "Cannot find module {}", moduleName);
    return {};
  }

  auto functionDef = make_handle(m_core->getFunctionDefinition(
      module.get(), materialName, transaction.get()));
  if (!functionDef.is_valid_interface()) {
    m_core->logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
        "Cannot find function {} definition in module {}",
        materialName,
        moduleName);
    return {};
  }

  auto compiledMaterial =
      make_handle(m_core->getCompiledMaterial(functionDef.get()));
  if (!compiledMaterial.is_valid_interface()) {
    m_core->logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
        "Failed compiling material {} for module {}",
        materialName,
        moduleName);
    return {};
  }

  // Get the compiled material hash and its target code. Get the default and
  // body resources state from that.
  auto uuid = compiledMaterial->get_hash();
  auto targetCode = make_handle(
      m_core->getPtxTargetCode(compiledMaterial.get(), transaction.get()));
  std::vector<libmdl::ArgumentBlockDescriptor::TextureDescriptor> textureDescs;

  for (auto i = 1ul; i < targetCode->get_texture_count(); ++i) {
    libmdl::ArgumentBlockDescriptor::TextureDescriptor textureDesc{
        targetCode->get_texture(i)};

    switch (targetCode->get_texture_shape(i)) {
    case mi::neuraylib::ITarget_code::Texture_shape_2d: {
      textureDesc.shape =
          libmdl::ArgumentBlockDescriptor::TextureDescriptor::Shape::TwoD;
      break;
    }
    case mi::neuraylib::ITarget_code::Texture_shape_3d: {
      textureDesc.shape =
          libmdl::ArgumentBlockDescriptor::TextureDescriptor::Shape::ThreeD;
      break;
    }
    case mi::neuraylib::ITarget_code::Texture_shape_cube: {
      textureDesc.shape =
          libmdl::ArgumentBlockDescriptor::TextureDescriptor::Shape::Cube;
      break;
    }
    case mi::neuraylib::ITarget_code::Texture_shape_bsdf_data: {
      textureDesc.shape =
          libmdl::ArgumentBlockDescriptor::TextureDescriptor::Shape::BsdfData;
      break;
    }
    case mi::neuraylib::ITarget_code::Texture_shape_ptex: {
      m_core->logMessage(mi::base::MESSAGE_SEVERITY_WARNING,
          "Ptex textures are not supported by VisRTX");
      textureDesc.shape =
          libmdl::ArgumentBlockDescriptor::TextureDescriptor::Shape::Unknown;
      break;
    }
    case mi::neuraylib::ITarget_code::Texture_shape_invalid:
    default: {
      textureDesc.shape =
          libmdl::ArgumentBlockDescriptor::TextureDescriptor::Shape::Unknown;
      break;
    }
    }

    if (targetCode->get_texture_shape(i)
        == mi::neuraylib::ITarget_code::Texture_shape_bsdf_data) {
      mi::Size x, y, z;
      const char* pixelFormat = {};
#if MI_NEURAYLIB_API_VERSION >= 56
      textureDesc.bsdf.data = targetCode->get_texture_df_data(i, x, y, z, pixelFormat);
#else
      textureDesc.bsdf.data = targetCode->get_texture_df_data(i, x, y, z);
#endif
      textureDesc.bsdf.dims[0] = x;
      textureDesc.bsdf.dims[1] = y;
      textureDesc.bsdf.dims[2] = z;
      textureDesc.bsdf.pixelFormat = pixelFormat;
      textureDesc.colorSpace = libmdl::ArgumentBlockDescriptor::
          TextureDescriptor::ColorSpace::Linear;
      textureDesc.url =
          fmt::format("bsdf_data_{:0x}", fmt::ptr(textureDesc.bsdf.data));
    } else {
      auto moduleOwner = targetCode->get_texture_owner_module(i);
      textureDesc.url =
          m_core->resolveResource(targetCode->get_texture_url(i), moduleOwner);

      switch (targetCode->get_texture_gamma(i)) {
      case mi::neuraylib::ITarget_code::GM_GAMMA_DEFAULT:
      case mi::neuraylib::ITarget_code::GM_GAMMA_LINEAR:
      case mi::neuraylib::ITarget_code::GM_GAMMA_UNKNOWN: {
        textureDesc.colorSpace = libmdl::ArgumentBlockDescriptor::
            TextureDescriptor::ColorSpace::Linear;
        break;
      }
      case mi::neuraylib::ITarget_code::GM_GAMMA_SRGB: {
        textureDesc.colorSpace = libmdl::ArgumentBlockDescriptor::
            TextureDescriptor::ColorSpace::sRGB;
        break;
      }
      case mi::neuraylib::ITarget_code::GM_FORCE_32_BIT: {
        assert(false);
        break;
      }
      }
    }

    textureDescs.push_back(textureDesc);
  }

  libmdl::ArgumentBlockDescriptor argBlockDesc(m_core,
      compiledMaterial.get(),
      targetCode.get(),
      std::move(textureDescs));

  // Reuse an existing targetCode and its matching ptx generated code if we
  // already have it.
  if (auto it = m_uuidToIndex.find(uuid); it != std::cend(m_uuidToIndex)) {
    ++m_targetCodes[it->second].refCount;
    return {uuid, argBlockDesc};
  }

  // First time we hit this. Build a complete PTX shader from the generated
  // blob.
  auto ptxBlob = libmdl::stitchPTXs(std::vector{
      nonstd::span{reinterpret_cast<const char *>(ptx::MDLTexture.ptr),
          ptx::MDLTexture.size},
      nonstd::span{targetCode->get_code(), targetCode->get_code_size()},
      nonstd::span{
          reinterpret_cast<const char *>(ptx::MDLShaderEvalSurfaceMaterial.ptr),
          ptx::MDLShaderEvalSurfaceMaterial.size},
  });

  // Find an empty slot if possible
  auto targetIt = std::find_if(std::begin(m_targetCodes),
      std::end(m_targetCodes),
      [](const auto &v) { return v.refCount == 0; });

  if (targetIt == std::end(m_targetCodes)) {
    targetIt = m_targetCodes.insert(std::end(m_targetCodes), {ptxBlob, 1});
  }

  auto targetIndex = std::distance(std::begin(m_targetCodes), targetIt);

  m_uuidToIndex.insert({uuid, targetIndex});
  m_materialNameToUuid.insert(
      {fullMaterialName, std::tuple{uuid, argBlockDesc}});

  m_lastUpdateTS = libmdl::newTimeStamp();

  // Make sure we commit the transaction.
  m_core->logMessage(mi::base::MESSAGE_SEVERITY_DEBUG,
      "Acquired material {} with uuid {:04x}-{:04x}-{:04x}-{:04x}",
      fullMaterialName,
      uuid.m_id1,
      uuid.m_id2,
      uuid.m_id3,
      uuid.m_id4);
  doCommit = true;
  return {uuid, argBlockDesc};
}

std::optional<libmdl::ArgumentBlockInstance>
MaterialRegistry::createArgumentBlock(
    const libmdl::ArgumentBlockDescriptor &argumentBlockDescriptor) const
{
  auto result = std::optional<libmdl::ArgumentBlockInstance>{};

  result = libmdl::ArgumentBlockInstance(argumentBlockDescriptor, m_core);

  return result;
}

void MaterialRegistry::releaseMaterial(const Uuid &uuid)
{
  if (auto it = m_uuidToIndex.find(uuid); it != std::end(m_uuidToIndex)) {
    m_core->logMessage(mi::base::MESSAGE_SEVERITY_DEBUG,
        "Releasing material with uuid {:04x}-{:04x}-{:04x}-{:04x}",
        uuid.m_id1,
        uuid.m_id2,
        uuid.m_id3,
        uuid.m_id4);
    if (--m_targetCodes[it->second].refCount == 0) {
      m_core->logMessage(mi::base::MESSAGE_SEVERITY_INFO,
          "Destroying material with uuid {:04x}-{:04x}-{:04x}-{:04x}",
          uuid.m_id1,
          uuid.m_id2,
          uuid.m_id3,
          uuid.m_id4);
      m_targetCodes[it->second] = {};
      m_uuidToIndex.erase(it);
      m_lastUpdateTS = libmdl::newTimeStamp();
    }
  } else {
    m_core->logMessage(mi::base::MESSAGE_SEVERITY_ERROR,
        "Cannot release material with UUID {:04x}-{:04x}-{:04x}-{:04x}",
        uuid.m_id1,
        uuid.m_id2,
        uuid.m_id3,
        uuid.m_id4);
  }
}

} // namespace visrtx::mdl
