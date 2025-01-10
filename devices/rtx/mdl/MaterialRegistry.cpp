#include "MaterialRegistry.h"

#include "ptx.h"

#include "libmdl/ArgumentBlockDescriptor.h"
#include "libmdl/ArgumentBlockInstance.h"
#include "libmdl/TimeStamp.h"
#include "libmdl/ptx.h"

#include <mi/base/enums.h>
#include <mi/base/handle.h>
#include <mi/base/ilogger.h>
#include <mi/neuraylib/icompiled_material.h>
#include <mi/neuraylib/iimage.h>
#include <mi/neuraylib/imdl_backend.h>

#include <fmt/base.h>
#include <mi/neuraylib/imdl_configuration.h>
#include <mi/neuraylib/itexture.h>
#include <glm/fwd.hpp>

#include <nonstd/scope.hpp>

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
{}

MaterialRegistry::~MaterialRegistry()
{
  m_core->removeScope(m_scope.get());
}

libmdl::Uuid MaterialRegistry::acquireMaterial(
    std::string_view moduleName, std::string_view materialName)
{
  using mi::base::make_handle;

  auto transaction = make_handle(m_core->createTransaction(m_scope.get()));
  bool doCommit = false;
  nonstd::scope_exit finalizeTransaction([transaction, &doCommit]() {
    if (doCommit)
      transaction->commit();
    else
      transaction->abort();
  });

  auto module = make_handle(m_core->loadModule(moduleName, transaction.get()));
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
  auto uuid = compiledMaterial->get_hash();

  if (auto it = m_uuidToIndex.find(uuid); it != cend(m_uuidToIndex)) {
    ++m_targetCodes[it->second].refCount;
    return uuid;
  }

  auto targetCode = make_handle(
      m_core->getPtxTargetCode(compiledMaterial.get(), transaction.get()));

  auto ptxBlob = libmdl::stitchPTXs(std::vector{
      nonstd::span{reinterpret_cast<const char *>(ptx::MDLTexture.ptr),
          ptx::MDLTexture.size},
      nonstd::span{targetCode->get_code(), targetCode->get_code_size()},
      nonstd::span{
          reinterpret_cast<const char *>(ptx::MDLShaderEvalSurfaceMaterial.ptr),
          ptx::MDLShaderEvalSurfaceMaterial.size},
  });

  auto targetIt = std::find_if(begin(m_targetCodes),
      end(m_targetCodes),
      [](const auto &v) { return v.refCount == 0; });

  if (targetIt == end(m_targetCodes)) {
    std::vector<std::string> textureNames({""});
    // That's a new texture. Let's parse the default and body texture DB names.
    for (auto i = 1ul; i < targetCode->get_texture_count(); ++i) {
      auto textureDbName = targetCode->get_texture(i);
      textureNames.push_back(textureDbName);
    }

    targetIt = m_targetCodes.insert(end(m_targetCodes),
        {libmdl::ArgumentBlockDescriptor(
             compiledMaterial.get(), targetCode.get(), std::move(textureNames)),
            targetCode,
            ptxBlob,
            1});
  } else {
    ++targetIt->refCount;
  }
  auto targetIndex = std::distance(begin(m_targetCodes), targetIt);
  m_uuidToIndex.insert({uuid, targetIndex});

  m_lastUpdateTS = libmdl::newTimeStamp();

  doCommit = true;
  return uuid;
}

std::optional<libmdl::ArgumentBlockInstance>
MaterialRegistry::createArgumentBlock(const Uuid &uuid) const
{
  auto result = std::optional<libmdl::ArgumentBlockInstance>{};

  auto transaction = make_handle(m_core->createTransaction());
  if (auto it = m_uuidToIndex.find(uuid); it != cend(m_uuidToIndex)) {
    result = libmdl::ArgumentBlockInstance(
        &m_targetCodes[it->second].argumentBlockDescriptor, m_core);
    transaction->commit();
  } else {
    transaction->abort();
  }

  return result;
}

void MaterialRegistry::releaseMaterial(const Uuid &uuid)
{
  if (auto it = m_uuidToIndex.find(uuid); it != end(m_uuidToIndex)) {
    if (--m_targetCodes[it->second].refCount == 0) {
      m_targetCodes[it->second] = {};
      m_uuidToIndex.erase(it);
      m_lastUpdateTS = libmdl::newTimeStamp();
    }
  }
}

} // namespace visrtx::mdl
