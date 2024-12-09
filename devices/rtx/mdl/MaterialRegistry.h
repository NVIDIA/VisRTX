#pragma once

#include "libmdl/ArgumentBlockDescriptor.h"
#include "libmdl/ArgumentBlockInstance.h"
#include "libmdl/Core.h"
#include "libmdl/TimeStamp.h"
#include "libmdl/uuid.h"

#include <mi/base/handle.h>
#include <mi/neuraylib/icompiled_material.h>
#include <mi/neuraylib/imdl_backend.h>
#include <mi/neuraylib/imdl_factory.h>
#include <mi/neuraylib/iscope.h>
#include <mi/neuraylib/itransaction.h>

#include <limits>
#include <optional>
#include <string_view>
#include <unordered_map>

namespace visrtx::mdl {

class MaterialRegistry
{
 public:
  using Uuid = libmdl::Uuid;

  MaterialRegistry(libmdl::Core *core);
  ~MaterialRegistry();

  mi::neuraylib::ITransaction *createTransaction() const
  {
    return m_core->createTransaction(m_scope.get());
  }

  mi::neuraylib::IMdl_factory *getMdlFactory() const
  {
    return m_core->getMdlFactory();
  }

  // Material code
  std::tuple<libmdl::Uuid, libmdl::ArgumentBlockDescriptor> acquireMaterial(
      std::string_view moduleName, std::string_view materialName);
  void releaseMaterial(const Uuid &uuid);

  // For SBT management
  libmdl::TimeStamp getLastUpdateTime() const
  {
    return m_lastUpdateTS;
  }

  using ImplementationIndex = std::uint32_t;
  static constexpr const auto INVALID_IMPLEMENTATION_INDEX =
      std::numeric_limits<ImplementationIndex>::max();
  ImplementationIndex getMaterialImplementationIndex(
      const libmdl::Uuid &uuid) const
  {
    if (auto it = m_uuidToIndex.find(uuid); it != cend(m_uuidToIndex)) {
      return it->second;
    } else {
      return INVALID_IMPLEMENTATION_INDEX;
    }
  }

  const std::vector<nonstd::span<const char>> getPtxBlobs() const
  {
    std::vector<nonstd::span<const char>> res;
    for (const auto &target : m_targetCodes) {
      res.push_back({target.ptxBlob});
    }
    return res;
  }

  // Per material instance data
  std::optional<libmdl::ArgumentBlockInstance> createArgumentBlock(
      const libmdl::ArgumentBlockDescriptor &uuid) const;

 private:
  libmdl::Core *m_core;
  mi::base::Handle<mi::neuraylib::IScope> m_scope;

  struct TargetCode
  {
    // mi::base::Handle<const mi::neuraylib::ITarget_code> targetCode;
    std::vector<char> ptxBlob;
    int refCount{};
  };

  // Per material PTX blobs. Stored in Sbt order. Sparse structure depending on
  // acquire/release calls.
  std::vector<TargetCode> m_targetCodes;

  std::unordered_map<std::string, 
    std::tuple<libmdl::Uuid, libmdl::ArgumentBlockDescriptor> > m_materialNameToUuid;
  std::unordered_map<libmdl::Uuid, std::size_t, libmdl::UuidHasher>
      m_uuidToIndex;

  libmdl::TimeStamp m_lastUpdateTS{};
};

} // namespace visrtx::mdl
