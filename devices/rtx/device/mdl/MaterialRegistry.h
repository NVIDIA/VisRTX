/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "MdlCompileCoordinator.h"

#include "libmdl/ArgumentBlockDescriptor.h"
#include "libmdl/ArgumentBlockInstance.h"
#include "libmdl/Core.h"
#include "libmdl/EmissionIR.h"
#include "libmdl/SceneDataDescriptor.h"
#include "libmdl/TimeStamp.h"
#include "libmdl/uuid.h"

#include <mi/base/handle.h>
#include <mi/neuraylib/icompiled_material.h>
#include <mi/neuraylib/imdl_backend.h>
#include <mi/neuraylib/imdl_factory.h>
#include <mi/neuraylib/iscope.h>
#include <mi/neuraylib/itransaction.h>

#include <atomic>
#include <cstdint>
#include <future>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace visrtx::mdl {

class MaterialRegistry
{
 public:
  using Uuid = libmdl::Uuid;

  struct AcquiredMaterial
  {
    Uuid uuid{};
    libmdl::ArgumentBlockDescriptor arguments;
    libmdl::SceneDataDescriptor sceneData;
  };

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

  // Split-phase acquire (ADR 0009). Returns immediately with a future the
  // caller collects during finalize. The name-cache lookup, module preload and
  // registry insert run on the coordinator thread; the expensive compile runs
  // on a pool worker, so materials committed in one flush compile in parallel.
  // This is the only acquire path, including MDLE and fallback requests.
  // Collect the future on the commit thread, never inside coordinator.run().
  // `moduleOrSource` is a module name or MDLE path (fromCode=false), or inline
  // MDL source (fromCode=true).
  std::future<AcquiredMaterial> acquireMaterialAsync(
      MdlCompileCoordinator &coordinator,
      std::string moduleOrSource,
      std::string materialName,
      bool fromCode);

  void releaseMaterial(const Uuid &uuid);

  // Lifetime high-water mark measured at compileMaterial entry/exit. Used by
  // the serial-execution regression gate, not inferred from pool size.
  std::uint32_t maxConcurrentCompiles() const;

  // For SBT management
  libmdl::TimeStamp getLastUpdateTime() const
  {
    return m_lastUpdateTS;
  }

  // Live slots (distinct compiled materials currently acquired). Test seam for
  // acquire/release balance, exposed as the device property
  // `numRegisteredMdlMaterials`.
  std::size_t numRegisteredMaterials() const
  {
    return m_uuidToIndex.size();
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

  // Owned emission IR of a compiled material (ADR 0007), extracted while the
  // compiled material was alive; empty when the uuid is unknown. The material
  // folds it against its live arguments at finalize.
  libmdl::EmissionIR getEmissionIR(const libmdl::Uuid &uuid) const
  {
    if (auto it = m_uuidToIndex.find(uuid); it != cend(m_uuidToIndex))
      return m_targetCodes[it->second].emission;
    return {};
  }

  struct PtxSnapshot
  {
    // Preserve empty slots: these indices are the material callable offsets.
    std::vector<std::vector<char>> blobs;
    libmdl::TimeStamp timestamp{};
  };

  // Call on the coordinator. Both bytes and timestamp belong to the same
  // registry version; the result stays valid after slots are released/reused.
  PtxSnapshot ptxSnapshot() const;

  // Order-independent content hash of every compiled material's PTX. Parallel
  // compilation that produced different code changes this even when the render
  // does not (ADR 0009 silent-miscompile gate). XOR combines slots so it is
  // independent of slot order; released (empty) slots are skipped.
  std::uint64_t ptxFingerprint() const
  {
    std::uint64_t acc = 0;
    for (const auto &target : m_targetCodes) {
      if (target.ptxBlob.empty())
        continue;
      std::uint64_t h = 1469598103934665603ull; // FNV-1a offset basis
      for (char c : target.ptxBlob) {
        h ^= static_cast<unsigned char>(c);
        h *= 1099511628211ull; // FNV-1a prime
      }
      acc ^= h;
    }
    return acc;
  }

  // Per material instance data
  std::optional<libmdl::ArgumentBlockInstance> createArgumentBlock(
      const libmdl::ArgumentBlockDescriptor &uuid) const;

 private:
  // Owned compile output passed from a worker to the coordinator.
  struct CompileProduct
  {
    libmdl::Uuid uuid;
    std::vector<char> ptxBlob;
    libmdl::EmissionIR emission;
    libmdl::ArgumentBlockDescriptor argBlockDescriptor;
    libmdl::SceneDataDescriptor sceneData;
  };

  // Return a previously compiled material, bumping its refcount. Empty on a
  // cache miss. No transaction needed -- the lookup is keyed on the name.
  std::optional<AcquiredMaterial> reuseCompiledMaterial(
      const std::string &fullMaterialName);

  // Load and commit on the coordinator, returning the actual database name
  // (including any canonical-name recovery). Empty on failure. Workers must
  // access this committed element directly, never invoke module loading APIs.
  std::optional<std::string> preloadModule(
      const std::string &moduleOrSource, bool fromCode);

  // Compile `materialName` from a committed database module into a
  // transaction-independent product. Opens its own transaction and accesses
  // the module by database name, without loading or resolving it again.
  // Runs on a pool worker in parallel with other compiles. Empty on failure.
  // `coordinator` serializes texture-URL resolution (the shared entity resolver
  // is not safe to hit from parallel workers).
  std::optional<CompileProduct> compileMaterial(const std::string &moduleDbName,
      const std::string &materialName,
      MdlCompileCoordinator &coordinator);

  // Register a compiled product under `fullMaterialName` (dedup by uuid) and
  // return the acquired material. Coordinator thread (mutates the registry).
  AcquiredMaterial insertCompiled(
      const std::string &fullMaterialName, CompileProduct product);

  libmdl::Core *m_core;
  mi::base::Handle<mi::neuraylib::IScope> m_scope;

  struct TargetCode
  {
    std::vector<char> ptxBlob;
    int refCount{};
    // Extracted at compile time, while the compiled material is alive (it is
    // not retained past compilation); evicted with the slot on release.
    libmdl::EmissionIR emission;
    // String IDs must match this slot's PTX, including on a UUID cache hit.
    libmdl::SceneDataDescriptor sceneData;
  };

  // Per material PTX blobs. Sparse: empty slots are reused across
  // acquire/release, and new slots are assigned in compile-completion order
  // (see insertCompiled). Consumers index by uuid via
  // getMaterialImplementationIndex, not by position, so slot order is not
  // load-bearing.
  std::vector<TargetCode> m_targetCodes;

  std::unordered_map<std::string, AcquiredMaterial> m_materialNameToUuid;
  std::unordered_map<libmdl::Uuid, std::size_t, libmdl::UuidHasher>
      m_uuidToIndex;

  libmdl::TimeStamp m_lastUpdateTS{};
  std::atomic<std::uint32_t> m_activeCompiles{0};
  std::atomic<std::uint32_t> m_maxConcurrentCompiles{0};
};

} // namespace visrtx::mdl
