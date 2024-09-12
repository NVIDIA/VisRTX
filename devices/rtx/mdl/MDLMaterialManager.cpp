/*
 * Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "MDLMaterialManager.h"
#include "MDLShader_ptx.h"
#include "mdl/MDLMaterialInfo.h"
#include "mdl/MDLSDK.h"
#include "optix_visrtx.h"
#include "shaders/MDLShaderData.cuh"
#include "utility/DeviceBuffer.h"

#include <anari/frontend/anari_enums.h>

#include <mi/base/handle.h>
#include <mi/neuraylib/imdl_backend.h>
#include <mi/neuraylib/itransaction.h>

#include <algorithm>
#include <filesystem>
#include <iterator>
#include <optional>
#include <vector>

namespace visrtx {

MDLMaterialManager::MDLMaterialManager(MDLSDK *mdlSdk) : m_mdlSdk(mdlSdk)
{
  auto path = std::filesystem::current_path() / "shaders";
  m_mdlSdk->addMdlSearchPath(path);
}

MDLMaterialManager::~MDLMaterialManager()
{
  auto path = std::filesystem::current_path() / "shaders";
  m_mdlSdk->removeMdlSearchPath(path);
}

MDLMaterialManager::Uuid MDLMaterialManager::acquireModule(
    const char *materialName)
{
  auto tx = m_mdlSdk->createTransaction();

  // The entry points we want expose.
  std::vector<mi::neuraylib::Target_function_description> descs{
      {"surface.scattering", "mdlBsdf"},
      {"thin_walled", "mdl_isThinWalled"},
  };

#if 0
  // Some other entry points. Just there for the sake of not searching for them again.
  std::vector<mi::neuraylib::Target_function_description> descs;
  descs.push_back(
      mi::neuraylib::Target_function_description("init"));
  descs.push_back(
      mi::neuraylib::Target_function_description("surface.scattering"));
  descs.push_back(
      mi::neuraylib::Target_function_description("surface.emission.emission"));
  descs.push_back(
      mi::neuraylib::Target_function_description("surface.emission.intensity"));
  descs.push_back(
      mi::neuraylib::Target_function_description("volume.absorption_coefficient"));
  descs.push_back(
      mi::neuraylib::Target_function_description("thin_walled"));
  descs.push_back(
      mi::neuraylib::Target_function_description("geometry.cutout_opacity"));
  descs.push_back(
      mi::neuraylib::Target_function_description("hair"));
  descs.push_back(
      mi::neuraylib::Target_function_description("backface.scattering"));
  descs.push_back(
      mi::neuraylib::Target_function_description("backface.emission.emission"));
  descs.push_back(
      mi::neuraylib::Target_function_description("backface.emission.intensity"));
#endif

  m_mdlSdk->reportMessage(
      ANARI_SEVERITY_INFO, "Compiling material %s", materialName);
  auto compilationResult =
      m_mdlSdk->compileMaterial(tx.get(), materialName, descs, true);

  tx->commit();

  if (!compilationResult.has_value()
      || !compilationResult->targetCode.is_valid_interface()) {
    m_mdlSdk->reportMessage(
        ANARI_SEVERITY_ERROR, "Failed compiling MDL: %s", materialName);
    return {};
  }
  m_mdlSdk->reportMessage(
      ANARI_SEVERITY_INFO, "Successfully compiled %s", materialName);

  Uuid uuid = compilationResult->uuid;

  // PTX blob from the compiled code.
  auto targetCode = compilationResult->targetCode->get_code();
  auto targetCodeSize = compilationResult->targetCode->get_code_size();
  std::vector<unsigned char> ptxBlob(
      reinterpret_cast<const uint8_t *>(targetCode),
      reinterpret_cast<const uint8_t *>(targetCode) + targetCodeSize

  );

  // Hack time. An alternative would be to compile shaderMain as LLVM byte code
  // and then embed it using "llvm_render_module" parameter to MDL's ptx
  // backend. Lets save that for later if we want and add such a dependency on
  // LLVM.
  std::vector<unsigned char> shaderMain(
      MDLShader_ptx, MDLShader_ptx + sizeof(MDLShader_ptx));

  // As we are blending to separate compilation unit PTXs, make sure headers are
  // not conflicting.
  static const std::string dotkwords[] = {
      "\n.version ",
      "\n.target",
      "\n.address_size",
  };

  // Cleanup both ptxBlob and shaderMain for headers.
  // Keep the highest version and target and insert those after the final string
  // is built.
  std::string version;
  std::string target;
  std::string addressSize = ".address_size 64";
  for (auto dotkword : dotkwords) {
    if (auto it = std::search(begin(shaderMain),
            end(shaderMain),
            cbegin(dotkword),
            cend(dotkword));
        it != end(shaderMain)) {
      auto eolIt = std::find(it + 1, end(shaderMain), '\n');
      std::string sub(it, eolIt);
      shaderMain.erase(it, eolIt);
      if (dotkword == dotkwords[0]) { // .version
        if (sub > version)
          version = sub;
      } else if (dotkword == dotkwords[1]) { //.target
        if (sub > target)
          target = sub;
      }
    }
    if (auto it = std::search(
            begin(ptxBlob), end(ptxBlob), cbegin(dotkword), cend(dotkword));
        it != end(ptxBlob)) {
      auto eolIt = std::find(it + 1, end(ptxBlob), '\n');
      std::string sub(it, eolIt);
      ptxBlob.erase(it, eolIt);
      if (dotkword == dotkwords[0]) { // .version
        if (sub > version)
          version = sub;
      } else if (dotkword == dotkwords[1]) { //.target
        if (sub > target)
          target = sub;
      }
    }
  }

  // Same cleanup with forward decls possibly conflicting with actual
  // declarations.
  static const std::string mdlPrefixes[] = {
      "\n.extern .func mdlBsdf_",
      "\n.extern .func  (.param .b32 func_retval0) mdl_isThinWalled\n",
  };
  for (const auto &mdlPrefix : mdlPrefixes) {
    for (auto it = std::search(begin(shaderMain),
             end(shaderMain),
             cbegin(mdlPrefix),
             cend(mdlPrefix));
        it != end(shaderMain);) {
      auto semiColonIt = std::find(it, end(shaderMain), ';');
      if (semiColonIt != end(shaderMain)) {
        shaderMain.erase(it, ++semiColonIt);
      }
      it = std::search(it, end(shaderMain), cbegin(mdlPrefix), cend(mdlPrefix));
    }
  }

  // And ditto with colliding symbol names (focusing on constant strings for
  // now).
  static const std::string strPrefix = "$str";
  for (auto it = std::search(begin(shaderMain),
           end(shaderMain),
           cbegin(strPrefix),
           cend(strPrefix));
      it != end(shaderMain);) {
    *++it = 'S';
    it = std::search(it, end(shaderMain), cbegin(strPrefix), cend(strPrefix));
  }

  ptxBlob.insert(cend(ptxBlob), cbegin(shaderMain), cend(shaderMain));
  std::string header = version + target + addressSize;
  ptxBlob.insert(begin(ptxBlob), cbegin(header), cend(header));

  // MaterialInfo
  MDLMaterialInfo materialInfo(
      compilationResult->targetCode->get_argument_block_count() == 1
          ? compilationResult->targetCode->get_argument_block(0)
          : nullptr);

  // Get a free slot in the material implementations list and get its index.
  auto it = std::find_if(begin(m_materialImplementations),
      end(m_materialImplementations),
      [](const auto &v) { return v.ptxBlob.empty(); });
  if (it == end(m_materialImplementations)) {
    it = m_materialImplementations.insert(it,
        {
            .materialInfo = std::move(materialInfo),
            .ptxBlob = std::move(ptxBlob),
        });
  } else {
    *it = {
        .materialInfo = std::move(materialInfo),
        .ptxBlob = std::move(ptxBlob),
    };
  }

  m_uuidToIndex[uuid] = std::distance(begin(m_materialImplementations), it);

  return uuid;
}

std::vector<ptx_blob> MDLMaterialManager::getPTXBlobs()
{
  std::vector<ptx_blob> blobs;

  // for (const auto& compiledResult : m_compiledResults) {
  for (const auto &materialImplementation : m_materialImplementations) {
    blobs.push_back({
        .ptr = data(materialImplementation.ptxBlob),
        .size = size(materialImplementation.ptxBlob),
    });
  }

  return blobs;
}

std::vector<MaterialSbtData> MDLMaterialManager::getMaterialSbtEntries()
{
  if (m_materialImplementations.empty())
    return {};

  std::vector<MaterialSbtData> entries;
  std::vector<uint64_t> offsets;

  std::vector<char> payloads; // Akin to MDLMaterialData, serialized below.

  for (const auto &materialImpl : m_materialImplementations) {
    if (auto argblockData = materialImpl.materialInfo.getArgumentBlockData();
        !argblockData.empty()) {
      offsets.push_back(payloads.size());
      payloads.insert(payloads.end(), cbegin(argblockData), cend(argblockData));
    } else {
      offsets.push_back(-1ul);
    }
  }

  m_argblocksBuffer.upload(payloads.data(), payloads.size());

  const uint8_t *baseAddr = m_argblocksBuffer.ptrAs<const uint8_t>();

  for (auto &offset : offsets) {
    entries.push_back(
        {.mdl = {.materialData = offset != -1ul
                 ? reinterpret_cast<const MDLMaterialData *>(baseAddr + offset)
                 : nullptr}});
  }

  return entries;
}

} // namespace visrtx
