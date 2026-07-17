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

#include "MaterialX.h"

#include "materialx/Transcoder.h"
#include "optix_visrtx.h"
#include "sampler/CompressedImage2D.h"
#include "sampler/Image2D.h"

#include <anari/frontend/anari_enums.h>

#include <filesystem>
#include <set>

namespace visrtx {

namespace {
// A spliced MDL image node consumes a 2D texture lookup, so a textured input
// only binds when the sampler is one of the 2D image samplers.
bool is2DImageSampler(const Sampler *s)
{
  return dynamic_cast<const Image2D *>(s)
      || dynamic_cast<const CompressedImage2D *>(s);
}

} // namespace

MaterialX::MaterialX(DeviceGlobalState *d) : MDL(d)
{
  d->materialx.consumers.insert(this);
}

MaterialX::~MaterialX()
{
  deviceState()->materialx.consumers.erase(this);
}

bool MaterialX::needsRetranscode()
{
  // A successful transcode overwrites source/sourceType/materialName with the
  // generated MDL handoff (sourceType="code"). Recognize our own write so it is
  // not mistaken for new user input; otherwise distinguish the app's values.
  auto liveSource = getParamString("source", "");
  auto liveSourceType = getParamString("sourceType", "documentFile");

  // `code` is exclusively our handoff sentinel — never a valid app sourceType —
  // so it alone marks our write and the cached scheme is restored, INDEPENDENT
  // of `source`. (An app editing inline text re-sets `source` but not
  // `sourceType`; keying this on `source` too would misread our "code" as an
  // unknown value and silently fall back to documentFile, treating the new
  // inline XML as a file path.)
  std::string userSourceType;
  if (liveSourceType == "code") {
    userSourceType = m_userSourceType;
  } else if (liveSourceType == "documentFile" || liveSourceType == "documentInline") {
    userSourceType = liveSourceType;
  } else {
    if (liveSourceType != m_userSourceType)
      reportMessage(ANARI_SEVERITY_WARNING,
          "MaterialX: unknown sourceType '%s'; using 'documentFile' "
          "(valid: documentFile, documentInline)",
          liveSourceType.c_str());
    userSourceType = "documentFile";
  }

  // `source` carries the generated MDL after our handoff; recognize that so our
  // own output is not re-read as a user document.
  std::string userPath = (liveSource == m_generatedSource) ? m_userPath : liveSource;

  std::optional<std::string> userSelected;
  if (hasParam("materialName")) {
    auto liveName = getParamString("materialName", "");
    // "" is never a valid material name — treat it as no selection (apps
    // commonly stage a default-empty materialName param) rather than a lookup
    // that can only fail.
    if (liveName.empty())
      userSelected = std::nullopt;
    else
      userSelected = (liveName == m_generatedName)
          ? m_userSelected
          : std::optional<std::string>(liveName);
  }

  bool changed = userPath != m_userPath || userSelected != m_userSelected
      || userSourceType != m_userSourceType;

  // A re-resolved distribution (root changed at device commit — including
  // unresolved -> resolved recovery) invalidates the generated MDL even when
  // the user's inputs are untouched: the MDL search path now serves
  // ::materialx::* from the new root.
  changed =
      changed || deviceState()->materialx.generation != m_distributionGeneration;

  // Inline content lives in `source` (captured by userPath above); only a file
  // source has an on-disk mtime to watch.
  if (userSourceType == "documentFile") {
    std::error_code ec;
    auto write = std::filesystem::last_write_time(userPath, ec);
    if (!ec) {
      changed = changed || write != m_userPathWrite;
      m_userPathWrite = write;
    }
  }

  m_userPath = userPath;
  m_userSelected = userSelected;
  m_userSourceType = userSourceType;
  return changed;
}

void MaterialX::transcode(const std::vector<std::string> &texturedOrigins)
{
  const bool inlineDoc = m_userSourceType == "documentInline";
  auto src = inlineDoc ? materialx::DocumentSource::inlineText(m_userPath)
                       : materialx::DocumentSource::file(m_userPath);

  const auto &distribution = deviceState()->materialx;
  m_distributionGeneration = distribution.generation;
  if (distribution.root.empty()) {
    reportMessage(ANARI_SEVERITY_ERROR,
        "MaterialX: no distribution found; set the materialxSearchPaths device "
        "parameter or MATERIALX_SEARCH_PATH.%s",
        distribution.trace.c_str());
    reportMessage(
        ANARI_SEVERITY_WARNING, "MaterialX: falling back to default material");
    // FREEZE the derived state (m_paramMap/m_materialNames/m_texturedOrigins):
    // nothing can be recomputed without a distribution, and the routed args it
    // keeps alive are the only surviving copy of the user's values (routing
    // consumed the clean-name params on an earlier commit). Clearing here
    // would let routeParameters tear them down and recovery would render
    // nodedef defaults. Only the generated MDL is dropped — that alone drives
    // the fallback. If the user ALSO changed the document while unresolved,
    // the frozen mapping is stale for it; the recovery retranscode (generation
    // bump) rebuilds it and routing's teardown then cleans any orphans.
    m_generatedSource.clear();
    m_generatedName.clear();
    return;
  }

  std::vector<std::filesystem::path> libs = {distribution.root / "libraries"};
  if (!inlineDoc)
    libs.push_back(std::filesystem::path(m_userPath).parent_path());

  auto r = materialx::transcodeMaterialXToMdl(
      src, m_userSelected, libs, texturedOrigins);
  if (!r.error.empty() || r.mdlSource.empty()) {
    const char *label = inlineDoc ? "<inline document>" : m_userPath.c_str();
    reportMessage(ANARI_SEVERITY_ERROR, "MaterialX: failed to transcode '%s': %s",
        label, r.error.c_str());
    reportMessage(ANARI_SEVERITY_WARNING, "MaterialX: falling back to default material");
    // FREEZE, same invariant as the unresolved branch above: a failed
    // transcode returns an empty mapping, and adopting it would let the
    // routed-arg teardown delete the only copy of the user's values. Keep the
    // last-known-good derived state; only the generated MDL drops.
    m_generatedSource.clear();
    m_generatedName.clear();
    return;
  }
  m_materialNames = std::move(r.available);
  m_paramMap = std::move(r.paramMap);
  m_texturedOrigins = texturedOrigins;
  m_generatedSource = std::move(r.mdlSource);
  m_generatedName = std::move(r.materialName);
}

std::vector<std::string> MaterialX::desiredTexturedOrigins() const
{
  // Start from the persisted set so an absent (already-routed) clean param
  // leaves its origin unchanged. Recomputing purely from present params would
  // read "absent" as "no longer textured" and revert the topology on any commit
  // that does not re-supply the clean name (e.g. a host editing another input).
  std::set<std::string> origins(
      m_texturedOrigins.begin(), m_texturedOrigins.end());
  for (const auto &m : m_paramMap) {
    // Wired-texture rows (valueArg empty, textureArg set) are graph textures, not
    // host-toggleable constants: their texture_2d arg always exists, so a sampler
    // binds by pure routing. Exempt them from the topology set — otherwise a bound
    // sampler would force a spurious retranscode and the splice path would re-wrap
    // the already-wired file port, corrupting the graph.
    if (m.valueArg.empty())
      continue;
    if (!hasParam(m.cleanName))
      continue;
    if (getParamDirect(m.cleanName).type() == ANARI_SAMPLER)
      origins.insert(m.originPath);
    else
      origins.erase(m.originPath); // explicit value overrides a textured input
  }
  return {origins.begin(), origins.end()};
}

void MaterialX::routeParameters()
{
  std::set<std::string> wanted; // arg params that must remain set after routing
  for (const auto &m : m_paramMap) {
    if (!hasParam(m.cleanName)) {
      // Clean name consumed by a prior commit: preserve whichever arg already
      // carries its value so an unrelated re-commit keeps the binding.
      if (!m.textureArg.empty() && hasParam(m.textureArg))
        wanted.insert(m.textureArg);
      if (!m.valueArg.empty() && hasParam(m.valueArg))
        wanted.insert(m.valueArg);
      continue;
    }
    auto value = getParamDirect(m.cleanName);
    if (value.type() == ANARI_SAMPLER) {
      auto *s = value.getObject<Sampler>();
      if (m.textureArg.empty() || !s || !is2DImageSampler(s)) {
        reportMessage(ANARI_SEVERITY_WARNING,
            "MaterialX: '%s' sampler not bound (image2D textured input required)",
            m.cleanName.c_str());
        removeParam(m.cleanName);
        continue;
      }
      setParamDirect(m.textureArg, value);
      wanted.insert(m.textureArg);
    } else if (!m.valueArg.empty()) {
      setParamDirect(m.valueArg, value);
      wanted.insert(m.valueArg);
    }
    removeParam(m.cleanName);
  }
  // Tear down args a previous commit routed that the current topology no longer
  // uses, so a changed/reverted topology leaves no orphan param mismatching the
  // freshly compiled arg block (which would warn and leak the sampler).
  for (const auto &arg : m_routedArgs)
    if (!wanted.count(arg))
      removeParam(arg);
  m_routedArgs.assign(wanted.begin(), wanted.end());
}

void MaterialX::commitParameters()
{
  // 1. Ensure m_paramMap exists (need origins/types to decide what to texture).
  //    Pass the PERSISTED textured set, never {}: the clean sampler params were
  //    consumed by routing on an earlier commit, so a wiped set is not
  //    reconstructable and the bound topology would silently revert to
  //    constants (a distribution-root change or document mtime touch must keep
  //    textures). Origins absent from a changed document are skipped by the
  //    splice. First commit: persisted set is empty = value-only, as before.
  bool sourceChanged = needsRetranscode(); // reads/updates source/material/mtime
  if (m_paramMap.empty() || sourceChanged)
    transcode(m_texturedOrigins);

  // 2. Desired textured set (read BEFORE routing removes the clean names).
  auto desired = desiredTexturedOrigins();

  // 3. Retranscode if the textured topology changed.
  if (desired != m_texturedOrigins)
    transcode(desired);

  // 4. Re-apply the MDL handoff on EVERY commit, not only when we re-transcoded.
  // An app re-setting `source` to the .mtlx path (or helium re-staging the
  // object) would otherwise leave the base reading a path as MDL code and
  // silently fall back to diffuse. MDL::syncSource has its own content-based
  // change detection, so an unchanged handoff is a cheap no-op (no recompile).
  setParam("sourceType", ANARI_STRING, "code");
  setParam("source", ANARI_STRING, m_generatedSource.c_str());
  if (!m_generatedName.empty()) {
    setParam("materialName", ANARI_STRING, m_generatedName.c_str());
  } else if (m_userSelected) {
    // Fallback active (no generated material): keep the USER's selection in
    // the param. Removing it would make the next needsRetranscode misread the
    // absence as "user unset" and recover on the document's first material.
    setParam("materialName", ANARI_STRING, m_userSelected->c_str());
  } else {
    removeParam("materialName");
  }

  // 5. Route clean params -> value/texture args, then delegate. While the
  // fallback is active (no generated material) routing would CONSUME clean
  // params against a mapping the fallback cannot use — a sampler bound during
  // the outage would be dropped unrecoverably. Leave everything staged; the
  // recovery retranscode reads and routes it.
  if (!m_generatedSource.empty())
    routeParameters();
  MDL::commitParameters();
}

bool MaterialX::getProperty(const std::string_view &name, ANARIDataType type,
    void *ptr, uint64_t size, uint32_t flags)
{
  if (name == "materialNames" && type == ANARI_STRING_LIST) {
    if (m_materialNames.empty())
      return false; // not available before a successful commit
    m_materialNamePtrs.clear();
    for (const auto &s : m_materialNames)
      m_materialNamePtrs.push_back(s.c_str());
    m_materialNamePtrs.push_back(nullptr);
    auto **out = static_cast<const char ***>(ptr);
    *out = m_materialNamePtrs.data();
    return true;
  }
  if (name == "textureInputs" && type == ANARI_STRING_LIST) {
    m_textureInputNames.clear();
    for (const auto &m : m_paramMap)
      if (m.valueArg.empty() && !m.textureArg.empty())
        m_textureInputNames.push_back(m.cleanName);
    if (m_textureInputNames.empty())
      return false; // not available before a successful commit
    m_textureInputPtrs.clear();
    for (const auto &s : m_textureInputNames)
      m_textureInputPtrs.push_back(s.c_str());
    m_textureInputPtrs.push_back(nullptr);
    auto **out = static_cast<const char ***>(ptr);
    *out = m_textureInputPtrs.data();
    return true;
  }
  return MDL::getProperty(name, type, ptr, size, flags);
}

} // namespace visrtx
