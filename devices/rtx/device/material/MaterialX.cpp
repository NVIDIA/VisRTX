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

// The TSD StandardSurface preset (and apps) reference the bundled standard
// surface by a builtin logical name rather than an install path, mirroring
// PhysicallyBasedMDL's builtin MDL module. Resolve it to the shipped .mtlx.
constexpr const char *kStandardSurfaceSource = "visrtx::standard_surface";
std::string resolveMaterialXSource(const std::string &source)
{
  if (source == kStandardSurfaceSource)
    return VISRTX_MATERIALX_STD_SURFACE_MTLX;
  return source;
}
} // namespace

MaterialX::MaterialX(DeviceGlobalState *d) : MDL(d) {}

bool MaterialX::needsRetranscode()
{
  // A successful transcode overwrites `source`/`materialName` with generated
  // MDL and its material name. Recognize that here so our OWN output is never
  // mistaken for new user input — otherwise the next commit would feed the MDL
  // code back in as a .mtlx path, fail to parse, and silently fall back to
  // diffuse. The app changing the path re-sets `source`, which then differs
  // from m_generatedSource and is adopted as the new user path.
  auto liveSource = getParamString("source", "");
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

  std::error_code ec;
  auto write = std::filesystem::last_write_time(userPath, ec);

  const bool changed = userPath != m_userPath || userSelected != m_userSelected
      || (!ec && write != m_userPathWrite);

  m_userPath = userPath;
  m_userSelected = userSelected;
  if (!ec)
    m_userPathWrite = write;
  return changed;
}

void MaterialX::transcode(const std::vector<std::string> &texturedOrigins)
{
  auto resolved = resolveMaterialXSource(m_userPath);
  std::vector<std::filesystem::path> libs = {MATERIALX_LIBRARIES_DIR,
      std::filesystem::path(resolved).parent_path()};
  auto r = materialx::transcodeMaterialXToMdl(
      resolved, m_userSelected, libs, texturedOrigins);
  m_materialNames = r.available;
  m_paramMap = std::move(r.paramMap);
  m_texturedOrigins = texturedOrigins;
  if (!r.error.empty() || r.mdlSource.empty()) {
    reportMessage(ANARI_SEVERITY_ERROR, "MaterialX: failed to transcode '%s': %s",
        m_userPath.c_str(), r.error.c_str());
    reportMessage(ANARI_SEVERITY_WARNING, "MaterialX: falling back to default material");
    m_generatedSource.clear();
    m_generatedName.clear();
  } else {
    m_generatedSource = std::move(r.mdlSource);
    m_generatedName = std::move(r.materialName);
  }
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
  //    On the first commit it is empty -> run a value-only transcode first.
  bool sourceChanged = needsRetranscode(); // reads/updates source/material/mtime
  if (m_paramMap.empty() || sourceChanged)
    transcode({}); // value-only; populates m_paramMap, m_generated*, clears textured

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
  if (m_generatedName.empty())
    removeParam("materialName");
  else
    setParam("materialName", ANARI_STRING, m_generatedName.c_str());

  // 5. Route clean params -> value/texture args, then delegate.
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
  return MDL::getProperty(name, type, ptr, size, flags);
}

} // namespace visrtx
