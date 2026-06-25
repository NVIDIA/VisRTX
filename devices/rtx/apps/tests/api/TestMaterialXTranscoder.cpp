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

#include "materialx/Transcoder.h"
#include <cstdio>
#include <filesystem>
#include <vector>

int main()
{
  const std::filesystem::path data =
      std::filesystem::path(MATERIALX_TEST_DATA_DIR) / "two_materials.mtlx";
  std::vector<std::filesystem::path> libs = {MATERIALX_LIBRARIES_DIR};

  auto names = visrtx::materialx::enumerateRenderableMaterials(
      visrtx::materialx::DocumentSource::file(data), libs);

  bool hasRed = false, hasBlue = false;
  for (const auto &n : names) {
    if (n.find("Mat_Red") != std::string::npos) hasRed = true;
    if (n.find("Mat_Blue") != std::string::npos) hasBlue = true;
  }
  if (names.size() != 2 || !hasRed || !hasBlue) {
    std::printf("FAIL: expected Mat_Red+Mat_Blue, got %zu names\n", names.size());
    return 1;
  }
  {
    const std::filesystem::path red =
        std::filesystem::path(MATERIALX_TEST_DATA_DIR) / "red_surface.mtlx";
    auto r = visrtx::materialx::transcodeMaterialXToMdl(
        visrtx::materialx::DocumentSource::file(red), std::nullopt, libs, {});
    if (!r.error.empty() || r.mdlSource.empty() || r.materialName.empty()) {
      std::printf("FAIL: red transcode error='%s'\n", r.error.c_str());
      return 1;
    }
    if (r.mdlSource.find("mdl 1.7") == std::string::npos) {
      std::printf("FAIL: generated MDL not pinned to 1.7\n");
      return 1;
    }
  }
  {
    auto r = visrtx::materialx::transcodeMaterialXToMdl(
        visrtx::materialx::DocumentSource::file("/no/such/file.mtlx"),
        std::nullopt, libs, {});
    if (r.error.empty() || !r.mdlSource.empty()) {
      std::printf("FAIL: missing-file should set error and empty source\n");
      return 1;
    }
  }
  {
    const std::filesystem::path red =
        std::filesystem::path(MATERIALX_TEST_DATA_DIR) / "red_surface.mtlx";
    auto r = visrtx::materialx::transcodeMaterialXToMdl(
        visrtx::materialx::DocumentSource::file(red), std::nullopt, libs, {});
    bool foundBaseColor = false;
    for (const auto &m : r.paramMap) {
      if (m.cleanName == "base_color") {
        foundBaseColor = true;
        if (m.valueArg != "srf_base_color") {
          std::printf("FAIL: base_color valueArg='%s', expected 'srf_base_color'\n",
              m.valueArg.c_str());
          return 1;
        }
        if (m.originPath != "srf/base_color") {
          std::printf("FAIL: base_color originPath='%s'\n", m.originPath.c_str());
          return 1;
        }
        if (m.type != "color3") {
          std::printf("FAIL: base_color type='%s'\n", m.type.c_str());
          return 1;
        }
        if (!m.textureArg.empty()) {
          std::printf("FAIL: base_color textureArg should be empty (no splice)\n");
          return 1;
        }
      }
    }
    if (!foundBaseColor) { std::printf("FAIL: paramMap missing base_color\n"); return 1; }
  }
  {
    const std::filesystem::path red =
        std::filesystem::path(MATERIALX_TEST_DATA_DIR) / "red_surface.mtlx";
    std::vector<std::string> textured = {"srf/base_color"};
    auto r = visrtx::materialx::transcodeMaterialXToMdl(
        visrtx::materialx::DocumentSource::file(red), std::nullopt, libs, textured);
    if (!r.error.empty()) { std::printf("FAIL: textured transcode: %s\n", r.error.c_str()); return 1; }
    bool ok = false;
    for (const auto &m : r.paramMap) {
      if (m.cleanName == "base_color" && m.originPath == "srf/base_color") {
        ok = !m.valueArg.empty() && !m.textureArg.empty() && m.type == "color3";
        if (!ok)
          std::printf("FAIL: base_color row valueArg='%s' textureArg='%s' type='%s'\n",
              m.valueArg.c_str(), m.textureArg.c_str(), m.type.c_str());
      }
    }
    if (!ok) { std::printf("FAIL: base_color not a paired textured row\n"); return 1; }
  }
  // Textured an INHERITED input. standard_surface's default nodedef declares
  // only base/base_color directly and inherits the rest, so emission_color must
  // be materialized via getActiveInput (not getInput) for the splice to fire.
  {
    const std::filesystem::path red =
        std::filesystem::path(MATERIALX_TEST_DATA_DIR) / "red_surface.mtlx";
    std::vector<std::string> textured = {"srf/emission_color"};
    auto r = visrtx::materialx::transcodeMaterialXToMdl(
        visrtx::materialx::DocumentSource::file(red), std::nullopt, libs, textured);
    if (!r.error.empty()) { std::printf("FAIL: emission textured transcode: %s\n", r.error.c_str()); return 1; }
    bool ok = false;
    for (const auto &m : r.paramMap)
      if (m.cleanName == "emission_color" && m.originPath == "srf/emission_color")
        ok = !m.textureArg.empty() && m.type == "color3";
    if (!ok) {
      std::printf("FAIL: emission_color (inherited) not a paired textured row\n");
      return 1;
    }
  }
  // --- Auto-instantiation: headline regression (distribution standard_surface
  // is nodedef-only with TWO overloaded surfaceshader nodedefs). materialName
  // unset must resolve to the default-version nodedef and transcode. ---
  {
    const std::filesystem::path ss =
        std::filesystem::path(MATERIALX_LIBRARIES_DIR) / "bxdf" / "standard_surface.mtlx";
    auto r = visrtx::materialx::transcodeMaterialXToMdl(
        visrtx::materialx::DocumentSource::file(ss), std::nullopt, libs, {});
    if (!r.error.empty() || r.mdlSource.empty() || r.materialName.empty()) {
      std::printf("FAIL: auto-instantiate (unset) error='%s' src.empty=%d\n",
          r.error.c_str(), (int)r.mdlSource.empty());
      return 1;
    }
    bool foundBaseColor = false;
    for (const auto &m : r.paramMap)
      if (m.cleanName == "base_color"
          && m.originPath == "standard_surface/base_color")
        foundBaseColor = true;
    if (!foundBaseColor) {
      std::printf("FAIL: auto-instantiate paramMap missing standard_surface/base_color\n");
      return 1;
    }
    if (!r.available.empty()) {
      std::printf("FAIL: auto-instantiate should report empty available, got %zu\n",
          r.available.size());
      return 1;
    }
  }
  // Same file, selected by node category and by exact nodedef name -> both resolve.
  {
    const std::filesystem::path ss =
        std::filesystem::path(MATERIALX_LIBRARIES_DIR) / "bxdf" / "standard_surface.mtlx";
    auto byCat = visrtx::materialx::transcodeMaterialXToMdl(
        visrtx::materialx::DocumentSource::file(ss),
        std::string("standard_surface"), libs, {});
    auto byName = visrtx::materialx::transcodeMaterialXToMdl(
        visrtx::materialx::DocumentSource::file(ss),
        std::string("ND_standard_surface_surfaceshader"), libs, {});
    if (!byCat.error.empty() || byCat.mdlSource.empty()) {
      std::printf("FAIL: auto-instantiate by category: %s\n", byCat.error.c_str());
      return 1;
    }
    if (!byName.error.empty() || byName.mdlSource.empty()) {
      std::printf("FAIL: auto-instantiate by nodedef name: %s\n", byName.error.c_str());
      return 1;
    }
  }
  // Error: file has nodedefs but none is a surfaceshader.
  {
    const std::filesystem::path f =
        std::filesystem::path(MATERIALX_TEST_DATA_DIR) / "no_surface_nodedef.mtlx";
    auto r = visrtx::materialx::transcodeMaterialXToMdl(
        visrtx::materialx::DocumentSource::file(f), std::nullopt, libs, {});
    if (r.error.empty() || !r.mdlSource.empty()) {
      std::printf("FAIL: no-surfaceshader should error with empty source\n");
      return 1;
    }
  }
  // Error: two distinct surfaceshader categories, materialName unset -> ambiguous.
  {
    const std::filesystem::path f =
        std::filesystem::path(MATERIALX_TEST_DATA_DIR) / "two_surface_nodedefs.mtlx";
    auto r = visrtx::materialx::transcodeMaterialXToMdl(
        visrtx::materialx::DocumentSource::file(f), std::nullopt, libs, {});
    if (r.error.empty() || !r.mdlSource.empty()) {
      std::printf("FAIL: multi-category unset should error with empty source\n");
      return 1;
    }
  }
  // Regression (Fix 1): a document-level nodegraph that outputs a surfaceshader
  // is a real material and must transcode normally, NOT trigger auto-instantiation.
  // A too-broad predicate (any NodeGraph parent) would divert it into
  // auto-instantiation and error with "no instantiable surfaceshader nodedef".
  {
    const std::filesystem::path f =
        std::filesystem::path(MATERIALX_TEST_DATA_DIR) / "graph_surface.mtlx";
    auto r = visrtx::materialx::transcodeMaterialXToMdl(
        visrtx::materialx::DocumentSource::file(f), std::nullopt, libs, {});
    if (!r.error.empty() || r.mdlSource.empty()) {
      std::printf("FAIL: graph_surface should transcode normally: '%s'\n", r.error.c_str());
      return 1;
    }
  }
  // Regression: a nodedef-only file loaded from a path OTHER than the library
  // dir (e.g. a separate MaterialX checkout, as a user would). Auto-instantiation
  // must still find the file's surfaceshader nodedef even though stdlib defines
  // the same names — the file must be read before stdlib is imported so its
  // nodedef keeps its own source URI rather than the build-time stdlib path.
  {
    const std::filesystem::path src =
        std::filesystem::path(MATERIALX_LIBRARIES_DIR) / "bxdf" / "standard_surface.mtlx";
    std::error_code ec;
    auto tmp = std::filesystem::temp_directory_path(ec)
        / "visrtx_mtlx_regression_standard_surface.mtlx";
    std::filesystem::copy_file(
        src, tmp, std::filesystem::copy_options::overwrite_existing, ec);
    if (ec) {
      std::printf("FAIL: could not stage temp .mtlx: %s\n", ec.message().c_str());
      return 1;
    }
    auto r = visrtx::materialx::transcodeMaterialXToMdl(
        visrtx::materialx::DocumentSource::file(tmp), std::nullopt, libs, {});
    std::filesystem::remove(tmp, ec);
    if (!r.error.empty() || r.mdlSource.empty()) {
      std::printf("FAIL: off-libdir nodedef file should auto-instantiate: '%s'\n",
          r.error.c_str());
      return 1;
    }
  }
  // Inline: a complete material given as XML text (no file) transcodes.
  {
    const std::string xml =
        "<?xml version=\"1.0\"?>\n"
        "<materialx version=\"1.39\">\n"
        "  <standard_surface name=\"srf\" type=\"surfaceshader\">\n"
        "    <input name=\"base_color\" type=\"color3\" value=\"0.1, 0.8, 0.2\"/>\n"
        "  </standard_surface>\n"
        "  <surfacematerial name=\"M\" type=\"material\">\n"
        "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"srf\"/>\n"
        "  </surfacematerial>\n"
        "</materialx>\n";
    auto r = visrtx::materialx::transcodeMaterialXToMdl(
        visrtx::materialx::DocumentSource::inlineText(xml), std::nullopt, libs, {});
    if (!r.error.empty() || r.mdlSource.empty()) {
      std::printf("FAIL: inline complete material: '%s'\n", r.error.c_str());
      return 1;
    }
    bool found = false;
    for (const auto &m : r.paramMap)
      if (m.cleanName == "base_color" && m.originPath == "srf/base_color")
        found = true;
    if (!found) { std::printf("FAIL: inline paramMap missing srf/base_color\n"); return 1; }
  }
  // Inline nodedef-only: a custom surfaceshader nodedef + nodegraph impl, no
  // renderable element -> auto-instantiation. Guards inline provenance (no URI).
  // The nodedef declares its type via an explicit <output>, not a type= attr.
  {
    const std::string xml =
        "<?xml version=\"1.0\"?>\n"
        "<materialx version=\"1.39\">\n"
        "  <nodedef name=\"ND_my_surface_vt\" node=\"my_surface_vt\">\n"
        "    <input name=\"base_color\" type=\"color3\" value=\"0.1, 0.8, 0.2\"/>\n"
        "    <output name=\"out\" type=\"surfaceshader\"/>\n"
        "  </nodedef>\n"
        "  <nodegraph name=\"IMP_my_surface_vt\" nodedef=\"ND_my_surface_vt\">\n"
        "    <standard_surface name=\"ss\" type=\"surfaceshader\">\n"
        "      <input name=\"base_color\" type=\"color3\" interfacename=\"base_color\"/>\n"
        "    </standard_surface>\n"
        "    <output name=\"out\" type=\"surfaceshader\" nodename=\"ss\"/>\n"
        "  </nodegraph>\n"
        "</materialx>\n";
    auto r = visrtx::materialx::transcodeMaterialXToMdl(
        visrtx::materialx::DocumentSource::inlineText(xml), std::nullopt, libs, {});
    if (!r.error.empty() || r.mdlSource.empty()) {
      std::printf("FAIL: inline nodedef-only auto-instantiate: '%s'\n", r.error.c_str());
      return 1;
    }
  }
  // Inline malformed XML -> error, empty source (caught by transcode try/catch).
  {
    auto r = visrtx::materialx::transcodeMaterialXToMdl(
        visrtx::materialx::DocumentSource::inlineText("<materialx not closed"),
        std::nullopt, libs, {});
    if (r.error.empty() || !r.mdlSource.empty()) {
      std::printf("FAIL: malformed inline should error with empty source\n");
      return 1;
    }
  }
  std::printf("PASS\n");
  return 0;
}
