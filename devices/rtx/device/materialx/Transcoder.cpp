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

#include <MaterialXCore/Document.h>
#include <MaterialXFormat/Util.h>
#include <MaterialXFormat/XmlIo.h>
#include <MaterialXGenMdl/MdlShaderGenerator.h>
#include <MaterialXGenShader/DefaultColorManagementSystem.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/Shader.h>
#include <MaterialXGenShader/UnitSystem.h>
#include <MaterialXGenShader/Util.h>

#include <memory>

namespace mx = MaterialX;

namespace visrtx::materialx {

namespace {
// Document/resource search path (for the .mtlx's includes + images).
mx::FileSearchPath toSearchPath(
    nonstd::span<const std::filesystem::path> paths)
{
  mx::FileSearchPath sp;
  for (const auto &p : paths)
    sp.append(mx::FilePath(p.string()));
  return sp;
}

// Load the MaterialX standard libraries. `librarySearchPaths.front()` is the
// data-library directory literally named "libraries". loadLibraries resolves
// the folder name "libraries" against a search path, so we search from its
// PARENT — passing the libraries dir itself would look for `.../libraries/
// libraries` and load nothing. (`getDefaultDataLibraryFolders()` is a
// Python-only helper; it does not exist in the C++ API.)
mx::DocumentPtr loadStdLibraries(
    nonstd::span<const std::filesystem::path> librarySearchPaths)
{
  auto stdLib = mx::createDocument();
  if (librarySearchPaths.empty())
    return stdLib;
  mx::FilePath librariesDir(librarySearchPaths.front().string());
  // .asString() is required: FileSearchPath has no FilePath ctor, only
  // FileSearchPath(const string&); FilePath→string is itself a user-defined
  // conversion, so two UDCs can't chain implicitly.
  mx::FileSearchPath libSearch(librariesDir.getParentPath().asString());
  mx::loadLibraries(mx::FilePathVec{mx::FilePath("libraries")}, libSearch, stdLib);
  return stdLib;
}
constexpr auto kMdlTargetVersion = mx::GenMdlOptions::MdlVersion::MDL_1_7;

mx::TypedElementPtr pickRenderable(const mx::DocumentPtr &doc,
    const std::optional<std::string> &selected, std::string &error)
{
  auto renderable = mx::findRenderableElements(doc);
  if (renderable.empty()) {
    error = "no renderable materials in document";
    return nullptr;
  }
  if (!selected)
    return renderable.front();
  for (const auto &e : renderable)
    if (e->getNamePath() == *selected || e->getName() == *selected)
      return e;
  error = "materialName '" + *selected + "' not found in document";
  return nullptr;
}
} // namespace

std::vector<std::string> enumerateRenderableMaterials(
    const std::filesystem::path &mtlxFile,
    nonstd::span<const std::filesystem::path> librarySearchPaths)
{
  try {
    auto stdLib = loadStdLibraries(librarySearchPaths);
    auto doc = mx::createDocument();
    doc->importLibrary(stdLib);
    mx::readFromXmlFile(doc, mtlxFile.string(), toSearchPath(librarySearchPaths));

    std::vector<std::string> names;
    for (const auto &elem : mx::findRenderableElements(doc))
      names.push_back(elem->getNamePath());
    return names;
  } catch (const std::exception &) {
    return {};
  }
}

TranscodeResult transcodeMaterialXToMdl(
    const std::filesystem::path &mtlxFile,
    std::optional<std::string> selected,
    nonstd::span<const std::filesystem::path> librarySearchPaths)
{
  TranscodeResult result;
  try {
    auto searchPath = toSearchPath(librarySearchPaths);
    auto stdLib = loadStdLibraries(librarySearchPaths);

    auto doc = mx::createDocument();
    doc->importLibrary(stdLib);
    mx::readFromXmlFile(doc, mtlxFile.string(), searchPath);

    for (const auto &e : mx::findRenderableElements(doc))
      result.available.push_back(e->getNamePath());

    auto elem = pickRenderable(doc, selected, result.error);
    if (!elem)
      return result;

    auto gen = mx::MdlShaderGenerator::create();
    auto cms = mx::DefaultColorManagementSystem::create(gen->getTarget());
    cms->loadLibrary(doc);
    gen->setColorManagementSystem(cms);
    auto units = mx::UnitSystem::create(gen->getTarget());
    units->loadLibrary(stdLib);
    gen->setUnitSystem(units);

    mx::GenContext context(gen);
    context.registerSourceCodeSearchPath(searchPath);
    context.getOptions().targetDistanceUnit = "meter";
    // GenMdlOptions has NO static create() — only a default ctor + public
    // targetVersion member. pushUserData takes shared_ptr<GenUserData>;
    // GenMdlOptions derives from it.
    auto mdlOptions = std::make_shared<mx::GenMdlOptions>();
    mdlOptions->targetVersion = kMdlTargetVersion;
    context.pushUserData(mx::GenMdlOptions::GEN_CONTEXT_USER_DATA_KEY, mdlOptions);

    auto shader = gen->generate(elem->getName(), elem, context);
    result.mdlSource = shader->getSourceCode(mx::Stage::PIXEL);
    result.materialName = shader->getName();
  } catch (const std::exception &e) {
    result.error = e.what();
    result.mdlSource.clear();
    result.materialName.clear();
  }
  return result;
}

} // namespace visrtx::materialx
