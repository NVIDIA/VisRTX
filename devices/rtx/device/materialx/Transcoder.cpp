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

#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <vector>

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

std::string sourceLabel(const DocumentSource &source)
{
  return source.kind == DocumentSource::Kind::Inline ? "<inline document>"
                                                     : source.value;
}

struct LoadedDocument
{
  mx::DocumentPtr doc;
  std::vector<mx::NodeDefPtr> userSurfaceNodeDefs; // authored in the source, pre-import
};

// Read the user document (file or inline text) BEFORE importing stdlib, so the
// nodedefs present at that point are exactly the user-authored ones (snapshot).
// Importing first would let a stdlib copy win a shared name and hide the user's.
LoadedDocument loadDocument(const DocumentSource &source,
    const mx::DocumentPtr &stdLib, const mx::FileSearchPath &searchPath)
{
  LoadedDocument loaded;
  loaded.doc = mx::createDocument();
  if (source.kind == DocumentSource::Kind::Inline)
    mx::readFromXmlString(loaded.doc, source.value, searchPath);
  else
    mx::readFromXmlFile(loaded.doc, source.value, searchPath);
  for (const auto &nd : loaded.doc->getNodeDefs())
    if (nd->getType() == mx::SURFACE_SHADER_TYPE_STRING)
      loaded.userSurfaceNodeDefs.push_back(nd);
  loaded.doc->importLibrary(stdLib);
  return loaded;
}

std::string joinNodeDefNames(const std::vector<mx::NodeDefPtr> &nds)
{
  std::string s;
  for (const auto &nd : nds) {
    if (!s.empty())
      s += ", ";
    s += nd->getName();
  }
  return s;
}

// An output parented to a nodedef-implementation nodegraph is internal, not a
// user-facing material. A free-standing document-level nodegraph
// (getNodeDef() == null) that outputs a surfaceshader IS a user material.
bool isImplementationOutput(const mx::TypedElementPtr &e)
{
  auto ng = e->getParent() ? e->getParent()->asA<mx::NodeGraph>() : nullptr;
  return ng && ng->getNodeDef() != nullptr;
}

// Among same-category overloads, the default-version nodedef; the sole member
// if there is only one; nullptr if several and none is flagged default.
mx::NodeDefPtr preferDefaultVersion(const std::vector<mx::NodeDefPtr> &nds)
{
  if (nds.size() == 1)
    return nds.front();
  // First default-flagged wins; two defaults in one category is an invalid
  // document we do not diagnose here.
  for (const auto &nd : nds)
    if (nd->getDefaultVersion())
      return nd;
  return nullptr;
}

// When a document has no user-authored renderable element, synthesize one from
// a user-snapshot surfaceshader nodedef: instantiate it and wrap it in a
// material node (a bare shader instance is not returned by
// findRenderableElements). Returns the material element, or nullptr with
// `error` set.
mx::TypedElementPtr autoInstantiateNodeDef(const mx::DocumentPtr &doc,
    const std::vector<mx::NodeDefPtr> &candidates,
    const std::optional<std::string> &selected,
    const std::string &sourceLabel,
    std::string &error)
{
  if (candidates.empty()) {
    error = "no renderable materials in '" + sourceLabel
        + "'; no instantiable surfaceshader nodedef in document";
    return nullptr;
  }

  mx::NodeDefPtr pick;
  if (selected) {
    for (const auto &nd : candidates)
      if (nd->getName() == *selected) {
        pick = nd;
        break;
      }
    if (!pick) {
      std::vector<mx::NodeDefPtr> byCategory;
      for (const auto &nd : candidates)
        if (nd->getNodeString() == *selected)
          byCategory.push_back(nd);
      if (byCategory.empty()) {
        error = "materialName '" + *selected
            + "' not found among instantiable surfaceshader nodedefs in '"
            + sourceLabel + "': " + joinNodeDefNames(candidates);
        return nullptr;
      }
      pick = preferDefaultVersion(byCategory);
      if (!pick) {
        error = "materialName '" + *selected + "' is ambiguous (overloads: "
            + joinNodeDefNames(byCategory) + "); use an exact nodedef name";
        return nullptr;
      }
    }
  } else {
    std::set<std::string> categories;
    for (const auto &nd : candidates)
      categories.insert(nd->getNodeString());
    if (categories.size() > 1) {
      error = "multiple instantiable surfaceshader nodedefs in '"
          + sourceLabel + "'; set materialName to one of: "
          + joinNodeDefNames(candidates);
      return nullptr;
    }
    pick = preferDefaultVersion(candidates);
    if (!pick) {
      error = "ambiguous surfaceshader overloads in '" + sourceLabel
          + "' (none flagged default); set materialName to an exact nodedef name: "
          + joinNodeDefNames(candidates);
      return nullptr;
    }
  }

  auto shader = doc->addNodeInstance(
      pick, doc->createValidChildName(pick->getNodeString()));
  return doc->addMaterialNode(
      doc->createValidChildName(pick->getNodeString() + "_material"), shader);
}

mx::TypedElementPtr pickRenderable(const mx::DocumentPtr &doc,
    const std::vector<mx::NodeDefPtr> &candidates,
    const std::optional<std::string> &selected,
    const std::string &sourceLabel,
    std::string &error)
{
  auto renderable = mx::findRenderableElements(doc);
  std::vector<mx::TypedElementPtr> materials;
  for (const auto &e : renderable)
    if (!isImplementationOutput(e))
      materials.push_back(e);
  if (materials.empty())
    return autoInstantiateNodeDef(doc, candidates, selected, sourceLabel, error);
  if (!selected)
    return materials.front();
  for (const auto &e : materials)
    if (e->getNamePath() == *selected || e->getName() == *selected)
      return e;
  error = "materialName '" + *selected + "' not found in document";
  return nullptr;
}
} // namespace

// image/tiledimage configuration ports that must NOT be exposed as params for a
// wired texture: the sampler supplies texels only; these stay at authored/MDL
// defaults (mirrors the splice path, which keeps only file/default). Validated
// against MaterialX 1.39.x image node inputs.
static const std::set<std::string> kImageAuxPorts = {
    "layer", "uaddressmode", "vaddressmode", "filtertype",
    "framerange", "frameoffset", "frameendaction",
    "uvtiling", "uvoffset", "realworldimagesize", "realworldtilesize"};

std::vector<std::string> enumerateRenderableMaterials(
    const DocumentSource &source,
    nonstd::span<const std::filesystem::path> librarySearchPaths)
{
  try {
    auto stdLib = loadStdLibraries(librarySearchPaths);
    auto loaded = loadDocument(source, stdLib, toSearchPath(librarySearchPaths));
    std::vector<std::string> names;
    for (const auto &elem : mx::findRenderableElements(loaded.doc))
      if (!isImplementationOutput(elem))
        names.push_back(elem->getNamePath());
    return names;
  } catch (const std::exception &) {
    return {};
  }
}

TranscodeResult transcodeMaterialXToMdl(
    const DocumentSource &source,
    std::optional<std::string> selected,
    nonstd::span<const std::filesystem::path> librarySearchPaths,
    nonstd::span<const std::string> texturedOriginPaths)
{
  TranscodeResult result;
  try {
    auto searchPath = toSearchPath(librarySearchPaths);
    auto stdLib = loadStdLibraries(librarySearchPaths);
    auto loaded = loadDocument(source, stdLib, searchPath);
    auto &doc = loaded.doc;

    for (const auto &e : mx::findRenderableElements(doc))
      if (!isImplementationOutput(e))
        result.available.push_back(e->getNamePath());

    auto elem = pickRenderable(
        doc, loaded.userSurfaceNodeDefs, selected, sourceLabel(source), result.error);
    if (!elem)
      return result;

    // Splice a type-matched <image> in front of each requested constant input.
    // Record imageNode -> (cleanName, originPath, type) for post-generate pairing.
    struct Spliced { std::string cleanName, originPath, type; };
    std::map<std::string, Spliced> splicedByImageNode; // image node name -> info
    for (const auto &origin : texturedOriginPaths) {
      auto slash = origin.find_last_of('/');
      if (slash == std::string::npos) continue;
      auto nodeName = origin.substr(0, slash);
      auto inputName = origin.substr(slash + 1);
      auto node = doc->getNode(nodeName);
      if (!node) continue;
      auto input = node->getInput(inputName);
      if (!input) {
        // Input absent from document (uses NodeDef default) — create it so we
        // can connect the image node without mutating the node's category name.
        // getActiveInput (not getInput) resolves inputs inherited from a base
        // nodedef: standard_surface's default nodedef declares only `base`/
        // `base_color` directly and inherits the rest (emission_color, etc.).
        auto nd = node->getNodeDef();
        if (!nd) continue;
        auto ndIn = nd->getActiveInput(inputName);
        if (!ndIn) continue;
        input = node->addInput(inputName, ndIn->getType());
        auto defVal = ndIn->getValueString();
        if (!defVal.empty()) input->setValueString(defVal);
      }
      if (!input || input->getConnectedNode())
        continue; // not texturable: missing or already author-connected
      const std::string T = input->getType();
      const std::string constVal = input->getValueString();
      auto imgName = doc->createValidChildName(inputName + "_tex");
      auto img = doc->addNode("image", imgName, T);
      auto fileIn = img->addInput("file", "filename");
      fileIn->setColorSpace("none"); // sampler delivers linear; no MDL re-decode
      auto defIn = img->addInput("default", T);
      if (!constVal.empty())
        defIn->setValueString(constVal);
      node->setConnectedNode(inputName, img);
      splicedByImageNode[imgName] = {inputName, origin, T};
    }

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

    // The single picked element is generated into its own MDL module, so the
    // module exposes exactly one material — the bare name always resolves to a
    // single overload in getFunctionDefinition; no ambiguity is possible even
    // for multi-material .mtlx documents.
    auto shader = gen->generate(elem->getName(), elem, context);
    result.mdlSource = shader->getSourceCode(mx::Stage::PIXEL);
    result.materialName = shader->getName();

    // Build paramMap from MDL::INPUTS, pairing spliced image ports per input.
    std::map<std::string, ParamMapping> texturedRows; // cleanName -> paired row
    const auto &pixelStage = shader->getStage(mx::Stage::PIXEL);
    if (pixelStage.getInputBlocks().count(mx::MDL::INPUTS)) {
      const auto &block = pixelStage.getInputBlock(mx::MDL::INPUTS);
      for (size_t i = 0; i < block.size(); ++i) {
        const mx::ShaderPort *port = block[i];
        const std::string &arg = port->getVariable();
        const std::string &path = port->getPath();
        if (arg.empty() || path.empty()) continue;
        auto slash = path.find_last_of('/');
        if (slash == std::string::npos) continue;
        auto nodePart = path.substr(0, slash);
        auto leaf = path.substr(slash + 1);
        auto it = splicedByImageNode.find(nodePart);
        if (it != splicedByImageNode.end()) {
          // A spliced image's port. Keep file/default; suppress aux ports.
          auto &row = texturedRows[it->second.cleanName];
          row.cleanName = it->second.cleanName;
          row.originPath = it->second.originPath;
          row.type = it->second.type;
          if (leaf == "file") row.textureArg = arg;
          else if (leaf == "default") row.valueArg = arg;
          // uaddressmode/vaddressmode/filtertype/layer/frame* -> ignored
          continue;
        }
        // Wired (author-connected) image/tiledimage node. Drop aux config ports
        // (left at authored/MDL defaults, mirroring the splice path's file/default
        // filtering); the file port is a texture_2d MDL arg already in the graph.
        std::string category;
        if (auto n = doc->getNode(nodePart)) category = n->getCategory();
        const bool isImageNode = category == "image" || category == "tiledimage";
        if (isImageNode && kImageAuxPorts.count(leaf))
          continue;

        std::string type;
        if (auto e = doc->getDescendant(path)) type = e->getAttribute("type");
        // Classification keys on the explicit MaterialX type attribute: a file
        // input without type="filename" falls to the value-row branch below
        // (unbindable). Documents from MaterialXGenMdl / HdMtlx always set it.
        if (type == "filename") {
          // texture_2d arg already compiled in: sampler-bindable by pure routing.
          // cleanName = path -> collision-free + host-predictable. valueArg empty
          // -> routeParameters takes the sampler branch and desiredTexturedOrigins
          // exempts it (no splice, no retranscode on bind/unbind).
          result.paramMap.push_back({path, path, type, /*valueArg=*/"", /*textureArg=*/arg});
        } else {
          result.paramMap.push_back({leaf, path, type, arg, ""});
        }
      }
    }
    for (auto &[name, row] : texturedRows)
      result.paramMap.push_back(row);
  } catch (const std::exception &e) {
    result.error = e.what();
    result.mdlSource.clear();
    result.materialName.clear();
  }
  return result;
}

} // namespace visrtx::materialx
