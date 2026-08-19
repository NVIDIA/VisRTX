// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/importers/detail/usd/UsdMaterials.h"
// usd
#include <pxr/imaging/hd/materialConnectionSchema.h>
#include <pxr/imaging/hd/materialNetworkSchema.h>
#include <pxr/imaging/hd/materialNodeParameterSchema.h>
#include <pxr/imaging/hd/materialNodeSchema.h>
#include <pxr/imaging/hd/materialSchema.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/usdShade/connectableAPI.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/shader.h>
#if TSD_USD_HAS_MATERIALX
#include <MaterialXCore/Document.h>
#include <MaterialXCore/Node.h>
#include <MaterialXFormat/XmlIo.h>
#include <pxr/imaging/hd/dataSourceMaterialNetworkInterface.h>
#include <pxr/imaging/hdMtlx/hdMtlx.h>
#endif
// std
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

namespace tsd::io::usd {

using namespace tsd::core;

namespace {

// USD shader node identifiers this converter understands.
const pxr::TfToken PREVIEW_SURFACE_ID("UsdPreviewSurface");
const pxr::TfToken UV_TEXTURE_ID("UsdUVTexture");
const pxr::TfToken PRIMVAR_READER_ID("UsdPrimvarReader_float2");
const pxr::TfToken TRANSFORM_2D_ID("UsdTransform2d");

// The Render Context OpenUSD publishes MaterialX networks under.
const pxr::TfToken MATERIALX_CONTEXT("mtlx");

// One resolved UsdPreviewSurface network, walked lazily out of the Hydra
// material network container.
struct NetworkWalker
{
  pxr::HdMaterialNetworkSchema network;

  pxr::HdMaterialNodeSchema node(const pxr::TfToken &path) const;
  pxr::TfToken nodeId(const pxr::TfToken &path) const;
  pxr::VtValue parameter(
      const pxr::TfToken &nodePath, const char *paramName) const;
  std::string stringParameter(
      const pxr::TfToken &nodePath, const char *paramName) const;
  pxr::TfToken connectedNode(
      const pxr::TfToken &nodePath, const char *inputName) const;
};

pxr::HdMaterialNodeSchema NetworkWalker::node(const pxr::TfToken &path) const
{
  return network.GetNodes().Get(path);
}

pxr::TfToken NetworkWalker::nodeId(const pxr::TfToken &path) const
{
  auto n = node(path);
  if (!n)
    return {};
  auto id = n.GetNodeIdentifier();
  return id ? id->GetTypedValue(0) : pxr::TfToken();
}

pxr::VtValue NetworkWalker::parameter(
    const pxr::TfToken &nodePath, const char *paramName) const
{
  auto n = node(nodePath);
  if (!n)
    return {};
  auto param = n.GetParameters().Get(pxr::TfToken(paramName));
  if (!param)
    return {};
  auto value = param.GetValue();
  return value ? value->GetValue(0) : pxr::VtValue();
}

// Shader inputs that name something -- a wrap mode, a colour space, a primvar
// -- are authored as a TfToken by some exporters and a plain string by others,
// so both spellings have to be accepted wherever one is read.
std::string NetworkWalker::stringParameter(
    const pxr::TfToken &nodePath, const char *paramName) const
{
  const auto value = parameter(nodePath, paramName);
  if (value.IsHolding<pxr::TfToken>())
    return value.UncheckedGet<pxr::TfToken>().GetString();
  if (value.IsHolding<std::string>())
    return value.UncheckedGet<std::string>();
  return {};
}

pxr::TfToken NetworkWalker::connectedNode(
    const pxr::TfToken &nodePath, const char *inputName) const
{
  auto n = node(nodePath);
  if (!n)
    return {};
  auto connections = n.GetInputConnections().Get(pxr::TfToken(inputName));
  if (!connections || connections.GetNumElements() == 0)
    return {};
  auto upstream = connections.GetElement(0).GetUpstreamNodePath();
  return upstream ? upstream->GetTypedValue(0) : pxr::TfToken();
}

// The UV primvar a texture reads, found by following the texture's `st` input
// back to whatever primvar reader ultimately feeds it. This is what lets an
// asset that does not use the conventional primvar name still get its
// textures.
std::string uvPrimvarOfTexture(
    const NetworkWalker &walker, const pxr::TfToken &texturePath)
{
  auto current = walker.connectedNode(texturePath, "st");
  // A UsdTransform2d may sit between the texture and the reader.
  for (int hop = 0; hop < 4 && !current.IsEmpty(); ++hop) {
    const auto id = walker.nodeId(current);
    if (id == PRIMVAR_READER_ID)
      return walker.stringParameter(current, "varname");
    if (id != TRANSFORM_2D_ID)
      return {};
    current = walker.connectedNode(current, "in");
  }
  return {};
}

// The UsdTransform2d feeding a texture's `st`, in the form ANARI takes it.
// USD applies it to the authored v-up coordinates, and the geometry converter
// has already reversed each vertex's `v` into ANARI's convention, so `v` is
// conjugated by that reversal: 1 - (s*(1 - v) + t) == s*v + (1 - s - t).
std::optional<UvTransform> uvTransformOfTexture(
    const NetworkWalker &walker, const pxr::TfToken &texturePath)
{
  auto transformNode = walker.connectedNode(texturePath, "st");
  if (walker.nodeId(transformNode) != TRANSFORM_2D_ID)
    return {};

  math::float2 s(1.f, 1.f);
  const auto scale = walker.parameter(transformNode, "scale");
  if (scale.IsHolding<pxr::GfVec2f>()) {
    const auto value = scale.UncheckedGet<pxr::GfVec2f>();
    s = math::float2(value[0], value[1]);
  }
  math::float2 t(0.f, 0.f);
  const auto translation = walker.parameter(transformNode, "translation");
  if (translation.IsHolding<pxr::GfVec2f>()) {
    const auto value = translation.UncheckedGet<pxr::GfVec2f>();
    t = math::float2(value[0], value[1]);
  }

  auto retval = math::IDENTITY_MAT4;
  retval[0][0] = s.x;
  retval[1][1] = s.y;
  return UvTransform{retval, math::float4(t.x, 1.f - s.y - t.y, 0.f, 0.f)};
}

std::string wrapModeOf(const NetworkWalker &walker,
    const pxr::TfToken &texturePath,
    const char *input)
{
  const auto mode = walker.stringParameter(texturePath, input);
  if (mode == "clamp")
    return "clampToEdge";
  if (mode == "mirror")
    return "mirrorRepeat";
  if (mode == "black")
    return "clampToBorder";
  return "repeat";
}

// Textures whose colour space is not sRGB carry data rather than colour and
// must not be de-gamma'd on load.
bool textureIsLinear(const NetworkWalker &walker,
    const pxr::TfToken &texturePath,
    bool colorRole)
{
  const auto space = walker.stringParameter(texturePath, "sourceColorSpace");
  if (space == "raw")
    return true;
  if (space == "sRGB")
    return false;
  // "auto" and unauthored: colour inputs are sRGB, data inputs are not.
  return !colorRole;
}

float3 asFloat3(const pxr::VtValue &v, const float3 &alt)
{
  if (v.IsHolding<pxr::GfVec3f>()) {
    const auto c = v.UncheckedGet<pxr::GfVec3f>();
    return float3(c[0], c[1], c[2]);
  }
  if (v.IsHolding<float>()) {
    const auto f = v.UncheckedGet<float>();
    return float3(f, f, f);
  }
  return alt;
}

// The absolute path an asset-valued input names, empty when it names nothing.
// The Stage's resolver produces a path for anything it could find; what it
// could not -- a UDIM tile set names no file -- is anchored to the Stage's own
// directory instead.
std::string anchoredAssetPath(
    ImportContext &ctx, const pxr::SdfAssetPath &assetPath)
{
  auto file = assetPath.GetResolvedPath();
  if (file.empty())
    file = assetPath.GetAssetPath();
  if (file.empty())
    return {};
  if (!isAbsolute(file))
    file = ctx.basePath + file;
  return file;
}

// Native MDL passthrough, read from the retained Stage because the MDL source
// asset and its sub-identifier are UsdShade concepts rather than something the
// resolved network models portably. Returns a null ref when the material has
// no MDL network, so the caller can fall back to a portable mapping.
MaterialRef tryMdlPassthrough(
    ImportContext &ctx, const pxr::SdfPath &materialPath)
{
  auto usdPrim = ctx.stage->GetPrimAtPath(materialPath);
  if (!usdPrim)
    return {};

  pxr::UsdShadeMaterial usdMaterial(usdPrim);
  if (!usdMaterial)
    return {};

  auto mdlOutput = usdMaterial.GetSurfaceOutput(pxr::TfToken("mdl"));
  if (!mdlOutput)
    return {};

  for (const auto &connection : mdlOutput.GetConnectedSources()) {
    pxr::UsdShadeShader shader(connection.source.GetPrim());
    if (!shader)
      continue;

    pxr::SdfAssetPath sourceAsset;
    if (!shader.GetSourceAsset(&sourceAsset, pxr::TfToken("mdl")))
      continue;

    auto module = sourceAsset.GetResolvedPath();
    if (module.empty())
      module = sourceAsset.GetAssetPath();
    if (module.empty())
      continue;

    pxr::TfToken subIdentifier;
    shader.GetSourceAssetSubIdentifier(&subIdentifier, pxr::TfToken("mdl"));

    auto material = ctx.scene->createObject<Material>(tokens::material::mdl);
    material->setName(materialPath.GetString().c_str());
    material->setParameter("sourceType", "module");
    material->setParameter("source", module.c_str());
    material->setParameter("materialName", subIdentifier.GetText());

    // Carry the shader's own scalar and colour inputs through as parameters;
    // the mdl subtype supports arbitrary parameter passthrough.
    for (const auto &input : shader.GetInputs()) {
      const auto name = input.GetBaseName().GetString();
      float scalar = 0.f;
      pxr::GfVec3f color;
      if (input.Get(&color))
        material->setParameter(
            Token(name.c_str()), float3(color[0], color[1], color[2]));
      else if (input.Get(&scalar))
        material->setParameter(Token(name.c_str()), scalar);
    }

    return material;
  }

  return {};
}

// OmniPBR mapping ////////////////////////////////////////////////////////////

// The value an input actually produces, which for an Omniverse asset is often
// published on the Material's own interface input rather than authored on the
// shader. GetValueProducingAttributes() walks that connection for us.
template <typename T>
std::optional<T> shaderInputValue(
    const pxr::UsdShadeShader &shader, const char *name, pxr::UsdTimeCode time)
{
  auto input = shader.GetInput(pxr::TfToken(name));
  if (!input)
    return {};
  const auto sources = input.GetValueProducingAttributes();
  const auto attribute = sources.empty() ? input.GetAttr() : sources.front();
  T value;
  if (attribute && attribute.Get(&value, time))
    return value;
  return {};
}

// The file an OmniPBR texture input names. OmniPBR usually names it on the
// input directly, but an asset authored through a Material Graph reaches it
// through a texture-reader node instead, whose own `file` input is where the
// path actually is.
std::optional<pxr::SdfAssetPath> omniPbrTextureAsset(
    const pxr::UsdShadeShader &shader, const char *name, pxr::UsdTimeCode time)
{
  if (auto asset = shaderInputValue<pxr::SdfAssetPath>(shader, name, time))
    return asset;

  auto input = shader.GetInput(pxr::TfToken(name));
  if (!input || !input.HasConnectedSource())
    return {};

  pxr::UsdShadeConnectableAPI source;
  pxr::TfToken sourceName;
  pxr::UsdShadeAttributeType sourceType;
  if (!input.GetConnectedSource(&source, &sourceName, &sourceType))
    return {};

  pxr::UsdShadeShader reader(source.GetPrim());
  if (!reader)
    return {};
  return shaderInputValue<pxr::SdfAssetPath>(reader, "file", time);
}

// The OmniPBR shader driving a material's MDL surface, or an invalid shader
// when the material is something else. OmniPBR is the shader Omniverse authors
// by default, and it is named exactly rather than by prefix: a module whose
// name merely starts with it is a different shader with input semantics of its
// own, and mapping it as OmniPBR would be a guess.
//
// Only the MDL Render Context's surface output is asked. The universal one is
// where a UsdPreviewSurface network would be, and that network is the reader
// below's to handle.
pxr::UsdShadeShader omniPbrShaderOf(const pxr::UsdShadeMaterial &usdMaterial)
{
  const pxr::TfToken mdl("mdl");

  auto isOmniPbr = [&](const pxr::UsdShadeShader &shader) {
    pxr::TfToken subIdentifier;
    if (shader.GetSourceAssetSubIdentifier(&subIdentifier, mdl)
        && subIdentifier == pxr::TfToken("OmniPBR"))
      return true;

    // An asset that named no sub-identifier is identified by its module.
    pxr::SdfAssetPath sourceAsset;
    if (!shader.GetSourceAsset(&sourceAsset, mdl))
      return false;
    // The authored path is what names the module; a resolved one would name
    // wherever this Stage's MDL search paths happened to find it.
    auto module = sourceAsset.GetAssetPath();
    if (module.empty())
      module = sourceAsset.GetResolvedPath();
    return std::filesystem::path(module).stem().string() == "OmniPBR";
  };

  auto output = usdMaterial.GetSurfaceOutput(mdl);
  if (output) {
    for (const auto &connection : output.GetConnectedSources()) {
      pxr::UsdShadeShader shader(connection.source.GetPrim());
      if (shader && isOmniPbr(shader))
        return shader;
    }
  }

  return pxr::UsdShadeShader();
}

// Map an OmniPBR shader onto a portable physically-based material, read from
// the retained Stage because OmniPBR's inputs are UsdShade concepts that the
// resolved network does not model portably -- the same reason
// tryMdlPassthrough() reads from there. Returns a null ref when the material
// is not OmniPBR, so the caller can fall through to the preview-surface
// reader.
//
// Only the inputs that carry over are read: the preview-surface reader would
// look for `diffuseColor`/`metallic`/`roughness` and find none of OmniPBR's
// own names, leaving the asset flat grey.
MaterialRef tryOmniPbrMapping(ImportContext &ctx,
    const pxr::SdfPath &materialPath,
    const pxr::TfToken &primType)
{
  auto usdPrim = ctx.stage->GetPrimAtPath(materialPath);
  if (!usdPrim)
    return {};

  pxr::UsdShadeMaterial usdMaterial(usdPrim);
  if (!usdMaterial)
    return {};

  auto shader = omniPbrShaderOf(usdMaterial);
  if (!shader)
    return {};

  auto material =
      ctx.scene->createObject<Material>(tokens::material::physicallyBased);
  material->setName(materialPath.GetString().c_str());

  // Every textured input takes precedence over its constant, which is what
  // OmniPBR itself does with them.
  auto bindTexture =
      [&](const char *usdName, const char *tsdName, bool colorRole) -> bool {
    const auto asset = omniPbrTextureAsset(shader, usdName, ctx.importTime);
    if (!asset)
      return false;
    const auto file = anchoredAssetPath(ctx, *asset);
    if (file.empty())
      return false;
    // OmniPBR names no colour space of its own, so the role of the input is
    // what says whether its texels must be de-gamma'd on load.
    auto sampler = importTexture(ctx.textureCache, file, !colorRole);
    if (!sampler) {
      ctx.reportSkip(materialPath,
          primType.GetString(),
          UsdSkipReason::TEXTURE_LOAD_FAILED,
          file);
      return false;
    }
    material->setParameterObject(Token(tsdName), *sampler);
    return true;
  };

  // alphaMode is a string selection, so the index has to move with the value
  // or the two disagree wherever the selection is what gets read.
  auto setAlphaMode = [&](const char *mode) {
    material->setParameter("alphaMode", mode);
    auto *parameter = material->parameter("alphaMode");
    const auto &modes = parameter->stringValues();
    for (size_t i = 0; i < modes.size(); ++i) {
      if (modes[i] == mode) {
        parameter->setStringSelection(int(i));
        break;
      }
    }
  };

  auto scalar = [&](const char *name) {
    return shaderInputValue<float>(shader, name, ctx.importTime);
  };
  auto color = [&](const char *name) -> std::optional<float3> {
    const auto value =
        shaderInputValue<pxr::GfVec3f>(shader, name, ctx.importTime);
    if (!value)
      return {};
    return float3((*value)[0], (*value)[1], (*value)[2]);
  };

  if (!bindTexture("diffuse_texture", "baseColor", true)) {
    if (const auto diffuse = color("diffuse_color_constant"))
      material->setParameter("baseColor", *diffuse);
  }

  // The colour is what OmniPBR emits at unit intensity; the two multiply.
  if (shaderInputValue<bool>(shader, "enable_emission", ctx.importTime)
          .value_or(false)) {
    const auto emissive = color("emissive_color").value_or(float3(0.f));
    const auto intensity = scalar("emissive_intensity").value_or(1.f);
    material->setParameter("emissive", emissive * intensity);
  }

  // OmniPBR's own defaults, which differ from the portable material's, so an
  // asset that leaves these unauthored still looks like it did in Omniverse.
  if (!bindTexture("metallic_texture", "metallic", false))
    material->setParameter(
        "metallic", scalar("metallic_constant").value_or(0.f));
  if (!bindTexture("reflectionroughness_texture", "roughness", false)) {
    material->setParameter(
        "roughness", scalar("reflection_roughness_constant").value_or(0.5f));
  }

  bindTexture("normalmap_texture", "normal", false);
  bindTexture("ao_texture", "occlusion", false);

  if (shaderInputValue<bool>(shader, "enable_opacity", ctx.importTime)
          .value_or(false)) {
    if (!bindTexture("opacity_texture", "opacity", false)) {
      if (const auto opacity = scalar("opacity_constant"))
        material->setParameter("opacity", *opacity);
    }
    // A threshold of zero means OmniPBR blends rather than cuts out.
    const auto threshold = scalar("opacity_threshold").value_or(0.f);
    if (threshold > 0.f) {
      setAlphaMode("mask");
      material->setParameter("alphaCutoff", threshold);
    } else
      setAlphaMode("blend");
  } else
    setAlphaMode("opaque");

  if (const auto ior = scalar("ior_constant"))
    material->setParameter("ior", *ior);
  if (const auto specular = scalar("specular_level"))
    material->setParameter("specular", *specular);

  return material;
}

// Try each Render Context in the caller's preference order, falling back per
// material so a Stage mixing network flavours resolves completely either way.
pxr::HdMaterialNetworkSchema selectNetwork(
    const pxr::HdMaterialSchema &material,
    const std::vector<std::string> &preference)
{
  for (const auto &context : preference) {
    auto network = material.GetMaterialNetwork(pxr::TfToken(context));
    if (network && network.GetNodes())
      return network;
  }
  // Nothing preferred matched: take whatever the material does have.
  for (const auto &context : material.GetRenderContexts()) {
    auto network = material.GetMaterialNetwork(context);
    if (network && network.GetNodes())
      return network;
  }
  return material.GetMaterialNetwork();
}

#if TSD_USD_HAS_MATERIALX

// The node a generated document holds under `name`, which OpenUSD's conversion
// places inside a node graph rather than at the document's top level.
MaterialX::NodePtr documentNode(
    const MaterialX::DocumentPtr &document, const std::string &name)
{
  if (auto node = document->getNode(name))
    return node;
  for (const auto &graph : document->getNodeGraphs()) {
    if (auto node = graph->getNode(name))
      return node;
  }
  return {};
}

// One texture a generated document reads, found on the way through.
struct DocumentTexture
{
  // The document-relative path of the `filename` input, which is the name the
  // device publishes the input under and expects a sampler bound to.
  std::string inputPath;
  std::string file;
  bool isLinear{true};
};

// Whether a texture carries colour rather than data, and so must be de-gamma'd
// on load. The document says so per input; MaterialX names the encoding, not
// the file format, so anything that is not an sRGB encoding is data.
bool inputIsLinear(const MaterialX::InputPtr &input)
{
  const auto colorSpace = input->getActiveColorSpace();
  return colorSpace != "srgb_texture" && colorSpace != "srgb_tx"
      && colorSpace != "sRGB";
}

// Rewrite the document's texture filenames to absolute paths, and collect what
// was found so the caller can bind samplers to it.
//
// OpenUSD's conversion writes SdfAssetPath::GetAssetPath() -- the path exactly
// as authored -- and leaves resolution to whoever consumes the document, which
// is why it also hands back the texture nodes it wrote. That contract does not
// survive TSD's handoff: the document travels to the device as inline text,
// with no file of its own for a relative path to be relative to. So the paths
// have to be absolute before they leave here.
//
// The anchor is the same one the rest of this importer uses for textures: the
// resolved path when the Stage's resolver produced one, and the Stage's own
// directory otherwise. The fallback is what carries UDIM sets, whose paths
// name no file that a resolver could have resolved.
std::vector<DocumentTexture> resolveTexturePaths(ImportContext &ctx,
    const pxr::SdfPath &materialPath,
    const std::string &primType,
    const MaterialX::DocumentPtr &document,
    const pxr::HdMtlxTexturePrimvarData &textures,
    pxr::HdDataSourceMaterialNetworkInterface &networkInterface)
{
  std::vector<DocumentTexture> retval;
  for (const auto &nodePath : textures.hdTextureNodes) {
    const auto nodeName = pxr::HdMtlxCreateNameFromPath(nodePath);
    auto inputNames = textures.mxHdTextureMap.find(nodeName);
    if (inputNames == textures.mxHdTextureMap.end())
      continue;

    auto node = documentNode(document, nodeName);
    if (!node)
      continue;

    for (const auto &inputName : inputNames->second) {
      auto input = node->getInput(inputName);
      if (!input)
        continue;

      const auto value = networkInterface.GetNodeParameterValue(
          pxr::TfToken(nodePath.GetString()), pxr::TfToken(inputName));
      if (!value.IsHolding<pxr::SdfAssetPath>())
        continue;

      const auto file =
          anchoredAssetPath(ctx, value.UncheckedGet<pxr::SdfAssetPath>());
      if (file.empty())
        continue;

      input->setValueString(file);

      // A path holding a MaterialX token names a set of tiles rather than a
      // file, so there is nothing to look for; anything else that is missing
      // now is worth saying, because the device that opens it later cannot
      // say which Stage prim asked for it. Reporting the tile set rather than
      // approximating it is a deliberate stance -- an ANARI sampler is a
      // single image and neither TSD nor the device has anywhere to send the
      // remaining tiles' texels. See ADR 0019.
      if (file.find('<') != std::string::npos) {
        ctx.reportSkip(materialPath,
            primType,
            UsdSkipReason::TEXTURE_LOAD_FAILED,
            file + " (tiled texture sets are not supported)");
        continue;
      }
      if (!std::filesystem::exists(file)) {
        ctx.reportSkip(
            materialPath, primType, UsdSkipReason::TEXTURE_LOAD_FAILED, file);
        continue;
      }

      retval.push_back({input->getNamePath(), file, inputIsLinear(input)});
    }
  }

  return retval;
}

// Write the generated document to `TSD_USD_MATERIALX_DUMP_DIR` when that is
// set, named after the material prim.
//
// The documents TSD emits are inline text handed straight to a device, so when
// one of them fails the device's shader generation there is otherwise nothing
// to look at: the error names a node inside a document nobody kept. Dumping
// here rather than device-side is deliberate -- the XML exists in full at this
// point, and TSD is where the node names being complained about are minted.
void dumpDocument(const pxr::SdfPath &materialPath, const std::string &xml)
{
  const char *dir = std::getenv("TSD_USD_MATERIALX_DUMP_DIR");
  if (dir == nullptr || *dir == '\0')
    return;

  std::error_code ec;
  std::filesystem::create_directories(dir, ec);

  auto name = materialPath.GetString();
  for (auto &c : name) {
    if (c == '/' || c == ':')
      c = '_';
  }
  const auto file = (std::filesystem::path(dir) / (name + ".mtlx")).string();

  std::ofstream out(file, std::ios::binary | std::ios::trunc);
  if (!out) {
    core::logWarning(
        "[import_USD] could not open '%s' to dump MaterialX document",
        file.c_str());
    return;
  }
  out << xml;
  core::logStatus("[import_USD] %s: MaterialX document dumped to %s",
      materialPath.GetText(),
      file.c_str());
}

// Whether every node in the document resolves to a MaterialX node definition,
// which is what a device's shader generator needs to compile it.
//
// MaterialX matches a node to its definition on category, type and the exact
// set of inputs, so a network that connects an input to an upstream output of
// a different type resolves to nothing. The document still writes out, and the
// failure surfaces only once it reaches the device, as `Could not find a
// nodedef for node '<name>'` -- after which shader generation stops and the
// prim silently renders with the default material. Checking here turns that
// into a reported skip and a portable-mapping fallback.
//
// The check runs on a copy, because resolution needs the standard libraries
// present in the document and importing them into the document TSD emits would
// inline the whole of MaterialX into the XML that travels to the device.
bool documentResolves(const MaterialX::DocumentPtr &document, std::string &why)
{
  auto probe = MaterialX::createDocument();
  probe->copyContentFrom(document);
  probe->importLibrary(pxr::HdMtlxStdLibraries());

  std::string unresolved;
  for (const auto &element : probe->traverseTree()) {
    auto node = element->asA<MaterialX::Node>();
    if (node && !node->getNodeDef()) {
      unresolved += (unresolved.empty() ? "" : ", ") + node->getName() + " <"
          + node->getCategory() + ">";
    }
  }
  if (unresolved.empty())
    return true;

  // MaterialX's own validation says which port is at fault, where the failed
  // lookup only says which node it gave up on. Report both.
  std::string message;
  probe->validate(&message);
  if (const auto end = message.find('\n'); end != std::string::npos)
    message.resize(end);

  why = "no MaterialX node definition for " + unresolved;
  if (!message.empty())
    why += " -- " + message;
  return false;
}

// Native MaterialX passthrough. The document is generated from the resolved
// network by OpenUSD's own conversion, so a MaterialX network passes through
// intact and a preview-surface network converts through its MaterialX node
// definitions. Returns a null ref when no network converts, so the caller can
// fall back to a portable mapping.
MaterialRef tryMaterialXPassthrough(ImportContext &ctx,
    const pxr::SdfPath &materialPath,
    const pxr::HdSceneIndexPrim &prim,
    const pxr::HdMaterialSchema &materialSchema)
{
  if (!materialSchema)
    return {};

  // Prefer an authored MaterialX network, but a preview-surface network also
  // converts through its own MaterialX node definitions.
  auto network = materialSchema.GetMaterialNetwork(MATERIALX_CONTEXT);
  if (!network || !network.GetNodes())
    network = selectNetwork(materialSchema, ctx.options->renderContexts);
  if (!network || !network.GetNodes())
    return {};

  auto surfaceTerminal =
      network.GetTerminals().Get(pxr::HdMaterialTerminalTokens->surface);
  if (!surfaceTerminal)
    return {};
  auto terminalPathSource = surfaceTerminal.GetUpstreamNodePath();
  if (!terminalPathSource)
    return {};
  const auto terminalNode = terminalPathSource->GetTypedValue(0);

  pxr::HdDataSourceMaterialNetworkInterface networkInterface(
      materialPath, network.GetContainer(), prim.dataSource);

  // Only a terminal MaterialX itself defines can be converted. A
  // UsdPreviewSurface terminal has no MaterialX node definition, and asking
  // for one anyway yields a document that fails MaterialX's own validation.
  if (!pxr::HdMtlxGetNodeDef(networkInterface.GetNodeType(terminalNode),
          pxr::HdMtlxStdLibraries()))
    return {};

  pxr::HdMtlxTexturePrimvarData textures;
  auto document = pxr::HdMtlxCreateMtlxDocumentFromHdMaterialNetworkInterface(
      &networkInterface,
      terminalNode,
      networkInterface.GetNodeInputConnectionNames(terminalNode),
      pxr::HdMtlxStdLibraries(),
      &textures);
  if (!document)
    return {};

  const auto documentTextures = resolveTexturePaths(ctx,
      materialPath,
      prim.primType.GetString(),
      document,
      textures,
      networkInterface);

  // The document names its own surface material node; TSD selects by that
  // name rather than assuming one derived from the prim path.
  const auto materialNodes = document->getMaterialNodes();
  std::string materialName;
  if (!materialNodes.empty())
    materialName = materialNodes.front()->getName();

  const auto xml = MaterialX::writeToXmlString(document);
  if (!xml.empty())
    dumpDocument(materialPath, xml);

  // Checked after the texture pass rather than before it, so that a document
  // being discarded does not take its tile-set and missing-texture reports
  // down with it -- the fallback mapping reads the network by
  // UsdPreviewSurface names and would report none of them. Checked before any
  // sampler is created, so nothing is bound to a document that is thrown away.
  if (std::string why; !documentResolves(document, why)) {
    ctx.reportSkip(materialPath,
        prim.primType.GetString(),
        UsdSkipReason::MATERIAL_RESOLUTION_FAILED,
        why);
    return {};
  }

  // One network converts to one material node in the normal case. More than
  // one means the name picked above is a guess, so say which names were on
  // offer rather than let a silently wrong pick reach the device.
  if (materialNodes.size() > 1) {
    std::string names;
    for (const auto &node : materialNodes)
      names += (names.empty() ? "" : ", ") + node->getName();
    core::logWarning(
        "[import_USD] %s: MaterialX document has %zu material nodes (%s); "
        "using '%s'",
        materialPath.GetText(),
        materialNodes.size(),
        names.c_str(),
        materialName.c_str());
  }

  if (materialName.empty() || xml.empty())
    return {};

  auto retval = ctx.scene->createObject<Material>(tokens::material::materialx);
  retval->setName(materialPath.GetString().c_str());
  retval->setParameter("sourceType", "documentInline");
  retval->setParameter("source", xml.c_str());
  retval->setParameter("materialName", materialName.c_str());

  // A device reads the document's texels from samplers bound to the `filename`
  // inputs by their document path, not by opening the files itself -- the
  // document is inline text and names no search root a renderer could resolve
  // against. TSD loads them here for the same reason it does for a preview
  // surface, and through the same cache, so a texture shared between materials
  // is read once.
  for (const auto &texture : documentTextures) {
    auto sampler =
        importTexture(ctx.textureCache, texture.file, texture.isLinear);
    if (!sampler) {
      ctx.reportSkip(materialPath,
          prim.primType.GetString(),
          UsdSkipReason::TEXTURE_LOAD_FAILED,
          texture.file);
      continue;
    }
    retval->setParameterObject(Token(texture.inputPath.c_str()), *sampler);
  }

  return retval;
}

#endif

// Material values are imported at one time, so say when the Stage animates
// them rather than leaving the difference to be noticed.
void reportAnimatedShaderInputs(ImportContext &ctx,
    const pxr::SdfPath &materialPath,
    const std::string &primType)
{
  if (auto usdPrim = ctx.stage->GetPrimAtPath(materialPath)) {
    for (const auto &descendant : usdPrim.GetDescendants()) {
      pxr::UsdShadeShader shader(descendant);
      if (!shader)
        continue;
      // Re-authoring an unchanged input at every frame is not a loss, so the
      // samples are compared rather than merely counted.
      bool animated = false;
      for (const auto &input : shader.GetInputs())
        animated = animated || attributeValueVaries(input.GetAttr());
      if (animated) {
        ctx.reportSkip(materialPath,
            primType,
            UsdSkipReason::TIME_VARYING_VALUE_DROPPED,
            "shader inputs are time-sampled; imported at one time");
        break;
      }
    }
  }
}

// The portable mapping: read the network's surface terminal as a
// UsdPreviewSurface and emit the physicallyBased material it describes. This
// is where every material not passed through natively ends up, so a Stage
// authored for another renderer still arrives with something bound.
ResolvedMaterial convertPreviewSurface(ImportContext &ctx,
    const pxr::HdMaterialSchema &materialSchema,
    const pxr::SdfPath &materialPath,
    const pxr::HdSceneIndexPrim &prim)
{
  if (!materialSchema) {
    ctx.reportSkip(materialPath,
        prim.primType.GetString(),
        UsdSkipReason::MATERIAL_RESOLUTION_FAILED,
        "no material network on the resolved prim");
    return {};
  }

  NetworkWalker walker{
      selectNetwork(materialSchema, ctx.options->renderContexts)};
  if (!walker.network) {
    ctx.reportSkip(materialPath,
        prim.primType.GetString(),
        UsdSkipReason::MATERIAL_RESOLUTION_FAILED,
        "no usable network for the requested Render Contexts");
    return {};
  }

  auto surfaceTerminal =
      walker.network.GetTerminals().Get(pxr::HdMaterialTerminalTokens->surface);
  if (!surfaceTerminal) {
    ctx.reportSkip(materialPath,
        prim.primType.GetString(),
        UsdSkipReason::MATERIAL_RESOLUTION_FAILED,
        "network has no surface terminal");
    return {};
  }

  auto terminalPathSource = surfaceTerminal.GetUpstreamNodePath();
  const auto surfacePath = terminalPathSource
      ? terminalPathSource->GetTypedValue(0)
      : pxr::TfToken();
  if (walker.nodeId(surfacePath) != PREVIEW_SURFACE_ID) {
    // Something richer than a preview surface is authored here. Emit what a
    // portable mapping can express and say so, rather than dropping it.
    ctx.reportSkip(materialPath,
        prim.primType.GetString(),
        UsdSkipReason::RICHER_MATERIAL_AVAILABLE,
        "surface terminal is '" + walker.nodeId(surfacePath).GetString()
            + "'; reading it as a preview surface");
  }

  auto material =
      ctx.scene->createObject<Material>(tokens::material::physicallyBased);
  material->setName(materialPath.GetString().c_str());

  ResolvedMaterial retval;
  retval.material = material;

  // Scalar inputs //

  auto setFloatIfPresent = [&](const char *usdName, const char *tsdName) {
    const auto value = walker.parameter(surfacePath, usdName);
    if (value.IsHolding<float>())
      material->setParameter(Token(tsdName), value.UncheckedGet<float>());
  };

  // Textured or constant inputs //

  auto bindTexture =
      [&](const char *usdName, const char *tsdName, bool colorRole) -> bool {
    const auto texturePath = walker.connectedNode(surfacePath, usdName);
    if (walker.nodeId(texturePath) != UV_TEXTURE_ID)
      return false;

    // The reader node names the primvar whether or not the image loads, so
    // record it before anything can fail.
    if (retval.uvPrimvarName.empty())
      retval.uvPrimvarName = uvPrimvarOfTexture(walker, texturePath);

    const auto fileValue = walker.parameter(texturePath, "file");
    if (!fileValue.IsHolding<pxr::SdfAssetPath>())
      return false;

    const auto file =
        anchoredAssetPath(ctx, fileValue.UncheckedGet<pxr::SdfAssetPath>());
    if (file.empty())
      return false;

    // Everything the binding varies goes in before the sampler is built:
    // makeImageSampler owns inTransform/inOffset, because an image that could
    // not be reordered needs a v-flip composed into them that a later
    // setParameter here would drop.
    const auto wrapS = wrapModeOf(walker, texturePath, "wrapS");
    const auto wrapT = wrapModeOf(walker, texturePath, "wrapT");
    SamplerSettings settings;
    settings.wrapMode1 = wrapS.c_str();
    settings.wrapMode2 = wrapT.c_str();
    settings.uvTransform = uvTransformOfTexture(walker, texturePath);

    const bool isLinear = textureIsLinear(walker, texturePath, colorRole);
    auto sampler = importTexture(ctx.textureCache, file, isLinear, settings);
    if (!sampler) {
      ctx.reportSkip(materialPath,
          prim.primType.GetString(),
          UsdSkipReason::TEXTURE_LOAD_FAILED,
          file);
      return false;
    }

    material->setParameterObject(Token(tsdName), *sampler);
    return true;
  };

  if (!bindTexture("diffuseColor", "baseColor", true)) {
    material->setParameter("baseColor",
        asFloat3(walker.parameter(surfacePath, "diffuseColor"),
            float3(0.18f, 0.18f, 0.18f)));
  }
  if (!bindTexture("emissiveColor", "emissive", true)) {
    const auto emissive = walker.parameter(surfacePath, "emissiveColor");
    if (!emissive.IsEmpty())
      material->setParameter("emissive", asFloat3(emissive, float3(0.f)));
  }
  // A normal map is optional; nothing else stands in for it.
  bindTexture("normal", "normal", false);
  if (!bindTexture("metallic", "metallic", false))
    setFloatIfPresent("metallic", "metallic");
  if (!bindTexture("roughness", "roughness", false))
    setFloatIfPresent("roughness", "roughness");
  if (!bindTexture("opacity", "opacity", false))
    setFloatIfPresent("opacity", "opacity");
  setFloatIfPresent("clearcoat", "clearcoat");
  setFloatIfPresent("clearcoatRoughness", "clearcoatRoughness");
  setFloatIfPresent("ior", "ior");

  return retval;
}

} // namespace

ResolvedMaterial resolveMaterial(ImportContext &ctx,
    const pxr::HdSceneIndexBaseRefPtr &sceneIndex,
    const pxr::SdfPath &materialPath)
{
  if (materialPath.IsEmpty())
    return {};

  const auto key = materialPath.GetString();
  if (auto found = ctx.materialCache.find(key);
      found != ctx.materialCache.end())
    return found->second;

  auto prim = sceneIndex->GetPrim(materialPath);
  const auto primType = prim.primType.GetString();
  auto materialSchema = pxr::HdMaterialSchema::GetFromParent(prim.dataSource);

  // Every exit path caches, failures included: a material that cannot be
  // resolved is resolved -- and reported -- once, not once per binding.
  auto cache = [&](ResolvedMaterial resolved) {
    ctx.materialCache[key] = resolved;
    return resolved;
  };

  // Native passthrough modes are opt-in; each falls back to the portable
  // mapping, saying so, rather than dropping the material.
  switch (ctx.options->materialMode) {
  case UsdMaterialMode::MDL:
    if (auto material = tryMdlPassthrough(ctx, materialPath))
      return cache({material});
    ctx.reportSkip(materialPath,
        primType,
        UsdSkipReason::RICHER_MATERIAL_AVAILABLE,
        "no MDL network authored; reading a portable mapping instead");
    break;
  case UsdMaterialMode::MATERIALX:
#if TSD_USD_HAS_MATERIALX
    if (auto material =
            tryMaterialXPassthrough(ctx, materialPath, prim, materialSchema))
      return cache({material});
    ctx.reportSkip(materialPath,
        primType,
        UsdSkipReason::RICHER_MATERIAL_AVAILABLE,
        "no network could be converted to a MaterialX document; reading a"
        " portable mapping instead");
#else
    // MaterialX passthrough needs OpenUSD's HdMtlx document conversion, which
    // this build of OpenUSD does not ship.
    ctx.reportSkip(materialPath,
        primType,
        UsdSkipReason::RICHER_MATERIAL_AVAILABLE,
        "MaterialX passthrough is unavailable in this OpenUSD build; "
        "reading a portable mapping instead");
#endif
    break;
  case UsdMaterialMode::PHYSICALLY_BASED:
    break;
  }

  reportAnimatedShaderInputs(ctx, materialPath, primType);

  // OmniPBR is part of the portable mapping rather than a passthrough mode: it
  // maps onto the same physicallyBased material the preview-surface reader
  // emits, and it has to be tried first because that reader would find none of
  // OmniPBR's input names and emit its defaults instead.
  if (auto material = tryOmniPbrMapping(ctx, materialPath, prim.primType))
    return cache({material});

  return cache(convertPreviewSurface(ctx, materialSchema, materialPath, prim));
}

} // namespace tsd::io::usd
