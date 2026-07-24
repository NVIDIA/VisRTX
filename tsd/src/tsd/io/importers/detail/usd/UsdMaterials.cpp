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
// std
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

// One resolved UsdPreviewSurface network, walked lazily out of the Hydra
// material network container.
struct NetworkWalker
{
  pxr::HdMaterialNetworkSchema network;

  pxr::HdMaterialNodeSchema node(const pxr::TfToken &path) const;
  pxr::TfToken nodeId(const pxr::TfToken &path) const;
  pxr::VtValue parameter(
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
    if (id == PRIMVAR_READER_ID) {
      const auto varname = walker.parameter(current, "varname");
      if (varname.IsHolding<std::string>())
        return varname.UncheckedGet<std::string>();
      if (varname.IsHolding<pxr::TfToken>())
        return varname.UncheckedGet<pxr::TfToken>().GetString();
      return {};
    }
    if (id != TRANSFORM_2D_ID)
      return {};
    current = walker.connectedNode(current, "in");
  }
  return {};
}

math::mat4 uvTransformOfTexture(
    const NetworkWalker &walker, const pxr::TfToken &texturePath)
{
  auto transformNode = walker.connectedNode(texturePath, "st");
  if (walker.nodeId(transformNode) != TRANSFORM_2D_ID)
    return math::IDENTITY_MAT4;

  auto retval = math::IDENTITY_MAT4;
  const auto scale = walker.parameter(transformNode, "scale");
  if (scale.IsHolding<pxr::GfVec2f>()) {
    const auto s = scale.UncheckedGet<pxr::GfVec2f>();
    retval[0][0] = s[0];
    retval[1][1] = s[1];
  }
  const auto translation = walker.parameter(transformNode, "translation");
  if (translation.IsHolding<pxr::GfVec2f>()) {
    const auto t = translation.UncheckedGet<pxr::GfVec2f>();
    retval[3][0] = t[0];
    retval[3][1] = t[1];
  }
  return retval;
}

std::string wrapModeOf(const NetworkWalker &walker,
    const pxr::TfToken &texturePath,
    const char *input)
{
  const auto value = walker.parameter(texturePath, input);
  std::string mode = "repeat";
  if (value.IsHolding<pxr::TfToken>())
    mode = value.UncheckedGet<pxr::TfToken>().GetString();
  else if (value.IsHolding<std::string>())
    mode = value.UncheckedGet<std::string>();
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
  const auto value = walker.parameter(texturePath, "sourceColorSpace");
  std::string space;
  if (value.IsHolding<pxr::TfToken>())
    space = value.UncheckedGet<pxr::TfToken>().GetString();
  else if (value.IsHolding<std::string>())
    space = value.UncheckedGet<std::string>();

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

    auto material = ctx.scene.createObject<Material>(tokens::material::mdl);
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

} // namespace

ResolvedMaterial resolveMaterial(ImportContext &ctx,
    const pxr::HdSceneIndexBaseRefPtr &sceneIndex,
    const pxr::SdfPath &materialPath)
{
  if (materialPath.IsEmpty())
    return {};

  const auto key = materialPath.GetString();
  if (auto found = ctx.materialCache.find(key);
      found != ctx.materialCache.end()) {
    ResolvedMaterial retval;
    retval.material = found->second;
    retval.uvPrimvarName = ctx.uvPrimvarCache[key];
    return retval;
  }

  auto prim = sceneIndex->GetPrim(materialPath);

  // Native passthrough modes are opt-in; each falls back to the portable
  // mapping, saying so, rather than dropping the material.
  if (ctx.options.materialMode == UsdMaterialMode::MDL) {
    if (auto material = tryMdlPassthrough(ctx, materialPath)) {
      ctx.materialCache[key] = material;
      ctx.uvPrimvarCache[key] = std::string();
      ResolvedMaterial retval;
      retval.material = material;
      return retval;
    }
    ctx.reportSkip(materialPath,
        prim.primType.GetString(),
        UsdSkipReason::RICHER_MATERIAL_AVAILABLE,
        "no MDL network authored; reading a portable mapping instead");
  } else if (ctx.options.materialMode == UsdMaterialMode::MATERIALX) {
    // MaterialX passthrough needs OpenUSD's HdMtlx document conversion, which
    // this build of OpenUSD does not ship.
    ctx.reportSkip(materialPath,
        prim.primType.GetString(),
        UsdSkipReason::RICHER_MATERIAL_AVAILABLE,
        "MaterialX passthrough is unavailable in this OpenUSD build; "
        "reading a portable mapping instead");
  }

  auto materialSchema = pxr::HdMaterialSchema::GetFromParent(prim.dataSource);
  if (!materialSchema) {
    ctx.reportSkip(materialPath,
        prim.primType.GetString(),
        UsdSkipReason::MATERIAL_RESOLUTION_FAILED,
        "no material network on the resolved prim");
    return {};
  }

  NetworkWalker walker{
      selectNetwork(materialSchema, ctx.options.renderContexts)};
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
      ctx.scene.createObject<Material>(tokens::material::physicallyBased);
  material->setName(key.c_str());

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

    const auto assetPath = fileValue.UncheckedGet<pxr::SdfAssetPath>();
    auto file = assetPath.GetResolvedPath();
    if (file.empty())
      file = assetPath.GetAssetPath();
    if (file.empty())
      return false;
    if (!isAbsolute(file))
      file = ctx.basePath + file;

    const bool isLinear = textureIsLinear(walker, texturePath, colorRole);
    auto sampler = importTexture(ctx.scene, file, ctx.textureCache, isLinear);
    if (!sampler) {
      ctx.reportSkip(materialPath,
          prim.primType.GetString(),
          UsdSkipReason::TEXTURE_LOAD_FAILED,
          file);
      return false;
    }

    sampler->setParameter(
        "wrapMode1", wrapModeOf(walker, texturePath, "wrapS").c_str());
    sampler->setParameter(
        "wrapMode2", wrapModeOf(walker, texturePath, "wrapT").c_str());
    sampler->setParameter("inAttribute", "attribute0");
    const auto uvTransform = uvTransformOfTexture(walker, texturePath);
    if (uvTransform != math::IDENTITY_MAT4)
      sampler->setParameter("inTransform", uvTransform);

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
  if (!bindTexture("normal", "normal", false))
    ; // no normal map authored
  if (!bindTexture("metallic", "metallic", false))
    setFloatIfPresent("metallic", "metallic");
  if (!bindTexture("roughness", "roughness", false))
    setFloatIfPresent("roughness", "roughness");
  if (!bindTexture("opacity", "opacity", false))
    setFloatIfPresent("opacity", "opacity");
  setFloatIfPresent("clearcoat", "clearcoat");
  setFloatIfPresent("clearcoatRoughness", "clearcoatRoughness");
  setFloatIfPresent("ior", "ior");

  ctx.materialCache[key] = material;
  ctx.uvPrimvarCache[key] = retval.uvPrimvarName;
  return retval;
}

} // namespace tsd::io::usd
