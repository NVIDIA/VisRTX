// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/Logging.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"

#include <anari/anari_cpp/ext/linalg.h>
#include <anari/frontend/anari_enums.h>

#include <tiny_gltf.h>

#include <cmath>
#include <cstdio>
#include <glm/fwd.hpp>
#include <glm/gtc/quaternion.hpp>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace std::string_literals;

namespace tsd::io {

using namespace tsd::core;
using namespace tsd::math;

template <typename T>
static T GetValueOrDefault(
    const tinygltf::Value &value, const T &defaultValue, const char *name)
{
  if (value.IsObject() && value.Has(name)) {
    auto subValue = value.Get(name);
    if constexpr (std::is_floating_point_v<T>) {
      if (subValue.IsNumber())
        return subValue.GetNumberAsDouble();
    } else if constexpr (std::is_integral_v<T>) {
      if (subValue.IsNumber())
        return subValue.GetNumberAsInt();
    } else if constexpr (std::is_same_v<T, float3>) {
      if (subValue.IsArray() && subValue.ArrayLen() >= 3)
        return float3(subValue.Get(0).GetNumberAsDouble(),
            subValue.Get(1).GetNumberAsDouble(),
            subValue.Get(2).GetNumberAsDouble());
    } else if constexpr (std::is_same_v<T, float4>) {
      if (subValue.IsArray() && subValue.ArrayLen() >= 4)
        return float4(subValue.Get(0).GetNumberAsDouble(),
            subValue.Get(1).GetNumberAsDouble(),
            subValue.Get(2).GetNumberAsDouble(),
            subValue.Get(3).GetNumberAsDouble());
    } else {
      // Note: If the type below fails to build for some types, specific
      // constexpr entries need to be added to handle those.
      static auto REF_VALUE = tinygltf::Value(T{});
      if (subValue.Type() == REF_VALUE.Type())
        return subValue.template Get<T>();
    }
  }

  return defaultValue;
}

template <typename T, typename... NAMES>
static T GetValueOrDefault(const tinygltf::Value &value,
    const T &defaultValue,
    const char *name,
    const NAMES &...names)
{
  static_assert(
      (std::is_same_v<std::remove_cv_t<std::decay_t<NAMES>>, char *> && ...),
      "All names must be C strings");
  if (value.IsObject() && value.Has(name)) {
    auto subValue = value.Get(name);
    return GetValueOrDefault<T>(
        subValue, defaultValue, std::forward<const NAMES &>(names)...);
  }

  return defaultValue;
}

static SamplerRef importGLTFTexture(Context &ctx,
    const tinygltf::Model &model,
    int textureIndex,
    TextureCache &cache,
    bool isLinear = false,
    bool flipNormalMapY = false,
    const char *samplerName = nullptr)
{
  if (textureIndex < 0 || textureIndex >= model.textures.size())
    return {};

  const auto &texture = model.textures[textureIndex];
  if (texture.source < 0 || texture.source >= model.images.size())
    return {};

  const auto &image = model.images[texture.source];

  std::string cacheKey = image.name.empty()
      ? "texture_"s + std::to_string(texture.source)
      : image.name;

  // Include linear/sRGB info in cache key to avoid conflicts
  if (isLinear) {
    cacheKey += "_linear";
  } else {
    cacheKey += "_srgb";
  }

  // Include normal map Y flip info in cache key
  if (flipNormalMapY) {
    cacheKey += "_yflip";
  }

  auto dataArray = cache[cacheKey];

  if (!dataArray.valid()) {
    if (image.image.empty()) {
      logWarning("[import_GLTF] empty image data for texture %d", textureIndex);
      return {};
    }

    switch (image.pixel_type) {
    case TINYGLTF_COMPONENT_TYPE_BYTE: {
      if (!isLinear)
        logWarning("[import_GLTF] signed byte textures not supported in sRGB");
      dataArray = ctx.createArray(
          ANARI_FIXED8 + (image.component - 1), image.width, image.height);
      break;
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
      if (isLinear)
        dataArray = ctx.createArray(
            ANARI_UFIXED8 + (image.component - 1), image.width, image.height);
      else
        dataArray =
            ctx.createArray(ANARI_UFIXED8_R_SRGB + (image.component - 1),
                image.width,
                image.height);
      break;
    }
    case TINYGLTF_COMPONENT_TYPE_SHORT: {
      if (!isLinear)
        logWarning("[import_GLTF] signed short textures not supported in sRGB");
      dataArray = ctx.createArray(
          ANARI_FIXED16 + (image.component - 1), image.width, image.height);
      break;
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
      if (!isLinear)
        logWarning(
            "[import_GLTF] unsigned short textures not supported in sRGB");
      dataArray = ctx.createArray(
          ANARI_UFIXED16 + (image.component - 1), image.width, image.height);
      break;
    }
    case TINYGLTF_COMPONENT_TYPE_INT: {
      if (!isLinear)
        logWarning("[import_GLTF] signed int textures not supported in sRGB");
      dataArray = ctx.createArray(
          ANARI_FIXED32 + (image.component - 1), image.width, image.height);
      break;
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
      if (!isLinear)
        logWarning("[import_GLTF] unsigned int textures not supported in sRGB");
      dataArray = ctx.createArray(
          ANARI_UFIXED32 + (image.component - 1), image.width, image.height);
      break;
    }
    case TINYGLTF_COMPONENT_TYPE_FLOAT: {
      if (!isLinear)
        logWarning("[import_GLTF] float textures not supported in sRGB");
      dataArray = ctx.createArray(
          ANARI_FLOAT32 + (image.component - 1), image.width, image.height);
      break;
    }
    case TINYGLTF_COMPONENT_TYPE_DOUBLE: {
      if (!isLinear)
        logWarning("[import_GLTF] double textures not supported in sRGB");
      dataArray = ctx.createArray(
          ANARI_FLOAT64 + (image.component - 1), image.width, image.height);
      break;
    }
    default: {
      logWarning("[import_GLTF] unsupported image component type texture: %d",
          image.pixel_type);
      return {};
    }
    }

    auto *outData = dataArray->map();
    std::memcpy(outData, image.image.data(), image.image.size());
    dataArray->unmap();

    cache[cacheKey] = dataArray;
  }

  auto sampler = ctx.createObject<Sampler>(tokens::sampler::image2D);
  sampler->setParameterObject("image"_t, *dataArray);
  sampler->setParameter("inAttribute"_t, "attribute0");

  // Apply sampler settings if available
  if (texture.sampler >= 0 && texture.sampler < model.samplers.size()) {
    const auto &gltfSampler = model.samplers[texture.sampler];

    // Wrap mode
    const char *wrapS = "repeat";
    const char *wrapT = "repeat";

    switch (gltfSampler.wrapS) {
    case TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE:
      wrapS = "clampToEdge";
      break;
    case TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT:
      wrapS = "mirror";
      break;
    case TINYGLTF_TEXTURE_WRAP_REPEAT:
      wrapS = "repeat";
      break;
    }

    switch (gltfSampler.wrapT) {
    case TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE:
      wrapT = "clampToEdge";
      break;
    case TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT:
      wrapT = "mirror";
      break;
    case TINYGLTF_TEXTURE_WRAP_REPEAT:
      wrapT = "repeat";
      break;
    }

    sampler->setParameter("wrapMode1"_t, wrapS);
    sampler->setParameter("wrapMode2"_t, wrapT);

    // Filter mode
    const char *filter = "linear";
    if (gltfSampler.magFilter == TINYGLTF_TEXTURE_FILTER_NEAREST
        || gltfSampler.minFilter == TINYGLTF_TEXTURE_FILTER_NEAREST) {
      filter = "nearest";
    }
    sampler->setParameter("filter"_t, filter);
  } else {
    sampler->setParameter("wrapMode1"_t, "repeat");
    sampler->setParameter("wrapMode2"_t, "repeat");
    sampler->setParameter("filter"_t, "linear");
  }

  // Set sampler name to reflect the input type if provided
  if (samplerName && samplerName[0] != '\0') {
    std::string fullName = std::string(samplerName) + ":" + cacheKey;
    sampler->setName(fullName.c_str());
  } else {
    sampler->setName(cacheKey.c_str());
  }
  return sampler;
}

static std::vector<MaterialRef> importGLTFMaterials(
    Context &ctx, const tinygltf::Model &model)
{
  // This function supports the following glTF material extensions:
  // - KHR_materials_transmission: transmission factor and texture
  // - KHR_materials_ior: index of refraction
  // - KHR_materials_volume: thickness, attenuation distance and color
  // - KHR_materials_clearcoat: clearcoat factor, texture, roughness
  // - KHR_materials_specular: specular factor, texture, color
  // - KHR_materials_sheen: sheen color and roughness
  // - KHR_materials_iridescence: iridescence factor, IOR, thickness

  std::vector<MaterialRef> materials;
  TextureCache cache;

  for (const auto &gltfMaterial : model.materials) {
    MaterialRef material;

    // Create PBR material by default
    material = ctx.createObject<Material>(tokens::material::physicallyBased);

    // Base color
    const auto &pbr = gltfMaterial.pbrMetallicRoughness;
    float4 baseColorFactor(pbr.baseColorFactor[0],
        pbr.baseColorFactor[1],
        pbr.baseColorFactor[2],
        pbr.baseColorFactor[3]);

    if (auto sampler = importGLTFTexture(ctx,
            model,
            pbr.baseColorTexture.index,
            cache,
            false,
            false,
            "baseColor")) {
      // Make this an opaque color. Opacity is handled below.
      sampler->setParameter("outTransform"_t,
          mat4({baseColorFactor.x, 0, 0, 0},
              {0, baseColorFactor.y, 0, 0},
              {0, 0, baseColorFactor.z, 0},
              {0, 0, 0, 0}));
      sampler->setParameter("outOffset"_t, float4(0, 0, 0, 1));
      material->setParameterObject("baseColor"_t, *sampler);
    } else {
      material->setParameter("baseColor"_t,
          float3(baseColorFactor[0], baseColorFactor[1], baseColorFactor[2]));
    }

    if (auto sampler = importGLTFTexture(ctx,
            model,
            pbr.baseColorTexture.index,
            cache,
            true,
            false,
            "opacity")) {
      sampler->setParameter("outTransform"_t,
          mat4({0, 0, 0, 0},
              {0, 0, 0, 0},
              {0, 0, 0, 0},
              {baseColorFactor.w, 0, 0, 1}));
      material->setParameterObject("opacity"_t, *sampler);
    } else {
      material->setParameter("opacity"_t, baseColorFactor.w);
    }

    // Metallic factor
    float metallicFactor = pbr.metallicFactor;
    if (auto sampler = importGLTFTexture(ctx,
            model,
            pbr.metallicRoughnessTexture.index,
            cache,
            true,
            false,
            "metallic")) {
      // Metallic is in the blue channel for glTF
      sampler->setParameter("outTransform"_t,
          mat4({0, 0, 0, 0},
              {0, 0, 0, 0},
              {metallicFactor, 0, 0, 0},
              {0, 0, 0, 1}));
      material->setParameterObject("metallic"_t, *sampler);
    } else {
      material->setParameter("metallic"_t, metallicFactor);
    }

    // Roughness factor
    float roughnessFactor = pbr.roughnessFactor;
    if (auto sampler = importGLTFTexture(ctx,
            model,
            pbr.metallicRoughnessTexture.index,
            cache,
            true,
            false,
            "roughness")) {
      // Roughness is in the green channel for glTF
      sampler->setParameter("outTransform"_t,
          mat4({0, 0, 0, 0},
              {roughnessFactor, 0, 0, 0},
              {0, 0, 0, 0},
              {0, 0, 0, 1}));
      material->setParameterObject("roughness"_t, *sampler);
    } else {
      material->setParameter("roughness"_t, roughnessFactor);
    }

    // Normal map
    if (auto sampler = importGLTFTexture(ctx,
            model,
            gltfMaterial.normalTexture.index,
            cache,
            true,
            false,
            "normal")) {
      float normalScale = gltfMaterial.normalTexture.scale;
      sampler->setParameter("outTransform"_t,
          mat4({normalScale, 0, 0, 0},
              {0, normalScale, 0, 0},
              {0, 0, 1, 0}, // Don't scale Z (blue) channel
              {0, 0, 0, 1}));
      material->setParameterObject("normal"_t, *sampler);
    }

    // Occlusion map
    if (auto sampler = importGLTFTexture(ctx,
            model,
            gltfMaterial.occlusionTexture.index,
            cache,
            true,
            false,
            "occlusion")) {
      material->setParameterObject("occlusion"_t, *sampler);
    }

    // Emissive
    float3 emissiveFactor(gltfMaterial.emissiveFactor[0],
        gltfMaterial.emissiveFactor[1],
        gltfMaterial.emissiveFactor[2]);
    // KHR_materials_emissive_strength
    if (auto emissiveStrengthIt =
            gltfMaterial.extensions.find("KHR_materials_emissive_strength");
        emissiveStrengthIt != gltfMaterial.extensions.end()) {
      const auto &emissiveStrengthExt = emissiveStrengthIt->second;

      emissiveFactor *=
          GetValueOrDefault(emissiveStrengthExt, 1.0f, "emissiveStrength");
    }

    if (auto sampler = importGLTFTexture(ctx,
            model,
            gltfMaterial.emissiveTexture.index,
            cache,
            false,
            false,
            "emissive")) {
      sampler->setParameter("outTransform"_t,
          mat4({emissiveFactor.x, 0, 0, 0},
              {0, emissiveFactor.y, 0, 0},
              {0, 0, emissiveFactor.z, 0},
              {0, 0, 0, 1}));
      material->setParameterObject("emissive"_t, *sampler);

    } else {
      material->setParameter("emissive"_t, ANARI_FLOAT32_VEC3, &emissiveFactor);
    }

    // Alpha mode and alpha cutoff
    if (gltfMaterial.alphaMode == "OPAQUE") {
      material->setParameter("alphaMode"_t, "opaque");
    } else if (gltfMaterial.alphaMode == "MASK") {
      material->setParameter("alphaMode"_t, "mask");
    } else if (gltfMaterial.alphaMode == "BLEND") {
      material->setParameter("alphaMode"_t, "blend");
    } else {
      logWarning("[import_GLTF] unknown alpha mode: %s",
          gltfMaterial.alphaMode.c_str());
    }
    material->setParameter("alphaCutoff"_t, float(gltfMaterial.alphaCutoff));

    // KHR_materials_transmission extension
    if (auto transmissionIt =
            gltfMaterial.extensions.find("KHR_materials_transmission");
        transmissionIt != gltfMaterial.extensions.end()) {
      const auto &transmissionExt = transmissionIt->second;

      // Transmission factor
      float transmissionFactor =
          GetValueOrDefault(transmissionExt, 0.0f, "transmissionFactor");

      // Transmission texture
      auto transmissionTextureIndex = GetValueOrDefault(
          transmissionExt, -1, "transmissionTexture", "index");
      if (auto sampler = importGLTFTexture(ctx,
              model,
              transmissionTextureIndex,
              cache,
              true,
              false,
              "transmission")) {
        sampler->setParameter("outTransform"_t,
            mat4({transmissionFactor, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 1}));
        material->setParameterObject("transmission"_t, *sampler);
      } else {
        material->setParameter("transmission"_t, transmissionFactor);
      }
    } else {
      // Default values
      material->setParameter("transmission"_t, 0.0f);
    }

    // KHR_materials_ior extension
    if (auto iorIt = gltfMaterial.extensions.find("KHR_materials_ior");
        iorIt != gltfMaterial.extensions.end()) {
      const auto &iorExt = iorIt->second;
      float ior = GetValueOrDefault(iorExt, 1.5f, "ior");
      material->setParameter("ior"_t, ior);
    } else {
      // Default values
      material->setParameter("ior"_t, 1.5f);
    }

    // KHR_materials_volume extension
    if (auto volumeIt = gltfMaterial.extensions.find("KHR_materials_volume");
        volumeIt != gltfMaterial.extensions.end()) {
      const auto &volumeExt = volumeIt->second;

      // Thickness factor
      float thicknessFactor =
          GetValueOrDefault(volumeExt, 0.0f, "thicknessFactor");

      // Thickness texture
      auto thicknessTextureIndex =
          GetValueOrDefault(volumeExt, -1, "thicknessTexture", "index");
      if (auto sampler = importGLTFTexture(ctx,
              model,
              thicknessTextureIndex,
              cache,
              true,
              false,
              "thickness")) {
        sampler->setParameter("outTransform"_t,
            mat4({0, thicknessFactor, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 1}));
        material->setParameterObject("thickness"_t, *sampler);
      } else {
        material->setParameter("thickness"_t, thicknessFactor);
      }

      // Attenuation distance
      float attenuationDistance =
          GetValueOrDefault(volumeExt, 0.0f, "attenuationDistance");
      material->setParameter("attenuationDistance"_t, attenuationDistance);

      // Attenuation color
      float3 attenuationColor = GetValueOrDefault(
          volumeExt, float3(1.0f, 1.0f, 1.0f), "attenuationColor");
      material->setParameter("attenuationColor"_t, attenuationColor);
    } else {
      // Default values
      material->setParameter("thickness"_t, 0.0f);
      material->setParameter(
          "attenuationDistance"_t, std::numeric_limits<float>::max());
      material->setParameter("attenuationColor"_t, float3(1.0f, 1.0f, 1.0f));
    }

    // KHR_materials_clearcoat extension
    if (auto clearcoatIt =
            gltfMaterial.extensions.find("KHR_materials_clearcoat");
        clearcoatIt != gltfMaterial.extensions.end()) {
      const auto &clearcoatExt = clearcoatIt->second;

      // Clearcoat factor
      float clearcoatFactor =
          GetValueOrDefault(clearcoatExt, 0.0f, "clearcoatFactor");

      // Clearcoat texture
      auto clearcoatTextureIndex =
          GetValueOrDefault(clearcoatExt, -1, "clearcoatTexture", "index");
      if (auto sampler = importGLTFTexture(ctx,
              model,
              clearcoatTextureIndex,
              cache,
              true,
              false,
              "clearcoat")) {
        sampler->setParameter("outTransform"_t,
            mat4({clearcoatFactor, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 1}));
        material->setParameterObject("clearcoat"_t, *sampler);
      } else {
        material->setParameter("clearcoat"_t, clearcoatFactor);
      }

      // Clearcoat roughness factor
      float clearcoatRoughnessFactor =
          GetValueOrDefault(clearcoatExt, 0.0f, "clearcoatRoughnessFactor");

      // Clearcoat roughness texture
      auto clearcoatRoughnessTextureIndex = GetValueOrDefault(
          clearcoatExt, -1, "clearcoatRoughnessTexture", "index");
      if (auto sampler = importGLTFTexture(ctx,
              model,
              clearcoatRoughnessTextureIndex,
              cache,
              true,
              false,
              "clearcoatRoughness")) {
        sampler->setParameter("outTransform"_t,
            mat4({0, 0, 0, 0},
                {clearcoatRoughnessFactor, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 1}));
        material->setParameterObject("clearcoatRoughness"_t, *sampler);
      } else {
        material->setParameter(
            "clearcoatRoughness"_t, clearcoatRoughnessFactor);
      }

      // Clearcoat normal texture
      auto clearcoatNormalTextureIndex = GetValueOrDefault(
          clearcoatExt, -1, "clearcoatNormalTexture", "index");
      if (auto sampler = importGLTFTexture(ctx,
              model,
              clearcoatNormalTextureIndex,
              cache,
              true,
              false,
              "clearcoatNormal")) {
        material->setParameterObject("clearcoatNormal"_t, *sampler);
      }
    } else {
      // Default values
      material->setParameter("clearcoat"_t, 0.0f);
      material->setParameter("clearcoatRoughness"_t, 0.0f);
      material->removeParameter("clearcoatNormal"_t);
    }

    // KHR_materials_specular extension
    if (auto specularIt =
            gltfMaterial.extensions.find("KHR_materials_specular");
        specularIt != gltfMaterial.extensions.end()) {
      const auto &specularExt = specularIt->second;

      float specularFactor =
          GetValueOrDefault(specularExt, 1.0f, "specularFactor");

      auto specularTextureIndex =
          GetValueOrDefault(specularExt, -1, "specularTexture", "index");
      if (auto sampler = importGLTFTexture(
              ctx, model, specularTextureIndex, cache, true)) {
        sampler->setParameter("outTransform"_t,
            mat4({0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {specularFactor, 0, 0, 0}));

        material->setParameterObject("specular"_t, *sampler);
      } else {
        material->setParameter("specular"_t, specularFactor);
      }

      // Handle specular color (factor and texture)
      float3 specularColorFactor = GetValueOrDefault(
          specularExt, float3(1.0f, 1.0f, 1.0f), "specularColorFactor");

      auto specularColorTextureIndex =
          GetValueOrDefault(specularExt, -1, "specularColorTexture", "index");
      if (auto sampler = importGLTFTexture(ctx,
              model,
              specularColorTextureIndex,
              cache,
              false,
              false,
              "specularColor")) {
        sampler->setParameter("outTransform"_t,
            mat4({specularColorFactor.x, 0, 0, 0},
                {0, specularColorFactor.y, 0, 0},
                {0, 0, specularColorFactor.z, 0},
                {0, 0, 0, 1}));

        material->setParameterObject("specularColor"_t, *sampler);
      } else {
        material->setParameter("specularColor"_t, specularColorFactor);
      }
    } else {
      // Default values
      material->setParameter("specular"_t, 1.0f);
      material->setParameter("specularColor"_t, float3(1.0f, 1.0f, 1.0f));
    }

    // KHR_materials_sheen extension
    if (auto sheenIt = gltfMaterial.extensions.find("KHR_materials_sheen");
        sheenIt != gltfMaterial.extensions.end()) {
      const auto &sheenExt = sheenIt->second;

      // Sheen color factor
      float3 sheenColorFactor = GetValueOrDefault(
          sheenExt, float3(0.0f, 0.0f, 0.0f), "sheenColorFactor");

      // Sheen color texture
      auto sheenColorTextureIndex =
          GetValueOrDefault(sheenExt, -1, "sheenColorTexture", "index");
      if (auto sampler = importGLTFTexture(ctx,
              model,
              sheenColorTextureIndex,
              cache,
              false,
              false,
              "sheenColor")) {
        sampler->setParameter("outTransform"_t,
            mat4({sheenColorFactor.x, 0, 0, 0},
                {0, sheenColorFactor.y, 0, 0},
                {0, 0, sheenColorFactor.z, 0},
                {0, 0, 0, 1}));
        material->setParameterObject("sheenColor"_t, *sampler);
      } else {
        material->setParameter("sheenColor"_t, sheenColorFactor);
      }

      // Sheen roughness factor
      float sheenRoughnessFactor =
          GetValueOrDefault(sheenExt, 0.f, "sheenRoughnessFactor");

      // Sheen roughness texture
      auto sheenRoughnessTextureIndex =
          GetValueOrDefault(sheenExt, -1, "sheenRoughnessTexture", "index");
      if (auto sampler = importGLTFTexture(ctx,
              model,
              sheenRoughnessTextureIndex,
              cache,
              true,
              false,
              "sheenRoughness")) {
        sampler->setParameter("outTransform"_t,
            mat4({0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {sheenRoughnessFactor, 0, 0, 1}));
        material->setParameterObject("sheenRoughness"_t, *sampler);
      } else {
        material->setParameter("sheenRoughness"_t, sheenRoughnessFactor);
      }
    } else {
      // Default values
      material->setParameter("sheenColor"_t, float3(0.0f, 0.0f, 0.0f));
      material->setParameter("sheenRoughness"_t, 0.0f);
    }

    // KHR_materials_iridescence extension
    if (auto iridescenceIt =
            gltfMaterial.extensions.find("KHR_materials_iridescence");
        iridescenceIt != gltfMaterial.extensions.end()) {
      const auto &iridescenceExt = iridescenceIt->second;

      // Iridescence factor
      float iridescenceFactor =
          GetValueOrDefault(iridescenceExt, 0.0f, "iridescenceFactor");

      // Iridescence texture
      auto iridescenceTextureIndex =
          GetValueOrDefault(iridescenceExt, -1, "iridescenceTexture", "index");
      if (auto sampler = importGLTFTexture(ctx,
              model,
              iridescenceTextureIndex,
              cache,
              true,
              false,
              "iridescence")) {
        sampler->setParameter("outTransform"_t,
            mat4({iridescenceFactor, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 1}));
        material->setParameterObject("iridescence"_t, *sampler);
      } else {
        material->setParameter("iridescence"_t, iridescenceFactor);
      }

      // Iridescence IOR
      float iridescenceIor =
          GetValueOrDefault(iridescenceExt, 0.0f, "iridescenceIor");
      material->setParameter("iridescenceIor"_t, iridescenceIor);

      // Iridescence thickness minimum
      float iridescenceThicknessMinimum = GetValueOrDefault(
          iridescenceExt, 100.0f, "iridescenceThicknessMinimum");

      // Iridescence thickness maximum
      float iridescenceThicknessMaximum = GetValueOrDefault(
          iridescenceExt, 400.0f, "iridescenceThicknessMaximum");

      // Iridescence thickness texture
      auto iridescenceThicknessTextureIndex = GetValueOrDefault(
          iridescenceExt, -1, "iridescenceThicknessTexture", "index");
      if (auto sampler = importGLTFTexture(ctx,
              model,
              iridescenceThicknessTextureIndex,
              cache,
              true,
              false,
              "iridescenceThickness")) {
        sampler->setParameter("outTransform"_t,
            mat4({iridescenceThicknessMaximum - iridescenceThicknessMinimum,
                     0,
                     0,
                     0},
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 1}));
        sampler->setParameter("outOffset"_t,
            float4(iridescenceThicknessMinimum, 0.0f, 0.0f, 0.0f));
        material->setParameterObject("iridescenceThickness"_t, *sampler);
      }
    } else {
      // Default values
      material->setParameter("iridescence"_t, float3(0.0f, 0.0f, 0.0f));
      material->setParameter("iridescenceIor"_t, 1.3f);
      material->setParameter("iridescenceThickness"_t, 0.0f);
    }

    material->setName(gltfMaterial.name.c_str());
    materials.push_back(material);
  }

  return materials;
}

template <typename T>
static const T *getAccessorData(const tinygltf::Model &model, int accessorIndex)
{
  if (accessorIndex < 0 || accessorIndex >= model.accessors.size())
    return nullptr;

  const auto &accessor = model.accessors[accessorIndex];
  const auto &bufferView = model.bufferViews[accessor.bufferView];
  const auto &buffer = model.buffers[bufferView.buffer];

  return reinterpret_cast<const T *>(
      buffer.data.data() + bufferView.byteOffset + accessor.byteOffset);
}

template <typename T>
static void copyStridedData(
    const tinygltf::Model &model, int accessorIndex, T *outData)
{
  if (accessorIndex < 0 || accessorIndex >= model.accessors.size())
    return;

  const auto &accessor = model.accessors[accessorIndex];
  const auto &bufferView = model.bufferViews[accessor.bufferView];
  const auto &buffer = model.buffers[bufferView.buffer];

  const uint8_t *sourceData =
      buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;

  // Check if data is interleaved (has a stride)
  if (bufferView.byteStride > 0) {
    // Calculate the size of one element based on accessor type and
    // component type
    size_t elementSize = tinygltf::GetNumComponentsInType(accessor.type);
    size_t componentSize =
        tinygltf::GetComponentSizeInBytes(accessor.componentType);

    size_t bytesPerElement = elementSize * componentSize;

    // Copy data with stride
    for (size_t i = 0; i < accessor.count; ++i) {
      std::memcpy(reinterpret_cast<uint8_t *>(outData) + i * bytesPerElement,
          sourceData + i * bufferView.byteStride,
          bytesPerElement);
    }
  } else {
    // Data is tightly packed, direct copy
    size_t bytesToCopy = accessor.count * sizeof(T);
    std::memcpy(outData, sourceData, bytesToCopy);
  }
}

static std::vector<SurfaceRef> importGLTFMeshes(Context &ctx,
    const tinygltf::Model &model,
    const std::vector<MaterialRef> &materials)
{
  std::vector<SurfaceRef> surfaces;

  for (const auto &mesh : model.meshes) {
    for (const auto &primitive : mesh.primitives) {
      if (primitive.mode != TINYGLTF_MODE_TRIANGLES) {
        logWarning("[import_GLTF] only triangle primitives are supported");
        continue;
      }

      auto geometry = ctx.createObject<Geometry>(tokens::geometry::triangle);

      // Position data
      auto posIt = primitive.attributes.find("POSITION");
      if (posIt == primitive.attributes.end()) {
        logWarning("[import_GLTF] primitive missing POSITION attribute");
        continue;
      }

      const auto &posAccessor = model.accessors[posIt->second];
      if (posAccessor.type != TINYGLTF_TYPE_VEC3
          || posAccessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {
        logWarning("[import_GLTF] unsupported position data format");
        continue;
      }

      auto vertexPositionArray =
          ctx.createArray(ANARI_FLOAT32_VEC3, posAccessor.count);
      auto *posDataOut = vertexPositionArray->mapAs<float3>();
      copyStridedData(model, posIt->second, posDataOut);
      vertexPositionArray->unmap();
      geometry->setParameterObject("vertex.position"_t, *vertexPositionArray);

      // Normal data
      auto normalIt = primitive.attributes.find("NORMAL");
      if (normalIt != primitive.attributes.end()) {
        const auto &normalAccessor = model.accessors[normalIt->second];
        if (normalAccessor.type == TINYGLTF_TYPE_VEC3
            && normalAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
          auto vertexNormalArray =
              ctx.createArray(ANARI_FLOAT32_VEC3, normalAccessor.count);
          auto *normalDataOut = vertexNormalArray->mapAs<float3>();
          copyStridedData(model, normalIt->second, normalDataOut);
          vertexNormalArray->unmap();
          geometry->setParameterObject("vertex.normal"_t, *vertexNormalArray);
        }
      }

      // Texture coordinate data
      auto texCoordIt = primitive.attributes.find("TEXCOORD_0");
      if (texCoordIt != primitive.attributes.end()) {
        const auto &texCoordAccessor = model.accessors[texCoordIt->second];
        if (texCoordAccessor.type == TINYGLTF_TYPE_VEC2
            && texCoordAccessor.componentType
                == TINYGLTF_COMPONENT_TYPE_FLOAT) {
          auto vertexTexCoordArray =
              ctx.createArray(ANARI_FLOAT32_VEC2, texCoordAccessor.count);
          auto *texCoordDataOut = vertexTexCoordArray->mapAs<float2>();
          copyStridedData(model, texCoordIt->second, texCoordDataOut);
          vertexTexCoordArray->unmap();
          geometry->setParameterObject(
              "vertex.attribute0"_t, *vertexTexCoordArray);
        }
      }

      // Tangent data
      auto tangentIt = primitive.attributes.find("TANGENT");
      if (tangentIt != primitive.attributes.end()) {
        const auto &tangentAccessor = model.accessors[tangentIt->second];
        if (tangentAccessor.type == TINYGLTF_TYPE_VEC4
            && tangentAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
          auto vertexTangentArray =
              ctx.createArray(ANARI_FLOAT32_VEC4, tangentAccessor.count);
          auto *tangentDataOut = vertexTangentArray->mapAs<float4>();
          copyStridedData(model, tangentIt->second, tangentDataOut);
          vertexTangentArray->unmap();
          geometry->setParameterObject("vertex.tangent"_t, *vertexTangentArray);
        }
      }

      // Color data
      auto colorIt = primitive.attributes.find("COLOR_0");
      if (colorIt != primitive.attributes.end()) {
        const auto &colorAccessor = model.accessors[colorIt->second];
        if (colorAccessor.type == TINYGLTF_TYPE_VEC3
            || colorAccessor.type == TINYGLTF_TYPE_VEC4) {
          if (colorAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
            auto vertexColorArray = ctx.createArray(
                colorAccessor.type == TINYGLTF_TYPE_VEC4 ? ANARI_FLOAT32_VEC4
                                                         : ANARI_FLOAT32_VEC3,
                colorAccessor.count);
            if (colorAccessor.type == TINYGLTF_TYPE_VEC4) {
              auto *colorDataOut = vertexColorArray->mapAs<float4>();
              copyStridedData(model, colorIt->second, colorDataOut);
            } else {
              auto *colorDataOut = vertexColorArray->mapAs<float3>();
              copyStridedData(model, colorIt->second, colorDataOut);
            }
            vertexColorArray->unmap();
            geometry->setParameterObject("vertex.color"_t, *vertexColorArray);
          }
        }
      }

      // Index data
      if (primitive.indices >= 0) {
        const auto &indexAccessor = model.accessors[primitive.indices];

        if (indexAccessor.componentType
            == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
          auto indexArray =
              ctx.createArray(ANARI_UINT32_VEC3, indexAccessor.count / 3);
          auto *outIndices = indexArray->mapAs<uint3>();

          // Check if we need to handle strided data
          const auto &indexBufferView =
              model.bufferViews[indexAccessor.bufferView];
          if (indexBufferView.byteStride > 0
              && indexBufferView.byteStride != sizeof(uint16_t)) {
            // Handle strided indices
            auto tempIndices = std::vector<uint16_t>(indexAccessor.count);
            copyStridedData(model, primitive.indices, tempIndices.data());

            for (size_t i = 0; i < indexAccessor.count / 3; ++i) {
              outIndices[i] = uint3(tempIndices[i * 3],
                  tempIndices[i * 3 + 1],
                  tempIndices[i * 3 + 2]);
            }
          } else {
            // Direct access for tightly packed data
            const uint16_t *inIndices =
                getAccessorData<uint16_t>(model, primitive.indices);
            for (size_t i = 0; i < indexAccessor.count / 3; ++i) {
              outIndices[i] = uint3(
                  inIndices[i * 3], inIndices[i * 3 + 1], inIndices[i * 3 + 2]);
            }
          }

          indexArray->unmap();
          geometry->setParameterObject("primitive.index"_t, *indexArray);
        } else if (indexAccessor.componentType
            == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
          auto indexArray =
              ctx.createArray(ANARI_UINT32_VEC3, indexAccessor.count / 3);

          // Check if we need to handle strided data
          const auto &indexBufferView =
              model.bufferViews[indexAccessor.bufferView];
          if (indexBufferView.byteStride > 0
              && indexBufferView.byteStride != sizeof(uint32_t)) {
            // Handle strided indices
            auto tempIndices = std::vector<uint32_t>(indexAccessor.count);
            copyStridedData(model, primitive.indices, tempIndices.data());
            auto *outIndices = indexArray->mapAs<uint3>();

            for (size_t i = 0; i < indexAccessor.count / 3; ++i) {
              outIndices[i] = uint3(tempIndices[i * 3],
                  tempIndices[i * 3 + 1],
                  tempIndices[i * 3 + 2]);
            }
            indexArray->unmap();
          } else {
            // Direct copy for tightly packed data
            const uint32_t *indexData =
                getAccessorData<uint32_t>(model, primitive.indices);
            auto *outIndices = indexArray->mapAs<uint3>();
            std::memcpy(
                outIndices, indexData, indexAccessor.count * sizeof(uint32_t));
            indexArray->unmap();
          }
          geometry->setParameterObject("primitive.index"_t, *indexArray);
        } else {
          logWarning("[import_GLTF] unsupported index data type");
          continue;
        }
      }

      std::string geometryName = mesh.name + "_primitive_"
          + std::to_string(&primitive - &mesh.primitives[0]);
      geometry->setName(geometryName.c_str());

      // Calculate tangents if they weren't provided and we have the necessary
      // data
      if (tangentIt == primitive.attributes.end()) {
        // Check if we have all the required data for tangent calculation
        auto posIt = primitive.attributes.find("POSITION");
        auto normalIt = primitive.attributes.find("NORMAL");
        auto texCoordIt = primitive.attributes.find("TEXCOORD_0");

        if (posIt != primitive.attributes.end()
            && normalIt != primitive.attributes.end()
            && texCoordIt != primitive.attributes.end()) {
          // Get the accessors
          const auto &posAccessor = model.accessors[posIt->second];
          const auto &normalAccessor = model.accessors[normalIt->second];
          const auto &texCoordAccessor = model.accessors[texCoordIt->second];

          // Verify we have the right data types
          if (posAccessor.type == TINYGLTF_TYPE_VEC3
              && posAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT
              && normalAccessor.type == TINYGLTF_TYPE_VEC3
              && normalAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT
              && texCoordAccessor.type == TINYGLTF_TYPE_VEC2
              && texCoordAccessor.componentType
                  == TINYGLTF_COMPONENT_TYPE_FLOAT) {
            // Get the data
            const float3 *positions =
                getAccessorData<float3>(model, posIt->second);
            const float3 *normals =
                getAccessorData<float3>(model, normalIt->second);
            const float2 *texCoords =
                getAccessorData<float2>(model, texCoordIt->second);

            // Get or generate indices
            std::vector<uint3> indices;
            if (primitive.indices >= 0) {
              // Indexed geometry
              const auto &indexAccessor = model.accessors[primitive.indices];
              if (indexAccessor.componentType
                  == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                const uint16_t *indexData =
                    getAccessorData<uint16_t>(model, primitive.indices);
                indices.reserve(indexAccessor.count / 3);
                for (size_t i = 0; i < indexAccessor.count / 3; ++i) {
                  indices.push_back(uint3(indexData[i * 3],
                      indexData[i * 3 + 1],
                      indexData[i * 3 + 2]));
                }
              } else if (indexAccessor.componentType
                  == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                const uint32_t *indexData =
                    getAccessorData<uint32_t>(model, primitive.indices);
                indices.reserve(indexAccessor.count / 3);
                for (size_t i = 0; i < indexAccessor.count / 3; ++i) {
                  indices.push_back(uint3(indexData[i * 3],
                      indexData[i * 3 + 1],
                      indexData[i * 3 + 2]));
                }
              }
            } else {
              // Non-indexed geometry (triangle soup) - generate sequential
              // indices. calcTangentsForTriangleMesh should be adapted to not
              // need that. Let's save that change for later.
              size_t numTriangles = posAccessor.count / 3;
              indices.reserve(numTriangles);
              for (size_t i = 0; i < numTriangles; ++i) {
                indices.push_back(uint3(i * 3, i * 3 + 1, i * 3 + 2));
              }
            }

            if (!indices.empty()) {
              // Create tangent array and compute tangents
              auto vertexTangentArray =
                  ctx.createArray(ANARI_FLOAT32_VEC4, posAccessor.count);
              auto *tangents = vertexTangentArray->mapAs<float4>();

              bool success = calcTangentsForTriangleMesh(indices.data(),
                  positions,
                  normals,
                  texCoords,
                  tangents,
                  indices.size(),
                  posAccessor.count);

              vertexTangentArray->unmap();

              if (success) {
                geometry->setParameterObject(
                    "vertex.tangent"_t, *vertexTangentArray);
                logInfo(
                    "[import_GLTF] Computed tangents for geometry '%s' with %zu vertices and %zu triangles",
                    geometryName.c_str(),
                    posAccessor.count,
                    indices.size());
              } else {
                logWarning(
                    "[import_GLTF] Failed to compute tangents for geometry '%s'",
                    geometryName.c_str());
              }
            }
          } else {
            logDebug(
                "[import_GLTF] Skipping tangent computation for geometry '%s': incompatible data types",
                geometryName.c_str());
          }
        } else {
          logDebug(
              "[import_GLTF] Skipping tangent computation for geometry '%s': missing required attributes (position=%s, normal=%s, texcoord=%s)",
              geometryName.c_str(),
              (posIt != primitive.attributes.end()) ? "yes" : "no",
              (normalIt != primitive.attributes.end()) ? "yes" : "no",
              (texCoordIt != primitive.attributes.end()) ? "yes" : "no");
        }
      }

      // Create surface with material
      MaterialRef material;
      if (primitive.material >= 0 && primitive.material < materials.size()) {
        material = materials[primitive.material];
      } else {
        material = ctx.defaultMaterial();
      }

      auto surface =
          ctx.createSurface(geometryName.c_str(), geometry, material);
      surfaces.push_back(surface);
    }
  }

  return surfaces;
}

static std::vector<LightRef> importGLTFLights(
    Context &ctx, const tinygltf::Model &model)
{
  std::vector<LightRef> lights;

  // Check for KHR_lights_punctual extension
  auto extensionIt = model.extensions.find("KHR_lights_punctual");
  if (extensionIt == model.extensions.end())
    return lights;

  const auto &lightExtension = extensionIt->second;
  if (!lightExtension.Has("lights"))
    return lights;

  const auto &lightsArray = lightExtension.Get("lights");
  if (!lightsArray.IsArray())
    return lights;

  for (size_t i = 0; i < lightsArray.ArrayLen(); ++i) {
    const auto &lightValue = lightsArray.Get(i);

    std::string type = lightValue.Get("type").Get<std::string>();

    LightRef light;

    if (type == "directional") {
      light = ctx.createObject<Light>(tokens::light::directional);
      light->setParameter("direction"_t, float3(0, 0, -1)); // Default direction
    } else if (type == "point") {
      light = ctx.createObject<Light>(tokens::light::point);
    } else if (type == "spot") {
      light = ctx.createObject<Light>(tokens::light::spot);
      light->setParameter("direction"_t, float3(0, 0, -1)); // Default direction

      if (lightValue.Has("spot")) {
        const auto &spot = lightValue.Get("spot");
        if (spot.Has("outerConeAngle")) {
          double outerCone = spot.Get("outerConeAngle").GetNumberAsDouble();
          light->setParameter("openingAngle"_t, float(outerCone));
        }
        if (spot.Has("innerConeAngle")) {
          double innerCone = spot.Get("innerConeAngle").GetNumberAsDouble();
          double outerCone = spot.Has("outerConeAngle")
              ? spot.Get("outerConeAngle").GetNumberAsDouble()
              : M_PI / 4.0;
          light->setParameter(
              "falloffAngle"_t, float((outerCone - innerCone) / 2.0));
        }
      }
    }

    if (light) {
      // Set color
      if (lightValue.Has("color")) {
        const auto &color = lightValue.Get("color");
        float3 lightColor(color.Get(0).GetNumberAsDouble(),
            color.Get(1).GetNumberAsDouble(),
            color.Get(2).GetNumberAsDouble());
        light->setParameter("color"_t, lightColor);
      } else {
        light->setParameter("color"_t, float3(1, 1, 1));
      }

      // Set intensity
      if (lightValue.Has("intensity")) {
        double intensity = lightValue.Get("intensity").GetNumberAsDouble();
        light->setParameter("intensity"_t, float(intensity));
      } else {
        light->setParameter("intensity"_t, 1.0f);
      }

      // Set name
      if (lightValue.Has("name")) {
        std::string name = lightValue.Get("name").Get<std::string>();
        light->setName(name.c_str());
      }

      lights.push_back(light);
    }
  }

  return lights;
}

static void populateGLTFLayer(Context &ctx,
    LayerNodeRef parentNode,
    const tinygltf::Model &model,
    const std::vector<SurfaceRef> &surfaces,
    const std::vector<LightRef> &lights,
    int nodeIndex)
{
  if (nodeIndex < 0 || nodeIndex >= model.nodes.size())
    return;

  const auto &node = model.nodes[nodeIndex];

  // Calculate transformation matrix
  mat4 transform = IDENTITY_MAT4;

  if (node.matrix.size() == 16) {
    // Matrix is provided directly
    transform = mat4(
        float4(node.matrix[0], node.matrix[1], node.matrix[2], node.matrix[3]),
        float4(node.matrix[4], node.matrix[5], node.matrix[6], node.matrix[7]),
        float4(
            node.matrix[8], node.matrix[9], node.matrix[10], node.matrix[11]),
        float4(node.matrix[12],
            node.matrix[13],
            node.matrix[14],
            node.matrix[15]));
  } else {
    // Compose from T * R * S
    mat4 translation = IDENTITY_MAT4;
    mat4 rotation = IDENTITY_MAT4;
    mat4 scale = IDENTITY_MAT4;

    if (node.translation.size() == 3) {
      translation = translation_matrix(float3(
          node.translation[0], node.translation[1], node.translation[2]));
    }

    if (node.rotation.size() == 4) {
      // Quaternion to matrix conversion
      rotation = rotation_matrix(float4(node.rotation[0],
          node.rotation[1],
          node.rotation[2],
          node.rotation[3]));
    }

    if (node.scale.size() == 3) {
      scale =
          scaling_matrix(float3(node.scale[0], node.scale[1], node.scale[2]));
    }

    transform = mul(translation, mul(rotation, scale));
  }

  // Create node in the hierarchy
  auto nodeRef = parentNode->insert_last_child({transform, node.name.c_str()});

  // Add mesh if present
  if (node.mesh >= 0 && node.mesh < model.meshes.size()) {
    const auto &mesh = model.meshes[node.mesh];
    for (size_t i = 0; i < mesh.primitives.size(); ++i) {
      // Find corresponding surface (each primitive becomes a surface)
      size_t surfaceIndex = 0;
      for (size_t meshIdx = 0; meshIdx < node.mesh; ++meshIdx) {
        surfaceIndex += model.meshes[meshIdx].primitives.size();
      }
      surfaceIndex += i;

      if (surfaceIndex < surfaces.size()) {
        auto surface = surfaces[surfaceIndex];
        nodeRef->insert_last_child(
            {Any(ANARI_SURFACE, surface.index()), surface->name().c_str()});
      }
    }
  }

  // Add light if present (KHR_lights_punctual extension)
  if (node.light >= 0 && node.light < lights.size()) {
    auto light = lights[node.light];
    nodeRef->insert_first_child(Any(ANARI_LIGHT, light->index()));
  }

  // Process children
  for (int childIndex : node.children) {
    populateGLTFLayer(ctx, nodeRef, model, surfaces, lights, childIndex);
  }
}

void import_GLTF(Context &ctx, const char *filename, LayerNodeRef location)
{
  tinygltf::Model model;
  tinygltf::TinyGLTF loader;
  std::string err;
  std::string warn;

  bool success = false;
  std::string ext = extensionOf(filename);

  if (ext == ".gltf") {
    success = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
  } else if (ext == ".glb") {
    success = loader.LoadBinaryFromFile(&model, &err, &warn, filename);
  } else {
    logError("[import_GLTF] unsupported file extension: %s", ext.c_str());
    return;
  }

  if (!warn.empty()) {
    logWarning("[import_GLTF] %s", warn.c_str());
  }

  if (!success) {
    logError("[import_GLTF] failed to load %s: %s", filename, err.c_str());
    return;
  }

  // Import materials
  auto materials = importGLTFMaterials(ctx, model);

  // Import meshes
  auto surfaces = importGLTFMeshes(ctx, model, materials);

  // Import lights
  auto lights = importGLTFLights(ctx, model);

  // Build scene hierarchy
  LayerNodeRef targetLocation =
      location ? location : ctx.defaultLayer()->root();

  // Create a root node for the entire glTF file
  std::string fileName = fileOf(filename);
  // Remove extension to get just the basename
  size_t dotPos = fileName.find_last_of('.');
  if (dotPos != std::string::npos) {
    fileName = fileName.substr(0, dotPos);
  }

  // Create root transformation node
  auto rootNode =
      targetLocation->insert_last_child({IDENTITY_MAT4, fileName.c_str()});

  if (model.defaultScene >= 0 && model.defaultScene < model.scenes.size()) {
    const auto &scene = model.scenes[model.defaultScene];
    for (int nodeIndex : scene.nodes) {
      populateGLTFLayer(ctx, rootNode, model, surfaces, lights, nodeIndex);
    }
  } else if (!model.scenes.empty()) {
    // Use first scene if no default is specified
    const auto &scene = model.scenes[0];
    for (int nodeIndex : scene.nodes) {
      populateGLTFLayer(ctx, rootNode, model, surfaces, lights, nodeIndex);
    }
  } else {
    logWarning("[import_GLTF] no scenes found in glTF file");
  }
}

} // namespace tsd::io
