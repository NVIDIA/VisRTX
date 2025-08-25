// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef TSD_USE_ASSIMP
#define TSD_USE_ASSIMP 1
#endif

#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
#include "tsd/core/Logging.hpp"
#if TSD_USE_ASSIMP
// assimp
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/DefaultLogger.hpp>
#include <assimp/Importer.hpp>
#endif

namespace tsd::io {

using namespace tsd::core;

#if TSD_USE_ASSIMP

static SamplerRef importEmbeddedTexture(
    Context &ctx, const aiTexture *embeddedTexture, TextureCache &cache)
{
  std::string filepath = embeddedTexture->mFilename.C_Str();
  const bool validTexture =
      embeddedTexture->mHeight != 0 && embeddedTexture->pcData != nullptr;
  logDebug("[import_ASSIMP] embedded '%s' texture | valid: %i height: %i",
      filepath.c_str(),
      int(validTexture),
      int(embeddedTexture->mHeight));

  auto dataArray = cache[filepath];

  if (!validTexture)
    return {};

  if (!dataArray.valid()) {
    dataArray = ctx.createArray(
        ANARI_UFIXED8_VEC4, embeddedTexture->mWidth, embeddedTexture->mHeight);
    dataArray->setData(embeddedTexture->pcData);
    cache[filepath] = dataArray;
  }

  auto tex = ctx.createObject<Sampler>(tokens::sampler::image2D);
  tex->setParameterObject("image"_t, *dataArray);
  tex->setParameter("inAttribute"_t, "attribute0");
  tex->setParameter("wrapMode1"_t, "repeat");
  tex->setParameter("wrapMode2"_t, "repeat");
  tex->setParameter("filter"_t, "linear");
  tex->setName(fileOf(filepath).c_str());

  return tex;
}

static std::vector<SurfaceRef> importASSIMPSurfaces(Context &ctx,
    const std::vector<MaterialRef> &materials,
    const aiScene *scene)
{
  std::vector<SurfaceRef> tsdMeshes;

  for (unsigned i = 0; i < scene->mNumMeshes; ++i) {
    aiMesh *mesh = scene->mMeshes[i];

    auto tsdMesh = ctx.createObject<Geometry>(tokens::geometry::triangle);

    unsigned numVertices = mesh->mNumVertices;
    auto vertexPositionArray = ctx.createArray(ANARI_FLOAT32_VEC3, numVertices);
    auto *outVertices = vertexPositionArray->mapAs<float3>();

    auto vertexNormalArray = ctx.createArray(
        ANARI_FLOAT32_VEC3, mesh->HasNormals() ? numVertices : 0);
    float3 *outNormals =
        vertexNormalArray ? vertexNormalArray->mapAs<float3>() : nullptr;

    auto vertexTexCoordArray = ctx.createArray(ANARI_FLOAT32_VEC3,
        mesh->HasTextureCoords(0 /*texcord set*/) ? numVertices : 0);
    float3 *outTexCoords =
        vertexTexCoordArray ? vertexTexCoordArray->mapAs<float3>() : nullptr;

    auto vertexTangentArray = ctx.createArray(
        ANARI_FLOAT32_VEC4, mesh->HasTangentsAndBitangents() ? numVertices : 0);
    float4 *outTangents =
        vertexTangentArray ? vertexTangentArray->mapAs<float4>() : nullptr;

    // TODO: test for AI_MAX_NUMBER_OF_COLOR_SETS, import all..
    auto vertexColorArray =
        ctx.createArray(ANARI_FLOAT32_VEC4, mesh->mColors[0] ? numVertices : 0);
    float4 *outColors =
        vertexColorArray ? vertexColorArray->mapAs<float4>() : nullptr;

    for (unsigned j = 0; j < mesh->mNumVertices; ++j) {
      aiVector3D v = mesh->mVertices[j];
      outVertices[j] = float3(v.x, v.y, v.z);

      if (mesh->HasNormals() && outNormals) {
        aiVector3D n = mesh->mNormals[j];
        outNormals[j] = float3(n.x, n.y, n.z);
      }

      // TODO: import tc sets > 0 accordingly..
      if (mesh->HasTextureCoords(0) && outTexCoords) {
        aiVector3D tc = mesh->mTextureCoords[0][j];
        outTexCoords[j] = float3(tc.x, tc.y, tc.z);
      }

      if (mesh->HasTangentsAndBitangents() && mesh->HasNormals()
          && outTangents) {
        aiVector3D tng = mesh->mTangents[j];
        aiVector3D btng = mesh->mBitangents[j];
        aiVector3D n = mesh->mNormals[j];

        // Convert to ANARI/glTF format where handedness is stored in
        // tangent's w-coord!

        // Gram-Schmidt orthogonalize
        tng = (tng - n * (n * tng)).Normalize();

        float handedness = copysign(1.0f, (n ^ tng) * btng);
        outTangents[j] = float4(tng.x, tng.y, tng.z, handedness);
      }

      // TODO: import color sets > 0 accordingly..
      if (mesh->mColors[0] && outColors) {
        aiColor4D c = mesh->mColors[0][j];
        outColors[j] = float4(c.r, c.g, c.b, c.a);
      }
    }

    unsigned numIndices = mesh->mNumFaces;
    auto indexArray = ctx.createArray(ANARI_UINT32_VEC3, numIndices);
    auto *outIndices = indexArray->mapAs<uint3>();

    for (unsigned j = 0; j < mesh->mNumFaces; ++j) {
      outIndices[j] = uint3(mesh->mFaces[j].mIndices[0],
          mesh->mFaces[j].mIndices[1],
          mesh->mFaces[j].mIndices[2]);
    }

    vertexPositionArray->unmap();
    tsdMesh->setParameterObject("vertex.position"_t, *vertexPositionArray);

    indexArray->unmap();
    tsdMesh->setParameterObject("primitive.index"_t, *indexArray);

    if (outNormals) {
      vertexNormalArray->unmap();
      tsdMesh->setParameterObject("vertex.normal"_t, *vertexNormalArray);
    }

    if (outTexCoords) {
      vertexTexCoordArray->unmap();
      tsdMesh->setParameterObject("vertex.attribute0"_t, *vertexTexCoordArray);
    }

    if (outTangents) {
      vertexTangentArray->unmap();
      tsdMesh->setParameterObject("vertex.tangent"_t, *vertexTangentArray);
    }

    if (outColors) {
      vertexColorArray->unmap();
      tsdMesh->setParameterObject("vertex.color"_t, *vertexColorArray);
    }

    // Calculate tangents if not supplied by mesh
    if (!outTangents) {
      auto vertexTangentArray =
          ctx.createArray(ANARI_FLOAT32_VEC4, numVertices);
      auto outTangents = vertexTangentArray->mapAs<float4>();

      calcTangentsForTriangleMesh(outIndices,
          outVertices,
          outNormals,
          outTexCoords,
          outTangents,
          numIndices,
          numVertices);

      vertexTangentArray->unmap();
      tsdMesh->setParameterObject("vertex.tangent"_t, *vertexTangentArray);
    }

    tsdMesh->setName((std::string(mesh->mName.C_Str()) + "_geometry").c_str());

    unsigned matID = mesh->mMaterialIndex;
    auto tsdMat = matID < 0 ? ctx.defaultMaterial() : materials[size_t(matID)];
    tsdMeshes.push_back(
        ctx.createSurface(mesh->mName.C_Str(), tsdMesh, tsdMat));
  }

  return tsdMeshes;
}

static std::vector<MaterialRef> importASSIMPMaterials(
    Context &ctx, const aiScene *scene, const std::string &filename)
{
  std::vector<MaterialRef> materials;

  TextureCache cache;

  std::string basePath = pathOf(filename);

  for (unsigned i = 0; i < scene->mNumMaterials; ++i) {
    aiMaterial *assimpMat = scene->mMaterials[i];
    ai_int matType;
    assimpMat->Get(AI_MATKEY_SHADING_MODEL, matType);

    MaterialRef m;

    auto loadTexture = [&](const aiString &texName,
                           bool isLinear = false) -> SamplerRef {
      SamplerRef tex;
      if (texName.length != 0) {
        auto *embeddedTexture = scene->GetEmbeddedTexture(texName.C_Str());
        if (embeddedTexture)
          tex = importEmbeddedTexture(ctx, embeddedTexture, cache);
        else
          tex = importTexture(ctx, basePath + texName.C_Str(), cache, isLinear);
      }

      return tex;
    };

    auto getTextureUVTransform = [&](const char *pKey,
                                     unsigned int type,
                                     unsigned int index = 0) -> mat4 {
      aiUVTransform uvTransform;
      if (aiGetMaterialUVTransform(assimpMat, pKey, type, index, &uvTransform)
          == AI_SUCCESS) {
        return mat4(
            {uvTransform.mScaling.x, 0.f, 0.f, uvTransform.mTranslation.x},
            {0.f, uvTransform.mScaling.y, 0.f, uvTransform.mTranslation.y},
            {0.f, 0.f, 1.f, 0.f},
            {0.0f, 0.0f, 0.f, 1.f});
      }
      return {{1.0f, 0.0f, 0.0f, 0.0f},
          {0.0f, 1.0f, 0.0f, 0.0f},
          {0.0f, 0.0f, 1.0f, 0.0f},
          {0.0f, 0.0f, 0.0f, 1.0f}};
    };

    if (matType == aiShadingMode_PBR_BRDF) {
      m = ctx.createObject<Material>(tokens::material::physicallyBased);

      // Diffuse color handling
      if (aiString baseColorTexture;
          assimpMat->GetTexture(AI_MATKEY_BASE_COLOR_TEXTURE, &baseColorTexture)
          == AI_SUCCESS) {
        if (auto sampler = loadTexture(baseColorTexture); sampler) {
          auto tx = getTextureUVTransform(
              AI_MATKEY_UVTRANSFORM(aiTextureType_BASE_COLOR, 0));
          sampler->setParameter("inTransform"_t, tx);
          m->setParameterObject("baseColor"_t, *sampler);
        }
      } else if (aiColor3D baseColor;
          assimpMat->Get(AI_MATKEY_BASE_COLOR, baseColor) == AI_SUCCESS) {
        m->setParameter("baseColor"_t, ANARI_FLOAT32_VEC3, &baseColor);
      }

      // Metallic/Roughness handling
      if (aiString metallicTexture;
          assimpMat->GetTexture(AI_MATKEY_METALLIC_TEXTURE, &metallicTexture)
          == AI_SUCCESS) {
        if (auto sampler = loadTexture(metallicTexture, true); sampler) {
          auto tx = getTextureUVTransform(
              AI_MATKEY_UVTRANSFORM(aiTextureType_METALNESS, 0));
          sampler->setParameter("inTransform"_t, tx);
          // - Metallic is blue
          sampler->setParameter("outTransform"_t,
              mat4({0, 0, 0, 0}, {0, 0, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 1}));
          m->setParameterObject("metallic"_t, *sampler);
        }
      } else if (ai_real metallicFactor;
          assimpMat->Get(AI_MATKEY_METALLIC_FACTOR, metallicFactor)
          == AI_SUCCESS) {
        m->setParameter("metallic"_t, ANARI_FLOAT32, &metallicFactor);
      }

      if (aiString roughnessTexture;
          assimpMat->GetTexture(AI_MATKEY_ROUGHNESS_TEXTURE, &roughnessTexture)
          == AI_SUCCESS) {
        if (auto sampler = loadTexture(roughnessTexture, true); sampler) {
          // Map red to red/blue as expected by our gltf pbr implementation
          auto tx = getTextureUVTransform(
              AI_MATKEY_UVTRANSFORM(aiTextureType_DIFFUSE_ROUGHNESS, 0));
          sampler->setParameter("inTransform"_t, tx);
          // - Roughness is green
          sampler->setParameter("outTransform"_t,
              mat4({0, 0, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 1}));
          m->setParameterObject("roughness"_t, *sampler);
        }
      } else if (ai_real roughnessFactor;
          assimpMat->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughnessFactor)
          == AI_SUCCESS) {
        m->setParameter("roughness"_t, ANARI_FLOAT32, &roughnessFactor);
      }

      // Specular workflow
      if (aiColor3D specularColor;
          assimpMat->Get(AI_MATKEY_COLOR_SPECULAR, specularColor)
          == AI_SUCCESS) {
        m->setParameter("specularcColor"_t, ANARI_FLOAT32_VEC3, &specularColor);
      }
      if (ai_real specularFactor;
          assimpMat->Get(AI_MATKEY_SPECULAR_FACTOR, specularFactor)
          == AI_SUCCESS) {
        m->setParameter("specular"_t, ANARI_FLOAT32, &specularFactor);
      }

      // Sheen handling
      if (aiString sheenColorTexture;
          assimpMat->GetTexture(
              AI_MATKEY_SHEEN_COLOR_TEXTURE, &sheenColorTexture)
          == AI_SUCCESS) {
        if (auto sampler = loadTexture(sheenColorTexture); sampler) {
          auto tx = getTextureUVTransform(
              AI_MATKEY_UVTRANSFORM(aiTextureType_SHEEN, 0));
          sampler->setParameter("inTransform"_t, tx);
          m->setParameterObject("sheenColor.texture"_t, *sampler);
        }
      } else if (aiColor3D sheenColor;
          assimpMat->Get(AI_MATKEY_SHEEN_COLOR_FACTOR, sheenColor)
          == AI_SUCCESS) {
        m->setParameter("sheenColor.value"_t, ANARI_FLOAT32_VEC3, &sheenColor);
      }

      if (aiString sheenRoughnessTexture;
          assimpMat->GetTexture(
              AI_MATKEY_SHEEN_ROUGHNESS_TEXTURE, &sheenRoughnessTexture)
          == AI_SUCCESS) {
        if (auto sampler = loadTexture(sheenRoughnessTexture, true); sampler) {
          auto tx = getTextureUVTransform(
              AI_MATKEY_UVTRANSFORM(aiTextureType_SHEEN, 1));
          sampler->setParameter("inTransform"_t, tx);
          m->setParameterObject("sheenRoughness.texture"_t, *sampler);
        }
      } else if (ai_real sheenRoughnessFactor;
          assimpMat->Get(AI_MATKEY_SHEEN_ROUGHNESS_FACTOR, sheenRoughnessFactor)
          == AI_SUCCESS) {
        m->setParameter(
            "sheenRoughness.value"_t, ANARI_FLOAT32, &sheenRoughnessFactor);
      }

      // Clearcoat handling
      if (aiString clearcoatTexture;
          assimpMat->GetTexture(AI_MATKEY_CLEARCOAT_TEXTURE, &clearcoatTexture)
          == AI_SUCCESS) {
        if (auto sampler = loadTexture(clearcoatTexture, true); sampler) {
          auto tx = getTextureUVTransform(
              AI_MATKEY_UVTRANSFORM(aiTextureType_CLEARCOAT, 0));
          sampler->setParameter("inTransform"_t, tx);
          m->setParameterObject("clearcoat.texture"_t, *sampler);
        }
      } else if (ai_real clearcoatFactor;
          assimpMat->Get(AI_MATKEY_CLEARCOAT_FACTOR, clearcoatFactor)
          == AI_SUCCESS) {
        m->setParameter("clearcoat.value"_t, ANARI_FLOAT32, &clearcoatFactor);
      }

      if (aiString clearcoatRoughnessTexture;
          assimpMat->GetTexture(
              AI_MATKEY_CLEARCOAT_ROUGHNESS_TEXTURE, &clearcoatRoughnessTexture)
          == AI_SUCCESS) {
        if (auto sampler = loadTexture(clearcoatRoughnessTexture, true);
            sampler) {
          auto tx = getTextureUVTransform(
              AI_MATKEY_UVTRANSFORM(aiTextureType_CLEARCOAT, 1));
          sampler->setParameter("inTransform"_t, tx);
          m->setParameterObject("clearcoatRoughness.texture"_t, *sampler);
        }
      } else if (ai_real clearcoatRoughnessFactor;
          assimpMat->Get(
              AI_MATKEY_CLEARCOAT_ROUGHNESS_FACTOR, clearcoatRoughnessFactor)
          == AI_SUCCESS) {
        m->setParameter("clearcoatRoughness.value"_t,
            ANARI_FLOAT32,
            &clearcoatRoughnessFactor);
      }

      if (aiString clearcoatNormalTexture;
          assimpMat->GetTexture(
              AI_MATKEY_CLEARCOAT_NORMAL_TEXTURE, &clearcoatNormalTexture)
          == AI_SUCCESS) {
        if (auto sampler = loadTexture(clearcoatNormalTexture, true); sampler) {
          auto tx = getTextureUVTransform(
              AI_MATKEY_UVTRANSFORM(aiTextureType_CLEARCOAT, 2));
          sampler->setParameter("inTransform"_t, tx);
          m->setParameterObject("clearcoatNormal"_t, *sampler);
        }
      }

      // Emssive handling
      if (aiString emissiveTexture;
          assimpMat->GetTexture(aiTextureType_EMISSIVE, 0, &emissiveTexture)
          == AI_SUCCESS) {
        if (auto sampler = loadTexture(emissiveTexture); sampler) {
          auto tx = getTextureUVTransform(
              AI_MATKEY_UVTRANSFORM(aiTextureType_EMISSIVE, 0));
          sampler->setParameter("inTransform"_t, tx);
          m->setParameterObject("emissive"_t, *sampler);
        }
      } else if (aiColor3D emissiveColor;
          assimpMat->Get(AI_MATKEY_COLOR_EMISSIVE, emissiveColor)
          == AI_SUCCESS) {
        m->setParameter("emissive"_t, ANARI_FLOAT32_VEC3, &emissiveColor);
      }

      // Opacity handling
      if (ai_real opacity;
          assimpMat->Get(AI_MATKEY_OPACITY, opacity) == AI_SUCCESS) {
        m->setParameter("opacity"_t, ANARI_FLOAT32, &opacity);
      }

      // Occlusion handling
      if (aiString occlusionTexture;
          assimpMat->GetTexture(
              aiTextureType_AMBIENT_OCCLUSION, 0, &occlusionTexture)
          == AI_SUCCESS) {
        if (auto sampler = loadTexture(occlusionTexture, true); sampler) {
          auto tx = getTextureUVTransform(
              AI_MATKEY_UVTRANSFORM(aiTextureType_AMBIENT_OCCLUSION, 0));
          sampler->setParameter("inTransform"_t, tx);
          m->setParameterObject("occlusion"_t, *sampler);
        }
      }

      // Normal handling
      if (aiString normalTexture;
          assimpMat->GetTexture(aiTextureType_NORMALS, 0, &normalTexture)
          == AI_SUCCESS) {
        if (auto sampler = loadTexture(normalTexture, true); sampler) {
          auto tx = getTextureUVTransform(
              AI_MATKEY_UVTRANSFORM(aiTextureType_NORMALS, 0));
          sampler->setParameter("inTransform"_t, tx);
          m->setParameterObject("normal"_t, *sampler);
        }
      }

      // transmission handling
      if (ai_real transmissionFactor;
          assimpMat->Get(AI_MATKEY_TRANSMISSION_FACTOR, transmissionFactor)
          == AI_SUCCESS) {
        m->setParameter("transmission"_t, ANARI_FLOAT32, &transmissionFactor);
      }
      if (aiString transmissionTexture;
          assimpMat->GetTexture(
              AI_MATKEY_TRANSMISSION_TEXTURE, &transmissionTexture)
          == AI_SUCCESS) {
        if (auto sampler = loadTexture(transmissionTexture, true); sampler) {
          auto tx = getTextureUVTransform(
              AI_MATKEY_UVTRANSFORM(aiTextureType_TRANSMISSION, 0));
          sampler->setParameter("inTransform"_t, tx);
          m->setParameterObject("transmission"_t, *sampler);
        }
      }
    } else { // GL-like dflt. material
      aiColor3D col;
      assimpMat->Get(AI_MATKEY_COLOR_DIFFUSE, col);
      ai_real opacity;
      assimpMat->Get(AI_MATKEY_OPACITY, opacity);

      m = ctx.createObject<Material>(tokens::material::matte);
      m->setParameter("color"_t, ANARI_FLOAT32_VEC3, &col);
      m->setParameter("opacity"_t, opacity);
    }

    aiString name;
    assimpMat->Get(AI_MATKEY_NAME, name);
    m->setName(name.C_Str());

    materials.push_back(m);
  }

  return materials;
}

static std::vector<LightRef> importASSIMPLights(
    Context &ctx, const aiScene *scene)
{
  std::vector<LightRef> lights;

  for (unsigned i = 0; i < scene->mNumLights; ++i) {
    aiLight *assimpLight = scene->mLights[i];
    LightRef lightRef;

    float intensity =
        assimpLight->mColorDiffuse.r > assimpLight->mColorDiffuse.b
        ? assimpLight->mColorDiffuse.r > assimpLight->mColorDiffuse.g
            ? assimpLight->mColorDiffuse.r
            : assimpLight->mColorDiffuse.g
        : assimpLight->mColorDiffuse.b;

    if (intensity == 0.f)
      intensity = 1.f;

    tsd::math::float3 color(assimpLight->mColorDiffuse.r / intensity,
        assimpLight->mColorDiffuse.g / intensity,
        assimpLight->mColorDiffuse.b / intensity);

    switch (assimpLight->mType) {
    case aiLightSource_DIRECTIONAL:
      lightRef = ctx.createObject<Light>(tokens::light::directional);
      lightRef->setParameter("direction",
          tsd::math::float3(assimpLight->mDirection.x,
              assimpLight->mDirection.y,
              assimpLight->mDirection.z));
      lightRef->setParameter("intensity", intensity);
      break;
    case aiLightSource_POINT:
      lightRef = ctx.createObject<Light>(tokens::light::point);
      lightRef->setParameter("position",
          tsd::math::float3(assimpLight->mPosition.x,
              assimpLight->mPosition.y,
              assimpLight->mPosition.z));
      lightRef->setParameter("intensity", intensity);
      break;
    case aiLightSource_SPOT:
      lightRef = ctx.createObject<Light>(tokens::light::spot);
      lightRef->setParameter("position",
          tsd::math::float3(assimpLight->mPosition.x,
              assimpLight->mPosition.y,
              assimpLight->mPosition.z));
      lightRef->setParameter("direction",
          tsd::math::float3(assimpLight->mDirection.x,
              assimpLight->mDirection.y,
              assimpLight->mDirection.z));
      lightRef->setParameter("openingAngle", assimpLight->mAngleOuterCone);
      lightRef->setParameter("falloffAngle",
          (assimpLight->mAngleOuterCone - assimpLight->mAngleInnerCone) / 2.f);
      lightRef->setParameter("intensity", intensity);
      break;
    default:
      break;
    }

    if (lightRef) {
      lightRef->setParameter("color", color);
      aiString name = assimpLight->mName;
      lightRef->setName(name.C_Str());
      lights.push_back(lightRef);
    }
  }

  return lights;
}

static void populateASSIMPLayer(Context &ctx,
    LayerNodeRef tsdLayerRef,
    const std::vector<SurfaceRef> &surfaces,
    const std::vector<LightRef> &lights,
    const aiNode *node)
{
  static_assert(
      sizeof(tsd::math::mat4) == sizeof(aiMatrix4x4), "matrix type size mismatch");
  tsd::math::mat4 mat;
  std::memcpy(&mat, &node->mTransformation, sizeof(mat));
  mat = tsd::math::transpose(mat);
  auto tr = tsdLayerRef->insert_last_child({mat, node->mName.C_Str()});

  for (unsigned int i = 0; i < node->mNumMeshes; i++) {
    auto mesh = surfaces.at(node->mMeshes[i]);
    tr->insert_last_child(
        {Any(ANARI_SURFACE, mesh.index()), mesh->name().c_str()});
  }

  // https://github.com/assimp/assimp/issues/1168#issuecomment-278673292
  // We won't find the light directly on the node, but matching names
  // indicate we're supposed to associate the light with the transform
  std::string name(node->mName.C_Str());
  auto it = std::find_if(lights.begin(),
      lights.end(),
      [name](const LightRef &lightRef) { return lightRef->name() == name; });

  if (it != lights.end()) {
    tr->insert_first_child(tsd::core::Any(ANARI_LIGHT, (*it)->index()));
  }

  for (unsigned int i = 0; i < node->mNumChildren; i++)
    populateASSIMPLayer(ctx, tr, surfaces, lights, node->mChildren[i]);
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void import_ASSIMP(
    Context &ctx, const char *filename, LayerNodeRef location, bool flatten)
{
  Assimp::DefaultLogger::create("", Assimp::Logger::VERBOSE);

  Assimp::Importer importer;

  auto importFlags = aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_FlipUVs;
  if (flatten)
    importFlags |= aiProcess_PreTransformVertices;

  const aiScene *scene = importer.ReadFile(filename, importFlags);

  if (scene == nullptr) {
    Assimp::DefaultLogger::get()->error(importer.GetErrorString());
    return;
  }

  auto lights = importASSIMPLights(ctx, scene);
  auto materials = importASSIMPMaterials(ctx, scene, filename);
  auto meshes = importASSIMPSurfaces(ctx, materials, scene);

  populateASSIMPLayer(ctx,
      location ? location : ctx.defaultLayer()->root(),
      meshes,
      lights,
      scene->mRootNode);
}
#else
void import_ASSIMP(
    Context &ctx, const char *filename, LayerNodeRef location, bool flatten)
{
  logError("[import_ASSIMP] ASSIMP not enabled in TSD build.");
}
#endif

} // namespace tsd
