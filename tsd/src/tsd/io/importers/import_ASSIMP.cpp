// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef TSD_USE_ASSIMP
#define TSD_USE_ASSIMP 1
#endif

#include "tsd/core/Logging.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
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
    Scene &scene, const aiTexture *embeddedTexture, TextureCache &cache)
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
    dataArray = scene.createArray(
        ANARI_UFIXED8_VEC4, embeddedTexture->mWidth, embeddedTexture->mHeight);
    dataArray->setData(embeddedTexture->pcData);
    cache[filepath] = dataArray;
  }

  auto tex = scene.createObject<Sampler>(tokens::sampler::image2D);
  tex->setParameterObject("image", *dataArray);
  tex->setParameter("inAttribute", "attribute0");
  tex->setParameter("wrapMode1", "repeat");
  tex->setParameter("wrapMode2", "repeat");
  tex->setParameter("filter", "linear");
  tex->setName(fileOf(filepath).c_str());

  return tex;
}

static std::vector<SurfaceRef> importASSIMPSurfaces(Scene &scene,
    const std::vector<MaterialRef> &materials,
    const aiScene *a_scene)
{
  std::vector<SurfaceRef> tsdMeshes;

  for (unsigned i = 0; i < a_scene->mNumMeshes; ++i) {
    aiMesh *mesh = a_scene->mMeshes[i];

    auto tsdMesh = scene.createObject<Geometry>(tokens::geometry::triangle);

    unsigned numVertices = mesh->mNumVertices;
    auto vertexPositionArray =
        scene.createArray(ANARI_FLOAT32_VEC3, numVertices);
    auto *outVertices = vertexPositionArray->mapAs<float3>();

    auto vertexNormalArray = scene.createArray(
        ANARI_FLOAT32_VEC3, mesh->HasNormals() ? numVertices : 0);
    float3 *outNormals =
        vertexNormalArray ? vertexNormalArray->mapAs<float3>() : nullptr;

    auto vertexTexCoordArray = scene.createArray(ANARI_FLOAT32_VEC2,
        mesh->HasTextureCoords(0 /*texcord set*/) ? numVertices : 0);
    float2 *outTexCoords =
        vertexTexCoordArray ? vertexTexCoordArray->mapAs<float2>() : nullptr;

    auto vertexTangentArray = scene.createArray(
        ANARI_FLOAT32_VEC4, mesh->HasTangentsAndBitangents() ? numVertices : 0);
    float4 *outTangents =
        vertexTangentArray ? vertexTangentArray->mapAs<float4>() : nullptr;

    // TODO: test for AI_MAX_NUMBER_OF_COLOR_SETS, import all..
    auto vertexColorArray = scene.createArray(
        ANARI_FLOAT32_VEC4, mesh->mColors[0] ? numVertices : 0);
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
        outTexCoords[j] = float2(tc.x, tc.y);
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
    auto indexArray = scene.createArray(ANARI_UINT32_VEC3, numIndices);
    auto *outIndices = indexArray->mapAs<uint3>();

    for (unsigned j = 0; j < mesh->mNumFaces; ++j) {
      outIndices[j] = uint3(mesh->mFaces[j].mIndices[0],
          mesh->mFaces[j].mIndices[1],
          mesh->mFaces[j].mIndices[2]);
    }

    vertexPositionArray->unmap();
    tsdMesh->setParameterObject("vertex.position", *vertexPositionArray);

    indexArray->unmap();
    tsdMesh->setParameterObject("primitive.index", *indexArray);

    if (outNormals) {
      vertexNormalArray->unmap();
      tsdMesh->setParameterObject("vertex.normal", *vertexNormalArray);
    }

    if (outTexCoords) {
      vertexTexCoordArray->unmap();
      tsdMesh->setParameterObject("vertex.attribute0", *vertexTexCoordArray);
    }

    if (outTangents) {
      vertexTangentArray->unmap();
      tsdMesh->setParameterObject("vertex.tangent", *vertexTangentArray);
    }

    if (outColors) {
      vertexColorArray->unmap();
      tsdMesh->setParameterObject("vertex.color", *vertexColorArray);
    }

    // Calculate tangents if not supplied by mesh
    if (!outTangents) {
      auto vertexTangentArray =
          scene.createArray(ANARI_FLOAT32_VEC4, numVertices);
      auto outTangents = vertexTangentArray->mapAs<float4>();

      calcTangentsForTriangleMesh(outIndices,
          outVertices,
          outNormals,
          outTexCoords,
          outTangents,
          numIndices,
          numVertices);

      vertexTangentArray->unmap();
      tsdMesh->setParameterObject("vertex.tangent", *vertexTangentArray);
    }

    tsdMesh->setName((std::string(mesh->mName.C_Str()) + "_geometry").c_str());

    unsigned matID = mesh->mMaterialIndex;
    auto tsdMat =
        matID < 0 ? scene.defaultMaterial() : materials[size_t(matID)];
    tsdMeshes.push_back(
        scene.createSurface(mesh->mName.C_Str(), tsdMesh, tsdMat));
  }

  return tsdMeshes;
}

static std::vector<MaterialRef> importASSIMPMaterials(
    Scene &scene, const aiScene *a_scene, const std::string &filename)
{
  std::vector<MaterialRef> materials;

  TextureCache cache;

  std::string basePath = pathOf(filename);

  for (unsigned i = 0; i < a_scene->mNumMaterials; ++i) {
    aiMaterial *assimpMat = a_scene->mMaterials[i];
    ai_int matType;
    assimpMat->Get(AI_MATKEY_SHADING_MODEL, matType);

    MaterialRef m;

    auto loadTexture = [&](const aiString &texName,
                           bool isLinear = false) -> SamplerRef {
      SamplerRef tex;
      if (texName.length != 0) {
        auto *embeddedTexture = a_scene->GetEmbeddedTexture(texName.C_Str());
        if (embeddedTexture)
          tex = importEmbeddedTexture(scene, embeddedTexture, cache);
        else
          tex =
              importTexture(scene, basePath + texName.C_Str(), cache, isLinear);
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
      m = scene.createObject<Material>(tokens::material::physicallyBased);

      // Diffuse color handling
      if (aiString baseColorTexture;
          assimpMat->GetTexture(AI_MATKEY_BASE_COLOR_TEXTURE, &baseColorTexture)
          == AI_SUCCESS) {
        if (auto sampler = loadTexture(baseColorTexture); sampler) {
          auto tx = getTextureUVTransform(
              AI_MATKEY_UVTRANSFORM(aiTextureType_BASE_COLOR, 0));
          sampler->setParameter("inTransform", tx);
          m->setParameterObject("baseColor", *sampler);
        }
      } else if (aiColor3D baseColor;
          assimpMat->Get(AI_MATKEY_BASE_COLOR, baseColor) == AI_SUCCESS) {
        m->setParameter("baseColor", ANARI_FLOAT32_VEC3, &baseColor);
      }

      // Metallic/Roughness handling
      if (aiString metallicTexture;
          assimpMat->GetTexture(AI_MATKEY_METALLIC_TEXTURE, &metallicTexture)
          == AI_SUCCESS) {
        if (auto sampler = loadTexture(metallicTexture, true); sampler) {
          auto tx = getTextureUVTransform(
              AI_MATKEY_UVTRANSFORM(aiTextureType_METALNESS, 0));
          sampler->setParameter("inTransform", tx);
          // - Metallic is blue
          sampler->setParameter("outTransform",
              mat4({0, 0, 0, 0}, {0, 0, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 1}));
          m->setParameterObject("metallic", *sampler);
        }
      } else if (ai_real metallicFactor;
          assimpMat->Get(AI_MATKEY_METALLIC_FACTOR, metallicFactor)
          == AI_SUCCESS) {
        m->setParameter("metallic", ANARI_FLOAT32, &metallicFactor);
      }

      if (aiString roughnessTexture;
          assimpMat->GetTexture(AI_MATKEY_ROUGHNESS_TEXTURE, &roughnessTexture)
          == AI_SUCCESS) {
        if (auto sampler = loadTexture(roughnessTexture, true); sampler) {
          // Map red to red/blue as expected by our gltf pbr implementation
          auto tx = getTextureUVTransform(
              AI_MATKEY_UVTRANSFORM(aiTextureType_DIFFUSE_ROUGHNESS, 0));
          sampler->setParameter("inTransform", tx);
          // - Roughness is green
          sampler->setParameter("outTransform",
              mat4({0, 0, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 1}));
          m->setParameterObject("roughness", *sampler);
        }
      } else if (ai_real roughnessFactor;
          assimpMat->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughnessFactor)
          == AI_SUCCESS) {
        m->setParameter("roughness", ANARI_FLOAT32, &roughnessFactor);
      }

#ifdef AI_MATKEY_ANISOTROPY_TEXTURE
      // anisotropic texture added with assimp 6.0
      if (aiString anisotropyTexture;
          assimpMat->GetTexture(
              AI_MATKEY_ANISOTROPY_TEXTURE, &anisotropyTexture)
          == AI_SUCCESS) {
        if (auto sampler = loadTexture(anisotropyTexture, true); sampler) {
          // Map red to red/green/blue as expected by our gltf pbr
          // implementation
          auto tx = getTextureUVTransform(
              AI_MATKEY_UVTRANSFORM(aiTextureType_ANISOTROPY, 0));
          sampler->setParameter("inTransform", tx);
          // - Tangent/bitangent Direction is red/green
          //   and remap from [0:1] to [-1:1]
          sampler->setParameter("outTransform",
              mat4({2, 0, 0, 0}, {0, 2, 0, 0}, {0, 0, 0, 0}, {-1, -1, 0, 1}));
          m->setParameterObject("anisotropyDirection", *sampler);
        }
        if (auto sampler = loadTexture(anisotropyTexture, true); sampler) {
          // Map red to red/green/blue as expected by our gltf pbr
          // implementation
          auto tx = getTextureUVTransform(
              AI_MATKEY_UVTRANSFORM(aiTextureType_ANISOTROPY, 0));
          sampler->setParameter("inTransform", tx);
          // - Strength is blue
          sampler->setParameter("outTransform",
              mat4({0, 0, 0, 0}, {0, 0, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 1}));
          m->setParameterObject("anisotropyStrength", *sampler);
        }
      } else
#endif
          if (ai_real anisotropyFactor;
              assimpMat->Get(AI_MATKEY_ANISOTROPY_FACTOR, anisotropyFactor)
              == AI_SUCCESS) {
        m->setParameter(
            "anisotropyStrength", ANARI_FLOAT32, &anisotropyFactor);
      }

#ifdef AI_MATKEY_ANISOTROPY_ROTATION
      // anisotropic rotation added with assimp 6.0
      if (ai_real anisotropyRotation;
          assimpMat->Get(AI_MATKEY_ANISOTROPY_ROTATION, anisotropyRotation)
          == AI_SUCCESS) {
        m->setParameter(
            "anisotropyRotation", ANARI_FLOAT32, &anisotropyRotation);
      }
#endif

      // Specular workflow
      if (aiColor3D specularColor;
          assimpMat->Get(AI_MATKEY_COLOR_SPECULAR, specularColor)
          == AI_SUCCESS) {
        m->setParameter("specularColor", ANARI_FLOAT32_VEC3, &specularColor);
      }
      if (ai_real specularFactor;
          assimpMat->Get(AI_MATKEY_SPECULAR_FACTOR, specularFactor)
          == AI_SUCCESS) {
        m->setParameter("specular", ANARI_FLOAT32, &specularFactor);
      }

      // Sheen handling
      if (aiString sheenColorTexture;
          assimpMat->GetTexture(
              AI_MATKEY_SHEEN_COLOR_TEXTURE, &sheenColorTexture)
          == AI_SUCCESS) {
        if (auto sampler = loadTexture(sheenColorTexture); sampler) {
          auto tx = getTextureUVTransform(
              AI_MATKEY_UVTRANSFORM(aiTextureType_SHEEN, 0));
          sampler->setParameter("inTransform", tx);
          m->setParameterObject("sheenColor", *sampler);
        }
      } else if (aiColor3D sheenColor;
          assimpMat->Get(AI_MATKEY_SHEEN_COLOR_FACTOR, sheenColor)
          == AI_SUCCESS) {
        m->setParameter("sheenColor", ANARI_FLOAT32_VEC3, &sheenColor);
      }

      if (aiString sheenRoughnessTexture;
          assimpMat->GetTexture(
              AI_MATKEY_SHEEN_ROUGHNESS_TEXTURE, &sheenRoughnessTexture)
          == AI_SUCCESS) {
        if (auto sampler = loadTexture(sheenRoughnessTexture, true); sampler) {
          auto tx = getTextureUVTransform(
              AI_MATKEY_UVTRANSFORM(aiTextureType_SHEEN, 1));
          sampler->setParameter("inTransform", tx);
          m->setParameterObject("sheenRoughness", *sampler);
        }
      } else if (ai_real sheenRoughnessFactor;
          assimpMat->Get(AI_MATKEY_SHEEN_ROUGHNESS_FACTOR, sheenRoughnessFactor)
          == AI_SUCCESS) {
        m->setParameter(
            "sheenRoughness", ANARI_FLOAT32, &sheenRoughnessFactor);
      }

      // Clearcoat handling
      if (aiString clearcoatTexture;
          assimpMat->GetTexture(AI_MATKEY_CLEARCOAT_TEXTURE, &clearcoatTexture)
          == AI_SUCCESS) {
        if (auto sampler = loadTexture(clearcoatTexture, true); sampler) {
          auto tx = getTextureUVTransform(
              AI_MATKEY_UVTRANSFORM(aiTextureType_CLEARCOAT, 0));
          sampler->setParameter("inTransform", tx);
          m->setParameterObject("clearcoat", *sampler);
        }
      } else if (ai_real clearcoatFactor;
          assimpMat->Get(AI_MATKEY_CLEARCOAT_FACTOR, clearcoatFactor)
          == AI_SUCCESS) {
        m->setParameter("clearcoat", ANARI_FLOAT32, &clearcoatFactor);
      }

      if (aiString clearcoatRoughnessTexture;
          assimpMat->GetTexture(
              AI_MATKEY_CLEARCOAT_ROUGHNESS_TEXTURE, &clearcoatRoughnessTexture)
          == AI_SUCCESS) {
        if (auto sampler = loadTexture(clearcoatRoughnessTexture, true);
            sampler) {
          auto tx = getTextureUVTransform(
              AI_MATKEY_UVTRANSFORM(aiTextureType_CLEARCOAT, 1));
          sampler->setParameter("inTransform", tx);
          m->setParameterObject("clearcoatRoughness", *sampler);
        }
      } else if (ai_real clearcoatRoughnessFactor;
          assimpMat->Get(
              AI_MATKEY_CLEARCOAT_ROUGHNESS_FACTOR, clearcoatRoughnessFactor)
          == AI_SUCCESS) {
        m->setParameter(
            "clearcoatRoughness", ANARI_FLOAT32, &clearcoatRoughnessFactor);
      }

      if (aiString clearcoatNormalTexture;
          assimpMat->GetTexture(
              AI_MATKEY_CLEARCOAT_NORMAL_TEXTURE, &clearcoatNormalTexture)
          == AI_SUCCESS) {
        if (auto sampler = loadTexture(clearcoatNormalTexture, true); sampler) {
          auto tx = getTextureUVTransform(
              AI_MATKEY_UVTRANSFORM(aiTextureType_CLEARCOAT, 2));
          sampler->setParameter("inTransform", tx);
          m->setParameterObject("clearcoatNormal", *sampler);
        }
      }

      // Emssive handling
      if (aiString emissiveTexture;
          assimpMat->GetTexture(aiTextureType_EMISSIVE, 0, &emissiveTexture)
          == AI_SUCCESS) {
        if (auto sampler = loadTexture(emissiveTexture); sampler) {
          auto tx = getTextureUVTransform(
              AI_MATKEY_UVTRANSFORM(aiTextureType_EMISSIVE, 0));
          sampler->setParameter("inTransform", tx);
          m->setParameterObject("emissive", *sampler);
        }
      } else if (aiColor3D emissiveColor;
          assimpMat->Get(AI_MATKEY_COLOR_EMISSIVE, emissiveColor)
          == AI_SUCCESS) {
        m->setParameter("emissive", ANARI_FLOAT32_VEC3, &emissiveColor);
      }

      // Opacity handling
      if (ai_real opacity;
          assimpMat->Get(AI_MATKEY_OPACITY, opacity) == AI_SUCCESS) {
        m->setParameter("opacity", ANARI_FLOAT32, &opacity);
      }

      // Occlusion handling
      if (aiString occlusionTexture;
          assimpMat->GetTexture(
              aiTextureType_AMBIENT_OCCLUSION, 0, &occlusionTexture)
          == AI_SUCCESS) {
        if (auto sampler = loadTexture(occlusionTexture, true); sampler) {
          auto tx = getTextureUVTransform(
              AI_MATKEY_UVTRANSFORM(aiTextureType_AMBIENT_OCCLUSION, 0));
          sampler->setParameter("inTransform", tx);
          m->setParameterObject("occlusion", *sampler);
        }
      }

      // Normal handling
      if (aiString normalTexture;
          assimpMat->GetTexture(aiTextureType_NORMALS, 0, &normalTexture)
          == AI_SUCCESS) {
        if (auto sampler = loadTexture(normalTexture, true); sampler) {
          auto tx = getTextureUVTransform(
              AI_MATKEY_UVTRANSFORM(aiTextureType_NORMALS, 0));
          sampler->setParameter("inTransform", tx);
          m->setParameterObject("normal", *sampler);
        }
      }

      // transmission handling
      if (ai_real transmissionFactor;
          assimpMat->Get(AI_MATKEY_TRANSMISSION_FACTOR, transmissionFactor)
          == AI_SUCCESS) {
        m->setParameter("transmission", ANARI_FLOAT32, &transmissionFactor);
      }
      if (aiString transmissionTexture;
          assimpMat->GetTexture(
              AI_MATKEY_TRANSMISSION_TEXTURE, &transmissionTexture)
          == AI_SUCCESS) {
        if (auto sampler = loadTexture(transmissionTexture, true); sampler) {
          auto tx = getTextureUVTransform(
              AI_MATKEY_UVTRANSFORM(aiTextureType_TRANSMISSION, 0));
          sampler->setParameter("inTransform", tx);
          m->setParameterObject("transmission", *sampler);
        }
      }
    } else { // GL-like dflt. material
      aiColor3D col;
      assimpMat->Get(AI_MATKEY_COLOR_DIFFUSE, col);
      ai_real opacity;
      assimpMat->Get(AI_MATKEY_OPACITY, opacity);

      m = scene.createObject<Material>(tokens::material::matte);
      m->setParameter("color", ANARI_FLOAT32_VEC3, &col);
      m->setParameter("opacity", opacity);
    }

    aiString name;
    assimpMat->Get(AI_MATKEY_NAME, name);
    m->setName(name.C_Str());

    materials.push_back(m);
  }

  return materials;
}

static std::vector<LightRef> importASSIMPLights(
    Scene &scene, const aiScene *a_scene)
{
  std::vector<LightRef> lights;

  for (unsigned i = 0; i < a_scene->mNumLights; ++i) {
    aiLight *assimpLight = a_scene->mLights[i];
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
      lightRef = scene.createObject<Light>(tokens::light::directional);
      lightRef->setParameter("direction",
          tsd::math::float3(assimpLight->mDirection.x,
              assimpLight->mDirection.y,
              assimpLight->mDirection.z));
      lightRef->setParameter("intensity", intensity);
      break;
    case aiLightSource_POINT:
      lightRef = scene.createObject<Light>(tokens::light::point);
      lightRef->setParameter("position",
          tsd::math::float3(assimpLight->mPosition.x,
              assimpLight->mPosition.y,
              assimpLight->mPosition.z));
      lightRef->setParameter("intensity", intensity);
      break;
    case aiLightSource_SPOT:
      lightRef = scene.createObject<Light>(tokens::light::spot);
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

static void populateASSIMPLayer(Scene &scene,
    LayerNodeRef tsdLayerRef,
    const std::vector<SurfaceRef> &surfaces,
    const std::vector<LightRef> &lights,
    const aiNode *node)
{
  static_assert(sizeof(tsd::math::mat4) == sizeof(aiMatrix4x4),
      "matrix type size mismatch");
  tsd::math::mat4 mat;
  std::memcpy(&mat, &node->mTransformation, sizeof(mat));
  mat = tsd::math::transpose(mat);
  auto tr = tsdLayerRef->insert_last_child({mat, node->mName.C_Str()});

  for (unsigned int i = 0; i < node->mNumMeshes; i++) {
    auto mesh = surfaces.at(node->mMeshes[i]);
    tr->insert_last_child({mesh, mesh->name().c_str()});
  }

  // https://github.com/assimp/assimp/issues/1168#issuecomment-278673292
  // We won't find the light directly on the node, but matching names
  // indicate we're supposed to associate the light with the transform
  std::string name(node->mName.C_Str());
  auto it = std::find_if(lights.begin(),
      lights.end(),
      [name](const LightRef &lightRef) { return lightRef->name() == name; });

  if (it != lights.end())
    tr->insert_first_child({ANARI_LIGHT, (*it)->index(), &scene});

  for (unsigned int i = 0; i < node->mNumChildren; i++)
    populateASSIMPLayer(scene, tr, surfaces, lights, node->mChildren[i]);
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void import_ASSIMP(
    Scene &scene, const char *filename, LayerNodeRef location, bool flatten)
{
  Assimp::DefaultLogger::create("", Assimp::Logger::VERBOSE);

  Assimp::Importer importer;

  auto importFlags = aiProcess_Triangulate | aiProcess_JoinIdenticalVertices
      | aiProcess_FlipUVs;
  if (flatten)
    importFlags |= aiProcess_PreTransformVertices;

  const aiScene *a_scene = importer.ReadFile(filename, importFlags);

  if (a_scene == nullptr) {
    Assimp::DefaultLogger::get()->error(importer.GetErrorString());
    return;
  }

  auto lights = importASSIMPLights(scene, a_scene);
  auto materials = importASSIMPMaterials(scene, a_scene, filename);
  auto meshes = importASSIMPSurfaces(scene, materials, a_scene);

  populateASSIMPLayer(scene,
      location ? location : scene.defaultLayer()->root(),
      meshes,
      lights,
      a_scene->mRootNode);
}
#else
void import_ASSIMP(
    Scene &scene, const char *filename, LayerNodeRef location, bool flatten)
{
  logError("[import_ASSIMP] ASSIMP not enabled in TSD build.");
}
#endif

} // namespace tsd::io
