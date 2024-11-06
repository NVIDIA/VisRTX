// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/authoring/importers.hpp"
#include "tsd/authoring/importers/detail/importer_common.hpp"
// assimp
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/DefaultLogger.hpp>
#include <assimp/Importer.hpp>

namespace tsd {

static IndexedVectorRef<Sampler> importEmbeddedTexture(
    Context &ctx, const aiTexture *embeddedTexture, TextureCache &cache)
{
  std::string filepath = embeddedTexture->mFilename.C_Str();
  const bool validTexture =
      embeddedTexture->mHeight != 0 && embeddedTexture->pcData != nullptr;
#if 0
  printf("EMBEDDED '%s' TEXTURE VALID: %i HEIGHT: %i\n",
      filepath.c_str(),
      int(validTexture),
      int(embeddedTexture->mHeight));
#endif

  auto tex = cache[filepath];

  if (!tex && validTexture) {
    tex = ctx.createObject<Sampler>(tokens::sampler::image2D);

    auto dataArray = ctx.createArray(
        ANARI_UFIXED8_VEC4, embeddedTexture->mWidth, embeddedTexture->mHeight);
    dataArray->setData(embeddedTexture->pcData);

    tex->setParameterObject("image"_t, *dataArray);
    tex->setParameter("inAttribute"_t, "attribute0");
    tex->setParameter("wrapMode1"_t, "repeat");
    tex->setParameter("wrapMode2"_t, "repeat");
    tex->setParameter("filter"_t, "linear");
    tex->setName(fileOf(filepath).c_str());

    cache[filepath] = tex;
  }

  return tex;
}

static std::vector<IndexedVectorRef<Surface>> importASSIMPSurfaces(Context &ctx,
    const std::vector<IndexedVectorRef<Material>> &materials,
    const aiScene *scene)
{
  std::vector<IndexedVectorRef<Surface>> tsdMeshes;

  for (unsigned i = 0; i < scene->mNumMeshes; ++i) {
    aiMesh *mesh = scene->mMeshes[i];

    auto tsdMesh = ctx.createObject<Geometry>(tokens::geometry::triangle);

    unsigned numVertices = mesh->mNumVertices;
    auto vertexPositionArray = ctx.createArray(ANARI_FLOAT32_VEC3, numVertices);
    auto *outVertices = vertexPositionArray->mapAs<float3>();

    auto vertexNormalArray = ctx.createArray(
        ANARI_FLOAT32_VEC3, mesh->HasNormals() ? numVertices : 0);
    auto outNormals = vertexNormalArray->mapAs<float3>();

    auto vertexTexCoordArray = ctx.createArray(ANARI_FLOAT32_VEC3,
        mesh->HasTextureCoords(0 /*texcord set*/) ? numVertices : 0);
    auto outTexCoords = vertexTexCoordArray->mapAs<float3>();

    auto vertexTangentArray = ctx.createArray(
        ANARI_FLOAT32_VEC4, mesh->HasTangentsAndBitangents() ? numVertices : 0);
    auto outTangents = vertexTangentArray->mapAs<float4>();

    // TODO: test for AI_MAX_NUMBER_OF_COLOR_SETS, import all..
    auto vertexColorArray =
        ctx.createArray(ANARI_FLOAT32_VEC4, mesh->mColors[0] ? numVertices : 0);
    auto *outColors = vertexColorArray->mapAs<float4>();

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
        auto nCrossT = n ^ tng;
        aiVector3D hndVec = btng / nCrossT;

        if (!isfinite(hndVec.x)) {
          // happens on rare occasions, try this way combination:
          auto tCrossB = tng ^ btng;
          hndVec = n / tCrossB;
        }

        if (!isfinite(hndVec.x)) {
          // and this one:
          auto bCrossN = btng ^ n;
          hndVec = tng / bCrossN;
        }

        assert(isfinite(hndVec.x) && isfinite(hndVec.y) && isfinite(hndVec.z));

        // should all be same
        constexpr static float eps = 1e-2f;
        assert(fabsf(hndVec.x - hndVec.y) < eps
            && fabsf(hndVec.x - hndVec.z) < eps);

        // should be close to +/-1
        float handedness = hndVec.x;
        assert(handedness < -1.f + eps || handedness > 1.f - eps);

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

    tsdMesh->setName((std::string(mesh->mName.C_Str()) + "_geometry").c_str());

    unsigned matID = mesh->mMaterialIndex;
    auto tsdMat = matID < 0 ? ctx.defaultMaterial() : materials[size_t(matID)];
    tsdMeshes.push_back(
        ctx.createSurface(mesh->mName.C_Str(), tsdMesh, tsdMat));
  }

  return tsdMeshes;
}

static std::vector<IndexedVectorRef<Material>> importASSIMPMaterials(
    Context &ctx, const aiScene *scene, const std::string &filename)
{
  std::vector<IndexedVectorRef<Material>> materials;

  TextureCache cache;

  std::string basePath = pathOf(filename);

  for (unsigned i = 0; i < scene->mNumMaterials; ++i) {
    aiMaterial *assimpMat = scene->mMaterials[i];
    ai_int matType;
    assimpMat->Get(AI_MATKEY_SHADING_MODEL, matType);

    IndexedVectorRef<Material> m;

    if (matType == aiShadingMode_PBR_BRDF) {
      aiColor3D baseColor;
      ai_real metallic, roughness;
      aiString baseColorTexture, metallicTexture, roughnessTexture,
          metallicRoughnessTexture, normalTexture;

      assimpMat->Get(AI_MATKEY_BASE_COLOR, baseColor);
      assimpMat->Get(AI_MATKEY_METALLIC_FACTOR, metallic);
      assimpMat->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness);

      assimpMat->GetTexture(AI_MATKEY_BASE_COLOR_TEXTURE, &baseColorTexture);
      assimpMat->GetTexture(AI_MATKEY_METALLIC_TEXTURE, &metallicTexture);
      assimpMat->GetTexture(AI_MATKEY_ROUGHNESS_TEXTURE, &roughnessTexture);
      assimpMat->GetTexture(AI_MATKEY_ROUGHNESS_TEXTURE, &roughnessTexture);
      // glTF has a combined metallic/roughness texture
      assimpMat->GetTexture(aiTextureType_UNKNOWN,
          0,
          &metallicRoughnessTexture,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr);
      // glTF: base color is stored in "diffuse"
      if (baseColorTexture.length != 0) {
        assimpMat->GetTexture(aiTextureType_DIFFUSE,
            0,
            &baseColorTexture,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr);
      }
      // normals:
      assimpMat->GetTexture(aiTextureType_NORMALS,
          0,
          &normalTexture,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr);

      m = ctx.createObject<Material>(tokens::material::physicallyBased);
      m->setParameter("baseColor"_t, ANARI_FLOAT32_VEC3, &baseColor);
      m->setParameter("metallic"_t, ANARI_FLOAT32, &metallic);
      m->setParameter("roughness"_t, ANARI_FLOAT32, &roughness);

      auto isEmbedded = [](const char *name) { return name[0] == '*'; };

      auto loadTexture = [&](const char *paramName, const aiString &texName) {
        if (texName.length != 0) {
          IndexedVectorRef<Sampler> tex;
          auto *embeddedTexture = scene->GetEmbeddedTexture(texName.C_Str());
          if (embeddedTexture)
            tex = importEmbeddedTexture(ctx, embeddedTexture, cache);
          else
            tex = importTexture(ctx, basePath + texName.C_Str(), cache);

          if (tex)
            m->setParameterObject(Token(paramName), *tex);
        }
      };

      // PBR textures:
      loadTexture("baseColor", baseColorTexture);
      loadTexture("metallic", metallicTexture);
      loadTexture("roughness", roughnessTexture);
      loadTexture("normal", normalTexture);

      if (metallicRoughnessTexture.length != 0) {
        auto texG = importTexture(
            ctx, basePath + metallicRoughnessTexture.C_Str(), cache);
        auto texB = importTexture(
            ctx, basePath + metallicRoughnessTexture.C_Str(), cache);
        if (texG && texB) {
          texG->setParameter("outTransform"_t,
              transpose(mat4(
                  {0, 1, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 1})));
          texB->setParameter("outTransform"_t,
              transpose(mat4(
                  {0, 0, 1, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 1})));

          m->setParameterObject("roughness"_t, *texG);
          m->setParameterObject("metallic"_t, *texB);
        }
      }
    } else { // GL-like dflt. material
      aiColor3D col;
      assimpMat->Get(AI_MATKEY_COLOR_DIFFUSE, col);

      m = ctx.createObject<Material>(tokens::material::matte);
      m->setParameter("color"_t, ANARI_FLOAT32_VEC3, &col);
      // m->setParameter("opacity"_t, mat.dissolve);
    }

    aiString name;
    assimpMat->Get(AI_MATKEY_NAME, name);
    m->setName(name.C_Str());

    materials.push_back(m);
  }

  return materials;
}

static std::vector<IndexedVectorRef<Light>> importASSIMPLights(
    Context &ctx, const aiScene *scene)
{
  std::vector<IndexedVectorRef<Light>> lights;

  for (unsigned i = 0; i < scene->mNumLights; ++i) {
    aiLight *assimpLight = scene->mLights[i];
    // TODO:
    switch (assimpLight->mType) {
    case aiLightSource_POINT:
      break;
    default:
      break;
    }
  }

  return lights;
}

static void populateASSIMPInstanceTree(Context &ctx,
    InstanceNodeRef tsdTreeRef,
    const std::vector<IndexedVectorRef<Surface>> &surfaces,
    const aiNode *node)
{
  static_assert(
      sizeof(tsd::mat4) == sizeof(aiMatrix4x4), "matrix type size mismatch");
  tsd::mat4 mat;
  std::memcpy(&mat, &node->mTransformation, sizeof(mat));
  mat = tsd::math::transpose(mat);
  auto tr = ctx.tree.insert_last_child(tsdTreeRef, {mat, node->mName.C_Str()});

  for (unsigned int i = 0; i < node->mNumMeshes; i++) {
    auto mesh = surfaces.at(node->mMeshes[i]);
    ctx.tree.insert_last_child(
        tr, {utility::Any(ANARI_SURFACE, mesh.index()), mesh->name().c_str()});
  }

  for (unsigned int i = 0; i < node->mNumChildren; i++)
    populateASSIMPInstanceTree(ctx, tr, surfaces, node->mChildren[i]);
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void import_ASSIMP(Context &ctx, const char *filename, bool flatten)
{
  Assimp::DefaultLogger::create("", Assimp::Logger::VERBOSE);

  Assimp::Importer importer;

  auto importFlags = aiProcess_Triangulate | aiProcess_JoinIdenticalVertices;
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

  populateASSIMPInstanceTree(ctx, ctx.tree.root(), meshes, scene->mRootNode);
}

} // namespace tsd
