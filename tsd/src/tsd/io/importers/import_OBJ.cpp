// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
// tiny_obj_importer
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

namespace tsd::io {

using namespace tsd::core;

struct OBJData
{
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
};

void import_OBJ(Scene &scene,
    const char *filepath,
    LayerNodeRef location,
    bool useDefaultMaterial)
{
  OBJData objdata;
  std::string warn;
  std::string err;
  const std::string basePath = pathOf(filepath);
  const std::string file = fileOf(filepath);

  auto retval = tinyobj::LoadObj(&objdata.attrib,
      &objdata.shapes,
      &objdata.materials,
      &warn,
      &err,
      filepath,
      basePath.c_str(),
      true);

  if (!retval) {
    std::stringstream ss;
    ss << "failed to open/parse obj file '" << filepath << "'";
    return;
  }

  auto obj_root = scene.insertChildNode(
      location ? location : scene.defaultLayer()->root(), file.c_str());

  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////

  std::vector<MaterialRef> materials;
  materials.resize(objdata.materials.size());

  TextureCache cache;

  auto getMaterial = [&](size_t i) -> MaterialRef {
    auto &m = materials[i];
    if (!m) {
      auto &mat = objdata.materials[i];

      m = scene.createObject<Material>(tokens::material::matte);
      m->setParameter("color", ANARI_FLOAT32_VEC3, mat.diffuse);
      m->setParameter("opacity", mat.dissolve);
      m->setName(mat.name.c_str());

      if (!mat.diffuse_texname.empty()) {
        auto tex = importTexture(scene, basePath + mat.diffuse_texname, cache);
        if (tex)
          m->setParameterObject("color", *tex);
      }
    }

    return m;
  };

  /////////////////////////////////////////////////////////////////////////////

  auto *vertices = objdata.attrib.vertices.data();
  auto *texcoords = objdata.attrib.texcoords.data();
  auto *normals = objdata.attrib.normals.data();

  for (auto &shape : objdata.shapes) {
    const size_t numIndices = shape.mesh.indices.size();
    if (numIndices == 0)
      continue;

    const size_t numVertices = numIndices;

    auto mesh = scene.createObject<Geometry>(tokens::geometry::triangle);

    auto vertexPositionArray = scene.createArray(ANARI_FLOAT32_VEC3, numVertices);
    auto *outVertices = vertexPositionArray->mapAs<float3>();

    float2 *outTexcoords = nullptr;
    ArrayRef texcoordArray;
    if (texcoords) {
      texcoordArray = scene.createArray(ANARI_FLOAT32_VEC2, numVertices);
      outTexcoords = texcoordArray->mapAs<float2>();
    }

    float3 *outNormals = nullptr;
    ArrayRef normalsArray;
    if (normals) {
      normalsArray = scene.createArray(ANARI_FLOAT32_VEC3, numVertices);
      outNormals = normalsArray->mapAs<float3>();
    }

    for (size_t i = 0; i < numIndices; i += 3) {
      const auto i0 = shape.mesh.indices[i + 0].vertex_index;
      const auto i1 = shape.mesh.indices[i + 1].vertex_index;
      const auto i2 = shape.mesh.indices[i + 2].vertex_index;
      const auto *v0 = vertices + (i0 * 3);
      const auto *v1 = vertices + (i1 * 3);
      const auto *v2 = vertices + (i2 * 3);
      outVertices[i + 0] = float3(v0[0], v0[1], v0[2]);
      outVertices[i + 1] = float3(v1[0], v1[1], v1[2]);
      outVertices[i + 2] = float3(v2[0], v2[1], v2[2]);

      if (texcoords) {
        const auto ti0 = shape.mesh.indices[i + 0].texcoord_index;
        const auto ti1 = shape.mesh.indices[i + 1].texcoord_index;
        const auto ti2 = shape.mesh.indices[i + 2].texcoord_index;
        const auto *t0 = texcoords + (ti0 * 2);
        const auto *t1 = texcoords + (ti1 * 2);
        const auto *t2 = texcoords + (ti2 * 2);
        outTexcoords[i + 0] = ti0 >= 0 ? float2(t0[0], t0[1]) : float2(0.f);
        outTexcoords[i + 1] = ti1 >= 0 ? float2(t1[0], t1[1]) : float2(0.f);
        outTexcoords[i + 2] = ti2 >= 0 ? float2(t2[0], t2[1]) : float2(0.f);
      }

      if (normals) {
        const auto ni0 = shape.mesh.indices[i + 0].normal_index;
        const auto ni1 = shape.mesh.indices[i + 1].normal_index;
        const auto ni2 = shape.mesh.indices[i + 2].normal_index;
        const auto *n0 = normals + (ni0 * 3);
        const auto *n1 = normals + (ni1 * 3);
        const auto *n2 = normals + (ni2 * 3);
        outNormals[i + 0] =
            ni0 >= 0 ? float3(n0[0], n0[1], n0[2]) : float3(1.f);
        outNormals[i + 1] =
            ni1 >= 0 ? float3(n1[0], n1[1], n1[2]) : float3(1.f);
        outNormals[i + 2] =
            ni2 >= 0 ? float3(n2[0], n2[1], n2[2]) : float3(1.f);
      }
    }

    vertexPositionArray->unmap();
    mesh->setParameterObject("vertex.position", *vertexPositionArray);

    if (texcoords) {
      texcoordArray->unmap();
      mesh->setParameterObject("vertex.attribute0", *texcoordArray);
    }

    if (normals) {
      normalsArray->unmap();
      mesh->setParameterObject("vertex.normal", *normalsArray);
    }

    auto name = shape.name;
    if (name.empty())
      name = "<unnamed_mesh>";
    mesh->setName(name.c_str());

    const int matID = shape.mesh.material_ids[0];
    const bool useDefault = useDefaultMaterial || matID < 0;
    auto mat = useDefault ? scene.defaultMaterial() : getMaterial(size_t(matID));

    auto surface = scene.createSurface(name.c_str(), mesh, mat);
    scene.insertChildObjectNode(obj_root, surface);
  }
}

} // namespace tsd
