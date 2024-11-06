// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/authoring/importers.hpp"
#include "tsd/authoring/importers/detail/importer_common.hpp"
// tiny_obj_importer
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

namespace tsd {

struct OBJData
{
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
};

void import_OBJ(Context &ctx, const char *filepath, bool useDefaultMaterial)
{
  OBJData objdata;
  std::string warn;
  std::string err;
  std::string basePath = pathOf(filepath);
  std::string file = fileOf(filepath);

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

  auto obj_root = ctx.tree.insert_last_child(
      ctx.tree.root(), {tsd::mat4(tsd::math::identity), file.c_str()});

  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////

  std::vector<IndexedVectorRef<Material>> materials;

  TextureCache cache;

  for (auto &mat : objdata.materials) {
    auto m = ctx.createObject<Material>(tokens::material::matte);

    m->setParameter("color"_t, ANARI_FLOAT32_VEC3, mat.diffuse);
    m->setParameter("opacity"_t, mat.dissolve);
    m->setName(mat.name.c_str());

    if (!mat.diffuse_texname.empty()) {
      auto tex = importTexture(ctx, basePath + mat.diffuse_texname, cache);
      if (tex)
        m->setParameterObject("color"_t, *tex);
    }

    // TODO other textures

    materials.push_back(m);
  }

  /////////////////////////////////////////////////////////////////////////////

  auto *vertices = objdata.attrib.vertices.data();
  auto *texcoords = objdata.attrib.texcoords.data();

  for (auto &shape : objdata.shapes) {
    size_t numIndices = shape.mesh.indices.size();
    if (numIndices == 0)
      continue;

    size_t numVertices = numIndices * 3;

    auto mesh = ctx.createObject<Geometry>(tokens::geometry::triangle);

    auto vertexPositionArray = ctx.createArray(ANARI_FLOAT32_VEC3, numVertices);
    auto *outVertices = vertexPositionArray->mapAs<float3>();

    float2 *outTexcoords = nullptr;
    IndexedVectorRef<Array> texcoordArray;

    if (texcoords) {
      texcoordArray = ctx.createArray(ANARI_FLOAT32_VEC2, numVertices);
      outTexcoords = texcoordArray->mapAs<float2>();
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
    }

    vertexPositionArray->unmap();
    mesh->setParameterObject("vertex.position"_t, *vertexPositionArray);

    if (texcoords) {
      texcoordArray->unmap();
      mesh->setParameterObject("vertex.attribute0"_t, *texcoordArray);
    }

    auto name = shape.name;
    if (name.empty())
      name = "<unnamed_mesh>";
    mesh->setName(name.c_str());

    const int matID = shape.mesh.material_ids[0];
    const bool useDefault = useDefaultMaterial || matID < 0;
    auto mat = useDefault ? ctx.defaultMaterial() : materials[size_t(matID)];

    auto surface = ctx.createSurface(name.c_str(), mesh, mat);
    ctx.tree.insert_last_child(
        obj_root, utility::Any(ANARI_SURFACE, surface.index()));
  }
}

} // namespace tsd
