// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd_io
#include <tsd/io/importers/detail/importer_common.hpp>
// tiny_obj_loader
#include "tiny_obj_loader.h"
// std
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, const char *argv[])
{
  if (argc < 3) {
    printf("usage: ./obj2header file.obj outfile.h\n");
    return 1;
  }

  std::string inputFile = argv[1];
  std::string outputFile = argv[2];

  printf("Loading file '%s'...", inputFile.c_str());
  fflush(stdout);

  struct
  {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
  } objdata;

  std::string warn;
  std::string err;
  std::string basePath = tsd::io::pathOf(inputFile);
  std::string file = tsd::io::fileOf(inputFile);

  auto retval = tinyobj::LoadObj(&objdata.attrib,
      &objdata.shapes,
      &objdata.materials,
      &warn,
      &err,
      inputFile.c_str(),
      basePath.c_str(),
      true);

  if (!retval) {
    printf("\nERROR: failed to open/parse obj file");
    return 1;
  }

  printf("done!\n");

  printf("Reformatting data to ANARI's spec...");
  fflush(stdout);

  std::vector<tsd::math::float3> v;
  std::vector<tsd::math::float3> vn;
  std::vector<tsd::math::float2> vt;

  v.reserve(objdata.attrib.vertices.size());
  vn.reserve(objdata.attrib.normals.size());
  vt.reserve(objdata.attrib.texcoords.size());

  for (auto &shape : objdata.shapes) {
    size_t numIndices = shape.mesh.indices.size();
    if (numIndices == 0)
      continue;

    for (size_t i = 0; i < numIndices; i += 3) {
      auto vi0 = shape.mesh.indices[i + 0].vertex_index;
      auto vi1 = shape.mesh.indices[i + 1].vertex_index;
      auto vi2 = shape.mesh.indices[i + 2].vertex_index;

      auto &v_in = objdata.attrib.vertices;
      auto v0 = tsd::math::float3(
          v_in[vi0 * 3 + 0], v_in[vi0 * 3 + 1], v_in[vi0 * 3 + 2]);
      auto v1 = tsd::math::float3(
          v_in[vi1 * 3 + 0], v_in[vi1 * 3 + 1], v_in[vi1 * 3 + 2]);
      auto v2 = tsd::math::float3(
          v_in[vi2 * 3 + 0], v_in[vi2 * 3 + 1], v_in[vi2 * 3 + 2]);

      v.push_back(v0);
      v.push_back(v1);
      v.push_back(v2);
    }

    for (size_t i = 0; i < numIndices; i += 3) {
      auto vni0 = shape.mesh.indices[i + 0].normal_index;
      auto vni1 = shape.mesh.indices[i + 1].normal_index;
      auto vni2 = shape.mesh.indices[i + 2].normal_index;

      auto &vn_in = objdata.attrib.normals;
      auto vn0 = vni0 < 0
          ? tsd::math::float3(0.f)
          : tsd::math::float3(
                vn_in[vni0 * 3 + 0], vn_in[vni0 * 3 + 1], vn_in[vni0 * 3 + 2]);
      auto vn1 = vni1 < 0
          ? tsd::math::float3(0.f)
          : tsd::math::float3(
                vn_in[vni1 * 3 + 0], vn_in[vni1 * 3 + 1], vn_in[vni1 * 3 + 2]);
      auto vn2 = vni2 < 0
          ? tsd::math::float3(0.f)
          : tsd::math::float3(
                vn_in[vni2 * 3 + 0], vn_in[vni2 * 3 + 1], vn_in[vni2 * 3 + 2]);

      vn.push_back(vn0);
      vn.push_back(vn1);
      vn.push_back(vn2);
    }

    for (size_t i = 0; i < numIndices; i += 3) {
      auto vti0 = shape.mesh.indices[i + 0].texcoord_index;
      auto vti1 = shape.mesh.indices[i + 1].texcoord_index;
      auto vti2 = shape.mesh.indices[i + 2].texcoord_index;

      auto &tc_in = objdata.attrib.texcoords;
      auto vt0 = vti0 < 0
          ? tsd::math::float2(0.f)
          : tsd::math::float2(tc_in[vti0 * 2 + 0], tc_in[vti0 * 2 + 1]);
      auto vt1 = vti1 < 0
          ? tsd::math::float2(0.f)
          : tsd::math::float2(tc_in[vti1 * 2 + 0], tc_in[vti1 * 2 + 1]);
      auto vt2 = vti2 < 0
          ? tsd::math::float2(0.f)
          : tsd::math::float2(tc_in[vti2 * 2 + 0], tc_in[vti2 * 2 + 1]);

      vt.push_back(vt0);
      vt.push_back(vt1);
      vt.push_back(vt2);
    }
  }

  printf("done!\n");

  printf("Writing file '%s'...", outputFile.c_str());
  fflush(stdout);

  auto *fout = std::fopen(outputFile.c_str(), "w");

  if (!fout) {
    printf("\nERROR: failed to open output file for writing");
    return 1;
  }

  fprintf(fout, "// Copyright 2024-2025 NVIDIA Corporation\n");
  fprintf(fout, "// SPDX-License-Identifier: Apache-2.0\n");
  fprintf(fout, "\n");
  fprintf(fout, "#include <cstdint>\n");
  fprintf(fout, "\n");
  fprintf(fout, "namespace obj2header {\n"); // namespace

  fprintf(fout, "\n");
  fprintf(fout, "// clang-format off\n");
  fprintf(fout, "\n");

  // Vertex array //

  fprintf(fout, "static float vertex_position[] = {\n");
  for (auto &a : v)
    fprintf(fout, "    %ff, %ff, %ff,\n", a.x, a.y, a.z);
  fprintf(fout, "}; // vertex_position\n"); // vertices

  fprintf(fout, "\n");

  // Normals array //

  fprintf(fout, "static float vertex_normal[] = {\n");
  for (auto &a : vn)
    fprintf(fout, "    %ff, %ff, %ff,\n", a.x, a.y, a.z);
  fprintf(fout, "}; // vertex_normal\n"); // vertices

  fprintf(fout, "\n");

  // Texcoord array //

  fprintf(fout, "static float vertex_uv[] = {\n");
  for (auto &a : vt)
    fprintf(fout, "    %ff, %ff,\n", a.x, a.y);
  fprintf(fout, "}; // vertex_uv\n"); // vertices

  fprintf(fout, "\n");
  fprintf(fout, "// clang-format on\n");
  fprintf(fout, "\n");

  fprintf(fout, "} // namespace obj2header\n");

  std::fclose(fout);

  printf("done!\n");

  return 0;
}
