// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/authoring/importers.hpp"
#include "tsd/authoring/importers/detail/importer_common.hpp"
// tinyply
#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"
// std
#include <fstream>

namespace tsd {

using namespace tinyply;

void import_PLY(Context &ctx, const char *filename)
{
  std::unique_ptr<std::istream> file_stream;
  std::vector<uint8_t> byte_buffer;

  try {
    file_stream.reset(new std::ifstream(filename, std::ios::binary));

    if (!file_stream || file_stream->fail()) {
      throw std::runtime_error(
          "file_stream failed to open " + std::string(filename));
    }

    file_stream->seekg(0, std::ios::end);
    const float size_mb = file_stream->tellg() * float(1e-6);
    file_stream->seekg(0, std::ios::beg);

    PlyFile file;
    file.parse_header(*file_stream);

    std::cout << "\t[ply_header] Type: "
              << (file.is_binary_file() ? "binary" : "ascii") << std::endl;
    for (const auto &c : file.get_comments())
      std::cout << "\t[ply_header] Comment: " << c << std::endl;
    for (const auto &c : file.get_info())
      std::cout << "\t[ply_header] Info: " << c << std::endl;

    for (const auto &e : file.get_elements()) {
      std::cout << "\t[ply_header] element: " << e.name << " (" << e.size << ")"
                << std::endl;
      for (const auto &p : e.properties) {
        std::cout << "\t[ply_header] \tproperty: " << p.name
                  << " (type=" << tinyply::PropertyTable[p.propertyType].str
                  << ")";
        if (p.isList)
          std::cout << " (list_type=" << tinyply::PropertyTable[p.listType].str
                    << ")";
        std::cout << std::endl;
      }
    }

    // Because most people have their own mesh types, tinyply treats parsed data
    // as structured/typed byte buffers. See examples below on how to marry your
    // own application-specific data structures with this one.
    std::shared_ptr<tinyply::PlyData> vertices, normals, colors, texcoords,
        faces, tripstrip;

    // The header information can be used to programmatically extract properties
    // on elements known to exist in the header prior to reading the data. For
    // brevity of this sample, properties like vertex position are hard-coded:
    try {
      vertices =
          file.request_properties_from_element("vertex", {"x", "y", "z"});
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    try {
      normals =
          file.request_properties_from_element("vertex", {"nx", "ny", "nz"});
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    try {
      colors = file.request_properties_from_element(
          "vertex", {"red", "green", "blue", "alpha"});
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    if (!colors) {
      try {
        colors = file.request_properties_from_element(
            "vertex", {"r", "g", "b", "a"});
      } catch (const std::exception &e) {
        std::cerr << "tinyply exception: " << e.what() << std::endl;
      }
    }

    try {
      texcoords = file.request_properties_from_element("vertex", {"u", "v"});
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    // Providing a list size hint (the last argument) is a 2x performance
    // improvement. If you have arbitrary ply files, it is best to leave this 0.
    try {
      faces =
          file.request_properties_from_element("face", {"vertex_indices"}, 3);
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    // Tristrips must always be read with a 0 list size hint (unless you know
    // exactly how many elements are specifically in the file, which is
    // unlikely);
    try {
      tripstrip = file.request_properties_from_element(
          "tristrips", {"vertex_indices"}, 0);
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    file.read(*file_stream);

    if (vertices)
      std::cout << "\tRead " << vertices->count << " total vertices "
                << std::endl;
    if (normals)
      std::cout << "\tRead " << normals->count << " total vertex normals "
                << std::endl;
    if (colors)
      std::cout << "\tRead " << colors->count << " total vertex colors "
                << std::endl;
    if (texcoords)
      std::cout << "\tRead " << texcoords->count << " total vertex texcoords "
                << std::endl;
    if (faces)
      std::cout << "\tRead " << faces->count << " total faces (triangles) "
                << std::endl;
    if (tripstrip)
      std::cout << "\tRead "
                << (tripstrip->buffer.size_bytes()
                       / tinyply::PropertyTable[tripstrip->t].stride)
                << " total indicies (tristrip) " << std::endl;

    if (!vertices || vertices->t == tinyply::Type::FLOAT64)
      return;

    ///////////////////////////////////////////////////////////////////////////

    auto objectName = fileOf(std::string(filename)) + " (PLY file)";

    // Material //

    auto mat = ctx.createObject<Material>(tokens::material::matte);
    mat->setParameter("color"_t, float3(0.8f));
    mat->setParameter("opacity"_t, 1.f);
    mat->setParameter("alphaMode"_t, "opaque");
    mat->parameter("alphaMode"_t)->setStringSelection(0);
    mat->setName((objectName + " material").c_str());

    // Mesh //

    auto ply_root =
        ctx.insertChildNode(ctx.tree.root(), fileOf(filename).c_str());
    auto mesh = ctx.createObject<Geometry>(tokens::geometry::triangle);

    auto makeArray1DForMesh = [&](Token parameterName,
                                  anari::DataType type,
                                  const void *ptr,
                                  size_t size) {
      auto arr = ctx.createArray(type, size);
      arr->setData(ptr);
      mesh->setParameterObject(parameterName, *arr);
    };

    makeArray1DForMesh("vertex.position"_t,
        ANARI_FLOAT32_VEC3,
        vertices->buffer.get(),
        vertices->count);

    if (normals && normals->t == tinyply::Type::FLOAT32) {
      makeArray1DForMesh("vertex.normal"_t,
          ANARI_FLOAT32_VEC3,
          normals->buffer.get(),
          normals->count);
    }

    if (colors && colors->t == tinyply::Type::UINT8) {
      makeArray1DForMesh("vertex.color"_t,
          ANARI_UFIXED8_VEC4,
          colors->buffer.get(),
          colors->count);
      mat->setParameter("color"_t, "color");
    }

    if (faces) {
      makeArray1DForMesh("primitive.index"_t,
          ANARI_UINT32_VEC3,
          faces->buffer.get(),
          faces->count);
    }

    mesh->setName((objectName + "_mesh").c_str());

    auto surface = ctx.createSurface(objectName.c_str(), mesh, mat);
    ctx.tree.insert_last_child(
        ply_root, utility::Any(ANARI_SURFACE, surface.index()));

  } catch (const std::exception &e) {
    std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
  }
}

} // namespace tsd