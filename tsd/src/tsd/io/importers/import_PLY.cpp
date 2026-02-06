// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/Logging.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
// tinyply
#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"
// std
#include <fstream>

namespace tsd::io {

using namespace tinyply;

void import_PLY(Scene &scene, const char *filename, LayerNodeRef location)
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

#if 0
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
#endif

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
      logError("[import_PLY] tinyply exception: %s", e.what());
    }

    try {
      normals =
          file.request_properties_from_element("vertex", {"nx", "ny", "nz"});
    } catch (const std::exception &e) {
      logError("[import_PLY] tinyply exception: %s", e.what());
    }

    try {
      colors = file.request_properties_from_element(
          "vertex", {"red", "green", "blue", "alpha"});
    } catch (const std::exception &e) {
      logError("[import_PLY] tinyply exception: %s", e.what());
    }

    if (!colors) {
      try {
        colors = file.request_properties_from_element(
            "vertex", {"r", "g", "b", "a"});
      } catch (const std::exception &e) {
        logError("[import_PLY] tinyply exception: %s", e.what());
      }
    }

    try {
      texcoords = file.request_properties_from_element("vertex", {"u", "v"});
    } catch (const std::exception &e) {
      logError("[import_PLY] tinyply exception: %s", e.what());
    }

    // Providing a list size hint (the last argument) is a 2x performance
    // improvement. If you have arbitrary ply files, it is best to leave this 0.
    try {
      faces =
          file.request_properties_from_element("face", {"vertex_indices"}, 3);
    } catch (const std::exception &e) {
      logError("[import_PLY] tinyply exception: %s", e.what());
    }

    // Tristrips must always be read with a 0 list size hint (unless you know
    // exactly how many elements are specifically in the file, which is
    // unlikely);
    try {
      tripstrip = file.request_properties_from_element(
          "tristrips", {"vertex_indices"}, 0);
    } catch (const std::exception &e) {
      logError("[import_PLY] tinyply exception: %s", e.what());
    }

    file.read(*file_stream);

    logInfo("[import_PLY] imported data info:");

    if (vertices)
      logInfo("    read %zu total vertices", vertices->count);
    if (normals)
      logInfo("    read %zu total normals", normals->count);
    if (colors)
      logInfo("    read %zu total colors", colors->count);
    if (texcoords)
      logInfo("    read %zu total texcoords", texcoords->count);
    if (faces)
      logInfo("    read %zu total faces (triangles)", faces->count);
    if (tripstrip)
      logInfo("    read %zu total indices (tristrip)",
          (tripstrip->buffer.size_bytes()
              / tinyply::PropertyTable[tripstrip->t].stride));

    if (!vertices || vertices->t == tinyply::Type::FLOAT64) {
      logWarning(
          "[import_PLY] float64 vertices not supported, import not successful");
      return;
    }

    ///////////////////////////////////////////////////////////////////////////

    auto objectName = fileOf(std::string(filename)) + " (PLY file)";

    // Material //

    auto mat = scene.createObject<Material>(tokens::material::matte);
    mat->setParameter("color", float3(0.8f));
    mat->setParameter("opacity", 1.f);
    mat->setParameter("alphaMode", "opaque");
    mat->parameter("alphaMode")->setStringSelection(0);
    mat->setName((objectName + " material").c_str());

    // Mesh //

    auto ply_root = scene.insertChildNode(
        location ? location : scene.defaultLayer()->root(),
        fileOf(filename).c_str());
    auto mesh = scene.createObject<Geometry>(tokens::geometry::triangle);

    auto makeArray1DForMesh = [&](Token parameterName,
                                  anari::DataType type,
                                  const void *ptr,
                                  size_t size) {
      auto arr = scene.createArray(type, size);
      arr->setData(ptr);
      mesh->setParameterObject(parameterName, *arr);
    };

    makeArray1DForMesh("vertex.position",
        ANARI_FLOAT32_VEC3,
        vertices->buffer.get(),
        vertices->count);

    if (normals && normals->t == tinyply::Type::FLOAT32) {
      makeArray1DForMesh("vertex.normal",
          ANARI_FLOAT32_VEC3,
          normals->buffer.get(),
          normals->count);
    }

    if (colors && colors->t == tinyply::Type::UINT8) {
      makeArray1DForMesh("vertex.color",
          ANARI_UFIXED8_VEC4,
          colors->buffer.get(),
          colors->count);
      mat->setParameter("color", "color");
    }

    if (faces) {
      makeArray1DForMesh("primitive.index",
          ANARI_UINT32_VEC3,
          faces->buffer.get(),
          faces->count);
    }

    mesh->setName((objectName + "_mesh").c_str());

    auto surface = scene.createSurface(objectName.c_str(), mesh, mat);
    ply_root->insert_last_child({surface});

  } catch (const std::exception &e) {
    logError("[import_PLY] caught tinyply exception: %s", e.what());
  }
}

} // namespace tsd::io