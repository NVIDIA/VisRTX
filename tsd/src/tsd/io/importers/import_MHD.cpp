// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
#include "tsd/core/Logging.hpp"
// std
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>

using namespace std;
namespace fs = filesystem;

namespace tsd::io {

using namespace tsd::core;

struct MHDHeader
{
  uint3 dims; // Dimensions
  double3 spacing; // Voxel spacing
  anari::DataType elementType; // Data type
  string dataFile; // Raw data filename
  bool isBigEndian; // Byte order
};

MHDHeader readMHDHeader(const string &filename)
{
  MHDHeader header;
  ifstream file(filename);
  string line;

  while (getline(file, line)) {
    // Parse key-value pairs
    size_t delimPos = line.find("=");
    if (delimPos != string::npos) {
      string key = line.substr(0, delimPos);
      string value = line.substr(delimPos + 1);

      // Trim whitespace
      key.erase(0, key.find_first_not_of(" \t"));
      key.erase(key.find_last_not_of(" \t") + 1);
      value.erase(0, value.find_first_not_of(" \t"));
      value.erase(value.find_last_not_of(" \t") + 1);

      // Parse common fields
      if (key == "DimSize") {
        sscanf(value.c_str(),
            "%u %u %u",
            &header.dims.x,
            &header.dims.y,
            &header.dims.z);
      } else if (key == "ElementSpacing") {
        sscanf(value.c_str(),
            "%lf %lf %lf",
            &header.spacing.x,
            &header.spacing.y,
            &header.spacing.z);
      } else if (key == "ElementType") {
        if (value == "MET_UCHAR") {
          header.elementType = ANARI_UFIXED8;
        } else if (value == "MET_SHORT") {
          header.elementType = ANARI_UFIXED16;
        } else if (value == "MET_FLOAT") {
          header.elementType = ANARI_FLOAT32;
        }

      } else if (key == "ElementDataFile") {
        header.dataFile = value;
      } else if (key == "BinaryData") {
        // Usually "True"
      } else if (key == "BinaryDataByteOrderMSB") {
        header.isBigEndian = (value == "True");
      }
    }
  }

  return header;
}

SpatialFieldRef import_MHD(Scene &scene, const char *filepath)
{
  const auto header = readMHDHeader(filepath);

  const auto dataFilepath =
      fs::path(filepath).parent_path().string() + "/" + header.dataFile;
  auto field =
      scene.createObject<SpatialField>(tokens::spatial_field::structuredRegular);
  field->setName(dataFilepath.c_str());

  auto voxelArray = scene.createArray(
      header.elementType, header.dims.x, header.dims.y, header.dims.z);
  auto *voxelData = voxelArray->map();

  auto fileHandle = fopen(dataFilepath.c_str(), "rb");
  size_t size = header.dims[0] * size_t(header.dims[1]) * header.dims[2]
      * anari::sizeOf(header.elementType);
  if (!fread((char *)voxelData, size, 1, fileHandle)) {
    logError(
        "[import_RAW] unable to open RAW file: '%s'", dataFilepath.c_str());
    voxelArray->unmap();
    scene.removeObject(voxelArray.data());
    scene.removeObject(field.data());
    fclose(fileHandle);
    return {};
  }

  fclose(fileHandle);
  voxelArray->unmap();

  field->setParameterObject("data", *voxelArray);

  return field;
}

} // namespace tsd
