// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/authoring/importers.hpp"
#include "tsd/authoring/importers/detail/importer_common.hpp"
#include "tsd/core/Logging.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace tsd {

std::vector<std::string> listFiles(
    const std::string &folderPath, const std::string &extension)
{
  std::vector<std::string> files;
  for (const auto &entry : std::filesystem::directory_iterator(folderPath)) {
    if (entry.is_regular_file()) {
      std::string ext = entry.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      if (ext == extension) {
        files.push_back(entry.path().filename().string());
      }
    }
  }

  return files;
}

struct SWCPoint
{
  int id;
  int type;
  double x, y, z;
  double radius;
  int parent;
};

void readSWCFile(Context &ctx,
    const std::string &filename,
    InstanceNode::Ref location,
    const std::string &name = "morphology")
{
  std::ifstream file(filename);
  std::string line;

  if (!file.is_open()) {
    logError("Error opening file: %s", filename);
    return;
  }

  std::map<uint64_t, SWCPoint> points;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#')
      continue;

    std::istringstream iss(line);
    SWCPoint point;
    if (iss >> point.id >> point.type >> point.x >> point.y >> point.z
        >> point.radius >> point.parent) {
      points[point.id] = point;
    }
  }
  file.close();

  if (!location)
    location = ctx.tree.root();

  // Material //

  auto material = ctx.defaultMaterial();

  const auto numPoints = points.size();

  // Generate spheres //

  auto spheres = ctx.createObject<tsd::Geometry>(tsd::tokens::geometry::sphere);

  spheres->setName("spheres_geometry_t");

  std::vector<float3> spherePositions;
  spherePositions.reserve(numPoints);
  std::vector<float> sphereRadii;
  sphereRadii.reserve(numPoints);

  for (const auto &p : points) {
    spherePositions.push_back(tsd::float3(p.second.x, p.second.y, p.second.z));
    sphereRadii.push_back(p.second.radius * 0.5);
  }

  auto spherePositionArray = ctx.createArray(ANARI_FLOAT32_VEC3, numPoints);
  auto sphereRadiusArray = ctx.createArray(ANARI_FLOAT32, numPoints);

  spherePositionArray->setData(spherePositions);
  sphereRadiusArray->setData(sphereRadii);

  spheres->setParameterObject("vertex.position"_t, *spherePositionArray);
  spheres->setParameterObject("vertex.radius"_t, *sphereRadiusArray);

  // Generate cones //

  auto cones = ctx.createObject<Geometry>(tokens::geometry::cone);

  cones->setName("cones_geometry_t");

  std::vector<float3> conePositions;
  std::vector<float> coneRadii;

  uint64_t numcones = 0;
  for (const auto &p : points) {
    if (p.second.parent == -1)
      continue;

    conePositions.push_back(tsd::float3(p.second.x, p.second.y, p.second.z));
    coneRadii.push_back(p.second.radius * 0.5f);

    const auto &p1 = points[p.second.parent];
    conePositions.push_back(tsd::float3(p1.x, p1.y, p1.z));
    coneRadii.push_back(p1.radius * 0.5f);
    ++numcones;
  }

  auto positionArray = ctx.createArray(ANARI_FLOAT32_VEC3, 2 * numcones);
  auto radiiArray = ctx.createArray(ANARI_FLOAT32, 2 * numcones);

  positionArray->setData(conePositions);
  radiiArray->setData(coneRadii);

  cones->setParameterObject("vertex.position"_t, *positionArray);
  cones->setParameterObject("vertex.radius"_t, *radiiArray);

  // Surfaces //

  const std::string conesName = name + "_cones";
  auto conseSurface = ctx.createSurface(conesName.c_str(), cones, material);
  ctx.insertChildObjectNode(location, conseSurface);
  const std::string spheresName = name + "_spheres";
  auto sphereSurface =
      ctx.createSurface(spheresName.c_str(), spheres, material);
  ctx.insertChildObjectNode(location, sphereSurface);
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void import_SWC(Context &ctx, const char *filename, InstanceNode::Ref location)
{
  readSWCFile(ctx, filename, location);
}

void import_SWCs(
    Context &ctx, const char *folderPath, InstanceNode::Ref location)
{
  std::vector<std::string> swcFiles = listFiles(folderPath, ".swc");
  for (const auto &swcFile : swcFiles) {
    const fs::path fileName = swcFile;
    const fs::path fullPath = folderPath / fileName;
    readSWCFile(ctx, fullPath, location, swcFile);
  }
}

} // namespace tsd
