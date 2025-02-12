// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/authoring/importers.hpp"
#include "tsd/core/Logging.hpp"

#include <filesystem>
#include <fstream>
#include <map>
#include <random>

namespace fs = std::filesystem;

namespace tsd {

/**
 * Lists all files with a specified extension in a given folder.
 *
 * @param folderPath Path to the folder containing files.
 * @param extension The file extension to search for (e.g. ".swc").
 *
 * @return A vector of file names with the specified extension.
 */
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

/**
 * Represents a point in a SWC (Standard Warehouse Connector) file.
 *
 * A SWC file is a text file that describes a collection of points and their
 * connections. Each point is represented by six values: an ID, a type, and x,
 * y, z coordinates.
 *
 * @struct SWCPoint
 */
struct SWCPoint
{
  int id; ///< Unique identifier for the point.
  int type; ///< Type of point (e.g. 0 for neurite, 1 for dendrite, 2 for axon).
  double x, y, z; ///< 3D coordinates of the point in space.
  double radius; ///< Radius of the point.
  int parent; ///< ID of the parent point (or -1 if it is a root point).
};

/**
 * Reads a SWC file and generates a 3D representation.
 *
 * @param ctx Context in which to create the 3D representation.
 * @param filename Path to the SWC file to read.
 * @param location Node in the scene graph where the 3D representation should be
 * added.
 * @param name A default name for the 3D representation.
 */
void readSWCFile(Context &ctx,
    const std::string &filename,
    InstanceNode::Ref location,
    const std::string &name = "morphology")
{
  // Open the SWC file and check for errors
  std::ifstream file(filename);
  std::string line;

  if (!file.is_open()) {
    logError("Error opening file: %s", filename);
    return;
  }

  // Read the file and store the points in a map
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

  // Get the location node if not already provided
  if (!location)
    location = ctx.tree.root();

  // Default material for the 3D representation
  auto material = ctx.defaultMaterial();

  // Count the number of points in the file
  const auto numPoints = points.size();

  // Generate spheres for each point
  auto spheres = ctx.createObject<tsd::Geometry>(tsd::tokens::geometry::sphere);
  spheres->setName("spheres_geometry_t");

  // Initialize the positions and radii of the spheres
  std::vector<float3> spherePositions;
  spherePositions.reserve(numPoints);
  std::vector<float> sphereRadii;
  sphereRadii.reserve(numPoints);

  for (const auto &p : points) {
    spherePositions.push_back(tsd::float3(p.second.x, p.second.y, p.second.z));
    sphereRadii.push_back(p.second.radius * 0.5);
  }

  // Create arrays to store the positions and radii of the spheres
  auto spherePositionArray = ctx.createArray(ANARI_FLOAT32_VEC3, numPoints);
  auto sphereRadiusArray = ctx.createArray(ANARI_FLOAT32, numPoints);

  spherePositionArray->setData(spherePositions);
  sphereRadiusArray->setData(sphereRadii);

  // Set the positions and radii of the spheres
  spheres->setParameterObject("vertex.position"_t, *spherePositionArray);
  spheres->setParameterObject("vertex.radius"_t, *sphereRadiusArray);

  // Generate cones to connect the points
  auto cones = ctx.createObject<Geometry>(tokens::geometry::cone);
  cones->setName("cones_geometry_t");

  // Initialize the positions and radii of the cones
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

  // Create arrays to store the positions and radii of the cones
  auto positionArray = ctx.createArray(ANARI_FLOAT32_VEC3, 2 * numcones);
  auto radiiArray = ctx.createArray(ANARI_FLOAT32, 2 * numcones);

  positionArray->setData(conePositions);
  radiiArray->setData(coneRadii);

  // Set the positions and radii of the cones
  cones->setParameterObject("vertex.position"_t, *positionArray);
  cones->setParameterObject("vertex.radius"_t, *radiiArray);

  // Material properties
  auto m = ctx.createObject<Material>(tokens::material::physicallyBased);

  // Randomly generate base color, metallic, and roughness values
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 0.5);
  tsd::float3 baseColor(0.5 + dis(gen), 0.5 + dis(gen), 0.5 + dis(gen));

  const float metallic = 0.5;
  const float roughness = 0.5;

  // Set the material properties
  m->setParameter("baseColor"_t, ANARI_FLOAT32_VEC3, &baseColor);
  m->setParameter("metallic"_t, ANARI_FLOAT32, &metallic);
  m->setParameter("roughness"_t, ANARI_FLOAT32, &roughness);

  // Create surfaces for the spheres and cones
  const std::string conesName = name + "_cones";
  auto conesSurface = ctx.createSurface(conesName.c_str(), cones, m);
  ctx.insertChildObjectNode(location, conesSurface);

  const std::string spheresName = name + "_spheres";
  auto sphereSurface = ctx.createSurface(spheresName.c_str(), spheres, m);
  ctx.insertChildObjectNode(location, sphereSurface);
}

/**
 * Imports a single SWC file into the current context.
 *
 * @param ctx Context in which to import the SWC file.
 * @param filename Path to the SWC file to import.
 * @param location Node in the scene graph where the SWC file should be
 * imported.
 */
void import_SWC(Context &ctx, const char *filename, InstanceNode::Ref location)
{
  readSWCFile(ctx, filename, location);
}

/**
 * Imports all SWC files in a given folder into the current context.
 *
 * @param ctx Context in which to import the SWC files.
 * @param folderPath Path to the folder containing the SWC files to import.
 * @param location Node in the scene graph where the SWC files should be
 * imported.
 */
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