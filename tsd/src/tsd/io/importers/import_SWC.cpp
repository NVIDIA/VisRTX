// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/importers.hpp"
#include "tsd/core/Logging.hpp"

#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <sstream>

namespace {
const float DEFAULT_ROUGHNESS = 0.5f;
const float DEFAULT_METALLIC = 0.5f;
} // namespace

namespace tsd::io {

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
void readSWCFile(
    Context &ctx, const std::string &filename, LayerNodeRef location)
{
  // Open the SWC file and check for errors
  std::ifstream file(filename);
  std::string line;

  if (!file.is_open()) {
    logError("Error opening file: %s", filename.c_str());
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
    location = ctx.defaultLayer()->root();

  // Default material for the 3D representation
  auto material = ctx.defaultMaterial();

  // Count the number of points in the file
  const auto numPoints = points.size();

  // Generate spheres for each point
  auto spheres = ctx.createObject<Geometry>(tokens::geometry::sphere);
  spheres->setName("spheres_geometry_t");

  // Initialize the positions and radii of the spheres
  std::vector<float3> spherePositions;
  spherePositions.reserve(numPoints);
  std::vector<float> sphereRadii;
  sphereRadii.reserve(numPoints);

  for (const auto &p : points) {
    spherePositions.push_back(tsd::math::float3(p.second.x, p.second.y, p.second.z));
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

    conePositions.push_back(tsd::math::float3(p.second.x, p.second.y, p.second.z));
    coneRadii.push_back(p.second.radius * 0.5f);

    const auto &p1 = points[p.second.parent];
    conePositions.push_back(tsd::math::float3(p1.x, p1.y, p1.z));
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
  tsd::math::float3 baseColor(0.5 + dis(gen), 0.5 + dis(gen), 0.5 + dis(gen));

  const float metallic = DEFAULT_METALLIC;
  const float roughness = DEFAULT_ROUGHNESS;

  // Set the material properties
  m->setParameter("baseColor"_t, ANARI_FLOAT32_VEC3, &baseColor);
  m->setParameter("metallic"_t, ANARI_FLOAT32, &metallic);
  m->setParameter("roughness"_t, ANARI_FLOAT32, &roughness);

  const auto swcLocation =
      location->insert_first_child(tsd::core::Any(ANARI_GEOMETRY, 1));

  // Create surfaces for the spheres and cones
  const std::string basename =
      std::filesystem::path(filename).filename().string();

  const std::string conesName = basename + "_cones";
  auto conesSurface = ctx.createSurface(conesName.c_str(), cones, m);
  ctx.insertChildObjectNode(swcLocation, conesSurface);

  const std::string spheresName = basename + "_spheres";
  auto sphereSurface = ctx.createSurface(spheresName.c_str(), spheres, m);
  ctx.insertChildObjectNode(swcLocation, sphereSurface);
}

/**
 * Imports a single SWC file into the current context.
 *
 * @param ctx Context in which to import the SWC file.
 * @param filename Path to the SWC file to import.
 * @param location Node in the scene graph where the SWC file should be
 * imported.
 */
void import_SWC(Context &ctx, const char *filename, LayerNodeRef location)
{
  readSWCFile(ctx, filename, location);
}

} // namespace tsd
