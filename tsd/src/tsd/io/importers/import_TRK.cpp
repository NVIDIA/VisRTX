// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/Logging.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
// std
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace tsd::io {

using namespace tsd::core;

namespace {
const float DEFAULT_CURVE_RADIUS = 0.1f;
} // namespace

/**
 * TrackVis .trk file header structure
 * Based on the TrackVis format specification
 */
struct TrackVisHeader
{
  char id_string[6]; // ID string for track file. The first 5 characters must be
                     // "TRACK".
  int16_t dim[3]; // Dimension of the image volume
  float voxel_size[3]; // Voxel size of the image volume
  float origin[3]; // Origin of the image volume
  int16_t n_scalars; // Number of scalars saved at each track point
  char scalar_name[10][20]; // Name of each scalar
  int16_t n_properties; // Number of properties saved at each track
  char property_name[10][20]; // Name of each property
  float vox_to_ras[4][4]; // 4x4 matrix for voxel to RAS (right, anterior,
                          // superior)
  char reserved[444]; // Reserved space for future version
  char voxel_order[4]; // Voxel order. e.g. "LPS"
  char pad2[4]; // Paddings
  float image_orientation_patient[6]; // Image orientation of the original image
  char pad1[2]; // Paddings
  unsigned char invert_x; // Inversion/rotation flags
  unsigned char invert_y;
  unsigned char invert_z;
  unsigned char swap_xy;
  unsigned char swap_yz;
  unsigned char swap_zx;
  int32_t n_count; // Number of tracks stored in this track file. 0 means the
                   // number was not stored.
  int32_t version; // Version number
  int32_t hdr_size; // Size of the header. Used to determine byte swap.
};

/**
 * Reads a TrackVis .trk file and generates streamlines with directional colors.
 *
 * @param scene Scene in which to create the streamlines.
 * @param filename Path to the .trk file to read.
 * @param location Node in the scene graph where the streamlines should be
 * added.
 */
void readTrkFile(
    Scene &scene, const std::string &filename, LayerNodeRef location)
{
  // Open the file
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    logError("[import_TRK] Error opening file: %s", filename.c_str());
    return;
  }

  // Read header
  TrackVisHeader header;
  file.read(reinterpret_cast<char *>(&header), sizeof(TrackVisHeader));

  // Validate header
  if (std::strncmp(header.id_string, "TRACK", 5) != 0) {
    logError("[import_TRK] Invalid TrackVis file (bad header): %s",
        filename.c_str());
    file.close();
    return;
  }

  // Check for byte swapping
  bool needsByteSwap = (header.hdr_size != 1000);
  if (needsByteSwap) {
    logWarning(
        "[import_TRK] File may need byte swapping (hdr_size=%d), attempting to read anyway",
        header.hdr_size);
  }

  logInfo("[import_TRK] Loading streamlines from %s", filename.c_str());
  logInfo("[import_TRK]   Dimension: %d x %d x %d",
      header.dim[0],
      header.dim[1],
      header.dim[2]);
  logInfo("[import_TRK]   Voxel size: %f x %f x %f",
      header.voxel_size[0],
      header.voxel_size[1],
      header.voxel_size[2]);
  logInfo("[import_TRK]   Number of tracks: %d", header.n_count);
  logInfo("[import_TRK]   Number of scalars per point: %d", header.n_scalars);
  logInfo(
      "[import_TRK]   Number of properties per track: %d", header.n_properties);

  // Read all streamlines
  std::vector<float3> allPositions;
  std::vector<float4> allColors;
  std::vector<uint32_t> indices;

  int32_t streamlineCount = 0;
  const int32_t pointStride =
      3 + header.n_scalars; // 3 for xyz, plus any scalars
  const int32_t trackPropertySize = header.n_properties;

  while (file.good() && !file.eof()) {
    // Read number of points in this streamline
    int32_t numPoints;
    file.read(reinterpret_cast<char *>(&numPoints), sizeof(int32_t));
    if (file.eof() || numPoints <= 0 || numPoints > 1000000)
      break;

    // Reserve space for this streamline
    std::vector<float3> streamlinePoints;
    streamlinePoints.reserve(numPoints);

    // Read all points in the streamline
    for (int32_t i = 0; i < numPoints; ++i) {
      std::vector<float> pointData(pointStride);
      file.read(reinterpret_cast<char *>(pointData.data()),
          pointStride * sizeof(float));

      float3 position{pointData[0], pointData[1], pointData[2]};
      streamlinePoints.push_back(position);
    }

    // Read track properties (if any) - we don't use them but need to skip them
    if (trackPropertySize > 0) {
      std::vector<float> properties(trackPropertySize);
      file.read(reinterpret_cast<char *>(properties.data()),
          trackPropertySize * sizeof(float));
    }

    // Generate colors based on directional vectors between consecutive points
    if (streamlinePoints.size() >= 2) {
      const uint32_t baseIndex = allPositions.size();

      for (size_t i = 0; i < streamlinePoints.size(); ++i) {
        allPositions.push_back(streamlinePoints[i]);

        // Compute direction to next point (or use previous direction for last
        // point)
        float3 direction;
        if (i < streamlinePoints.size() - 1) {
          direction = streamlinePoints[i + 1] - streamlinePoints[i];
        } else {
          direction = streamlinePoints[i] - streamlinePoints[i - 1];
        }

        // Normalize direction
        float length = std::sqrt(direction.x * direction.x
            + direction.y * direction.y + direction.z * direction.z);
        if (length > 1e-6f) {
          direction.x /= length;
          direction.y /= length;
          direction.z /= length;
        }

        // Map direction to color: RGB = (x, y, z) * 0.5 + 0.5
        float4 color;
        color.x = direction.x * 0.5f + 0.5f; // Red
        color.y = direction.y * 0.5f + 0.5f; // Green
        color.z = direction.z * 0.5f + 0.5f; // Blue
        color.w = 1.0f; // Alpha

        allColors.push_back(color);
      }

      // Create curve indices - each pair of consecutive points forms a curve
      // segment
      for (size_t i = 0; i < streamlinePoints.size() - 1; ++i) {
        indices.push_back(baseIndex + i);
      }

      streamlineCount++;
    }
  }

  file.close();

  logInfo("[import_TRK] Loaded %d streamlines with %zu total points",
      streamlineCount,
      allPositions.size());

  if (allPositions.empty()) {
    logWarning("[import_TRK] No streamlines loaded from file");
    return;
  }

  // Get the location node if not already provided
  if (!location)
    location = scene.defaultLayer()->root();

  const std::string basename =
      std::filesystem::path(filename).filename().string();

  // Create transform node for the streamlines
  auto trackLocation = scene.insertChildTransformNode(location);
  (*trackLocation)->name() = "xfm";

  // Create curve geometry
  auto curves = scene.createObject<Geometry>(tokens::geometry::curve);
  curves->setName((basename).c_str());

  // Create arrays for positions, colors, and indices
  auto positionArray =
      scene.createArray(ANARI_FLOAT32_VEC3, allPositions.size());
  auto colorArray = scene.createArray(ANARI_FLOAT32_VEC4, allColors.size());
  auto indexArray = scene.createArray(ANARI_UINT32, indices.size());

  positionArray->setData(allPositions);
  colorArray->setData(allColors);
  indexArray->setData(indices);

  // Set geometry parameters
  curves->setParameterObject("vertex.position", *positionArray);
  curves->setParameterObject("vertex.color", *colorArray);
  curves->setParameterObject("primitive.index", *indexArray);
  curves->setParameter("radius", DEFAULT_CURVE_RADIUS);

  // Create material with vertex color support
  auto material = scene.createObject<Material>(tokens::material::matte);
  material->setParameter("color", "color");
  material->setName((basename + "_material").c_str());

  // Create surface and add to scene
  auto surface = scene.createSurface(
      (basename + "_streamlines").c_str(), curves, material);
  scene.insertChildObjectNode(trackLocation, surface);
}

/**
 * Imports a TrackVis .trk file into the scene.
 *
 * @param scene Scene in which to import the .trk file.
 * @param filename Path to the .trk file to import.
 * @param location Node in the scene graph where the streamlines should be
 * imported.
 */
void import_TRK(Scene &scene, const char *filename, LayerNodeRef location)
{
  readTrkFile(scene, filename, location);
}

} // namespace tsd::io
