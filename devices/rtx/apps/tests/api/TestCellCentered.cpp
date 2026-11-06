/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

// GPU end-to-end test for cell-centered structuredRegular spatial fields.
// Exercises the standardized KHR_SPATIAL_FIELD_STRUCTURED_REGULAR_CELL_CENTERED
// boolean parameter `cellCentered` alongside the VISRTX string parameter
// `dataCentering` (VISRTX_SPATIAL_FIELD_DATA_CENTERING), and their precedence
// rule (KHR wins when both are set).
//
// Setup mirrors TestIsosurfaceRamp: a z-ramp field (data[i,j,k] = k) with an
// isosurface selecting a z-plane, camera at z=-5 looking down +z.
//
// Node-centered (default): value v sits at z = v (samples on grid vertices),
// so isovalue 3.5 selects the plane z = 3.5 -> depth 8.5.
// Cell-centered: cell k has value k at z = k + 0.5, so isovalue 3.5 selects
// the plane z = 4.0 -> depth 9.0.

// anari_cpp
#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp/ext/std.h>
#include <anari/anari_cpp.hpp>
// VisRTX
#include <anari/ext/visrtx/makeVisRTXDevice.h>
// std
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using uvec2 = std::array<unsigned int, 2>;
using ivec3 = std::array<int, 3>;
using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;

static void statusFunc(const void *,
    ANARIDevice,
    ANARIObject source,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity == ANARI_SEVERITY_FATAL_ERROR) {
    fprintf(stderr, "[FATAL][%p] %s\n", source, message);
    std::exit(1);
  } else if (severity == ANARI_SEVERITY_ERROR)
    fprintf(stderr, "[ERROR][%p] %s\n", source, message);
}

enum class Centering
{
  DEFAULT, // no centering parameter (node-centered)
  KHR, // cellCentered = true
  VISRTX, // dataCentering = "cell"
  BOTH_CONFLICTING // cellCentered = true + dataCentering = "node" (KHR wins)
};

struct CenterProbe
{
  bool hit;
  float depth;
};

// Render the z-ramp isosurface with the given centering spelling and report
// the center-pixel hit + depth.
static CenterProbe centerProbe(anari::Device device, Centering centering)
{
  const uvec2 imageSize = {64, 64};
  const vec4 background = {0.f, 0.f, 0.f, 1.f};
  const ivec3 dims = {8, 8, 8};

  // Field data: a linear z-ramp, data[i,j,k] = float(k). ANARI Array3D is
  // stored x-fastest.
  std::vector<float> fieldData(size_t(dims[0]) * dims[1] * dims[2]);
  for (int k = 0; k < dims[2]; ++k)
    for (int j = 0; j < dims[1]; ++j)
      for (int i = 0; i < dims[0]; ++i)
        fieldData[size_t(i) + dims[0] * (size_t(j) + size_t(dims[1]) * k)] =
            float(k);

  auto field =
      anari::newObject<anari::SpatialField>(device, "structuredRegular");
  {
    auto data =
        anari::newArray3D(device, fieldData.data(), dims[0], dims[1], dims[2]);
    anari::setAndReleaseParameter(device, field, "data", data);
    anari::setParameter(device, field, "origin", vec3{0.f, 0.f, 0.f});
    anari::setParameter(device, field, "spacing", vec3{1.f, 1.f, 1.f});
    anari::setParameter(device, field, "filter", "linear");
    switch (centering) {
    case Centering::KHR:
      anari::setParameter(device, field, "cellCentered", true);
      break;
    case Centering::VISRTX:
      anari::setParameter(device, field, "dataCentering", "cell");
      break;
    case Centering::BOTH_CONFLICTING:
      anari::setParameter(device, field, "dataCentering", "node");
      anari::setParameter(device, field, "cellCentered", true);
      break;
    case Centering::DEFAULT:
      break;
    }
    // Commit the field BEFORE the geometry that references it: the isosurface
    // finalize needs the field's space-skipping grid, otherwise it warns
    // "space-skipping grid not ready" and renders nothing.
    anari::commitParameters(device, field);
  }

  auto geometry = anari::newObject<anari::Geometry>(device, "isosurface");
  anari::setParameter(device, geometry, "field", field);
  anari::setParameter(device, geometry, "isovalue", 3.5f);
  anari::commitParameters(device, geometry);
  anari::release(device, field);

  auto material = anari::newObject<anari::Material>(device, "matte");
  anari::setParameter(device, material, "color", vec3{1.f, 0.f, 0.f});
  anari::commitParameters(device, material);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geometry);
  anari::setAndReleaseParameter(device, surface, "material", material);
  anari::commitParameters(device, surface);

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "surface", &surface, 1);
  anari::release(device, surface);
  anari::commitParameters(device, world);

  // Camera at z=-5 looking down +z at the center of the iso-plane.
  auto camera = anari::newObject<anari::Camera>(device, "perspective");
  anari::setParameter(device, camera, "position", vec3{3.5f, 3.5f, -5.f});
  anari::setParameter(device, camera, "direction", vec3{0.f, 0.f, 1.f});
  anari::setParameter(device, camera, "up", vec3{0.f, 1.f, 0.f});
  anari::setParameter(
      device, camera, "aspect", imageSize[0] / float(imageSize[1]));
  anari::commitParameters(device, camera);

  auto renderer = anari::newObject<anari::Renderer>(device, "debug");
  anari::setParameter(device, renderer, "background", background);
  anari::setParameter(device, renderer, "method", "baseColor");
  anari::commitParameters(device, renderer);

  auto frame = anari::newObject<anari::Frame>(device);
  anari::setParameter(device, frame, "size", imageSize);
  anari::setParameter(device, frame, "channel.color", ANARI_UFIXED8_RGBA_SRGB);
  anari::setParameter(device, frame, "channel.depth", ANARI_FLOAT32);
  anari::setAndReleaseParameter(device, frame, "world", world);
  anari::setAndReleaseParameter(device, frame, "camera", camera);
  anari::setAndReleaseParameter(device, frame, "renderer", renderer);
  anari::commitParameters(device, frame);

  anari::render(device, frame);
  anari::wait(device, frame);

  const size_t centerIdx =
      (imageSize[1] / 2) * size_t(imageSize[0]) + imageSize[0] / 2;

  auto fb = anari::map<uint32_t>(device, frame, "channel.color");
  const uint32_t center = fb.data[centerIdx];
  const uint32_t red = center & 0xFFu; // R channel (RGBA8, little-endian)
  anari::unmap(device, frame, "channel.color");

  auto depthBuf = anari::map<float>(device, frame, "channel.depth");
  const float depth = depthBuf.data ? depthBuf.data[centerIdx] : -1.f;
  anari::unmap(device, frame, "channel.depth");

  anari::release(device, frame);

  return {red > 16u, depth}; // geometry (red) vs black background
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);

  const CenterProbe nodeCentered = centerProbe(device, Centering::DEFAULT);
  const CenterProbe khrCellCentered = centerProbe(device, Centering::KHR);
  const CenterProbe visrtxCellCentered = centerProbe(device, Centering::VISRTX);
  const CenterProbe conflicting =
      centerProbe(device, Centering::BOTH_CONFLICTING);

  anari::release(device, device);

  constexpr float kDepthTol = 0.15f;
  // Node-centered: value 3.5 lies at z=3.5, camera at z=-5 -> depth 8.5.
  // Cell-centered: value 3.5 lies halfway between cell centers z=3.5 and
  // z=4.5 -> z=4.0 -> depth 9.0.
  const float kNodeDepth = 8.5f;
  const float kCellDepth = 9.0f;
  int failures = 0;

  auto check = [](const char *label, const CenterProbe &p, float expected) {
    if (!p.hit) {
      fprintf(stderr, "FAIL: %s center is background\n", label);
      return 1;
    }
    if (std::fabs(p.depth - expected) > kDepthTol) {
      fprintf(stderr,
          "FAIL: %s depth %.3f, expected ~%.3f\n",
          label,
          p.depth,
          expected);
      return 1;
    }
    return 0;
  };

  // 1. No centering parameter: vertex-centered (default), depth 8.5.
  failures += check("node-centered (default)", nodeCentered, kNodeDepth);

  // 2. KHR spelling: cellCentered=true -> depth 9.0.
  failures += check("KHR cellCentered=true", khrCellCentered, kCellDepth);

  // 3. VISRTX spelling: dataCentering="cell" -> depth 9.0.
  failures +=
      check("VISRTX dataCentering=cell", visrtxCellCentered, kCellDepth);

  // 4. Conflicting spellings: cellCentered=true takes precedence over
  //    dataCentering="node" -> depth 9.0.
  failures += check("conflicting (KHR wins)", conflicting, kCellDepth);

  if (failures)
    return 1;
  printf("cell-centered structuredRegular field test passed\n");
  return 0;
}
