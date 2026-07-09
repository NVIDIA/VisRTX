/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

// Lights added AFTER the first render must take effect on the next render
// without a second unrelated scene edit — the failure the TSD viewer shows,
// where a light "does not work on subgraph creation" until a hide/show forces a
// rebuild. Two incremental cases, each rendering the SAME persistent frame twice
// with a commit in between:
//   1. a surface's material is switched to emissive -> a Geometry Light must
//      appear (exercises the material-driven light-set invalidation);
//   2. an authored quad light is added to the world -> it must appear.

#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp/ext/std.h>
#include <anari/anari_cpp.hpp>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>


using uvec2 = std::array<unsigned int, 2>;
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

static constexpr uvec2 IMAGE_SIZE = {256, 256};
static constexpr float RADIANCE = 8.f;
static constexpr float QUAD_Y = 1.5f;
static constexpr float QUAD_HALF = 0.5f;

static anari::Surface triangleSurface(ANARIDevice device,
    const std::array<vec3, 4> &pos,
    anari::Material mat)
{
  const std::array<std::array<unsigned, 3>, 2> idx = {
      std::array<unsigned, 3>{0, 1, 2}, std::array<unsigned, 3>{0, 2, 3}};
  auto geom = anari::newObject<anari::Geometry>(device, "triangle");
  anari::setParameterArray1D(device, geom, "vertex.position", pos.data(), 4);
  anari::setParameterArray1D(device, geom, "primitive.index", idx.data(), 2);
  anari::commitParameters(device, geom);
  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);
  return surface;
}

static const std::array<vec3, 4> FLOOR = {vec3{-6.f, 0.f, -6.f},
    vec3{6.f, 0.f, -6.f},
    vec3{6.f, 0.f, 6.f},
    vec3{-6.f, 0.f, 6.f}};
static const std::array<vec3, 4> QUAD = {vec3{-QUAD_HALF, QUAD_Y, -QUAD_HALF},
    vec3{QUAD_HALF, QUAD_Y, -QUAD_HALF},
    vec3{QUAD_HALF, QUAD_Y, QUAD_HALF},
    vec3{-QUAD_HALF, QUAD_Y, QUAD_HALF}};

static anari::Frame makeFrame(ANARIDevice device, anari::World world)
{
  auto camera = anari::newObject<anari::Camera>(device, "perspective");
  anari::setParameter(device, camera, "position", vec3{0.f, 0.5f, -3.f});
  anari::setParameter(device, camera, "direction", vec3{0.f, -0.15f, 1.f});
  anari::setParameter(device, camera, "up", vec3{0.f, 1.f, 0.f});
  anari::setParameter(
      device, camera, "aspect", IMAGE_SIZE[0] / float(IMAGE_SIZE[1]));
  anari::commitParameters(device, camera);

  auto renderer = anari::newObject<anari::Renderer>(device, "quality");
  anari::setParameter(device, renderer, "background", vec4{0.f, 0.f, 0.f, 1.f});
  anari::setParameter(device, renderer, "ambientRadiance", 0.f);
  anari::setParameter(device, renderer, "pixelSamples", 64);
  anari::setParameter(device, renderer, "fireflyFilterMode", "none");
  anari::commitParameters(device, renderer);

  auto frame = anari::newObject<anari::Frame>(device);
  anari::setParameter(device, frame, "size", IMAGE_SIZE);
  anari::setParameter(device, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setParameter(device, frame, "world", world);
  anari::setAndReleaseParameter(device, frame, "camera", camera);
  anari::setAndReleaseParameter(device, frame, "renderer", renderer);
  anari::commitParameters(device, frame);
  return frame;
}

static double floorPool(ANARIDevice device, anari::Frame frame)
{
  anari::render(device, frame);
  anari::wait(device, frame);
  auto fb = anari::map<vec4>(device, frame, "channel.color");
  double sum = 0.0;
  uint64_t n = 0;
  for (uint32_t y = IMAGE_SIZE[1] / 8; y < IMAGE_SIZE[1] / 2; ++y) {
    for (uint32_t x = 3 * IMAGE_SIZE[0] / 8; x < 5 * IMAGE_SIZE[0] / 8; ++x) {
      const vec4 &p = fb.data[y * IMAGE_SIZE[0] + x];
      sum += 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
      ++n;
    }
  }
  anari::unmap(device, frame, "channel.color");
  return n ? sum / double(n) : 0.0;
}

// Case 1: switch a committed surface's material to emissive after a render.
static bool testMaterialToEmissive(ANARIDevice device)
{
  auto floorMat = anari::newObject<anari::Material>(device, "physicallyBased");
  anari::setParameter(device, floorMat, "baseColor", vec3{0.6f, 0.6f, 0.6f});
  anari::commitParameters(device, floorMat);
  auto floor = triangleSurface(device, FLOOR, floorMat);

  // Start non-emissive (black baseColor, no emissive).
  auto quadMat = anari::newObject<anari::Material>(device, "physicallyBased");
  anari::setParameter(device, quadMat, "baseColor", vec3{0.f, 0.f, 0.f});
  anari::commitParameters(device, quadMat);
  auto quad = triangleSurface(device, QUAD, quadMat);

  const std::array<anari::Surface, 2> surfaces = {floor, quad};
  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "surface", surfaces.data(), 2);
  anari::commitParameters(device, world);

  auto frame = makeFrame(device, world);
  const double before = floorPool(device, frame);

  // Now make the quad emissive and commit ONLY the material (as an editor would).
  anari::setParameter(
      device, quadMat, "emissive", vec3{RADIANCE, RADIANCE, RADIANCE});
  anari::commitParameters(device, quadMat);
  const double after = floorPool(device, frame);

  printf("  material->emissive: before=%f after=%f\n", before, after);

  anari::release(device, floorMat);
  anari::release(device, quadMat);
  anari::release(device, floor);
  anari::release(device, quad);
  anari::release(device, world);
  anari::release(device, frame);

  if (after <= 0.01) {
    fprintf(stderr,
        "FAIL: making a material emissive after render did not light the floor "
        "(after=%f) — light set not rebuilt\n",
        after);
    return false;
  }
  return true;
}

// Case 2: add an authored quad light to the world after a render.
static bool testAddLight(ANARIDevice device)
{
  auto floorMat = anari::newObject<anari::Material>(device, "physicallyBased");
  anari::setParameter(device, floorMat, "baseColor", vec3{0.6f, 0.6f, 0.6f});
  anari::commitParameters(device, floorMat);
  auto floor = triangleSurface(device, FLOOR, floorMat);

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "surface", &floor, 1);
  anari::commitParameters(device, world);

  auto frame = makeFrame(device, world);
  const double before = floorPool(device, frame);

  auto light = anari::newObject<anari::Light>(device, "quad");
  anari::setParameter(device, light, "color", vec3{1.f, 1.f, 1.f});
  anari::setParameter(
      device, light, "position", vec3{-QUAD_HALF, QUAD_Y, -QUAD_HALF});
  anari::setParameter(device, light, "edge1", vec3{2.f * QUAD_HALF, 0.f, 0.f});
  anari::setParameter(device, light, "edge2", vec3{0.f, 0.f, 2.f * QUAD_HALF});
  anari::setParameter(device, light, "intensity", RADIANCE);
  anari::setParameter(device, light, "side", "both");
  anari::commitParameters(device, light);
  anari::setParameterArray1D(device, world, "light", &light, 1);
  anari::commitParameters(device, world);

  const double after = floorPool(device, frame);
  printf("  add quad light: before=%f after=%f\n", before, after);

  anari::release(device, floorMat);
  anari::release(device, floor);
  anari::release(device, light);
  anari::release(device, world);
  anari::release(device, frame);

  if (after <= 0.01) {
    fprintf(stderr,
        "FAIL: adding a quad light after render did not light the floor "
        "(after=%f) — light set not rebuilt\n",
        after);
    return false;
  }
  return true;
}

// Case 3 (the reviewer's Blocker 1): a second light that starts dark is toggled
// bright by committing ONLY the light — no world/group re-commit. The Stage 0
// pick CDF must rebuild, or the light's pick probability stays 0 and it is never
// sampled (invisible until an unrelated scene edit forces a rebuild).
static bool testToggleSecondLight(ANARIDevice device)
{
  auto floorMat = anari::newObject<anari::Material>(device, "physicallyBased");
  anari::setParameter(device, floorMat, "baseColor", vec3{0.6f, 0.6f, 0.6f});
  anari::commitParameters(device, floorMat);
  auto floor = triangleSurface(device, FLOOR, floorMat);

  auto lightA = anari::newObject<anari::Light>(device, "directional");
  anari::setParameter(device, lightA, "direction", vec3{0.3f, -1.f, 0.2f});
  anari::setParameter(device, lightA, "irradiance", 3.f);
  anari::commitParameters(device, lightA);

  auto lightB = anari::newObject<anari::Light>(device, "directional");
  anari::setParameter(device, lightB, "direction", vec3{-0.3f, -1.f, -0.2f});
  anari::setParameter(device, lightB, "irradiance", 0.f); // starts dark
  anari::commitParameters(device, lightB);

  const std::array<anari::Light, 2> lights = {lightA, lightB};
  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "surface", &floor, 1);
  anari::setParameterArray1D(device, world, "light", lights.data(), 2);
  anari::commitParameters(device, world);

  auto frame = makeFrame(device, world);
  const double before = floorPool(device, frame);

  // Toggle B bright by committing ONLY the light object.
  anari::setParameter(device, lightB, "irradiance", 3.f);
  anari::commitParameters(device, lightB);
  const double after = floorPool(device, frame);

  printf("  toggle 2nd light: before=%f after=%f\n", before, after);

  anari::release(device, floorMat);
  anari::release(device, floor);
  anari::release(device, lightA);
  anari::release(device, lightB);
  anari::release(device, world);
  anari::release(device, frame);

  // B roughly doubles the illuminated floor; a stale pick CDF leaves it near
  // `before`.
  if (after < before * 1.3) {
    fprintf(stderr,
        "FAIL: toggling a light bright via a light-only commit had no effect "
        "(before=%f after=%f) — pick CDF not rebuilt\n",
        before,
        after);
    return false;
  }
  return true;
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);
  bool ok = true;
  ok &= testMaterialToEmissive(device);
  ok &= testAddLight(device);
  ok &= testToggleSecondLight(device);
  anari::release(device, device);

  if (!ok)
    return 1;
  printf("emissive incremental light-set rebuild passed\n");
  return 0;
}
