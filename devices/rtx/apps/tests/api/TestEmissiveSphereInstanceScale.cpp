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

// Sphere Geometry Light sampling must be exact under an instance scale, and must
// resolve indexed primitives and per-vertex radii correctly. An emissive sphere
// SET (two indexed spheres, distinct per-vertex radii) baked at a 2x world
// footprint under an identity instance must light a floor identically to the
// same unit-footprint set stretched to that footprint by a uniform scale(2)
// instance. A uniform scale keeps each sphere a sphere, so the two describe the
// SAME world emitters; any pixel difference means the sphere area Jacobian
// (|cross(M·t1,M·t2)|), the outward-normal orientation, or the indexed/per-vertex
// radius resolution in the sampler / hit-side pdf is wrong. This exercises the
// paths TestEmissiveSphereLight (single, soup, identity) does not. Non-uniform
// scale turns a sphere into an ellipsoid, which has no closed-form reference
// light, so it is left to the triangle path's TestEmissiveInstanceScale.
// Rendered with 'quality', linear float buffer, firefly filter off.

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
using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;
using mat4 = std::array<float, 16>;

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
static constexpr int PIXEL_SAMPLES = 512;
static constexpr float EMISSIVE_RADIANCE = 8.f;
static constexpr float SCALE = 2.f;

// Unit-footprint sphere set (pre-scale). Two spheres, distinct radii, sitting
// above the origin; the "scaled" render places them here and stretches by a
// scale(2) instance, the "baked" render pre-multiplies center+radius by 2 with
// an identity instance. index = {1, 0} so index order != soup order, exercising
// the indexed path.
static constexpr std::array<vec3, 2> UNIT_CENTERS = {
    vec3{-0.4f, 0.75f, 0.f}, vec3{0.4f, 0.75f, 0.f}};
static constexpr std::array<float, 2> UNIT_RADII = {0.2f, 0.25f};
static constexpr std::array<unsigned, 2> SPHERE_INDEX = {1u, 0u};

static anari::Surface makeFloor(ANARIDevice device)
{
  const std::array<vec3, 4> pos = {vec3{-6.f, 0.f, -6.f},
      vec3{6.f, 0.f, -6.f},
      vec3{6.f, 0.f, 6.f},
      vec3{-6.f, 0.f, 6.f}};
  const std::array<std::array<unsigned, 3>, 2> idx = {
      std::array<unsigned, 3>{0, 1, 2}, std::array<unsigned, 3>{0, 2, 3}};

  auto geom = anari::newObject<anari::Geometry>(device, "triangle");
  anari::setParameterArray1D(device, geom, "vertex.position", pos.data(), 4);
  anari::setParameterArray1D(device, geom, "primitive.index", idx.data(), 2);
  anari::commitParameters(device, geom);

  auto mat = anari::newObject<anari::Material>(device, "physicallyBased");
  anari::setParameter(device, mat, "baseColor", vec3{0.6f, 0.6f, 0.6f});
  anari::setParameter(device, mat, "roughness", 1.f);
  anari::commitParameters(device, mat);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);
  return surface;
}

// Indexed, per-vertex-radius emissive sphere set. `s` scales both centers and
// radii (bake the instance scale into the geometry when s != 1).
static anari::Surface makeEmissiveSpheres(ANARIDevice device, float s)
{
  std::array<vec3, 2> centers = UNIT_CENTERS;
  std::array<float, 2> radii = UNIT_RADII;
  for (int i = 0; i < 2; ++i) {
    centers[i] = {centers[i][0] * s, centers[i][1] * s, centers[i][2] * s};
    radii[i] *= s;
  }

  auto geom = anari::newObject<anari::Geometry>(device, "sphere");
  anari::setParameterArray1D(
      device, geom, "vertex.position", centers.data(), 2);
  anari::setParameterArray1D(device, geom, "vertex.radius", radii.data(), 2);
  anari::setParameterArray1D(
      device, geom, "primitive.index", SPHERE_INDEX.data(), 2);
  anari::commitParameters(device, geom);

  auto mat = anari::newObject<anari::Material>(device, "physicallyBased");
  anari::setParameter(device, mat, "baseColor", vec3{0.f, 0.f, 0.f});
  anari::setParameter(device,
      mat,
      "emissive",
      vec3{EMISSIVE_RADIANCE, EMISSIVE_RADIANCE, EMISSIVE_RADIANCE});
  anari::commitParameters(device, mat);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);
  return surface;
}

// Render the floor plus the emissive sphere set. `scaled` selects the unit set
// stretched by a scale(2) instance; otherwise a baked 2x set with an identity
// instance. Both yield the same two world spheres.
static std::vector<vec4> render(ANARIDevice device, bool scaled)
{
  auto floor = makeFloor(device);
  auto emitter =
      scaled ? makeEmissiveSpheres(device, 1.f) : makeEmissiveSpheres(device, SCALE);

  auto floorGroup = anari::newObject<anari::Group>(device);
  anari::setParameterArray1D(device, floorGroup, "surface", &floor, 1);
  anari::release(device, floor);
  anari::commitParameters(device, floorGroup);

  auto emitterGroup = anari::newObject<anari::Group>(device);
  anari::setParameterArray1D(device, emitterGroup, "surface", &emitter, 1);
  anari::release(device, emitter);
  anari::commitParameters(device, emitterGroup);

  auto floorInstance = anari::newObject<anari::Instance>(device, "transform");
  anari::setAndReleaseParameter(device, floorInstance, "group", floorGroup);
  anari::commitParameters(device, floorInstance);

  auto emitterInstance = anari::newObject<anari::Instance>(device, "transform");
  anari::setAndReleaseParameter(device, emitterInstance, "group", emitterGroup);
  if (scaled) {
    // Column-major uniform scale(2).
    const mat4 xfm = {SCALE, 0.f, 0.f, 0.f, 0.f, SCALE, 0.f, 0.f, 0.f, 0.f,
        SCALE, 0.f, 0.f, 0.f, 0.f, 1.f};
    anari::setParameter(
        device, emitterInstance, "transform", ANARI_FLOAT32_MAT4, xfm.data());
  }
  anari::commitParameters(device, emitterInstance);

  const std::array<anari::Instance, 2> instances = {
      floorInstance, emitterInstance};

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(
      device, world, "instance", instances.data(), instances.size());
  anari::release(device, floorInstance);
  anari::release(device, emitterInstance);
  anari::commitParameters(device, world);

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
  anari::setParameter(device, renderer, "pixelSamples", PIXEL_SAMPLES);
  anari::setParameter(device, renderer, "fireflyFilterMode", "none");
  anari::commitParameters(device, renderer);

  auto frame = anari::newObject<anari::Frame>(device);
  anari::setParameter(device, frame, "size", IMAGE_SIZE);
  anari::setParameter(device, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setAndReleaseParameter(device, frame, "world", world);
  anari::setAndReleaseParameter(device, frame, "camera", camera);
  anari::setAndReleaseParameter(device, frame, "renderer", renderer);
  anari::commitParameters(device, frame);

  anari::render(device, frame);
  anari::wait(device, frame);

  auto fb = anari::map<vec4>(device, frame, "channel.color");
  std::vector<vec4> out(fb.data, fb.data + size_t(fb.width) * fb.height);
  anari::unmap(device, frame, "channel.color");
  anari::release(device, frame);
  return out;
}

// Mean linear luminance over the lit floor region (lower-center of the frame,
// ANARI's bottom-left framebuffer origin).
static double poolMeanLuminance(const std::vector<vec4> &fb)
{
  double sum = 0.0;
  uint64_t n = 0;
  for (uint32_t y = IMAGE_SIZE[1] / 8; y < IMAGE_SIZE[1] / 2; ++y) {
    for (uint32_t x = 3 * IMAGE_SIZE[0] / 8; x < 5 * IMAGE_SIZE[0] / 8; ++x) {
      const vec4 &p = fb[y * IMAGE_SIZE[0] + x];
      sum += 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
      ++n;
    }
  }
  return n ? sum / double(n) : 0.0;
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);

  const double baked = poolMeanLuminance(render(device, false));
  const double scaled = poolMeanLuminance(render(device, true));

  anari::release(device, device);

  const double relErr =
      baked > 0.0 ? std::abs(scaled - baked) / baked : (scaled > 0 ? 1.0 : 0.0);
  printf("baked=%f  scaledInstance=%f  relErr=%f\n", baked, scaled, relErr);

  if (baked <= 0.0 || scaled <= 0.0) {
    fprintf(stderr,
        "FAIL: floor pool unlit (baked=%f scaled=%f)\n",
        baked,
        scaled);
    return 1;
  }
  // 5% absorbs MC noise at 512 spp; a broken Jacobian, orientation, or index/
  // radius resolution shifts it far more.
  constexpr double TOLERANCE = 0.05;
  if (relErr > TOLERANCE) {
    fprintf(stderr,
        "FAIL: sphere instance scale / indexing biased the emitter (relErr=%f > %f)\n",
        relErr,
        TOLERANCE);
    return 1;
  }
  printf("emissive sphere instance-scale + indexing Jacobian passed\n");
  return 0;
}
