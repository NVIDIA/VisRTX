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

#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <anari/anari_cpp.hpp>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include "DeviceTestArray.h"

using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;
static int errors = 0;
static bool retireShared = false;
static bool expandedSdf = false;

struct ResidencyCase
{
  const char *name;
  int deviceInputs;
};

static constexpr std::array<ResidencyCase, 4> QUAD_CASES = {{{"host", 0},
    {"device-positions", 1},
    {"device-indices", 2},
    {"all-device", 3}}};
static constexpr std::array<ResidencyCase, 5> CURVE_CASES = {{{"host", 0},
    {"device-positions", 1},
    {"device-indices", 2},
    {"device-vertex-radii", 4},
    {"all-device", 7}}};
static constexpr auto CONE_CASES = CURVE_CASES;
static constexpr auto TRIANGLE_CASES = QUAD_CASES;
static constexpr std::array<ResidencyCase, 5> SPHERE_CASES = {{{"host", 0},
    {"device-positions", 1},
    {"device-indices", 2},
    {"device-vertex-radii", 4},
    {"all-device", 7}}};
static constexpr std::array<ResidencyCase, 5> CYLINDER_CASES = {{{"host", 0},
    {"device-positions", 1},
    {"device-indices", 2},
    {"device-primitive-radii", 4},
    {"all-device", 7}}};
static constexpr std::array<ResidencyCase, 2> SDF_CASES = {
    {{"host", 0}, {"device-primitive-bytes", 1}}};
static constexpr std::array<ResidencyCase, 4> EXPANDED_SDF_CASES = {
    {{"host", 0},
        {"device-primitive-bytes", 1},
        {"device-neighbors", 2},
        {"all-device", 3}}};
// Indices and radii always use shared GPU storage in retirement mode.
static constexpr std::array<ResidencyCase, 2> RETIREMENT_CASES = {
    {{"host-positions-shared-device-indices-radii", 0},
        {"device-positions-shared-device-indices-radii", 1}}};

template <typename Container>
static std::pair<anari::Array1D, void *> sharedArray(
    ANARIDevice d, anari::Geometry g, const char *name, const Container &values)
{
  using T = typename Container::value_type;
  void *ptr = nullptr;
  if (cudaMalloc(&ptr, values.size() * sizeof(T)) != cudaSuccess
      || cudaMemcpy(ptr,
             values.data(),
             values.size() * sizeof(T),
             cudaMemcpyHostToDevice)
          != cudaSuccess)
    std::exit(2);
  auto array = anariNewArray1D(
      d, ptr, nullptr, nullptr, anari::ANARITypeFor<T>::value, values.size());
  anari::setParameter(d, g, name, array);
  return {array, ptr};
}
static void status(const void *,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity <= ANARI_SEVERITY_ERROR) {
    ++errors;
    fprintf(stderr, "%s\n", message);
  }
}

struct Scene
{
  anari::World world;
  anari::Frame frame;
};
static Scene makeScene(ANARIDevice d, anari::Geometry g)
{
  auto material = anari::newObject<anari::Material>(d, "matte");
  anari::setParameter(d, material, "color", vec3{1, 0, 0});
  anari::commitParameters(d, material);
  auto surface = anari::newObject<anari::Surface>(d);
  anari::setParameter(d, surface, "geometry", g);
  anari::setAndReleaseParameter(d, surface, "material", material);
  anari::commitParameters(d, surface);
  auto world = anari::newObject<anari::World>(d);
  anari::setParameterArray1D(d, world, "surface", &surface, 1);
  anari::release(d, surface);
  anari::commitParameters(d, world);
  auto camera = anari::newObject<anari::Camera>(d, "perspective");
  anari::setParameter(d, camera, "position", vec3{0, 0, -3});
  anari::setParameter(d, camera, "direction", vec3{0, 0, 1});
  anari::commitParameters(d, camera);
  auto renderer = anari::newObject<anari::Renderer>(d, "debug");
  anari::setParameter(d, renderer, "method", "baseColor");
  anari::setParameter(d, renderer, "background", vec4{0, 0, 0, 1});
  anari::commitParameters(d, renderer);
  auto frame = anari::newObject<anari::Frame>(d);
  anari::setParameter(d, frame, "size", std::array<unsigned, 2>{64, 64});
  anari::setParameter(d, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setAndReleaseParameter(d, frame, "camera", camera);
  anari::setAndReleaseParameter(d, frame, "renderer", renderer);
  anari::setParameter(d, frame, "world", world);
  anari::commitParameters(d, frame);
  return {world, frame};
}

// Exact bounds and a visible/cleared center pixel are independent oracles.
static bool validate(ANARIDevice d,
    Scene scene,
    const std::array<float, 6> &expected,
    bool visible)
{
  anari::wait(d, scene.frame);
  std::array<float, 6> bounds{};
  bool ok = anariGetProperty(d,
      scene.world,
      "bounds",
      ANARI_FLOAT32_BOX3,
      bounds.data(),
      sizeof(bounds),
      ANARI_WAIT);
  for (size_t i = 0; i < bounds.size(); ++i)
    ok = std::isfinite(bounds[i]) && std::abs(bounds[i] - expected[i]) < 1e-4f
        && ok;
  auto fb = anari::map<vec4>(d, scene.frame, "channel.color");
  const float red = fb.data ? fb.data[32 * 64 + 32][0] : NAN;
  ok = std::isfinite(red) && (visible ? red > 0.5f : red < 0.01f) && ok;
  anari::unmap(d, scene.frame, "channel.color");
  printf("bounds=[%g %g %g %g %g %g] center=%g %s\n",
      bounds[0],
      bounds[1],
      bounds[2],
      bounds[3],
      bounds[4],
      bounds[5],
      red,
      ok ? "PASS" : "FAIL");
  return ok;
}

// Public API seam: exact geometric bounds and a visible center pixel are
// independent controls. 1e-4 absolute bounds tolerance permits OptiX rounding;
// moving the input by +2 in X must move the bounds and clear the center pixel.
static bool check(ANARIDevice d,
    const char *type,
    const ResidencyCase &residency,
    float shift)
{
  const int device = residency.deviceInputs;
  const char *scenario =
      retireShared ? "retirement" : (expandedSdf ? "sdf-expanded" : type);
  fprintf(stderr, "[case] %s/%s shift=%g\n", scenario, residency.name, shift);
  auto g = anari::newObject<anari::Geometry>(d, type);
  std::vector<std::pair<anari::Array1D, void *>> shared;
  float xExtent = 0.5f, yExtent = 0.5f, zExtent = 0.f;
  if (std::strcmp(type, "sphere") == 0) {
    // The unused large sphere makes ignoring the index observable.
    const std::array<vec3, 2> vertices = {
        {{shift + 4.f, 4.f, 0}, {shift, 0, 0}}};
    const std::array<unsigned, 1> indices = {1};
    const std::array<float, 2> radii = {1.f, 0.5f};
    setTestArray(d, g, "vertex.position", vertices, device & 1, 1);
    setTestArray(d, g, "primitive.index", indices, device & 2, 1);
    setTestArray(d, g, "vertex.radius", radii, device & 4, 1);
    zExtent = 0.5f;
  } else if (std::strcmp(type, "triangle") == 0) {
    const std::array<vec3, 4> vertices = {{{shift + 4.f, 4.f, 0},
        {shift - 0.5f, -0.5f, 0},
        {shift + 0.5f, -0.5f, 0},
        {shift, 0.5f, 0}}};
    const std::array<std::array<unsigned, 3>, 1> indices = {{{1, 2, 3}}};
    setTestArray(d, g, "vertex.position", vertices, device & 1, 1);
    setTestArray(d, g, "primitive.index", indices, device & 2, 1);
  } else if (std::strcmp(type, "quad") == 0) {
    const std::array<vec3, 4> vertices = {{{shift - 0.5f, -0.5f, 0},
        {shift + 0.5f, -0.5f, 0},
        {shift + 0.5f, 0.5f, 0},
        {shift - 0.5f, 0.5f, 0}}};
    const std::array<std::array<unsigned, 4>, 1> indices = {{{0, 1, 2, 3}}};
    setTestArray(d, g, "vertex.position", vertices, device & 1, 1);
    setTestArray(d, g, "primitive.index", indices, device & 2, 1);
  } else if (std::strcmp(type, "cone") == 0
      || std::strcmp(type, "cylinder") == 0) {
    const bool cone = std::strcmp(type, "cone") == 0;
    // Vertex 0 is deliberately unused. Selecting it instead of logical
    // vertices 2 and 1 must fail both the bounds and visibility controls.
    const std::array<vec3, 3> vertices = {
        {{shift + 4.f, 4.f, 0}, {shift + 0.5f, 0, 0}, {shift - 0.5f, 0, 0}}};
    const std::array<std::array<unsigned, 2>, 1> indices = {{{2, 1}}};
    const std::array<float, 3> vertexRadii = {0.75f, 0.25f, 0.125f};
    const std::array<float, 1> primitiveRadii = {0.25f};
    setTestArray(d, g, "vertex.position", vertices, device & 1, 1);
    setTestArray(d, g, "primitive.index", indices, device & 2, 1);
    if (cone)
      setTestArray(d, g, "vertex.radius", vertexRadii, device & 4, 1);
    else
      setTestArray(d, g, "primitive.radius", primitiveRadii, device & 4, 1);
    anari::setParameter(d, g, "caps", "both");
    // Conservative tube bounds: endpoints [-0.5,0.5], largest radius 0.25.
    // This literal oracle is independent of device bound-generation code.
    xExtent = 0.75f;
    yExtent = zExtent = 0.25f;
  } else if (std::strcmp(type, "curve") == 0) {
    const std::array<vec3, 2> vertices = {
        {{shift - 0.5f, 0, 0}, {shift + 0.5f, 0, 0}}};
    const std::array<unsigned, 1> indices = {0};
    const std::array<float, 2> radii = {0.125f, 0.125f};
    setTestArray(d, g, "vertex.position", vertices, device & 1, 1);
    if (retireShared) {
      shared.push_back(sharedArray(d, g, "primitive.index", indices));
      shared.push_back(sharedArray(d, g, "vertex.radius", radii));
    } else {
      setTestArray(d, g, "primitive.index", indices, device & 2, 1);
      setTestArray(d, g, "vertex.radius", radii, device & 4, 1);
    }
    xExtent = 0.625f;
    yExtent = zExtent = 0.125f;
  } else {
    // VISRTX_GEOMETRY_SDF documents N*72 raw bytes. Pack a sphere, without
    // depending on device classes or their private preprocessing buffers.
    std::array<uint8_t, 72> bytes{};
    const vec3 center = {shift, 0, 0};
    const float radius = 0.5f;
    std::memcpy(bytes.data() + 20, center.data(), sizeof(center));
    std::memcpy(bytes.data() + 44, &radius, sizeof(radius));
    std::memcpy(bytes.data() + 48, &radius, sizeof(radius));
    if (expandedSdf) {
      bytes[64] = 1;
      const std::array<uint64_t, 1> neighbours = {0};
      setTestArray(d, g, "primitive.neighbor", neighbours, device & 2, 1);
      anari::setParameter(d, g, "blendFactor", 0.5f);
      anari::setParameter(d, g, "noiseFactor", 0.5f);
      // 0.5 base + 0.5*0.5 blend + 0.5*0.5*0.1 noise.
      xExtent = yExtent = 0.775f;
    }
    setTestArray(d, g, "primitive.sdf", bytes, device & 1, 1);
    zExtent = xExtent;
  }
  anari::commitParameters(d, g);
  const auto scene = makeScene(d, g);
  anari::release(d, g);
  const auto frame = scene.frame;
  anari::render(d, frame);
  if (retireShared) {
    // Last PUBLIC release must finish outstanding consumers and rebind any
    // aliases before the application is allowed to retire its shared storage.
    for (auto entry : shared) {
      anari::release(d, entry.first);
      cudaFree(entry.second);
    }
    fprintf(stderr,
        "[case] %s/%s shift=%g after shared storage retirement\n",
        scenario,
        residency.name,
        shift);
    anari::setParameter(d, frame, "accumulationVersion", uint64_t(2));
    anari::commitParameters(d, frame);
    anari::render(d, frame);
  }
  const std::array<float, 6> expected = {
      shift - xExtent, -yExtent, -zExtent, shift + xExtent, yExtent, zExtent};
  const bool ok = validate(d, scene, expected, shift == 0.f);
  anari::release(d, frame);
  anari::release(d, scene.world);
  return ok;
}
// One array observed by two distinct geometries/worlds. Detaching one consumer
// must not detach the other or hide observer-only unmap updates.
static bool sharedInputs(ANARIDevice d)
{
  auto positions = anari::newArray1D(d, ANARI_FLOAT32_VEC3, 2);
  const auto update = [&](float y) {
    auto *p = anari::map<vec3>(d, positions);
    p[0] = {-0.5f, y, 0};
    p[1] = {0.5f, y, 0};
    anari::unmap(d, positions);
  };
  update(0);
  std::array<anari::Geometry, 2> geometries;
  std::array<Scene, 2> scenes;
  for (size_t i = 0; i < scenes.size(); ++i) {
    auto g = anari::newObject<anari::Geometry>(d, "curve");
    anari::setParameter(d, g, "vertex.position", positions);
    anari::setParameter(d, g, "radius", 0.125f);
    anari::commitParameters(d, g);
    geometries[i] = g;
    scenes[i] = makeScene(d, g);
  }
  const auto check = [&](size_t consumer, float y) {
    fprintf(stderr, "[case] shared-input consumer=%zu y=%g\n", consumer, y);
    anari::render(d, scenes[consumer].frame);
    return validate(d,
        scenes[consumer],
        {-0.625f, y - 0.125f, -0.125f, 0.625f, y + 0.125f, 0.125f},
        y == 0);
  };
  bool ok = check(0, 0);
  ok = check(1, 0) && ok;
  update(1); // no geometry/world recommit
  ok = check(0, 1) && ok;
  ok = check(1, 1) && ok;
  const std::array<vec3, 2> replacement = {{{-0.5f, 0, 0}, {0.5f, 0, 0}}};
  setTestArray(d, geometries[0], "vertex.position", replacement, true);
  anari::commitParameters(d, geometries[0]);
  ok = check(0, 0) && ok;
  ok = check(1, 1) && ok;
  for (size_t i = 0; i < scenes.size(); ++i) {
    anari::release(d, scenes[i].frame);
    anari::release(d, scenes[i].world);
    anari::release(d, geometries[i]);
  }
  anari::release(d, positions);
  return ok;
}
int main(int argc, char **argv)
{
  retireShared = argc > 1 && std::strcmp(argv[1], "--retire") == 0;
  expandedSdf = argc > 1 && std::strcmp(argv[1], "sdf-expanded") == 0;
  const char *type = retireShared
      ? "curve"
      : (expandedSdf ? "sdf" : (argc > 1 ? argv[1] : "quad"));
  auto d = makeVisRTXDevice(status);
  bool ok = true;
  const auto runCases = [&](const auto &cases) {
    for (const auto &residency : cases)
      for (float shift : {0.f, 2.f})
        ok = check(d, type, residency, shift) && ok;
  };
  if (std::strcmp(type, "--shared-inputs") == 0)
    ok = sharedInputs(d);
  else if (retireShared)
    runCases(RETIREMENT_CASES);
  else if (expandedSdf)
    runCases(EXPANDED_SDF_CASES);
  else if (std::strcmp(type, "triangle") == 0)
    runCases(TRIANGLE_CASES);
  else if (std::strcmp(type, "sphere") == 0)
    runCases(SPHERE_CASES);
  else if (std::strcmp(type, "quad") == 0)
    runCases(QUAD_CASES);
  else if (std::strcmp(type, "curve") == 0)
    runCases(CURVE_CASES);
  else if (std::strcmp(type, "cone") == 0)
    runCases(CONE_CASES);
  else if (std::strcmp(type, "cylinder") == 0)
    runCases(CYLINDER_CASES);
  else
    runCases(SDF_CASES);
  anari::release(d, d);
  return ok && errors == 0 ? 0 : 1;
}
