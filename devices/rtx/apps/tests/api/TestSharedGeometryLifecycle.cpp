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
#include <string>
#include "DeviceTestArray.h"
#include "ReceiverTestSupport.h"

using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;
using Pools = std::array<double, 2>;
static constexpr unsigned WIDTH = 128, HEIGHT = 128;
// Fixed before inspecting results: match the existing 512-spp triangle/quad
// oracle's 5% allowance. Omitting one light must fail this same threshold.
// Zero-area phases allow only 1e-6.
static constexpr double REL_TOL = 0.05, BLACK_TOL = 1e-6;

static std::array<vec3, 5> vertices(float halfWidth)
{
  return {{{-halfWidth, 1.5f, -0.5f},
      {halfWidth, 1.5f, -0.5f},
      {halfWidth, 1.5f, 0.5f},
      {-halfWidth, 1.5f, 0.5f},
      {-halfWidth, 1.5f, -0.25f}}};
}
static void setTriangles(ANARIDevice d,
    anari::Geometry g,
    float halfWidth,
    bool device = true,
    bool soup = false)
{
  const auto positions = vertices(halfWidth);
  // Unequal areas plus a zero-area triangle exercise the area/CDF sampler.
  const std::array<std::array<unsigned, 3>, 4> indices = {
      {{0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {4, 4, 4}}};
  if (soup) {
    std::array<vec3, 12> expanded;
    for (size_t i = 0; i < indices.size(); ++i)
      for (size_t c = 0; c < 3; ++c)
        expanded[3 * i + c] = positions[indices[i][c]];
    setTestArray(d, g, "vertex.position", expanded, device);
    anari::unsetParameter(d, g, "primitive.index");
  } else {
    setTestArray(d, g, "vertex.position", positions, device);
    setTestArray(d, g, "primitive.index", indices, device);
  }
  anari::commitParameters(d, g);
}
static anari::Geometry geometry(ANARIDevice d, float halfWidth)
{
  auto g = anari::newObject<anari::Geometry>(d, "triangle");
  setTriangles(d, g, halfWidth);
  return g;
}
static anari::Surface surface(ANARIDevice d, anari::Geometry g)
{
  auto m = anari::newObject<anari::Material>(d, "physicallyBased");
  anari::setParameter(d, m, "baseColor", vec3{0, 0, 0});
  anari::setParameter(d, m, "emissive", vec3{8, 8, 8});
  anari::commitParameters(d, m);
  auto s = anari::newObject<anari::Surface>(d);
  anari::setParameter(d, s, "geometry", g);
  anari::setAndReleaseParameter(d, s, "material", m);
  anari::commitParameters(d, s);
  return s;
}
static void translate(ANARIDevice d, anari::Instance instance, float x)
{
  const std::array<float, 16> matrix = {
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, 0, 0, 1};
  anari::setParameter(
      d, instance, "transform", ANARI_FLOAT32_MAT4, matrix.data());
  anari::commitParameters(d, instance);
}
static anari::Instance instanceOf(ANARIDevice d, anari::Surface s, float x)
{
  auto group = anari::newObject<anari::Group>(d);
  anari::setParameterArray1D(d, group, "surface", &s, 1);
  anari::commitParameters(d, group);
  auto instance = anari::newObject<anari::Instance>(d, "transform");
  anari::setAndReleaseParameter(d, instance, "group", group);
  translate(d, instance, x);
  return instance;
}
static anari::World worldWithFloor(ANARIDevice d)
{
  auto w = anari::newObject<anari::World>(d);
  auto floor = visrtx::test::receiverFloor(d);
  anari::setParameterArray1D(d, w, "surface", &floor, 1);
  anari::release(d, floor);
  return w;
}
static anari::Frame frame(ANARIDevice d, anari::World world)
{
  return visrtx::test::receiverFrame(
      d, world, {WIDTH, HEIGHT}, "interactive", 512);
}
static Pools render(ANARIDevice d, anari::Frame f)
{
  anari::render(d, f);
  anari::wait(d, f);
  const visrtx::test::FrameMapping<vec4> mapping(d, f, "channel.color");
  const auto *pixels = mapping.checkedData(WIDTH, HEIGHT, ANARI_FLOAT32_VEC4);
  Pools pools{};
  for (unsigned side = 0; side < 2; ++side)
    pools[side] = visrtx::test::meanLuminance(pixels,
        {WIDTH, HEIGHT},
        {WIDTH / 4 + side * WIDTH / 4,
            WIDTH / 2 + side * WIDTH / 4,
            HEIGHT / 8,
            HEIGHT / 2});
  return pools;
}
// Independent oracle: two authored analytic quads, not copies of the geometry
// preprocessing implementation. A missing light must fail the same comparator.
static Pools reference(ANARIDevice d,
    const std::array<float, 2> &widths,
    const std::array<float, 2> &offsets,
    bool omitRight = false)
{
  auto w = worldWithFloor(d);
  const visrtx::test::ObjectOwner worldOwner(d, w);
  std::vector<anari::Light> lights;
  for (unsigned i = 0; i < 2; ++i) {
    if (widths[i] == 0 || (omitRight && i == 1))
      continue;
    auto light = anari::newObject<anari::Light>(d, "quad");
    anari::setParameter(
        d, light, "position", vec3{offsets[i] - widths[i], 1.5f, -0.5f});
    anari::setParameter(d, light, "edge1", vec3{2 * widths[i], 0, 0});
    anari::setParameter(d, light, "edge2", vec3{0, 0, 1});
    anari::setParameter(d, light, "color", vec3{1, 1, 1});
    anari::setParameter(d, light, "intensity", 8.f);
    anari::setParameter(d, light, "side", "both");
    anari::commitParameters(d, light);
    lights.push_back(light);
  }
  if (!lights.empty())
    anari::setParameterArray1D(d, w, "light", lights.data(), lights.size());
  for (auto light : lights)
    anari::release(d, light);
  anari::commitParameters(d, w);
  auto f = frame(d, w);
  const visrtx::test::ObjectOwner frameOwner(d, f);
  return render(d, f);
}
static bool matches(const Pools &actual, const Pools &expected)
{
  for (unsigned i = 0; i < 2; ++i) {
    if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]))
      return false;
    if (expected[i] > BLACK_TOL) {
      if (std::abs(actual[i] - expected[i]) / expected[i] > REL_TOL)
        return false;
    } else if (std::abs(actual[i]) > BLACK_TOL)
      return false;
  }
  return true;
}
static bool phase(ANARIDevice d,
    anari::Frame f,
    anari::World world,
    const char *name,
    const std::array<float, 2> &widths,
    const std::array<float, 2> &offsets)
{
  fprintf(stderr, "[case] %s\n", name);
  const auto actual = render(d, f);
  uint32_t lights = 0;
  const uint32_t expectedLights =
      uint32_t(widths[0] > 0) + uint32_t(widths[1] > 0);
  const bool countMatches =
      anari::getProperty(d, world, "numLightInstances", lights)
      && lights == expectedLights;
  const auto expected = reference(d, widths, offsets);
  const bool lit = widths[0] > 0 || widths[1] > 0;
  bool ok = countMatches && matches(actual, expected);
  if (lit)
    for (double value : expected)
      ok = ok && std::isfinite(value) && value > BLACK_TOL;
  printf("%s actual=[%g,%g] analytic=[%g,%g] lights=%u %s\n",
      name,
      actual[0],
      actual[1],
      expected[0],
      expected[1],
      lights,
      ok ? "PASS" : "FAIL");
  return ok;
}
int main()
{
  auto d = makeVisRTXDevice(visrtx::test::exitOnErrorStatus);
  auto g = geometry(d, 0.5f);
  const std::array<anari::Surface, 2> surfaces = {surface(d, g), surface(d, g)};
  std::array<float, 2> offsets = {-0.9f, 0.9f};
  const std::array<anari::Instance, 2> instances = {
      instanceOf(d, surfaces[0], offsets[0]),
      instanceOf(d, surfaces[1], offsets[1])};
  auto w = worldWithFloor(d);
  anari::setParameterArray1D(
      d, w, "instance", instances.data(), instances.size());
  anari::commitParameters(d, w);
  auto f = frame(d, w);
  bool ok = true;
  // Share the analytic oracle with lifecycle checks; finish with device-backed
  // indexed triangles so the subsequent sequence exercises that path once.
  for (bool device : {false, true})
    for (bool soup : {true, false}) {
      setTriangles(d, g, 0.5f, device, soup);
      const std::string name = std::string(device ? "device/" : "host/")
          + (soup ? "triangle-soup" : "indexed-triangle");
      ok = phase(d, f, w, name.c_str(), {0.5f, 0.5f}, offsets) && ok;
    }
  const auto oracle = reference(d, {0.5f, 0.5f}, offsets);
  const auto missing = reference(d, {0.5f, 0.5f}, offsets, true);
  const bool validControl = std::isfinite(missing[0])
      && std::isfinite(missing[1]) && missing[0] > BLACK_TOL
      && missing[1] > BLACK_TOL;
  const bool controls = validControl && !matches(oracle, missing);
  printf("missing-instance=[%g,%g] control=%s\n",
      missing[0],
      missing[1],
      controls ? "PASS" : "FAIL");
  ok = controls && ok;

  // Shrink while the five-vertex/four-triangle area data is active. The same
  // geometry keeps its buffer capacity, but must publish only two triangles.
  // A changed footprint makes stale area weights/counts observable in both
  // pools against independently authored analytic lights.
  auto positions = anari::newArray1D(d, ANARI_FLOAT32_VEC3, 4);
  auto update = [&](float halfWidth) {
    const auto v = vertices(halfWidth);
    std::copy_n(v.begin(), 4, anari::map<vec3>(d, positions));
    anari::unmap(d, positions);
  };
  update(0.375f);
  const std::array<std::array<unsigned, 3>, 2> indices = {
      {{0, 1, 2}, {0, 2, 3}}};
  anari::setParameter(d, g, "vertex.position", positions);
  setTestArray(d, g, "primitive.index", indices, true);
  anari::commitParameters(d, g);
  ok = phase(d, f, w, "shared active count shrink", {0.375f, 0.375f}, offsets)
      && ok;
  update(0.f); // no geometry/surface/group/instance recommit
  ok = phase(d, f, w, "shared unmap to zero", {0, 0}, offsets) && ok;
  update(0.5f); // recovery also requires no higher-object recommit
  ok = phase(d, f, w, "shared unmap recovery", {0.5f, 0.5f}, offsets) && ok;
  setTestArray(d, g, "vertex.position", vertices(0.25f), true);
  anari::commitParameters(d, g);
  ok = phase(d, f, w, "shared device replacement", {0.25f, 0.25f}, offsets)
      && ok;
  auto detached = geometry(d, 0.375f);
  anari::setParameter(d, surfaces[1], "geometry", detached);
  anari::commitParameters(d, surfaces[1]);
  anari::release(d, detached);
  ok = phase(d, f, w, "detach right surface", {0.25f, 0.375f}, offsets) && ok;
  update(0.125f);
  anari::setParameter(d, g, "vertex.position", positions);
  anari::commitParameters(d, g);
  ok = phase(d, f, w, "update remaining consumer", {0.125f, 0.375f}, offsets)
      && ok;
  offsets[1] = 1.25f;
  translate(d, instances[1], offsets[1]);
  ok =
      phase(d, f, w, "move detached instance", {0.125f, 0.375f}, offsets) && ok;
  anari::release(d, f);
  anari::release(d, w);
  for (auto i : instances)
    anari::release(d, i);
  for (auto s : surfaces)
    anari::release(d, s);
  anari::release(d, g);
  anari::release(d, positions);
  anari::release(d, d);
  return ok ? 0 : 1;
}
