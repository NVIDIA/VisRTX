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

// Stage 2.5: instance-uniform attribute emission. An emissive material bound to
// the `color` attribute, with the color supplied by the INSTANCE (not per-vertex
// and not on the geometry), must light a floor the same as a constant emitter of
// that color. The path-hit deposit reads the color from the real instance; the
// next-event sampler must resolve it against the SAME instance (via the light
// record's surface-instance reference) or NEE and the deposit disagree — the
// exact case deferred from Stage 2a, where the sampler used a transform-only
// synthetic instance and would read the geometry default instead. Checked on
// 'quality' and 'interactive'; the emission is spatially uniform so there is no
// geometry-factor confound. Linear float, firefly off.

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
using mat4 = std::array<float, 16>; // column-major

// Column-major translation in X only.
static mat4 translateX(float tx)
{
  return mat4{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, tx, 0, 0, 1};
}

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
static constexpr float EMIT = 8.f;
static constexpr float QUAD_Y = 1.5f;
static constexpr float QUAD_HALF = 0.5f;

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

// Down-facing emissive emitter. `instanceAttr` binds emissive to the `color`
// attribute (supplied by the instance below); otherwise emissive is a constant.
// `sphere` selects a sphere (a "user" geometry, exercising the userSI slot of
// the surface-instance cursor) instead of a triangle quad. No vertex color
// either way.
static anari::Surface makeEmissiveEmitter(
    ANARIDevice device, bool instanceAttr, bool sphere)
{
  anari::Geometry geom;
  if (sphere) {
    const std::array<vec3, 1> center = {vec3{0.f, QUAD_Y, 0.f}};
    geom = anari::newObject<anari::Geometry>(device, "sphere");
    anari::setParameterArray1D(
        device, geom, "vertex.position", center.data(), 1);
    anari::setParameter(device, geom, "radius", QUAD_HALF);
    anari::commitParameters(device, geom);
  } else {
    const std::array<vec3, 4> pos = {vec3{-QUAD_HALF, QUAD_Y, -QUAD_HALF},
        vec3{QUAD_HALF, QUAD_Y, -QUAD_HALF},
        vec3{QUAD_HALF, QUAD_Y, QUAD_HALF},
        vec3{-QUAD_HALF, QUAD_Y, QUAD_HALF}};
    const std::array<std::array<unsigned, 3>, 2> idx = {
        std::array<unsigned, 3>{0, 1, 2}, std::array<unsigned, 3>{0, 2, 3}};
    geom = anari::newObject<anari::Geometry>(device, "triangle");
    anari::setParameterArray1D(device, geom, "vertex.position", pos.data(), 4);
    anari::setParameterArray1D(device, geom, "primitive.index", idx.data(), 2);
    anari::commitParameters(device, geom);
  }

  auto mat = anari::newObject<anari::Material>(device, "physicallyBased");
  anari::setParameter(device, mat, "baseColor", vec3{0.f, 0.f, 0.f});
  anari::setParameter(device, mat, "metallic", 0.f);
  anari::setParameter(device, mat, "roughness", 1.f);
  if (instanceAttr)
    anari::setParameter(device, mat, "emissive", "color");
  else
    anari::setParameter(device, mat, "emissive", vec3{EMIT, EMIT, EMIT});
  anari::commitParameters(device, mat);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);
  return surface;
}

// `numXfm > 1` supplies an ARRAY of transforms (numTransforms() > 1), so one
// instance emits several copies. The instance-uniform color is shared across
// them — this exercises the per-transform surface-instance cursor increment that
// the light-fill hand-mirrors from buildInstanceSurfaceGPUData.
static anari::Instance instanceOf(ANARIDevice device,
    anari::Surface surface,
    bool withColor,
    int numXfm = 1)
{
  auto group = anari::newObject<anari::Group>(device);
  anari::setParameterArray1D(device, group, "surface", &surface, 1);
  anari::release(device, surface);
  anari::commitParameters(device, group);

  auto inst = anari::newObject<anari::Instance>(device, "transform");
  anari::setAndReleaseParameter(device, inst, "group", group);
  if (numXfm > 1) {
    const std::array<mat4, 2> xfms = {translateX(-0.9f), translateX(0.9f)};
    anari::setParameterArray1D(
        device, inst, "transform", ANARI_FLOAT32_MAT4, xfms.data(), numXfm);
  }
  if (withColor) // instance-uniform emission color
    anari::setParameter(device, inst, "color", vec4{EMIT, EMIT, EMIT, 1.f});
  anari::commitParameters(device, inst);
  return inst;
}

static std::vector<vec4> render(ANARIDevice device,
    bool instanceAttr,
    bool sphere,
    const char *rendererType,
    bool multiXfm)
{
  const std::array<anari::Instance, 2> instances = {
      instanceOf(device, makeFloor(device), false),
      instanceOf(device,
          makeEmissiveEmitter(device, instanceAttr, sphere),
          instanceAttr,
          multiXfm ? 2 : 1)};

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(
      device, world, "instance", instances.data(), instances.size());
  for (auto i : instances)
    anari::release(device, i);
  anari::commitParameters(device, world);

  auto camera = anari::newObject<anari::Camera>(device, "perspective");
  anari::setParameter(device, camera, "position", vec3{0.f, 0.5f, -3.f});
  anari::setParameter(device, camera, "direction", vec3{0.f, -0.15f, 1.f});
  anari::setParameter(device, camera, "up", vec3{0.f, 1.f, 0.f});
  anari::setParameter(
      device, camera, "aspect", IMAGE_SIZE[0] / float(IMAGE_SIZE[1]));
  anari::commitParameters(device, camera);

  auto renderer = anari::newObject<anari::Renderer>(device, rendererType);
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

static bool checkRenderer(ANARIDevice device,
    bool sphere,
    const char *rendererType,
    double tol,
    bool multiXfm = false)
{
  const double attr =
      poolMeanLuminance(render(device, true, sphere, rendererType, multiXfm));
  const double constant =
      poolMeanLuminance(render(device, false, sphere, rendererType, multiXfm));
  const double relErr = constant > 0.0
      ? std::abs(attr - constant) / constant
      : (attr > 0 ? 1.0 : 0.0);
  const char *geom = sphere ? "sphere" : "quad";
  const char *xfm = multiXfm ? "x2" : "x1";
  printf("%s/%s/%s: instanceAttr=%f  constant=%f  relErr=%f\n",
      rendererType,
      geom,
      xfm,
      attr,
      constant,
      relErr);

  if (attr <= 0.0 || constant <= 0.0) {
    fprintf(stderr, "FAIL: %s/%s/%s floor pool unlit\n", rendererType, geom, xfm);
    return false;
  }
  if (relErr > tol) {
    fprintf(stderr,
        "FAIL: %s/%s/%s instance-attribute emission not equivalent to its "
        "constant (relErr=%f > %f) — NEE likely missed the instance-uniform "
        "color\n",
        rendererType,
        geom,
        xfm,
        relErr,
        tol);
    return false;
  }
  return true;
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);

  // Both a triangle quad (triangleSI slot) and a sphere (userSI slot) exercise
  // the surface-instance cursor's per-geometry-type correlation.
  bool ok = checkRenderer(device, /*sphere=*/false, "quality", 0.04);
  ok = checkRenderer(device, /*sphere=*/true, "quality", 0.04) && ok;
  ok = checkRenderer(device, /*sphere=*/false, "interactive", 0.08) && ok;
  ok = checkRenderer(device, /*sphere=*/true, "interactive", 0.08) && ok;

  // numTransforms() > 1: one instance emits two copies via a transform array,
  // exercising the per-transform surface-instance cursor increment.
  ok = checkRenderer(device, /*sphere=*/false, "quality", 0.04, /*multiXfm=*/true)
      && ok;
  ok = checkRenderer(
           device, /*sphere=*/true, "quality", 0.04, /*multiXfm=*/true)
      && ok;

  anari::release(device, device);

  if (!ok)
    return 1;
  printf("emissive instance-attribute geometry light equivalence passed\n");
  return 0;
}
