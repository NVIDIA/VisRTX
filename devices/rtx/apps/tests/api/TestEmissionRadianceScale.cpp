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

// Emission radiance scale. Primary-ray deposit of an emissive quad — no floor,
// so next-event sampling does not affect the measured deposit — isolates the
// emission evaluation. A Lambertian emitter of radiance V must read V at the
// center pixel for BOTH an inline color and a uniform emissive texture, and must
// read V regardless of view angle (an oblique camera pins that). Guards whichever
// backend `physicallyBased` resolves to; the MDL backend (built under the MDL
// config, like the rest of the emissive suite) is guarded against the
// radiant-exitance vs radiance mix-up that made MDL emission PI x too bright — the
// compiled EDF is authored in intensity_radiant_exitance, so the evaluate callable
// must return edf*intensity, not edf*cos*intensity/pdf. The oblique case also
// catches a lone `*cos` regression that head-on (cos=1) would hide.

// anari_cpp
#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp/ext/std.h>
#include <anari/anari_cpp.hpp>
// VisRTX
#include <anari/ext/visrtx/makeVisRTXDevice.h>
// std
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using uvec2 = std::array<unsigned int, 2>;
using vec2 = std::array<float, 2>;
using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;

static void statusFunc(const void *,
    ANARIDevice,
    ANARIObject src,
    ANARIDataType,
    ANARIStatusSeverity sev,
    ANARIStatusCode,
    const char *msg)
{
  if (sev == ANARI_SEVERITY_FATAL_ERROR) {
    fprintf(stderr, "[FATAL][%p] %s\n", src, msg);
    std::exit(1);
  } else if (sev == ANARI_SEVERITY_ERROR)
    fprintf(stderr, "[ERROR][%p] %s\n", src, msg);
}

static constexpr uvec2 IMAGE_SIZE = {64, 64};
static constexpr float EMIT = 1.0f;

// Center-pixel radiance of an emissive quad viewed from (camPos, camDir). The
// quad spans [-2,2] in the z=0 plane with a +Z normal, big enough that the center
// ray hits it from either camera.
static double renderCenter(
    ANARIDevice device, bool textured, vec3 camPos, vec3 camDir)
{
  const std::array<vec3, 4> pos = {vec3{-2, -2, 0}, vec3{2, -2, 0},
      vec3{2, 2, 0}, vec3{-2, 2, 0}};
  const std::array<vec2, 4> uv = {
      vec2{0, 0}, vec2{1, 0}, vec2{1, 1}, vec2{0, 1}};
  const std::array<std::array<unsigned, 3>, 2> idx = {
      std::array<unsigned, 3>{0, 1, 2}, std::array<unsigned, 3>{0, 2, 3}};

  auto geom = anari::newObject<anari::Geometry>(device, "triangle");
  anari::setParameterArray1D(device, geom, "vertex.position", pos.data(), 4);
  anari::setParameterArray1D(device, geom, "vertex.attribute0", uv.data(), 4);
  anari::setParameterArray1D(device, geom, "primitive.index", idx.data(), 2);
  anari::commitParameters(device, geom);

  auto mat = anari::newObject<anari::Material>(device, "physicallyBased");
  anari::setParameter(device, mat, "baseColor", vec3{0, 0, 0});
  anari::setParameter(device, mat, "metallic", 0.f);
  anari::setParameter(device, mat, "roughness", 1.f);
  if (textured) {
    const std::array<vec3, 4> texels = {vec3{EMIT, EMIT, EMIT},
        vec3{EMIT, EMIT, EMIT}, vec3{EMIT, EMIT, EMIT}, vec3{EMIT, EMIT, EMIT}};
    auto image = anari::newArray2D(device, texels.data(), 2, 2);
    auto sampler = anari::newObject<anari::Sampler>(device, "image2D");
    anari::setAndReleaseParameter(device, sampler, "image", image);
    anari::setParameter(device, sampler, "inAttribute", "attribute0");
    anari::commitParameters(device, sampler);
    anari::setAndReleaseParameter(device, mat, "emissive", sampler);
  } else {
    anari::setParameter(device, mat, "emissive", vec3{EMIT, EMIT, EMIT});
  }
  anari::commitParameters(device, mat);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "surface", &surface, 1);
  anari::release(device, surface);
  anari::commitParameters(device, world);

  auto camera = anari::newObject<anari::Camera>(device, "perspective");
  anari::setParameter(device, camera, "position", camPos);
  anari::setParameter(device, camera, "direction", camDir);
  anari::setParameter(device, camera, "up", vec3{0, 1, 0});
  anari::setParameter(device, camera, "aspect", 1.f);
  anari::commitParameters(device, camera);

  auto renderer = anari::newObject<anari::Renderer>(device, "quality");
  anari::setParameter(device, renderer, "background", vec4{0, 0, 0, 1});
  anari::setParameter(device, renderer, "ambientRadiance", 0.f);
  anari::setParameter(device, renderer, "pixelSamples", 16);
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
  const vec4 &p = fb.data[(IMAGE_SIZE[1] / 2) * IMAGE_SIZE[0] + IMAGE_SIZE[0] / 2];
  const double lum = 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
  anari::unmap(device, frame, "channel.color");
  anari::release(device, frame);
  return lum;
}

static bool check(const char *label, double lum)
{
  // Near-deterministic: uniform emitter, primary ray. A PI-scale regression lands
  // at ~3.14x and a lone `*cos` regression at ~0.89x (oblique) — both far outside.
  constexpr double TOL = 0.02;
  const bool ok = std::abs(lum - EMIT) <= TOL * EMIT;
  printf("%s: %.6f (%s)\n", label, lum, ok ? "ok" : "BAD");
  if (!ok)
    fprintf(stderr,
        "FAIL: %s emissive V must deposit radiance V (got %.4f, expected %.4f) — "
        "a ~PI factor means the emission mode/normalization is wrong; a ~0.89 "
        "means a spurious cosine\n",
        label,
        lum,
        EMIT);
  return ok;
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);

  const vec3 headPos = {0, 0, 3}, headDir = {0, 0, -1};
  // Oblique camera: its forward ray still hits the quad center at (0,0,0), but
  // k1 makes ~26.6 deg with the +Z normal (cos ~ 0.894), so a stray `*cos`
  // factor reads ~0.89 V here while head-on (cos=1) would not notice.
  const vec3 oblPos = {1.5f, 0, 3}, oblDir = {-1.5f, 0, -3};

  bool ok = check("inline/head-on", renderCenter(device, false, headPos, headDir));
  ok = check("textured/head-on", renderCenter(device, true, headPos, headDir))
      && ok;
  ok = check("inline/oblique", renderCenter(device, false, oblPos, oblDir)) && ok;

  anari::release(device, device);

  if (!ok)
    return 1;
  printf("emission radiance scale passed\n");
  return 0;
}
