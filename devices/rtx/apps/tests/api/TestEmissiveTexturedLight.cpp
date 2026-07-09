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

// Stage 2 textured emission. An emissive quad whose `emissive` is an image2D
// sampler (a high-contrast checker averaging to mid-gray) must light a floor the
// same as a constant emitter at the texture's AVERAGE color. For a small emitter
// far from the receiver each pool point integrates the whole emitter, so the
// textured and averaged-constant pools match — validating that the NEE sampler
// evaluates emission through the material's real entry point at the sampled point
// (a synthetic hit), and that hit-side MIS stays unbiased. Checked with BOTH
// 'quality' (path-traced deposit + MIS) and 'interactive' (single-bounce NEE):
// the latter is the case that previously left textured emitters dark. Emitter
// parked above frame; linear float buffer, firefly off.

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
#include <functional>
#include <vector>

using uvec2 = std::array<unsigned int, 2>;
using vec2 = std::array<float, 2>;
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
static constexpr int PIXEL_SAMPLES = 1024;
static constexpr float PEAK = 16.f; // checker bright texel
static constexpr float AVERAGE = PEAK / 2.f; // 8x8 checker => half bright
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
  anari::setParameter(device, mat, "metallic", 0.f);
  anari::setParameter(device, mat, "roughness", 1.f);
  anari::commitParameters(device, mat);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);
  return surface;
}

// image2D sampler over an 8x8 PEAK/0 checker (average = PEAK/2 per channel),
// nearest-filtered so the texels stay exact.
static anari::Sampler makeCheckerEmissiveSampler(ANARIDevice device)
{
  constexpr int N = 8;
  std::array<vec3, N * N> texels;
  for (int y = 0; y < N; ++y)
    for (int x = 0; x < N; ++x) {
      const float v = ((x + y) & 1) ? PEAK : 0.f;
      texels[y * N + x] = vec3{v, v, v};
    }
  auto image = anari::newArray2D(device, texels.data(), N, N);

  auto sampler = anari::newObject<anari::Sampler>(device, "image2D");
  anari::setAndReleaseParameter(device, sampler, "image", image);
  anari::setParameter(device, sampler, "inAttribute", "attribute0");
  anari::setParameter(device, sampler, "filter", "nearest");
  anari::commitParameters(device, sampler);
  return sampler;
}

enum EmitterKind
{
  CONSTANT, // emissive = AVERAGE
  TEXTURED, // emissive = checker sampler (mean AVERAGE)
  ATTRIBUTE // emissive = "color" vertex attribute (mean AVERAGE)
};

// Down-facing emissive quad carrying attribute0 texcoords and a vertex.color
// whose area-weighted mean over the two triangles is AVERAGE.
static anari::Surface makeEmissiveQuad(ANARIDevice device, EmitterKind kind)
{
  const std::array<vec3, 4> pos = {vec3{-QUAD_HALF, QUAD_Y, -QUAD_HALF},
      vec3{QUAD_HALF, QUAD_Y, -QUAD_HALF},
      vec3{QUAD_HALF, QUAD_Y, QUAD_HALF},
      vec3{-QUAD_HALF, QUAD_Y, QUAD_HALF}};
  const std::array<vec2, 4> uv = {
      vec2{0.f, 0.f}, vec2{1.f, 0.f}, vec2{1.f, 1.f}, vec2{0.f, 1.f}};
  // Per-corner emission for the ATTRIBUTE case: bright on the two diagonal
  // corners (v0,v2), dark on (v1,v3). Each triangle — (0,1,2) and (0,2,3) —
  // averages exactly AVERAGE, and the pattern is 180°-symmetric about the
  // emitter centre so the geometry-factor variation across it cancels, keeping
  // the cast pool equal to the averaged constant.
  const float hi = 1.5f * AVERAGE;
  const std::array<vec4, 4> color = {vec4{hi, hi, hi, 1.f},
      vec4{0.f, 0.f, 0.f, 1.f},
      vec4{hi, hi, hi, 1.f},
      vec4{0.f, 0.f, 0.f, 1.f}};
  const std::array<std::array<unsigned, 3>, 2> idx = {
      std::array<unsigned, 3>{0, 1, 2}, std::array<unsigned, 3>{0, 2, 3}};

  auto geom = anari::newObject<anari::Geometry>(device, "triangle");
  anari::setParameterArray1D(device, geom, "vertex.position", pos.data(), 4);
  anari::setParameterArray1D(device, geom, "vertex.attribute0", uv.data(), 4);
  anari::setParameterArray1D(device, geom, "vertex.color", color.data(), 4);
  anari::setParameterArray1D(device, geom, "primitive.index", idx.data(), 2);
  anari::commitParameters(device, geom);

  auto mat = anari::newObject<anari::Material>(device, "physicallyBased");
  anari::setParameter(device, mat, "baseColor", vec3{0.f, 0.f, 0.f});
  anari::setParameter(device, mat, "metallic", 0.f);
  anari::setParameter(device, mat, "roughness", 1.f);
  if (kind == TEXTURED) {
    auto sampler = makeCheckerEmissiveSampler(device);
    anari::setAndReleaseParameter(device, mat, "emissive", sampler);
  } else if (kind == ATTRIBUTE) {
    anari::setParameter(device, mat, "emissive", "color");
  } else {
    anari::setParameter(
        device, mat, "emissive", vec3{AVERAGE, AVERAGE, AVERAGE});
  }
  anari::commitParameters(device, mat);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);
  return surface;
}

// image2D sampler over a uniform AVERAGE texture: emission is spatially constant
// regardless of texcoords, so a textured emitter on any geometry must match a
// constant one — exercising the synthetic-hit texture eval per geometry type
// without a geometry-factor confound.
static anari::Sampler makeUniformEmissiveSampler(ANARIDevice device)
{
  const std::array<vec3, 4> texels = {vec3{AVERAGE, AVERAGE, AVERAGE},
      vec3{AVERAGE, AVERAGE, AVERAGE},
      vec3{AVERAGE, AVERAGE, AVERAGE},
      vec3{AVERAGE, AVERAGE, AVERAGE}};
  auto image = anari::newArray2D(device, texels.data(), 2, 2);

  auto sampler = anari::newObject<anari::Sampler>(device, "image2D");
  anari::setAndReleaseParameter(device, sampler, "image", image);
  anari::setParameter(device, sampler, "inAttribute", "attribute0");
  anari::commitParameters(device, sampler);
  return sampler;
}

// Emissive sphere at the emitter footprint. Sphere/cylinder/cone go through a
// different sampler path (finishAreaLightSample) than the triangle, so cover it:
// a uniform-textured sphere must match a constant one.
static anari::Surface makeEmissiveSphere(ANARIDevice device, bool textured)
{
  const std::array<vec3, 1> pos = {vec3{0.f, QUAD_Y, 0.f}};
  const std::array<vec2, 1> uv = {vec2{0.5f, 0.5f}};

  auto geom = anari::newObject<anari::Geometry>(device, "sphere");
  anari::setParameterArray1D(device, geom, "vertex.position", pos.data(), 1);
  anari::setParameterArray1D(device, geom, "vertex.attribute0", uv.data(), 1);
  anari::setParameter(device, geom, "radius", 0.4f);
  anari::commitParameters(device, geom);

  auto mat = anari::newObject<anari::Material>(device, "physicallyBased");
  anari::setParameter(device, mat, "baseColor", vec3{0.f, 0.f, 0.f});
  anari::setParameter(device, mat, "metallic", 0.f);
  anari::setParameter(device, mat, "roughness", 1.f);
  if (textured) {
    auto sampler = makeUniformEmissiveSampler(device);
    anari::setAndReleaseParameter(device, mat, "emissive", sampler);
  } else {
    anari::setParameter(
        device, mat, "emissive", vec3{AVERAGE, AVERAGE, AVERAGE});
  }
  anari::commitParameters(device, mat);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);
  return surface;
}

using EmitterFactory = std::function<anari::Surface(ANARIDevice)>;

static std::vector<vec4> render(
    ANARIDevice device, const EmitterFactory &emitter, const char *rendererType)
{
  const std::array<anari::Surface, 2> surfaces = {
      makeFloor(device), emitter(device)};

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(
      device, world, "surface", surfaces.data(), surfaces.size());
  for (auto s : surfaces)
    anari::release(device, s);
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

// A non-constant emitter (textured or attribute-driven) must light the floor the
// same as a constant emitter at its average — and, for `interactive`, must light
// it at all (the case textured emitters previously missed).
static bool checkEmitter(ANARIDevice device,
    const EmitterFactory &emitterFactory,
    const EmitterFactory &constantFactory,
    const char *label,
    const char *rendererType,
    double tol)
{
  const double emitter =
      poolMeanLuminance(render(device, emitterFactory, rendererType));
  const double constant =
      poolMeanLuminance(render(device, constantFactory, rendererType));
  const double relErr = constant > 0.0
      ? std::abs(emitter - constant) / constant
      : (emitter > 0 ? 1.0 : 0.0);
  printf("%s/%s: emitter=%f  constant(avg)=%f  relErr=%f\n",
      rendererType,
      label,
      emitter,
      constant,
      relErr);

  if (emitter <= 0.0) {
    fprintf(stderr,
        "FAIL: %s %s emitter did not light the floor\n",
        rendererType,
        label);
    return false;
  }
  if (constant <= 0.0) {
    fprintf(stderr, "FAIL: %s reference emitter unlit\n", rendererType);
    return false;
  }
  if (relErr > tol) {
    fprintf(stderr,
        "FAIL: %s %s emitter not equivalent to its averaged constant (relErr=%f > %f)\n",
        rendererType,
        label,
        relErr,
        tol);
    return false;
  }
  return true;
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);

  const EmitterFactory quadTex = [](ANARIDevice d) {
    return makeEmissiveQuad(d, TEXTURED);
  };
  const EmitterFactory quadAttr = [](ANARIDevice d) {
    return makeEmissiveQuad(d, ATTRIBUTE);
  };
  const EmitterFactory quadConst = [](ANARIDevice d) {
    return makeEmissiveQuad(d, CONSTANT);
  };
  const EmitterFactory sphereTex = [](ANARIDevice d) {
    return makeEmissiveSphere(d, true);
  };
  const EmitterFactory sphereConst = [](ANARIDevice d) {
    return makeEmissiveSphere(d, false);
  };

  // 4% for the path tracer; interactive's single-bounce direct term is noisier
  // and approximates differently, so it gets more headroom.
  bool ok = checkEmitter(device, quadTex, quadConst, "quad-textured", "quality", 0.04);
  ok = checkEmitter(device, quadAttr, quadConst, "quad-attribute", "quality", 0.04) && ok;
  ok = checkEmitter(device, sphereTex, sphereConst, "sphere-textured", "quality", 0.04) && ok;
  ok = checkEmitter(device, quadTex, quadConst, "quad-textured", "interactive", 0.08) && ok;
  ok = checkEmitter(device, quadAttr, quadConst, "quad-attribute", "interactive", 0.08) && ok;
  ok = checkEmitter(device, sphereTex, sphereConst, "sphere-textured", "interactive", 0.08) && ok;

  anari::release(device, device);

  if (!ok)
    return 1;
  printf("emissive textured/attribute geometry light equivalence passed\n");
  return 0;
}
