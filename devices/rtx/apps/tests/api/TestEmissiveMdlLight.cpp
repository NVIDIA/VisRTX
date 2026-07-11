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

// A raw `mdl` material with author-declared constant diffuse emission must be
// synthesized into a Geometry Light (per ADR 0006). The compiled EDF authors
// intensity as radiant exitance, so a body-literal `intensity: color(K)` emits
// radiance K/PI — the floor it lights must match a native physicallyBased
// constant emitter of radiance K/PI, on 'quality' (deposit + MIS) and
// 'interactive' (next-event only; the renderer that goes dark when the light is
// missing). An away-facing winding must light the floor identically (triangle
// Geometry Lights are double-sided; the synthetic next-event normal is oriented
// toward the receiver). A zero-intensity `mdl` material must synthesize NO
// light — asserted via the world's `numLightInstances` property in a
// no-authored-light scene, since a zero-radiance light is pixel-identical to
// no light. A texture-driven intensity cannot fold at compile time, so it is
// sampleable with a unit-luminance proxy Pick Power and the device evaluates
// the true per-point radiance at the sampled point: a uniform texture must
// match the constant reference, a checker must match the reference at its
// mean, and an UNBOUND texture still counts as a light (documented
// conservative over-inclusion). Linear float buffer, firefly off.

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
static constexpr int PIXEL_SAMPLES = 512;
// Body-literal MDL emission intensity (radiant exitance); emitted radiance is
// INTENSITY / PI. The native reference emitter is authored at that radiance.
// Must match the `color(16.0)` literal inside MDL_EMISSIVE below.
static constexpr float INTENSITY = 16.f;
static constexpr float PI = 3.14159265358979323846f;
static constexpr float QUAD_Y = 1.5f;
static constexpr float QUAD_HALF = 0.5f;

// Constant diffuse emission, body-literal intensity: the compile-time
// classification must fold it and register a Geometry Light.
static const char *MDL_EMISSIVE = R"mdl(mdl 1.6;
import ::df::*;
export material emissive() = material(
    surface: material_surface(
        emission: material_emission(
            emission: df::diffuse_edf(),
            intensity: color(16.0))));
)mdl";

// Same shape, provably-zero intensity: must NOT become a light.
static const char *MDL_DARK = R"mdl(mdl 1.6;
import ::df::*;
export material dark() = material(
    surface: material_surface(
        emission: material_emission(
            emission: df::diffuse_edf(),
            intensity: color(0.0))));
)mdl";

// Intensity folding to a non-finite constant: must NOT become a light — the
// classifier disqualifies it entirely, or the textured branch would make it
// sampleable and next-event estimation would spray the NaN/Inf to every
// receiver the pick selects. (If the MDL compiler ever rejects the overflow
// fold outright, the material fails to compile and is equally non-emissive.)
static const char *MDL_NONFINITE = R"mdl(mdl 1.6;
import ::df::*;
export material nonfinite() = material(
    surface: material_surface(
        emission: material_emission(
            emission: df::diffuse_edf(),
            intensity: color(1.0e38 * 1.0e38))));
)mdl";

// Texture-driven diffuse intensity: does not fold at compile time, so the
// classification records diffuse-but-textured — sampleable with a unit-proxy
// Pick Power, radiance evaluated on the device at the sampled point.
static const char *MDL_TEXTURED = R"mdl(mdl 1.6;
import ::df::*;
import ::tex::*;
import ::state::*;
export material emissive_tex(uniform texture_2d tex = texture_2d()) = material(
    surface: material_surface(
        emission: material_emission(
            emission: df::diffuse_edf(),
            intensity: tex::lookup_color(
                tex: tex,
                coord: float2(
                    state::texture_coordinate(0).x,
                    state::texture_coordinate(0).y)))));
)mdl";

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

enum class Emitter
{
  MDL,
  MDL_AWAY_FACING,
  MDL_TEXTURED_UNIFORM, // uniform INTENSITY texture: radiance INTENSITY/PI
  MDL_TEXTURED_CHECKER, // INTENSITY/0 checker: mean radiance INTENSITY/(2 PI)
  MDL_TEXTURED_AWAY, // uniform texture, reversed winding
  NATIVE_REFERENCE, // constant radiance INTENSITY/PI
  NATIVE_REFERENCE_HALF // constant radiance INTENSITY/(2 PI)
};

// image2D sampler for the MDL texture argument. `checker` selects a
// nearest-filtered NxN INTENSITY/0 checker (mean INTENSITY/2 per channel);
// otherwise every texel is INTENSITY. No `inAttribute`: the MDL expression
// supplies the coordinate (state::texture_coordinate(0)).
static anari::Sampler makeIntensitySampler(ANARIDevice device, bool checker)
{
  constexpr int N = 8;
  std::array<vec3, N * N> texels;
  for (int y = 0; y < N; ++y) {
    for (int x = 0; x < N; ++x) {
      const float v = (!checker || ((x + y) & 1)) ? INTENSITY : 0.f;
      texels[y * N + x] = vec3{v, v, v};
    }
  }
  auto image = anari::newArray2D(device, texels.data(), N, N);

  auto sampler = anari::newObject<anari::Sampler>(device, "image2D");
  anari::setAndReleaseParameter(device, sampler, "image", image);
  anari::setParameter(device, sampler, "filter", "nearest");
  anari::commitParameters(device, sampler);
  return sampler;
}

// Emissive quad above the floor. MDL variants use the inline `code` source;
// the native references are physicallyBased constant emitters at the matching
// mean radiance. AWAY variants reverse the winding so the geometric normal
// points up, away from the floor. The quad carries attribute0 texcoords
// spanning [0,1]^2 for the textured cases; the pool is far enough that each
// receiver point integrates the whole emitter, so a textured emitter matches
// the constant reference at its mean.
static anari::Surface makeEmissiveQuad(ANARIDevice device, Emitter kind)
{
  const std::array<vec3, 4> pos = {vec3{-QUAD_HALF, QUAD_Y, -QUAD_HALF},
      vec3{QUAD_HALF, QUAD_Y, -QUAD_HALF},
      vec3{QUAD_HALF, QUAD_Y, QUAD_HALF},
      vec3{-QUAD_HALF, QUAD_Y, QUAD_HALF}};
  const std::array<vec2, 4> uv = {
      vec2{0.f, 0.f}, vec2{1.f, 0.f}, vec2{1.f, 1.f}, vec2{0.f, 1.f}};
  const std::array<std::array<unsigned, 3>, 2> idxDown = {
      std::array<unsigned, 3>{0, 1, 2}, std::array<unsigned, 3>{0, 2, 3}};
  const std::array<std::array<unsigned, 3>, 2> idxUp = {
      std::array<unsigned, 3>{0, 2, 1}, std::array<unsigned, 3>{0, 3, 2}};
  const bool awayFacing =
      kind == Emitter::MDL_AWAY_FACING || kind == Emitter::MDL_TEXTURED_AWAY;
  const auto &idx = awayFacing ? idxUp : idxDown;

  auto geom = anari::newObject<anari::Geometry>(device, "triangle");
  anari::setParameterArray1D(device, geom, "vertex.position", pos.data(), 4);
  anari::setParameterArray1D(device, geom, "vertex.attribute0", uv.data(), 4);
  anari::setParameterArray1D(device, geom, "primitive.index", idx.data(), 2);
  anari::commitParameters(device, geom);

  anari::Material mat;
  switch (kind) {
  case Emitter::NATIVE_REFERENCE:
  case Emitter::NATIVE_REFERENCE_HALF: {
    mat = anari::newObject<anari::Material>(device, "physicallyBased");
    anari::setParameter(device, mat, "baseColor", vec3{0.f, 0.f, 0.f});
    anari::setParameter(device, mat, "metallic", 0.f);
    anari::setParameter(device, mat, "roughness", 1.f);
    const float radiance = kind == Emitter::NATIVE_REFERENCE_HALF
        ? INTENSITY / (2.f * PI)
        : INTENSITY / PI;
    anari::setParameter(
        device, mat, "emissive", vec3{radiance, radiance, radiance});
    break;
  }
  case Emitter::MDL_TEXTURED_UNIFORM:
  case Emitter::MDL_TEXTURED_CHECKER:
  case Emitter::MDL_TEXTURED_AWAY: {
    mat = anari::newObject<anari::Material>(device, "mdl");
    anari::setParameter(device, mat, "sourceType", "code");
    anari::setParameter(device, mat, "source", MDL_TEXTURED);
    anari::setParameter(device, mat, "materialName", "emissive_tex");
    anari::setAndReleaseParameter(device,
        mat,
        "tex",
        makeIntensitySampler(device, kind == Emitter::MDL_TEXTURED_CHECKER));
    break;
  }
  case Emitter::MDL:
  case Emitter::MDL_AWAY_FACING: {
    mat = anari::newObject<anari::Material>(device, "mdl");
    anari::setParameter(device, mat, "sourceType", "code");
    anari::setParameter(device, mat, "source", MDL_EMISSIVE);
    anari::setParameter(device, mat, "materialName", "emissive");
    break;
  }
  }
  anari::commitParameters(device, mat);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);
  return surface;
}

static anari::World makeWorld(ANARIDevice device, Emitter kind)
{
  const std::array<anari::Surface, 2> surfaces = {
      makeFloor(device), makeEmissiveQuad(device, kind)};
  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(
      device, world, "surface", surfaces.data(), surfaces.size());
  for (auto s : surfaces)
    anari::release(device, s);
  anari::commitParameters(device, world);
  return world;
}

static std::vector<vec4> render(
    ANARIDevice device, anari::World world, const char *rendererType)
{
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
  anari::setParameter(device, frame, "world", world);
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

static bool checkEquivalence(ANARIDevice device,
    Emitter kind,
    Emitter refKind,
    const char *label,
    const char *rendererType,
    double tol)
{
  auto mdlWorld = makeWorld(device, kind);
  auto refWorld = makeWorld(device, refKind);
  const double mdl = poolMeanLuminance(render(device, mdlWorld, rendererType));
  const double ref = poolMeanLuminance(render(device, refWorld, rendererType));
  anari::release(device, mdlWorld);
  anari::release(device, refWorld);

  const double relErr =
      ref > 0.0 ? std::abs(mdl - ref) / ref : (mdl > 0 ? 1.0 : 0.0);
  printf("%s/%s: mdl=%f  reference=%f  relErr=%f\n",
      rendererType,
      label,
      mdl,
      ref,
      relErr);

  // A NaN pool would sail through the sign checks below.
  if (!std::isfinite(mdl) || !std::isfinite(ref)) {
    fprintf(stderr,
        "FAIL: %s/%s produced a non-finite pool mean\n",
        rendererType,
        label);
    return false;
  }
  if (mdl <= 0.0) {
    fprintf(stderr,
        "FAIL: %s/%s mdl emitter left floor dark\n",
        rendererType,
        label);
    return false;
  }
  if (ref <= 0.0) {
    fprintf(stderr, "FAIL: %s reference emitter unlit\n", rendererType);
    return false;
  }
  if (relErr > tol) {
    fprintf(stderr,
        "FAIL: %s/%s mdl emission not equivalent to native reference "
        "(relErr=%f > %f)\n",
        rendererType,
        label,
        relErr,
        tol);
    return false;
  }
  return true;
}

// Light count in a no-authored-light scene == synthesized Geometry Lights.
static bool checkLightCount(
    ANARIDevice device, const char *source, const char *name, uint32_t expected)
{
  const std::array<vec3, 4> pos = {vec3{-QUAD_HALF, QUAD_Y, -QUAD_HALF},
      vec3{QUAD_HALF, QUAD_Y, -QUAD_HALF},
      vec3{QUAD_HALF, QUAD_Y, QUAD_HALF},
      vec3{-QUAD_HALF, QUAD_Y, QUAD_HALF}};
  const std::array<std::array<unsigned, 3>, 2> idx = {
      std::array<unsigned, 3>{0, 1, 2}, std::array<unsigned, 3>{0, 2, 3}};

  auto geom = anari::newObject<anari::Geometry>(device, "triangle");
  anari::setParameterArray1D(device, geom, "vertex.position", pos.data(), 4);
  anari::setParameterArray1D(device, geom, "primitive.index", idx.data(), 2);
  anari::commitParameters(device, geom);

  auto mat = anari::newObject<anari::Material>(device, "mdl");
  anari::setParameter(device, mat, "sourceType", "code");
  anari::setParameter(device, mat, "source", source);
  anari::setParameter(device, mat, "materialName", name);
  anari::commitParameters(device, mat);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "surface", &surface, 1);
  anari::release(device, surface);
  anari::commitParameters(device, world);

  uint32_t count = ~0u;
  const bool found =
      anari::getProperty(device, world, "numLightInstances", count, ANARI_WAIT);
  anari::release(device, world);

  printf("numLightInstances(%s)=%u (expected %u)\n", name, count, expected);
  if (!found) {
    fprintf(stderr, "FAIL: world has no numLightInstances property\n");
    return false;
  }
  if (count != expected) {
    fprintf(stderr,
        "FAIL: %s: expected %u synthesized light(s), got %u\n",
        name,
        expected,
        count);
    return false;
  }
  return true;
}

// Same exclusion assertion for the MDL-backed PhysicallyBased wrapper: its
// committed `emissive` value decides sampleability, so a black wrapper must
// synthesize no light (US6). Uses the always-exposed `physicallyBasedMDL`
// subtype so this holds regardless of the PBR-backend build option.
static bool checkWrapperLightCount(
    ANARIDevice device, float emissive, uint32_t expected)
{
  const std::array<vec3, 4> pos = {vec3{-QUAD_HALF, QUAD_Y, -QUAD_HALF},
      vec3{QUAD_HALF, QUAD_Y, -QUAD_HALF},
      vec3{QUAD_HALF, QUAD_Y, QUAD_HALF},
      vec3{-QUAD_HALF, QUAD_Y, QUAD_HALF}};
  const std::array<std::array<unsigned, 3>, 2> idx = {
      std::array<unsigned, 3>{0, 1, 2}, std::array<unsigned, 3>{0, 2, 3}};

  auto geom = anari::newObject<anari::Geometry>(device, "triangle");
  anari::setParameterArray1D(device, geom, "vertex.position", pos.data(), 4);
  anari::setParameterArray1D(device, geom, "primitive.index", idx.data(), 2);
  anari::commitParameters(device, geom);

  auto mat = anari::newObject<anari::Material>(device, "physicallyBasedMDL");
  anari::setParameter(device, mat, "baseColor", vec3{0.f, 0.f, 0.f});
  anari::setParameter(
      device, mat, "emissive", vec3{emissive, emissive, emissive});
  anari::commitParameters(device, mat);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "surface", &surface, 1);
  anari::release(device, surface);
  anari::commitParameters(device, world);

  uint32_t count = ~0u;
  const bool found =
      anari::getProperty(device, world, "numLightInstances", count, ANARI_WAIT);
  anari::release(device, world);

  printf("numLightInstances(wrapper emissive=%g)=%u (expected %u)\n",
      emissive,
      count,
      expected);
  if (!found) {
    fprintf(stderr, "FAIL: world has no numLightInstances property\n");
    return false;
  }
  if (count != expected) {
    fprintf(stderr,
        "FAIL: wrapper emissive=%g: expected %u synthesized light(s), got %u\n",
        emissive,
        expected,
        count);
    return false;
  }
  return true;
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);

  const auto REF = Emitter::NATIVE_REFERENCE;
  const auto REF_HALF = Emitter::NATIVE_REFERENCE_HALF;
  bool ok =
      checkEquivalence(device, Emitter::MDL, REF, "quad", "quality", 0.04);
  ok = checkEquivalence(device, Emitter::MDL, REF, "quad", "interactive", 0.08)
      && ok;
  // Away-facing winding: triangle Geometry Lights are double-sided, so the
  // floor must read the same. Interactive is the sharp signal — without the
  // receiver-facing orientation of the synthetic next-event normal, the
  // single-sided EDF returns 0 and the floor goes dark.
  ok =
      checkEquivalence(
          device, Emitter::MDL_AWAY_FACING, REF, "away-facing", "quality", 0.04)
      && ok;
  ok = checkEquivalence(device,
           Emitter::MDL_AWAY_FACING,
           REF,
           "away-facing",
           "interactive",
           0.08)
      && ok;

  // Textured intensity: the classification cannot fold it, so the light rides
  // the unit-proxy Pick Power and the device evaluates the true per-point
  // radiance at the sampled point. A uniform texture must match the constant
  // reference exactly; the checker must match the reference at its mean.
  ok = checkEquivalence(device,
           Emitter::MDL_TEXTURED_UNIFORM,
           REF,
           "textured-uniform",
           "quality",
           0.04)
      && ok;
  ok = checkEquivalence(device,
           Emitter::MDL_TEXTURED_UNIFORM,
           REF,
           "textured-uniform",
           "interactive",
           0.08)
      && ok;
  ok = checkEquivalence(device,
           Emitter::MDL_TEXTURED_CHECKER,
           REF_HALF,
           "textured-checker",
           "quality",
           0.04)
      && ok;
  ok = checkEquivalence(device,
           Emitter::MDL_TEXTURED_CHECKER,
           REF_HALF,
           "textured-checker",
           "interactive",
           0.08)
      && ok;
  ok = checkEquivalence(device,
           Emitter::MDL_TEXTURED_AWAY,
           REF,
           "textured-away",
           "quality",
           0.04)
      && ok;
  ok = checkEquivalence(device,
           Emitter::MDL_TEXTURED_AWAY,
           REF,
           "textured-away",
           "interactive",
           0.08)
      && ok;

  ok = checkLightCount(device, MDL_EMISSIVE, "emissive", 1) && ok;
  ok = checkLightCount(device, MDL_DARK, "dark", 0) && ok;
  // Textured intensity with an UNBOUND texture still classifies as a light —
  // the documented conservative over-inclusion (unbiased; it merely wastes a
  // pick slot while rendering black).
  ok = checkLightCount(device, MDL_TEXTURED, "emissive_tex", 1) && ok;
  ok = checkLightCount(device, MDL_NONFINITE, "nonfinite", 0) && ok;
  ok = checkWrapperLightCount(device, 5.f, 1) && ok;
  ok = checkWrapperLightCount(device, 0.f, 0) && ok;

  anari::release(device, device);

  if (!ok)
    return 1;
  printf("raw mdl emission Geometry Light equivalence passed\n");
  return 0;
}
