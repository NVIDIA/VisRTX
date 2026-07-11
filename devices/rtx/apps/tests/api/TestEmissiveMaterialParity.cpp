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

// Cross-implementation emission parity: the three emissive material
// implementations — physicallyBased, physicallyBasedMDL, and raw `mdl` (inline
// source) — must light the same floor identically for the same authored
// radiance, with both constant and sampled (textured) emission. The `mdl`
// sources author `intensity = value * PI` because MDL emission intensity is
// radiant EXITANCE (radiance = intensity/PI); omitting the PI factor makes an
// MDL emitter read ~3x dimmer than the equivalent physicallyBased one — the
// classic authoring trap this test also documents. The quantitative
// counterpart of generate_emissive_mdl_comparison (TSD procedural scene).
// Sampled (textured) emission is next-event sampled on ALL THREE
// implementations — the wrapper's bound emissive sampler included — so parity
// must hold under 'interactive' (next-event only) as well as 'quality'.
// Linear float buffer, firefly off.

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
// Authored emitted radiance, identical across implementations. The checker's
// bright texel is 2x so its mean matches the constant emitters.
static constexpr float RADIANCE = 8.f;
static constexpr float QUAD_Y = 1.5f;
static constexpr float QUAD_HALF = 0.5f;

// `value * math::PI`: intensity is radiant exitance, radiance = intensity/PI.
static const char *MDL_CONSTANT = R"mdl(mdl 1.6;
import ::df::*;
import ::math::*;
export material emissive(color value = color(1.0)) = material(
    surface: material_surface(
        emission: material_emission(
            emission: df::diffuse_edf(),
            intensity: value * math::PI)));
)mdl";

static const char *MDL_SAMPLED = R"mdl(mdl 1.6;
import ::df::*;
import ::math::*;
import ::state::*;
import ::tex::*;
export material emissive_tex(uniform texture_2d tex = texture_2d()) = material(
    surface: material_surface(
        emission: material_emission(
            emission: df::diffuse_edf(),
            intensity: tex::lookup_color(
                tex: tex,
                coord: float2(
                    state::texture_coordinate(0).x,
                    state::texture_coordinate(0).y)) * math::PI)));
)mdl";

enum class Impl
{
  PBR,
  WRAPPER,
  MDL
};
static const char *implName(Impl impl)
{
  switch (impl) {
  case Impl::PBR:
    return "physicallyBased";
  case Impl::WRAPPER:
    return "physicallyBasedMDL";
  default:
    return "mdl";
  }
}

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

// Shared 8x8 checker at 2*RADIANCE*scale / 0 (mean RADIANCE*scale), nearest so
// the texel values stay exact; scale 0 gives the all-black texture. It drives
// the native/wrapper samplers via `inAttribute`; the mdl source reads its
// coordinate from state::texture_coordinate(0) instead.
static anari::Sampler makeEmissionSampler(ANARIDevice device, float scale = 1.f)
{
  constexpr int N = 8;
  std::array<vec3, N * N> texels;
  for (int y = 0; y < N; ++y) {
    for (int x = 0; x < N; ++x) {
      const float v = ((x + y) & 1) ? 2.f * RADIANCE * scale : 0.f;
      texels[y * N + x] = vec3{v, v, v};
    }
  }
  auto image = anari::newArray2D(device, texels.data(), N, N);

  auto sampler = anari::newObject<anari::Sampler>(device, "image2D");
  anari::setAndReleaseParameter(device, sampler, "image", image);
  anari::setParameter(device, sampler, "inAttribute", "attribute0");
  anari::setParameter(device, sampler, "filter", "nearest");
  anari::commitParameters(device, sampler);
  return sampler;
}

static anari::Material makeEmitterMaterial(
    ANARIDevice device, Impl impl, bool sampled)
{
  anari::Material mat;
  if (impl == Impl::MDL) {
    mat = anari::newObject<anari::Material>(device, "mdl");
    anari::setParameter(device, mat, "sourceType", "code");
    if (sampled) {
      anari::setParameter(device, mat, "source", MDL_SAMPLED);
      anari::setParameter(device, mat, "materialName", "emissive_tex");
      anari::setAndReleaseParameter(
          device, mat, "tex", makeEmissionSampler(device));
    } else {
      anari::setParameter(device, mat, "source", MDL_CONSTANT);
      anari::setParameter(device, mat, "materialName", "emissive");
      anari::setParameter(
          device, mat, "value", vec3{RADIANCE, RADIANCE, RADIANCE});
    }
  } else {
    mat = anari::newObject<anari::Material>(device, implName(impl));
    anari::setParameter(device, mat, "baseColor", vec3{0.f, 0.f, 0.f});
    anari::setParameter(device, mat, "metallic", 0.f);
    anari::setParameter(device, mat, "roughness", 1.f);
    if (sampled) {
      anari::setAndReleaseParameter(
          device, mat, "emissive", makeEmissionSampler(device));
    } else {
      anari::setParameter(
          device, mat, "emissive", vec3{RADIANCE, RADIANCE, RADIANCE});
    }
  }
  anari::commitParameters(device, mat);
  return mat;
}

// Pick Power is variance-only, so no floor-pool parity can pin the LIVE
// sampler mean — pin it through the light count instead: an all-black bound
// emissive texture has a zero mean and must synthesize NO Geometry Light (a
// unit-proxy or stale-default mean would count one), while the checker must
// synthesize exactly one. Uses the always-exposed `physicallyBasedMDL` subtype
// so this holds regardless of the PBR-backend build option.
static bool checkSampledWrapperLightCount(
    ANARIDevice device, bool black, uint32_t expected)
{
  const std::array<vec3, 4> pos = {vec3{-QUAD_HALF, QUAD_Y, -QUAD_HALF},
      vec3{QUAD_HALF, QUAD_Y, -QUAD_HALF},
      vec3{QUAD_HALF, QUAD_Y, QUAD_HALF},
      vec3{-QUAD_HALF, QUAD_Y, QUAD_HALF}};
  const std::array<vec2, 4> uv = {
      vec2{0.f, 0.f}, vec2{1.f, 0.f}, vec2{1.f, 1.f}, vec2{0.f, 1.f}};
  const std::array<std::array<unsigned, 3>, 2> idx = {
      std::array<unsigned, 3>{0, 1, 2}, std::array<unsigned, 3>{0, 2, 3}};

  auto geom = anari::newObject<anari::Geometry>(device, "triangle");
  anari::setParameterArray1D(device, geom, "vertex.position", pos.data(), 4);
  anari::setParameterArray1D(device, geom, "vertex.attribute0", uv.data(), 4);
  anari::setParameterArray1D(device, geom, "primitive.index", idx.data(), 2);
  anari::commitParameters(device, geom);

  auto mat = anari::newObject<anari::Material>(device, "physicallyBasedMDL");
  anari::setParameter(device, mat, "baseColor", vec3{0.f, 0.f, 0.f});
  anari::setAndReleaseParameter(
      device, mat, "emissive", makeEmissionSampler(device, black ? 0.f : 1.f));
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

  printf("numLightInstances(wrapper sampled %s)=%u (expected %u)\n",
      black ? "black" : "checker",
      count,
      expected);
  if (!found) {
    fprintf(stderr, "FAIL: world has no numLightInstances property\n");
    return false;
  }
  if (count != expected) {
    fprintf(stderr,
        "FAIL: sampled wrapper (%s): expected %u synthesized light(s), got %u\n",
        black ? "black" : "checker",
        expected,
        count);
    return false;
  }
  return true;
}

static anari::World makeWorld(ANARIDevice device, Impl impl, bool sampled)
{
  const std::array<vec3, 4> pos = {vec3{-QUAD_HALF, QUAD_Y, -QUAD_HALF},
      vec3{QUAD_HALF, QUAD_Y, -QUAD_HALF},
      vec3{QUAD_HALF, QUAD_Y, QUAD_HALF},
      vec3{-QUAD_HALF, QUAD_Y, QUAD_HALF}};
  const std::array<vec2, 4> uv = {
      vec2{0.f, 0.f}, vec2{1.f, 0.f}, vec2{1.f, 1.f}, vec2{0.f, 1.f}};
  const std::array<std::array<unsigned, 3>, 2> idx = {
      std::array<unsigned, 3>{0, 1, 2}, std::array<unsigned, 3>{0, 2, 3}};

  auto geom = anari::newObject<anari::Geometry>(device, "triangle");
  anari::setParameterArray1D(device, geom, "vertex.position", pos.data(), 4);
  anari::setParameterArray1D(device, geom, "vertex.attribute0", uv.data(), 4);
  anari::setParameterArray1D(device, geom, "primitive.index", idx.data(), 2);
  anari::commitParameters(device, geom);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(
      device, surface, "material", makeEmitterMaterial(device, impl, sampled));
  anari::commitParameters(device, surface);

  const std::array<anari::Surface, 2> surfaces = {makeFloor(device), surface};
  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(
      device, world, "surface", surfaces.data(), surfaces.size());
  for (auto s : surfaces)
    anari::release(device, s);
  anari::commitParameters(device, world);
  return world;
}

static double poolMean(
    ANARIDevice device, Impl impl, bool sampled, const char *rendererType)
{
  auto world = makeWorld(device, impl, sampled);

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
  anari::release(device, frame);
  return n ? sum / double(n) : 0.0;
}

static bool checkParity(const char *label, double a, double b, double tol)
{
  const double relErr = b > 0.0 ? std::abs(a - b) / b : (a > 0 ? 1.0 : 0.0);
  printf("%s: %f vs %f  relErr=%f\n", label, a, b, relErr);
  if (!std::isfinite(a) || !std::isfinite(b)) {
    fprintf(stderr, "FAIL: %s produced a non-finite pool mean\n", label);
    return false;
  }
  if (a <= 0.0 || b <= 0.0) {
    fprintf(stderr, "FAIL: %s left the floor dark\n", label);
    return false;
  }
  if (relErr > tol) {
    fprintf(stderr,
        "FAIL: %s not equivalent (relErr=%f > %f)\n",
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

  bool ok = true;

  // Constant emission: all three implementations must agree, both renderers,
  // under either PBR backend.
  for (const char *renderer : {"quality", "interactive"}) {
    const double tol = renderer[0] == 'q' ? 0.04 : 0.08;
    const double cPbr = poolMean(device, Impl::PBR, false, renderer);
    const double cWrap = poolMean(device, Impl::WRAPPER, false, renderer);
    const double cMdl = poolMean(device, Impl::MDL, false, renderer);
    char label[128];
    snprintf(label, sizeof(label), "%s/constant wrapper-vs-pbr", renderer);
    ok = checkParity(label, cWrap, cPbr, tol) && ok;
    snprintf(label, sizeof(label), "%s/constant mdl-vs-pbr", renderer);
    ok = checkParity(label, cMdl, cPbr, tol) && ok;
  }

  // Sampled emission under 'quality': every implementation converges (the
  // wrapper through its path-hit deposit), and the checker mean must equal the
  // constant row.
  {
    const double cPbr = poolMean(device, Impl::PBR, false, "quality");
    const double sPbr = poolMean(device, Impl::PBR, true, "quality");
    const double sWrap = poolMean(device, Impl::WRAPPER, true, "quality");
    const double sMdl = poolMean(device, Impl::MDL, true, "quality");
    ok = checkParity("quality/sampled mdl-vs-pbr", sMdl, sPbr, 0.04) && ok;
    ok = checkParity("quality/sampled wrapper-vs-pbr", sWrap, sPbr, 0.04) && ok;
    ok = checkParity("quality/sampled-vs-constant pbr", sPbr, cPbr, 0.04) && ok;
  }

  // Sampled emission under 'interactive' (next-event only): raw mdl must match
  // its own constant row (its textured next-event path).
  {
    const double cMdl = poolMean(device, Impl::MDL, false, "interactive");
    const double sMdl = poolMean(device, Impl::MDL, true, "interactive");
    ok = checkParity("interactive/sampled-vs-constant mdl", sMdl, cMdl, 0.08)
        && ok;

    const double sPbr = poolMean(device, Impl::PBR, true, "interactive");
    ok = checkParity("interactive/sampled pbr-vs-mdl", sPbr, sMdl, 0.08) && ok;

    // Wrapper sampled emission is next-event sampled too (live sampler-mean
    // Pick Power, device EDF at the synthetic hit): full parity, no gap.
    const double sWrap = poolMean(device, Impl::WRAPPER, true, "interactive");
    ok = checkParity("interactive/sampled wrapper-vs-mdl", sWrap, sMdl, 0.08)
        && ok;
  }

  // The live sampler mean itself, via the light count (see the helper).
  ok = checkSampledWrapperLightCount(device, true, 0) && ok;
  ok = checkSampledWrapperLightCount(device, false, 1) && ok;

  anari::release(device, device);

  if (!ok)
    return 1;
  printf("emissive material parity passed\n");
  return 0;
}
