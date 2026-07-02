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

// Quantitative counterpart of generate_emissive_materialx_comparison: the
// `materialx` emissive material must reach the SAME Geometry Light parity the
// `mdl` and `physicallyBased` emitters already have. A standard_surface with
// `emission`>0 (or a wired emissive `emission_color` texture) must:
//   (a) synthesize exactly one Geometry Light (numLightInstances == 1), and
//   (b) light a shared floor to the same mean as the physicallyBased and raw
//       `mdl` emitters authored for the same radiance.
// MaterialX IS-A MDL, so its transcoded standard_surface emission goes through
// the MDL emission classifier; if that classifier does not recognize the
// pattern, no light is synthesized and this test fails at (a)/(b). This is the
// acceptance test for teaching the classifier the standard_surface pattern.
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

// `value * math::PI`: MDL intensity is radiant exitance, radiance = intensity/PI.
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

// MaterialX: standard_surface's shadergen owns the EDF normalization, so the
// target radiance goes straight into `emission` (no app-side PI). The literal
// below must equal RADIANCE.
static_assert(RADIANCE == 8.f, "MDX_CONSTANT emission literal is stale");
static const char *MDX_CONSTANT = R"mtlx(<?xml version="1.0"?>
<materialx version="1.39">
  <standard_surface name="srf" type="surfaceshader">
    <input name="base" type="float" value="0.0"/>
    <input name="specular" type="float" value="0.0"/>
    <input name="emission" type="float" value="8.0"/>
    <input name="emission_color" type="color3" value="1.0, 1.0, 1.0"/>
  </standard_surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="srf"/>
  </surfacematerial>
</materialx>
)mtlx";

// `emission_color` wired to an <image> whose `file` is bound to the shared
// checker sampler; `emission` weight is 1 so radiance == sampled color.
static const char *MDX_SAMPLED = R"mtlx(<?xml version="1.0"?>
<materialx version="1.39">
  <image name="emissionTex" type="color3">
    <input name="file" type="filename" value=""/>
  </image>
  <standard_surface name="srf" type="surfaceshader">
    <input name="base" type="float" value="0.0"/>
    <input name="specular" type="float" value="0.0"/>
    <input name="emission" type="float" value="1.0"/>
    <input name="emission_color" type="color3" nodename="emissionTex"/>
  </standard_surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="srf"/>
  </surfacematerial>
</materialx>
)mtlx";

enum class Impl
{
  PBR,
  MDL,
  MATERIALX
};
static const char *implName(Impl impl)
{
  switch (impl) {
  case Impl::PBR:
    return "physicallyBased";
  case Impl::MDL:
    return "mdl";
  default:
    return "materialx";
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

// Shared 8x8 checker at 2*RADIANCE / 0 (mean RADIANCE), nearest so the texel
// values stay exact. Drives the native/mdl/materialx samplers.
static anari::Sampler makeEmissionSampler(ANARIDevice device)
{
  constexpr int N = 8;
  std::array<vec3, N * N> texels;
  for (int y = 0; y < N; ++y) {
    for (int x = 0; x < N; ++x) {
      const float v = ((x + y) & 1) ? 2.f * RADIANCE : 0.f;
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
  } else if (impl == Impl::MATERIALX) {
    mat = anari::newObject<anari::Material>(device, "materialx");
    anari::setParameter(device, mat, "sourceType", "documentInline");
    if (sampled) {
      anari::setParameter(device, mat, "source", MDX_SAMPLED);
      anari::setAndReleaseParameter(
          device, mat, "emissionTex/file", makeEmissionSampler(device));
    } else {
      anari::setParameter(device, mat, "source", MDX_CONSTANT);
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

static anari::Surface makeEmitter(ANARIDevice device, Impl impl, bool sampled)
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
  return surface;
}

// The emitter must synthesize exactly one Geometry Light. This is the crisp
// signal for the classifier gap: a MaterialX emissive surface that fails to
// register a light reports 0 here.
static bool checkLightCount(ANARIDevice device, Impl impl, bool sampled)
{
  auto surface = makeEmitter(device, impl, sampled);
  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "surface", &surface, 1);
  anari::release(device, surface);
  anari::commitParameters(device, world);

  uint32_t count = ~0u;
  const bool found =
      anari::getProperty(device, world, "numLightInstances", count, ANARI_WAIT);
  anari::release(device, world);

  printf("numLightInstances(%s %s)=%u (expected 1)\n",
      implName(impl),
      sampled ? "sampled" : "constant",
      count);
  if (!found) {
    fprintf(stderr, "FAIL: world has no numLightInstances property\n");
    return false;
  }
  if (count != 1u) {
    fprintf(stderr,
        "FAIL: %s %s synthesized %u Geometry Lights, expected 1\n",
        implName(impl),
        sampled ? "sampled" : "constant",
        count);
    return false;
  }
  return true;
}

static anari::World makeWorld(ANARIDevice device, Impl impl, bool sampled)
{
  const std::array<anari::Surface, 2> surfaces = {
      makeFloor(device), makeEmitter(device, impl, sampled)};
  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(
      device, world, "surface", surfaces.data(), surfaces.size());
  for (auto s : surfaces)
    anari::release(device, s);
  anari::commitParameters(device, world);
  return world;
}

// Mean floor luminance under a single emitter of the given implementation.
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
    fprintf(
        stderr, "FAIL: %s not equivalent (relErr=%f > %f)\n", label, relErr, tol);
    return false;
  }
  return true;
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);

  bool ok = true;

  // (a) The Geometry Light must be synthesized at all — the crisp signal.
  for (bool sampled : {false, true}) {
    ok = checkLightCount(device, Impl::PBR, sampled) && ok;
    ok = checkLightCount(device, Impl::MDL, sampled) && ok;
    ok = checkLightCount(device, Impl::MATERIALX, sampled) && ok;
  }

  // (b) Floor-pool parity against physicallyBased and raw mdl, both renderers.
  for (const char *renderer : {"quality", "interactive"}) {
    const double tol = renderer[0] == 'q' ? 0.04 : 0.08;
    for (bool sampled : {false, true}) {
      const double pbr = poolMean(device, Impl::PBR, sampled, renderer);
      const double mdl = poolMean(device, Impl::MDL, sampled, renderer);
      const double mdx = poolMean(device, Impl::MATERIALX, sampled, renderer);
      char label[128];
      snprintf(label,
          sizeof(label),
          "%s/%s materialx-vs-pbr",
          renderer,
          sampled ? "sampled" : "constant");
      ok = checkParity(label, mdx, pbr, tol) && ok;
      snprintf(label,
          sizeof(label),
          "%s/%s materialx-vs-mdl",
          renderer,
          sampled ? "sampled" : "constant");
      ok = checkParity(label, mdx, mdl, tol) && ok;
    }
  }

  anari::release(device, device);

  if (!ok)
    return 1;
  printf("emissive materialx parity passed\n");
  return 0;
}
