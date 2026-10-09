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

// Exercise the public MDL scene-data/time contract without external modules.
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp/ext/std.h>
#include <anari/anari_cpp.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

using vec2 = std::array<float, 2>;
using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;
using uvec2 = std::array<unsigned, 2>;
static int errors = 0;
static int timeWarnings = 0;
static int metadataWarnings = 0;

static void statusFunc(const void *,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity <= ANARI_SEVERITY_ERROR) {
    ++errors;
    fprintf(stderr, "ANARI error: %s\n", message);
  }
  if (severity == ANARI_SEVERITY_WARNING
      && std::strstr(message, "mdl.animationTime must be finite"))
    ++timeWarnings;
  if (severity == ANARI_SEVERITY_WARNING && std::strstr(message, "sceneData.")
      && std::strstr(message, "is not a valid parameter for an MDL material")) {
    ++metadataWarnings;
    fprintf(stderr, "ANARI invalid metadata: %s\n", message);
  }
}

static void require(bool ok, const char *message)
{
  if (!ok)
    throw std::runtime_error(message);
}

static const char *source = R"mdl(mdl 1.7;
import ::df::*;
import ::scene::*;
import ::state::*;
export material main(uniform int mode = 0, uniform float gain = 1) = let {
  float scalar = scene::data_lookup_float("scalar", 0.25);
  float2 pair = scene::data_lookup_float2("pair", float2(0.1, 0.2));
  float3 triple = scene::data_lookup_float3("triple", float3(0.1, 0.2, 0.3));
  float4 quad = scene::data_lookup_float4("quad", float4(0.1, 0.2, 0.3, 0.4));
  color tint = scene::data_lookup_color("tint", color(0.1, 0.2, 0.3));
  uniform float constant = scene::data_lookup_uniform_float("scalar", 0.125);
  color value = mode == 0 ? color(scalar)
      : mode == 1 ? color(pair.x, pair.y, 0)
      : mode == 2 ? color(triple)
      : mode == 3 ? color(quad.x, quad.y, quad.w)
      : mode == 4 ? tint
      : mode == 5 ? color(constant)
      : mode == 6 ? color(scene::data_isvalid("scalar") ? 1 : 0)
      : mode == 7 ? color(state::animation_time())
      : mode == 8 ? scene::data_lookup_color("missing", color(0.7, 0.1, 0.8))
      : mode == 9 ? color(float(scene::data_lookup_int("scalar", 2)) * 0.1)
      : color(scene::data_lookup_uniform_float("scalar.uniform", 0.15));
} in material(surface: material_surface(
    scattering: df::diffuse_reflection_bsdf(tint: color(0)),
    emission: material_emission(emission: df::diffuse_edf(),
        intensity: value * gain * 3.14159265)));
)mdl";

struct Scene
{
  ANARIDevice device = makeVisRTXDevice(statusFunc);
  anari::Geometry geometry =
      anari::newObject<anari::Geometry>(device, "triangle");
  anari::Material material = anari::newObject<anari::Material>(device, "mdl");
  anari::Surface surface = anari::newObject<anari::Surface>(device);
  anari::Group group = anari::newObject<anari::Group>(device);
  anari::Instance instance =
      anari::newObject<anari::Instance>(device, "transform");
  anari::World world = anari::newObject<anari::World>(device);
  anari::Camera camera =
      anari::newObject<anari::Camera>(device, "orthographic");
  anari::Renderer renderer =
      anari::newObject<anari::Renderer>(device, "quality");
  anari::Frame frame = anari::newObject<anari::Frame>(device);

  Scene()
  {
    const char **extensions = nullptr;
    require(anariGetProperty(device,
                device,
                "extension",
                ANARI_STRING_LIST,
                &extensions,
                sizeof(extensions),
                ANARI_WAIT)
            != 0,
        "device extension query failed");
    for (const char *name :
        {"ANARI_VISRTX_MDL_SCENE_DATA", "ANARI_VISRTX_MDL_ANIMATION_TIME"}) {
      bool found = false;
      for (auto p = extensions; p && *p; ++p)
        found |= std::strcmp(*p, name) == 0;
      require(found, name);
    }
    const std::array<vec3, 4> positions = {
        vec3{-1, -1, 0}, vec3{1, -1, 0}, vec3{1, 1, 0}, vec3{-1, 1, 0}};
    const std::array<std::array<unsigned, 3>, 2> indices = {
        std::array<unsigned, 3>{0, 1, 2}, std::array<unsigned, 3>{0, 2, 3}};
    anari::setParameterArray1D(
        device, geometry, "vertex.position", positions.data(), 4);
    anari::setParameterArray1D(
        device, geometry, "primitive.index", indices.data(), 2);
    anari::setParameter(device, geometry, "attribute0", vec4{0.8f, 0, 0, 1});
    anari::commitParameters(device, geometry);
    anari::setParameter(device, material, "sourceType", "code");
    anari::setParameter(device, material, "source", source);
    anari::setParameter(device, material, "materialName", "main");
    anari::setParameter(device, material, "sceneData.scalar", "attribute0");
    anari::setParameter(device, material, "sceneData.pair", "attribute1");
    anari::setParameter(device, material, "sceneData.triple", "attribute2");
    anari::setParameter(device, material, "sceneData.quad", "attribute3");
    anari::setParameter(device, material, "sceneData.tint", "color");
    anari::commitParameters(device, material);
    anari::setParameter(device, surface, "geometry", geometry);
    anari::setParameter(device, surface, "material", material);
    anari::commitParameters(device, surface);
    anari::setParameterArray1D(device, group, "surface", &surface, 1);
    anari::commitParameters(device, group);
    anari::setParameter(device, instance, "group", group);
    anari::commitParameters(device, instance);
    anari::setParameterArray1D(device, world, "instance", &instance, 1);
    anari::commitParameters(device, world);
    anari::setParameter(device, camera, "position", vec3{0, 0, 3});
    anari::setParameter(device, camera, "direction", vec3{0, 0, -1});
    anari::setParameter(device, camera, "up", vec3{0, 1, 0});
    anari::setParameter(device, camera, "height", 2.f);
    anari::commitParameters(device, camera);
    anari::setParameter(device, renderer, "background", vec4{0, 0, 0, 1});
    anari::setParameter(device, renderer, "pixelSamples", 4);
    anari::setParameter(device, renderer, "ambientRadiance", 0.f);
    anari::setParameter(device, renderer, "fireflyFilterMode", "none");
    anari::commitParameters(device, renderer);
    configureFrame(frame);
  }

  void configureFrame(anari::Frame f)
  {
    anari::setParameter(device, f, "size", uvec2{64, 64});
    anari::setParameter(device, f, "channel.color", ANARI_FLOAT32_VEC4);
    anari::setParameter(device, f, "world", world);
    anari::setParameter(device, f, "camera", camera);
    anari::setParameter(device, f, "renderer", renderer);
    anari::commitParameters(device, f);
  }

  ~Scene()
  {
    for (auto o : {ANARIObject(frame),
             ANARIObject(renderer),
             ANARIObject(camera),
             ANARIObject(world),
             ANARIObject(instance),
             ANARIObject(group),
             ANARIObject(surface),
             ANARIObject(material),
             ANARIObject(geometry)})
      anari::release(device, o);
    anari::release(device, device);
  }

  void mode(int mode)
  {
    anari::setParameter(device, material, "mode", mode);
    anari::commitParameters(device, material);
  }

  void check(const char *label,
      vec3 expected,
      anari::Frame f = nullptr,
      unsigned xBegin = 16,
      unsigned xEnd = 48)
  {
    f = f ? f : frame;
    anari::render(device, f);
    anari::wait(device, f);
    auto fb = anari::map<vec4>(device, f, "channel.color");
    require(
        fb.data && fb.width == 64 && fb.height == 64, "invalid framebuffer");
    vec3 mean{};
    const float area = float((xEnd - xBegin) * 32);
    for (unsigned y = 16; y < 48; ++y)
      for (unsigned x = xBegin; x < xEnd; ++x)
        for (int c = 0; c < 3; ++c)
          mean[c] += fb.data[y * 64 + x][c] / area;
    anari::unmap(device, f, "channel.color");
    printf("%s: %.4f %.4f %.4f\n", label, mean[0], mean[1], mean[2]);
    for (int c = 0; c < 3; ++c)
      require(
          std::isfinite(mean[c]) && std::abs(mean[c] - expected[c]) < 0.035f,
          label);
    printf("samples=%d\n", samples(f));
    require(errors == 0, "renderer reported an error");
  }

  int samples(anari::Frame f = nullptr)
  {
    int n = 0;
    require(anariGetProperty(device,
                f ? f : frame,
                "numSamples",
                ANARI_INT32,
                &n,
                sizeof(n),
                ANARI_WAIT)
            != 0,
        "sample query failed");
    return n;
  }
};

static void testSceneData(Scene &s)
{
  auto d = s.device;
  s.check("constant scalar", {0.8f, 0.8f, 0.8f});
  anari::setParameter(d, s.instance, "attribute0", vec4{0.7f, 0, 0, 1});
  anari::commitParameters(d, s.instance);
  s.check("geometry constant overrides instance constant", {0.8f, 0.8f, 0.8f});
  s.mode(5);
  s.check("uniform geometry overrides instance constant", {0.8f, 0.8f, 0.8f});
  const vec4 overriddenInstanceValue{0.6f, 0, 0, 1};
  anari::setParameterArray1D(
      d, s.instance, "attribute0", &overriddenInstanceValue, 1);
  anari::commitParameters(d, s.instance);
  s.check("uniform geometry overrides instance array", {0.8f, 0.8f, 0.8f});
  s.mode(0);
  s.check("geometry constant overrides instance array", {0.8f, 0.8f, 0.8f});
  anari::setParameter(d, s.geometry, "attribute0", vec4{0, 0, 0, 1});
  anari::commitParameters(d, s.geometry);
  s.check("authored zero overrides instance array", {0, 0, 0});
  anari::setParameter(d, s.geometry, "attribute0", vec4{0.8f, 0, 0, 1});
  anari::commitParameters(d, s.geometry);
  anari::unsetParameter(d, s.instance, "attribute0");
  anari::commitParameters(d, s.instance);
  s.mode(5);
  s.check("inferred uniform constant", {0.8f, 0.8f, 0.8f});
  anari::setParameter(d, s.material, "sceneData.scalar.uniform", "attribute2");
  anari::setParameter(d, s.geometry, "attribute2", vec4{0.45f, 0, 0, 1});
  anari::commitParameters(d, s.geometry);
  anari::commitParameters(d, s.material);
  s.mode(10);
  s.check("name ending in uniform", {0.45f, 0.45f, 0.45f});
  s.mode(5);
  s.check("independent scalar binding", {0.8f, 0.8f, 0.8f});
  anari::unsetParameter(d, s.material, "sceneData.scalar.uniform");
  anari::unsetParameter(d, s.geometry, "attribute2");
  anari::commitParameters(d, s.material);
  anari::commitParameters(d, s.geometry);
  const std::array<float, 4> varying = {0.2f, 0.8f, 0.8f, 0.2f};
  anari::setParameterArray1D(
      d, s.geometry, "vertex.attribute0", varying.data(), 4);
  anari::commitParameters(d, s.geometry);
  s.check("uniform rejects varying override", {0.125f, 0.125f, 0.125f});
  s.mode(0);
  s.check("vertex interpolation", {0.5f, 0.5f, 0.5f});
  const std::array<float, 6> face = {0.35f, 0.35f, 0.35f, 0.35f, 0.35f, 0.35f};
  anari::setParameterArray1D(
      d, s.geometry, "faceVarying.attribute0", face.data(), 6);
  anari::commitParameters(d, s.geometry);
  s.check("face-varying overrides vertex", {0.35f, 0.35f, 0.35f});
  s.mode(5);
  s.check("uniform rejects face-varying", {0.125f, 0.125f, 0.125f});
  s.mode(0);
  anari::unsetParameter(d, s.geometry, "faceVarying.attribute0");
  anari::unsetParameter(d, s.geometry, "vertex.attribute0");
  const std::array<float, 2> primitive = {0.2f, 0.8f};
  anari::setParameterArray1D(
      d, s.geometry, "primitive.attribute0", primitive.data(), 2);
  anari::commitParameters(d, s.geometry);
  s.check("primitive interpolation", {0.5f, 0.5f, 0.5f});
  s.mode(5);
  s.check("uniform rejects primitive override", {0.125f, 0.125f, 0.125f});
  s.mode(0);
  anari::unsetParameter(d, s.geometry, "primitive.attribute0");
  anari::unsetParameter(d, s.geometry, "attribute0");
  anari::commitParameters(d, s.geometry);
  s.check("absent geometry attribute", {0.25f, 0.25f, 0.25f});
  s.mode(6);
  s.check("isvalid absent", {0, 0, 0});
  anari::setParameter(d, s.instance, "attribute0", vec4{0.7f, 0, 0, 1});
  anari::commitParameters(d, s.instance);
  s.mode(0);
  s.check("instance constant", {0.7f, 0.7f, 0.7f});
  s.mode(5);
  s.check("uniform instance constant", {0.7f, 0.7f, 0.7f});
  const vec4 instanceValue{0.6f, 0, 0, 1};
  anari::setParameterArray1D(d, s.instance, "attribute0", &instanceValue, 1);
  anari::commitParameters(d, s.instance);
  s.check("inferred uniform instance array", {0.6f, 0.6f, 0.6f});
  s.mode(0);
  s.check("instance array", {0.6f, 0.6f, 0.6f});
  const std::array<std::array<float, 16>, 2> transforms = {
      std::array<float, 16>{0.5f, 0, 0, 0, 0, 1, 0, 0,
          0, 0, 1, 0, -0.5f, 0, 0, 1},
      std::array<float, 16>{0.5f, 0, 0, 0, 0, 1, 0, 0,
          0, 0, 1, 0, 0.5f, 0, 0, 1}};
  const std::array<vec4, 2> instanceValues = {
      vec4{0.2f, 0, 0, 1}, vec4{0.8f, 0, 0, 1}};
  anari::setParameterArray1D(d,
      s.instance,
      "transform",
      ANARI_FLOAT32_MAT4,
      transforms.data(),
      transforms.size());
  anari::setParameterArray1D(d,
      s.instance,
      "attribute0",
      instanceValues.data(),
      instanceValues.size());
  anari::commitParameters(d, s.instance);
  s.mode(5);
  s.check("first instance uniform value", {0.2f, 0.2f, 0.2f}, nullptr, 16, 28);
  s.check("second instance uniform value", {0.8f, 0.8f, 0.8f}, nullptr, 36, 48);
  anari::unsetParameter(d, s.instance, "transform");
  anari::unsetParameter(d, s.instance, "attribute0");
  anari::commitParameters(d, s.instance);
  s.mode(6);
  s.check("unset instance attribute", {0, 0, 0});
  anari::setParameter(d, s.geometry, "attribute0", vec4{0, 0, 0, 1});
  anari::commitParameters(d, s.geometry);
  s.check("isvalid authored zero", {1, 1, 1});
  const std::array<float, 4> partlyNonfinite = {
      std::numeric_limits<float>::quiet_NaN(), 0.8f,
      0.8f, std::numeric_limits<float>::quiet_NaN()};
  anari::setParameterArray1D(
      d, s.geometry, "vertex.attribute0", partlyNonfinite.data(), 4);
  anari::commitParameters(d, s.geometry);
  s.check("isvalid does not read varying values", {1, 1, 1});
  anari::unsetParameter(d, s.geometry, "vertex.attribute0");
  anari::commitParameters(d, s.geometry);
  s.mode(0);
  anari::unsetParameter(d, s.material, "sceneData.scalar");
  anari::commitParameters(d, s.material);
  s.check("unset binding", {0.25f, 0.25f, 0.25f});
  anari::setParameter(d, s.material, "sceneData.scalar", "attribute9");
  anari::commitParameters(d, s.material);
  s.check("invalid binding", {0.25f, 0.25f, 0.25f});
  anari::setParameter(d, s.material, "sceneData.scalar", "attribute0");
  anari::setParameter(d,
      s.geometry,
      "attribute0",
      vec4{std::numeric_limits<float>::quiet_NaN(), 0, 0, 1});
  anari::commitParameters(d, s.geometry);
  anari::commitParameters(d, s.material);
  s.check("nonfinite data", {0.25f, 0.25f, 0.25f});
  s.mode(6);
  s.check("isvalid nonfinite constant", {1, 1, 1});
  const std::array<vec2, 4> pair = {
      vec2{0.2f, 0.6f}, vec2{0.2f, 0.6f}, vec2{0.2f, 0.6f}, vec2{0.2f, 0.6f}};
  anari::setParameterArray1D(
      d, s.geometry, "vertex.attribute1", pair.data(), 4);
  anari::setParameter(d, s.geometry, "attribute2", vec4{0.3f, 0.6f, 0.9f, 1});
  anari::setParameter(
      d, s.geometry, "attribute3", vec4{0.2f, 0.4f, 0.6f, 0.8f});
  anari::setParameter(d, s.geometry, "color", vec4{0.9f, 0.3f, 0.1f, 1});
  anari::commitParameters(d, s.geometry);
  s.mode(1);
  s.check("float2 array", {0.2f, 0.6f, 0});
  s.mode(2);
  s.check("float3", {0.3f, 0.6f, 0.9f});
  s.mode(3);
  s.check("float4", {0.2f, 0.4f, 0.8f});
  s.mode(4);
  s.check("color", {0.9f, 0.3f, 0.1f});
  anari::setParameter(d, s.material, "gain", 0.5f);
  anari::commitParameters(d, s.material);
  s.check("retained argument", {0.45f, 0.15f, 0.05f});
  anari::unsetParameter(d, s.material, "gain");
  s.mode(8);
  s.check("unbound name", {0.7f, 0.1f, 0.8f});
  s.mode(9);
  s.check("integer default", {0.2f, 0.2f, 0.2f});
  // A different target-code string table must resolve independently.
  anari::setParameter(d, s.material, "source", R"mdl(mdl 1.7;
import ::df::*; import ::scene::*;
export material main() = material(surface: material_surface(
  emission: material_emission(emission: df::diffuse_edf(),
    intensity: scene::data_lookup_color("replacement", color(0)) * 3.14159265)));
)mdl");
  anari::unsetParameter(d, s.material, "mode");
  anari::setParameter(d, s.material, "sceneData.replacement", "color");
  anari::commitParameters(d, s.material);
  s.check("source replacement string IDs", {0.9f, 0.3f, 0.1f});
}

static void testCurve(Scene &s)
{
  auto d = s.device;
  auto previous = s.geometry;
  s.geometry = anari::newObject<anari::Geometry>(d, "curve");
  const std::array<vec3, 2> positions = {vec3{-1, 0, 0}, vec3{1, 0, 0}};
  const std::array<float, 2> values = {0.2f, 0.8f};
  const unsigned index = 0;
  anari::setParameterArray1D(
      d, s.geometry, "vertex.position", positions.data(), 2);
  anari::setParameterArray1D(
      d, s.geometry, "vertex.attribute0", values.data(), 2);
  anari::setParameterArray1D(d, s.geometry, "primitive.index", &index, 1);
  anari::setParameter(d, s.geometry, "radius", 0.8f);
  anari::commitParameters(d, s.geometry);
  anari::setParameter(d, s.surface, "geometry", s.geometry);
  anari::commitParameters(d, s.surface);
  anari::release(d, previous);
  anari::setParameter(d, s.material, "source", source);
  s.mode(0);
  s.check("curve vertex interpolation", {0.5f, 0.5f, 0.5f});
  s.mode(5);
  s.check("uniform rejects curve interpolation", {0.125f, 0.125f, 0.125f});
  anari::unsetParameter(d, s.geometry, "vertex.attribute0");
  anari::setParameter(d, s.geometry, "attribute0", vec4{0.45f, 0, 0, 1});
  anari::commitParameters(d, s.geometry);
  s.check("curve uniform constant", {0.45f, 0.45f, 0.45f});
}

static void testAnimation(Scene &s)
{
  auto d = s.device;
  anari::setParameter(d, s.material, "source", source);
  s.mode(7);
  s.check("default time", {0, 0, 0});
  // Settle the existing deferred scene-finalization reset before timing checks.
  s.check("default time settled", {0, 0, 0});
  anari::setParameter(d, s.frame, "mdl.animationTime", 0.25f);
  anari::commitParameters(d, s.frame);
  s.check("first time", {0.25f, 0.25f, 0.25f});
  const int n = s.samples();
  s.check("progressive same time", {0.25f, 0.25f, 0.25f});
  require(s.samples() > n, "same time stopped accumulation");
  anari::setParameter(d, s.frame, "mdl.animationTime", 0.75f);
  anari::commitParameters(d, s.frame);
  s.check("changed time clears accumulation", {0.75f, 0.75f, 0.75f});
  require(s.samples() == n, "changed time did not reset sample count");
  anari::setParameter(d, s.frame, "accumulationVersion", uint64_t(1));
  anari::commitParameters(d, s.frame);
  s.check("manual accumulation version", {0.75f, 0.75f, 0.75f});
  s.check("manual accumulation settled", {0.75f, 0.75f, 0.75f});
  anari::setParameter(d, s.frame, "mdl.animationTime", 0.5f);
  anari::commitParameters(d, s.frame);
  s.check("time resets manual accumulation", {0.5f, 0.5f, 0.5f});
  require(s.samples() == n, "manual accumulation did not reset on time change");
  anari::setParameter(d, s.frame, "mdl.animationTime", 0.5f);
  anari::commitParameters(d, s.frame);
  s.check("unchanged committed time accumulates", {0.5f, 0.5f, 0.5f});
  require(s.samples() > n, "unchanged time reset manual accumulation");
  anari::setParameter(d, s.frame, "mdl.animationTime", 0.75f);
  anari::commitParameters(d, s.frame);
  s.check("restore first frame time", {0.75f, 0.75f, 0.75f});
  auto other = anari::newObject<anari::Frame>(d);
  s.configureFrame(other);
  anari::setParameter(d, other, "mdl.animationTime", 0.1f);
  anari::commitParameters(d, other);
  s.check("second frame time", {0.1f, 0.1f, 0.1f}, other);
  s.check("first frame time remains local", {0.75f, 0.75f, 0.75f});
  anari::release(d, other);
  for (float t : {std::numeric_limits<float>::quiet_NaN(),
           std::numeric_limits<float>::infinity()}) {
    anari::setParameter(d, s.frame, "mdl.animationTime", t);
    anari::commitParameters(d, s.frame);
    s.check("nonfinite time uses zero", {0, 0, 0});
  }
  require(timeWarnings == 2, "missing invalid-time warnings");
}

int main()
{
  try {
    Scene scene;
    testSceneData(scene);
    testCurve(scene);
    testAnimation(scene);
    require(
        metadataWarnings == 0, "valid scene-data metadata reported invalid");
    puts("MDL scene data and animation time passed");
    return 0;
  } catch (const std::exception &e) {
    fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
