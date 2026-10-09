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

// A shading normal that leans away from a grazing viewer is raised toward the
// geometric normal until the view's mirror direction clears the surface
// (Cycles' ensure_valid_specular_reflection), in both the CUDA and the MDL
// physicallyBased backends. Without it MDL treats the viewer as inside the
// surface and the pixel goes black.
//
// A diffuse floor whose shading normal leans 35° away from a grazing camera,
// through a normal map, interpolated vertex normals or a clearcoat normal map,
// must be lit like the unmapped floor. Under the directional light, with
// direct lighting only, the expected value has a closed form, computed here on
// the host with the same adjustment and the backend's diffuse lobe. Every
// backend must match it, which is what makes CUDA (rtx build) and MDL (rtx_mdl
// build) agree. A top-down camera never triggers the adjustment and must shade
// the plain tilted normal, as before. The normal AOV reports the unadjusted
// normal.
//
// Runs against `physicallyBased`, and additionally against the always-MDL
// `physicallyBasedMDL` subtype when MDL support is compiled in. Linear float
// buffer, firefly filter off.

// anari_cpp
#include <anari/anari_cpp/ext/std.h>
#include <anari/anari_cpp.hpp>
// VisRTX
#include <anari/ext/visrtx/makeVisRTXDevice.h>
// std
#include <algorithm>
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

static constexpr uvec2 IMAGE_SIZE = {128, 128};
static constexpr int PIXEL_SAMPLES = 64;
static constexpr float ALBEDO = 0.6f;
static constexpr float IRRADIANCE = 2.f;
static constexpr float RADIANCE = 8.f;
static constexpr float TILT = 35.f * 3.14159265f / 180.f;
static constexpr float PI = 3.14159265f;

// The camera sees the floor's +y side; every tilt leans toward +z, away from
// the grazing camera at z = -3.
static const vec3 TILTED_NORMAL = {0.f, std::cos(TILT), std::sin(TILT)};

enum class Shading
{
  FLAT, // no map, vertex normals absent: Ns == Ng, never adjusted
  NORMAL_MAP, // `normal` map tilting the shading normal
  VERTEX_NORMALS, // smooth normals tilted, no map
  CLEARCOAT, // full clearcoat, no maps
  CLEARCOAT_MAP, // full clearcoat with a tilted `clearcoatNormal` map
};

enum class View
{
  GRAZING,
  TOP,
};

enum class Light
{
  DIRECTIONAL,
  EMITTER,
};

struct Case
{
  Shading shading;
  View view;
  Light light;
};

struct Result
{
  double luminance;
  vec3 normal; // mean normal AOV over the region
};

static float dot(const vec3 &a, const vec3 &b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static vec3 normalize(const vec3 &a)
{
  const float l = std::sqrt(dot(a, a));
  return {a[0] / l, a[1] / l, a[2] / l};
}

static vec3 cross(const vec3 &a, const vec3 &b)
{
  return {a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0]};
}

static void cameraFrame(View view, vec3 &pos, vec3 &dir, vec3 &up)
{
  if (view == View::GRAZING) {
    pos = {0.f, 0.5f, -3.f};
    dir = normalize(vec3{0.f, -0.15f, 1.f});
    up = {0.f, 1.f, 0.f};
  } else {
    pos = {0.f, 1.2f, 0.f};
    dir = {0.f, -1.f, 0.f};
    up = {0.f, 0.f, 1.f};
  }
}

// The measured region: where TestNormalMapDecode looks for the grazing view
// (floor in front of the camera, under the emitter), the middle for top-down.
static bool inRegion(View view, uint32_t x, uint32_t y)
{
  const uvec2 &s = IMAGE_SIZE;
  return view == View::GRAZING
      ? (y >= s[1] / 8 && y < s[1] / 2 && x >= 3 * s[0] / 8 && x < 5 * s[0] / 8)
      : (y >= s[1] / 4 && y < 3 * s[1] / 4 && x >= s[0] / 4
            && x < 3 * s[0] / 4);
}

///////////////////////////////////////////////////////////////////////////////
// Host reference ////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// Cycles' ensure_valid_specular_reflection, as ported to the device.
static vec3 adjustNormalForView(const vec3 &Ng, const vec3 &V, const vec3 &N)
{
  const float NdotV = dot(N, V);
  const vec3 R = {2.f * NdotV * N[0] - V[0],
      2.f * NdotV * N[1] - V[1],
      2.f * NdotV * N[2] - V[2]};
  const float VdotNg = std::max(dot(V, Ng), 0.f);
  const float threshold = std::min(0.9f * VdotNg, 0.01f);
  if (dot(Ng, R) >= threshold)
    return N;
  const float NdotNg = dot(N, Ng);
  vec3 X = {
      N[0] - NdotNg * Ng[0], N[1] - NdotNg * Ng[1], N[2] - NdotNg * Ng[2]};
  X = normalize(X);
  const float VdotX = dot(V, X);
  const float a = VdotX * VdotX + VdotNg * VdotNg;
  const float b = 2.f * (a + VdotNg * threshold);
  const float c = (threshold + VdotNg) * (threshold + VdotNg);
  const float root = std::sqrt(std::max(b * b - 4.f * a * c, 0.f));
  const float Nz2 = 0.25f * (VdotX < 0.f ? b + root : b - root) / a;
  const float Nz = std::sqrt(std::max(Nz2, 0.f));
  const float Nx = std::sqrt(std::max(1.f - Nz2, 0.f));
  return {
      X[0] * Nx + Ng[0] * Nz, X[1] * Nx + Ng[1] * Nz, X[2] * Nx + Ng[2] * Nz};
}

// Mean luminance of the diffuse floor under the overhead directional light,
// with the shading normal tilted and then adjusted for each view ray. 4x4
// stratified rays per pixel stand in for the renderer's pixel jitter. CUDA's
// diffuse lobe is Lambertian; MDL's (libbsdf diffuse_evaluate) scales it by
// 2 / (1 + N.Ng) to conserve energy when the shading normal leaves Ng.
static double expectedDirectional(View view, bool tilted, bool mdl)
{
  vec3 pos, dir, up;
  cameraFrame(view, pos, dir, up);
  const float aspect = IMAGE_SIZE[0] / float(IMAGE_SIZE[1]);
  const float h = 2.f * std::tan(0.5f * PI / 3.f);
  const vec3 du0 = normalize(cross(dir, up));
  const vec3 dv0 = normalize(cross(du0, dir));
  const vec3 Ng = {0.f, 1.f, 0.f};
  const vec3 N = tilted ? TILTED_NORMAL : Ng;
  double sum = 0.0;
  uint64_t n = 0;
  for (uint32_t y = 0; y < IMAGE_SIZE[1]; y++) {
    for (uint32_t x = 0; x < IMAGE_SIZE[0]; x++) {
      if (!inRegion(view, x, y))
        continue;
      for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
          const float sx = (x + (i + 0.5f) / 4.f) / IMAGE_SIZE[0] - 0.5f;
          const float sy = (y + (j + 0.5f) / 4.f) / IMAGE_SIZE[1] - 0.5f;
          vec3 d;
          for (int k = 0; k < 3; k++)
            d[k] = dir[k] + sx * h * aspect * du0[k] + sy * h * dv0[k];
          d = normalize(d);
          const vec3 V = {-d[0], -d[1], -d[2]};
          const vec3 Na = adjustNormalForView(Ng, V, N);
          const float scale = mdl ? 2.f / (1.f + std::max(Na[1], 0.f)) : 1.f;
          sum += ALBEDO / PI * IRRADIANCE * std::max(Na[1], 0.f) * scale;
          n++;
        }
      }
    }
  }
  return sum / double(n);
}

///////////////////////////////////////////////////////////////////////////////
// Scene /////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

static anari::Sampler makeTiltMap(ANARIDevice device)
{
  // Spec-decoded tangent-space normal: a pure bitangent tilt.
  const vec3 ts = {0.f, std::sin(TILT), std::cos(TILT)};
  const std::array<vec3, 4> img = {ts, ts, ts, ts};
  auto sampler = anari::newObject<anari::Sampler>(device, "image2D");
  anari::setParameter(device, sampler, "inAttribute", "attribute0");
  anari::setParameter(device, sampler, "filter", "nearest");
  anari::setParameterArray2D(device, sampler, "image", img.data(), 2, 2);
  anari::commitParameters(device, sampler);
  return sampler;
}

static anari::Surface makeFloor(
    ANARIDevice device, const char *subtype, Shading shading)
{
  const std::array<vec3, 4> pos = {vec3{-6.f, 0.f, -6.f},
      vec3{6.f, 0.f, -6.f},
      vec3{6.f, 0.f, 6.f},
      vec3{-6.f, 0.f, 6.f}};
  const std::array<vec2, 4> uv = {
      vec2{0.f, 0.f}, vec2{1.f, 0.f}, vec2{1.f, 1.f}, vec2{0.f, 1.f}};
  // Wound so +y is the front face.
  const std::array<std::array<unsigned, 3>, 2> idx = {
      std::array<unsigned, 3>{0, 2, 1}, std::array<unsigned, 3>{0, 3, 2}};
  // T = +x and B = w * cross(N, T) = -w * z, so w = -1 points the bitangent
  // (the map's +Y) at +z, away from the grazing camera.
  const std::array<vec4, 4> tangents = {vec4{1.f, 0.f, 0.f, -1.f},
      vec4{1.f, 0.f, 0.f, -1.f},
      vec4{1.f, 0.f, 0.f, -1.f},
      vec4{1.f, 0.f, 0.f, -1.f}};
  const std::array<vec3, 4> normals = {
      TILTED_NORMAL, TILTED_NORMAL, TILTED_NORMAL, TILTED_NORMAL};

  auto geom = anari::newObject<anari::Geometry>(device, "triangle");
  anari::setParameterArray1D(device, geom, "vertex.position", pos.data(), 4);
  anari::setParameterArray1D(device, geom, "vertex.attribute0", uv.data(), 4);
  anari::setParameterArray1D(
      device, geom, "vertex.tangent", tangents.data(), 4);
  anari::setParameterArray1D(device, geom, "primitive.index", idx.data(), 2);
  if (shading == Shading::VERTEX_NORMALS)
    anari::setParameterArray1D(
        device, geom, "vertex.normal", normals.data(), 4);
  anari::commitParameters(device, geom);

  auto mat = anari::newObject<anari::Material>(device, subtype);
  anari::setParameter(device, mat, "baseColor", vec3{ALBEDO, ALBEDO, ALBEDO});
  anari::setParameter(device, mat, "metallic", 0.f);
  anari::setParameter(device, mat, "roughness", 1.f);
  if (shading == Shading::NORMAL_MAP)
    anari::setAndReleaseParameter(device, mat, "normal", makeTiltMap(device));
  if (shading == Shading::CLEARCOAT || shading == Shading::CLEARCOAT_MAP) {
    anari::setParameter(device, mat, "clearcoat", 1.f);
    anari::setParameter(device, mat, "clearcoatRoughness", 0.5f);
  }
  if (shading == Shading::CLEARCOAT_MAP) {
    anari::setAndReleaseParameter(
        device, mat, "clearcoatNormal", makeTiltMap(device));
  }
  anari::commitParameters(device, mat);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);
  return surface;
}

// Constant emissive quad straight above the floor, facing down.
static anari::Surface makeEmitter(ANARIDevice device)
{
  const std::array<vec3, 4> pos = {vec3{-0.5f, 1.5f, -0.5f},
      vec3{0.5f, 1.5f, -0.5f},
      vec3{0.5f, 1.5f, 0.5f},
      vec3{-0.5f, 1.5f, 0.5f}};
  const std::array<std::array<unsigned, 3>, 2> idx = {
      std::array<unsigned, 3>{0, 1, 2}, std::array<unsigned, 3>{0, 2, 3}};

  auto geom = anari::newObject<anari::Geometry>(device, "triangle");
  anari::setParameterArray1D(device, geom, "vertex.position", pos.data(), 4);
  anari::setParameterArray1D(device, geom, "primitive.index", idx.data(), 2);
  anari::commitParameters(device, geom);

  auto mat = anari::newObject<anari::Material>(device, "physicallyBased");
  anari::setParameter(device, mat, "baseColor", vec3{0.f, 0.f, 0.f});
  anari::setParameter(
      device, mat, "emissive", vec3{RADIANCE, RADIANCE, RADIANCE});
  anari::commitParameters(device, mat);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);
  return surface;
}

static Result render(ANARIDevice device, const char *subtype, const Case &k)
{
  std::array<anari::Surface, 2> surfaces = {
      makeFloor(device, subtype, k.shading), nullptr};
  size_t numSurfaces = 1;
  if (k.light == Light::EMITTER)
    surfaces[numSurfaces++] = makeEmitter(device);

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(
      device, world, "surface", surfaces.data(), numSurfaces);
  for (size_t i = 0; i < numSurfaces; i++)
    anari::release(device, surfaces[i]);
  if (k.light == Light::DIRECTIONAL) {
    auto light = anari::newObject<anari::Light>(device, "directional");
    anari::setParameter(device, light, "direction", vec3{0.f, -1.f, 0.f});
    anari::setParameter(device, light, "irradiance", IRRADIANCE);
    anari::commitParameters(device, light);
    anari::setParameterArray1D(device, world, "light", &light, 1);
    anari::release(device, light);
  }
  anari::commitParameters(device, world);

  vec3 pos, dir, up;
  cameraFrame(k.view, pos, dir, up);
  auto camera = anari::newObject<anari::Camera>(device, "perspective");
  anari::setParameter(device, camera, "position", pos);
  anari::setParameter(device, camera, "direction", dir);
  anari::setParameter(device, camera, "up", up);
  anari::setParameter(
      device, camera, "aspect", IMAGE_SIZE[0] / float(IMAGE_SIZE[1]));
  anari::commitParameters(device, camera);

  auto renderer = anari::newObject<anari::Renderer>(device, "quality");
  anari::setParameter(device, renderer, "background", vec4{0.f, 0.f, 0.f, 1.f});
  anari::setParameter(device, renderer, "ambientRadiance", 0.f);
  anari::setParameter(device, renderer, "pixelSamples", PIXEL_SAMPLES);
  anari::setParameter(device, renderer, "fireflyFilterMode", "none");
  // Direct light only: a continuation ray sampled around a tilted shading
  // normal can head into the floor and light it again, which no closed form
  // here accounts for.
  anari::setParameter(device, renderer, "maxRayDepth", 1);
  anari::commitParameters(device, renderer);

  auto frame = anari::newObject<anari::Frame>(device);
  anari::setParameter(device, frame, "size", IMAGE_SIZE);
  anari::setParameter(device, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setParameter(device, frame, "channel.normal", ANARI_FLOAT32_VEC3);
  anari::setAndReleaseParameter(device, frame, "world", world);
  anari::setAndReleaseParameter(device, frame, "camera", camera);
  anari::setAndReleaseParameter(device, frame, "renderer", renderer);
  anari::commitParameters(device, frame);

  anari::render(device, frame);
  anari::wait(device, frame);

  Result result{0.0, {0.f, 0.f, 0.f}};
  uint64_t n = 0;
  auto color = anari::map<vec4>(device, frame, "channel.color");
  auto normal = anari::map<vec3>(device, frame, "channel.normal");
  for (uint32_t y = 0; y < IMAGE_SIZE[1]; y++) {
    for (uint32_t x = 0; x < IMAGE_SIZE[0]; x++) {
      if (!inRegion(k.view, x, y))
        continue;
      const vec4 &p = color.data[y * IMAGE_SIZE[0] + x];
      result.luminance += 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
      const vec3 &nn = normal.data[y * IMAGE_SIZE[0] + x];
      for (int i = 0; i < 3; i++)
        result.normal[i] += nn[i];
      n++;
    }
  }
  anari::unmap(device, frame, "channel.normal");
  anari::unmap(device, frame, "channel.color");
  anari::release(device, frame);

  result.luminance /= double(n);
  for (int i = 0; i < 3; i++)
    result.normal[i] /= float(n);
  return result;
}

///////////////////////////////////////////////////////////////////////////////
// Checks ////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

static bool checkNear(
    const char *label, double value, double expected, double tolerance)
{
  const double relErr = std::abs(value - expected) / expected;
  printf("%s: %f vs expected %f relErr=%f\n", label, value, expected, relErr);
  if (!std::isfinite(value) || relErr > tolerance) {
    fprintf(stderr,
        "FAIL: %s is %f, expected %f (relErr=%f > %f)\n",
        label,
        value,
        expected,
        relErr,
        tolerance);
    return false;
  }
  return true;
}

// Lit like the unmapped floor: not black, not blown out.
static bool checkLit(const char *label, double value, double flat)
{
  const double ratio = value / flat;
  printf("%s: %f vs unmapped %f ratio=%f\n", label, value, flat, ratio);
  if (!std::isfinite(value) || ratio < 0.5 || ratio > 1.5) {
    fprintf(stderr,
        "FAIL: %s is %f against the unmapped floor's %f (ratio %f outside "
        "[0.5, 1.5]) — a shading normal leaning away from the viewer was not "
        "kept valid\n",
        label,
        value,
        flat,
        ratio);
    return false;
  }
  return true;
}

// The AOV describes the surface: the tilted normal, not the adjusted one.
static bool checkNormalAOV(const char *label, const vec3 &aov)
{
  const float err = std::max({std::abs(aov[0] - TILTED_NORMAL[0]),
      std::abs(aov[1] - TILTED_NORMAL[1]),
      std::abs(aov[2] - TILTED_NORMAL[2])});
  printf("%s normal AOV: (%f, %f, %f)\n", label, aov[0], aov[1], aov[2]);
  if (err > 0.01f) {
    fprintf(stderr,
        "FAIL: %s normal AOV (%f, %f, %f) is not the unadjusted normal "
        "(%f, %f, %f)\n",
        label,
        aov[0],
        aov[1],
        aov[2],
        TILTED_NORMAL[0],
        TILTED_NORMAL[1],
        TILTED_NORMAL[2]);
    return false;
  }
  return true;
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);

  bool ok = true;
  const std::array<const char *, 2> subtypes = {"physicallyBased",
#ifdef VISRTX_TEST_MDL_WRAPPER
      "physicallyBasedMDL"
#else
      nullptr
#endif
  };
  const std::array<bool, 2> isMDL = {
#ifdef VISRTX_TEST_PHYSICALLY_BASED_IS_MDL
      true,
#else
      false,
#endif
      true};
  std::array<double, 2> grazing = {0.0, 0.0};
  for (size_t s = 0; s < subtypes.size(); s++) {
    const char *subtype = subtypes[s];
    if (!subtype)
      continue;
    // The same adjustment and diffuse lobe on the host: matching these is what
    // makes the backends agree.
    const double expectedGrazing =
        expectedDirectional(View::GRAZING, true, isMDL[s]);
    const double expectedTop = expectedDirectional(View::TOP, true, isMDL[s]);
    const double expectedFlat = expectedDirectional(View::TOP, false, isMDL[s]);
    char label[128];
    const auto run = [&](Shading shading, View view, Light light) {
      return render(device, subtype, Case{shading, view, light});
    };

    // Directional light: exact against the host reference.
    const Result flat = run(Shading::FLAT, View::GRAZING, Light::DIRECTIONAL);
    snprintf(label, sizeof(label), "%s/flat/top", subtype);
    ok = checkNear(label,
             run(Shading::FLAT, View::TOP, Light::DIRECTIONAL).luminance,
             expectedFlat,
             0.025)
        && ok;

    const Result mapped =
        run(Shading::NORMAL_MAP, View::GRAZING, Light::DIRECTIONAL);
    grazing[s] = mapped.luminance;
    snprintf(label, sizeof(label), "%s/normal-map/grazing", subtype);
    ok = checkLit(label, mapped.luminance, flat.luminance) && ok;
    ok = checkNear(label, mapped.luminance, expectedGrazing, 0.025) && ok;
    ok = checkNormalAOV(label, mapped.normal) && ok;

    snprintf(label, sizeof(label), "%s/normal-map/top", subtype);
    ok = checkNear(label,
             run(Shading::NORMAL_MAP, View::TOP, Light::DIRECTIONAL).luminance,
             expectedTop,
             0.025)
        && ok;

    const Result smooth =
        run(Shading::VERTEX_NORMALS, View::GRAZING, Light::DIRECTIONAL);
    snprintf(label, sizeof(label), "%s/vertex-normals/grazing", subtype);
    ok = checkLit(label, smooth.luminance, flat.luminance) && ok;
    ok = checkNear(label, smooth.luminance, expectedGrazing, 0.025) && ok;
    ok = checkNormalAOV(label, smooth.normal) && ok;

    snprintf(label, sizeof(label), "%s/vertex-normals/top", subtype);
    ok = checkNear(label,
             run(Shading::VERTEX_NORMALS, View::TOP, Light::DIRECTIONAL)
                 .luminance,
             expectedTop,
             0.025)
        && ok;

    // The clearcoat's own normal leans away; the base stays flat.
    const double coat =
        run(Shading::CLEARCOAT, View::GRAZING, Light::DIRECTIONAL).luminance;
    snprintf(label, sizeof(label), "%s/clearcoat-map/grazing", subtype);
    ok = checkLit(label,
             run(Shading::CLEARCOAT_MAP, View::GRAZING, Light::DIRECTIONAL)
                 .luminance,
             coat)
        && ok;

    // Emissive quad: no closed form, but lit like the unmapped floor. Not with
    // vertex normals: their shadow-terminator offset (shadingHitpoint) lifts
    // this huge, uniformly tilted floor toward the quad and brightens it.
    const double flatEmitter =
        run(Shading::FLAT, View::GRAZING, Light::EMITTER).luminance;
    snprintf(label, sizeof(label), "%s/normal-map/grazing/emitter", subtype);
    ok = checkLit(label,
             run(Shading::NORMAL_MAP, View::GRAZING, Light::EMITTER).luminance,
             flatEmitter)
        && ok;
  }

  if (subtypes[1])
    ok = checkNear("physicallyBasedMDL vs physicallyBased/normal-map/grazing",
             grazing[1],
             grazing[0],
             0.05)
        && ok;

  anari::release(device, device);

  if (!ok)
    return 1;
  printf("shading normal grazing passed\n");
  return 0;
}
