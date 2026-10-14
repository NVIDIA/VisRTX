/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <anari/anari_cpp.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using vec2 = std::array<float, 2>;
using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;
using uvec2 = std::array<unsigned, 2>;
static int g_errors = 0;

static void status(const void *,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity <= ANARI_SEVERITY_ERROR) {
    std::fprintf(stderr, "%s\n", message);
    ++g_errors;
  }
}

static double mean(const std::vector<vec4> &pixels, bool outer = false)
{
  double total = 0.;
  unsigned count = 0;
  for (unsigned y = 0; y < 64; ++y) {
    for (unsigned x = 0; x < 64; ++x) {
      if (outer && x >= 20 && x < 44 && y >= 20 && y < 44)
        continue;
      const auto &p = pixels[y * 64 + x];
      total += (p[0] + p[1] + p[2]) / 3.;
      ++count;
    }
  }
  return total / count;
}

int main()
{
  auto d = makeVisRTXDevice(status);
  const char **extensions = nullptr;
  anariGetProperty(d,
      d,
      "extension",
      ANARI_STRING_LIST,
      &extensions,
      sizeof(extensions),
      ANARI_WAIT);
  bool advertised = false;
  bool hasShaping = false;
  bool hasFocus = false;
  if (extensions) {
    for (const char **entry = extensions; *entry; ++entry) {
      advertised |= std::strcmp(*entry, "ANARI_VISRTX_LIGHT_QUAD_TEXTURE") == 0;
      hasShaping |= std::strcmp(*entry, "ANARI_VISRTX_LIGHT_QUAD_SHAPING") == 0;
      hasFocus |= std::strcmp(*entry, "ANARI_VISRTX_LIGHT_QUAD_FOCUS") == 0;
    }
  }
  if (!advertised) {
    std::fprintf(stderr, "quad texture extension is missing\n");
    anari::release(d, d);
    return 1;
  }

  auto light = anari::newObject<anari::Light>(d, "quad");
  anari::setParameter(d, light, "position", vec3{-0.1f, 1.5f, -0.1f});
  anari::setParameter(d, light, "edge1", vec3{0.f, 0.f, 0.2f});
  anari::setParameter(d, light, "edge2", vec3{0.2f, 0.f, 0.f});
  anari::setParameter(d, light, "intensity", 10.f);
  anari::commitParameters(d, light);
  const std::array<vec3, 4> positions = {vec3{-2.f, 0.f, -2.f},
      vec3{-2.f, 0.f, 2.f},
      vec3{2.f, 0.f, 2.f},
      vec3{2.f, 0.f, -2.f}};
  const std::array<std::array<unsigned, 3>, 2> indices = {
      std::array<unsigned, 3>{0, 1, 2}, std::array<unsigned, 3>{0, 2, 3}};
  auto geometry = anari::newObject<anari::Geometry>(d, "triangle");
  anari::setParameterArray1D(
      d, geometry, "vertex.position", positions.data(), 4);
  anari::setParameterArray1D(d, geometry, "primitive.index", indices.data(), 2);
  anari::commitParameters(d, geometry);
  auto material = anari::newObject<anari::Material>(d, "matte");
  anari::setParameter(d, material, "color", vec3{0.8f, 0.8f, 0.8f});
  anari::commitParameters(d, material);
  auto surface = anari::newObject<anari::Surface>(d);
  anari::setAndReleaseParameter(d, surface, "geometry", geometry);
  anari::setAndReleaseParameter(d, surface, "material", material);
  anari::commitParameters(d, surface);
  auto world = anari::newObject<anari::World>(d);
  anari::setParameterArray1D(d, world, "surface", &surface, 1);
  anari::setParameterArray1D(d, world, "light", &light, 1);
  anari::commitParameters(d, world);
  auto camera = anari::newObject<anari::Camera>(d, "orthographic");
  anari::setParameter(d, camera, "position", vec3{0.f, 1.f, 0.f});
  anari::setParameter(d, camera, "direction", vec3{0.f, -1.f, 0.f});
  anari::setParameter(d, camera, "up", vec3{0.f, 0.f, -1.f});
  anari::setParameter(d, camera, "height", 3.f);
  anari::setParameter(d, camera, "aspect", 1.f);
  anari::commitParameters(d, camera);
  auto renderer = anari::newObject<anari::Renderer>(d, "interactive");
  anari::setParameter(d, renderer, "background", vec4{0.f, 0.f, 0.f, 1.f});
  anari::setParameter(d, renderer, "ambientRadiance", 0.f);
  anari::setParameter(d, renderer, "pixelSamples", 32);
  anari::setParameter(d, renderer, "maxSampledLights", 0);
  anari::setParameter(d, renderer, "fireflyFilter", "none");
  anari::commitParameters(d, renderer);
  auto frame = anari::newObject<anari::Frame>(d);
  anari::setParameter(d, frame, "size", uvec2{64, 64});
  anari::setParameter(d, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setParameter(d, frame, "world", world);
  anari::setParameter(d, frame, "camera", camera);
  anari::setParameter(d, frame, "renderer", renderer);
  anari::commitParameters(d, frame);
  const auto render = [&]() {
    anari::render(d, frame);
    anari::wait(d, frame);
    auto mapped = anari::map<vec4>(d, frame, "channel.color");
    std::vector<vec4> pixels(
        mapped.data, mapped.data + mapped.width * mapped.height);
    anari::unmap(d, frame, "channel.color");
    return pixels;
  };

  const double baseline = mean(render());
  std::array<vec3, 4> white = {
      vec3{1, 1, 1}, vec3{1, 1, 1}, vec3{1, 1, 1}, vec3{1, 1, 1}};
  auto image = anari::newArray2D(d, white.data(), 2, 2);
  anari::setParameter(d, light, "colorMap", image);
  anari::commitParameters(d, light);
  const double mappedWhite = mean(render());
  auto *texels = static_cast<vec3 *>(anariMapArray(d, image));
  for (int i = 0; i < 4; ++i)
    texels[i] = vec3{.25f, .25f, .25f};
  anariUnmapArray(d, image);
  const double gray = mean(render());
  texels = static_cast<vec3 *>(anariMapArray(d, image));
  for (int i = 0; i < 4; ++i)
    texels[i] = vec3{0, 0, 0};
  anariUnmapArray(d, image);
  const double black = mean(render());
  anari::unsetParameter(d, light, "colorMap");
  anari::commitParameters(d, light);
  const double restored = mean(render());
  bool ok = baseline > 1e-3 && std::abs(mappedWhite - baseline) < .01 * baseline
      && std::abs(gray - .25 * baseline) < .01 * baseline && black < 1e-6
      && std::abs(restored - baseline) < .01 * baseline;
  std::printf(
      "quad texture baseline=%g white=%g gray=%g black=%g restored=%g\n",
      baseline,
      mappedWhite,
      gray,
      black,
      restored);

  // Rebind an asymmetric image: the camera looks up at the rectangle.
  texels = static_cast<vec3 *>(anariMapArray(d, image));
  texels[0] = vec3{1, 0, 0};
  texels[1] = vec3{0, 1, 0};
  texels[2] = vec3{0, 0, 1};
  texels[3] = vec3{1, 1, 1};
  anariUnmapArray(d, image);
  anari::setParameter(d, light, "colorMap", image);
  anari::commitParameters(d, light);
  anari::setParameter(d, camera, "direction", vec3{0, 1, 0});
  anari::setParameter(d, camera, "up", vec3{0, 0, 1});
  anari::setParameter(d, camera, "height", .2f);
  anari::commitParameters(d, camera);
  for (const char *mode : {"interactive", "quality"}) {
    auto renderer2 = anari::newObject<anari::Renderer>(d, mode);
    anari::setParameter(d, renderer2, "pixelSamples", 16);
    anari::setParameter(d, renderer2, "background", vec4{0, 0, 0, 1});
    anari::setParameter(d, renderer2, "fireflyFilter", "none");
    anari::commitParameters(d, renderer2);
    anari::setParameter(d, frame, "renderer", renderer2);
    anari::commitParameters(d, frame);
    const auto pixels = render();
    const auto &r = pixels[8 * 64 + 8], &b = pixels[8 * 64 + 55],
               &g = pixels[55 * 64 + 8];
    std::printf("%s texture proxy R=%g,%g,%g B=%g,%g,%g G=%g,%g,%g\n",
        mode,
        r[0],
        r[1],
        r[2],
        b[0],
        b[1],
        b[2],
        g[0],
        g[1],
        g[2]);
    ok &= r[0] > 5 && r[1] < 1e-4 && r[2] < 1e-4 && b[2] > 5 && b[0] < 1e-4
        && b[1] < 1e-4 && g[1] > 5 && g[0] < 1e-4 && g[2] < 1e-4;
    anari::setParameter(d, light, "visible", false);
    anari::commitParameters(d, light);
    ok &= mean(render()) < 1e-6;
    anari::setParameter(d, light, "visible", true);
    anari::commitParameters(d, light);
    anari::release(d, renderer2);
  }
  // Exercise composition when the independent shaping extensions are present.
  const bool testShaping = hasShaping && hasFocus;
  anari::setParameter(d, camera, "direction", vec3{0, -1, 0});
  anari::setParameter(d, camera, "up", vec3{0, 0, -1});
  anari::setParameter(d, camera, "height", 3.f);
  anari::commitParameters(d, camera);
  anari::setParameter(d, frame, "renderer", renderer);
  anari::commitParameters(d, frame);
  anari::unsetParameter(d, light, "colorMap");
  if (testShaping) {
    anari::setParameter(d, light, "coneAngle", .65f);
    anari::setParameter(d, light, "coneSoftness", .3f);
    anari::setParameter(d, light, "focus", 5.f);
  }
  anari::commitParameters(d, light);
  const double shaped = mean(render());
  const std::array<vec3, 4> replacementWhite = {
      vec3{1, 1, 1}, vec3{1, 1, 1}, vec3{1, 1, 1}, vec3{1, 1, 1}};
  auto replacement = anari::newArray2D(d, replacementWhite.data(), 2, 2);
  anari::setParameter(d, light, "colorMap", replacement);
  anari::commitParameters(d, light);
  const double shapedWhite = mean(render());
  ok &= shaped > 0 && (!testShaping || shaped < baseline)
      && std::abs(shapedWhite - shaped) < .01 * shaped;
  // Compare native textured radiance to an independent emissive mesh.
  if (testShaping) {
    anari::setParameter(d, light, "coneAngle", 1.57079632679f);
    anari::setParameter(d, light, "focus", 0.f);
  }
  anari::setParameter(d, light, "colorMap", image);
  anari::commitParameters(d, light);
  auto quality = anari::newObject<anari::Renderer>(d, "quality");
  anari::setParameter(d, quality, "pixelSamples", 128);
  anari::setParameter(d, quality, "fireflyFilter", "none");
  anari::setParameter(d, quality, "ambientRadiance", 0.f);
  anari::setParameter(d, quality, "background", vec4{0, 0, 0, 1});
  anari::commitParameters(d, quality);
  anari::setParameter(d, frame, "renderer", quality);
  anari::commitParameters(d, frame);
  const double analytic = mean(render());
  const std::array<vec3, 4> ep = {vec3{-.1f, 1.5f, -.1f},
      vec3{.1f, 1.5f, -.1f},
      vec3{.1f, 1.5f, .1f},
      vec3{-.1f, 1.5f, .1f}};
  const std::array<vec2, 4> uv = {
      vec2{0, 0}, vec2{0, 1}, vec2{1, 1}, vec2{1, 0}};
  auto eg = anari::newObject<anari::Geometry>(d, "triangle");
  anari::setParameterArray1D(d, eg, "vertex.position", ep.data(), 4);
  anari::setParameterArray1D(d, eg, "vertex.attribute0", uv.data(), 4);
  anari::setParameterArray1D(d, eg, "primitive.index", indices.data(), 2);
  anari::commitParameters(d, eg);
  auto sampler = anari::newObject<anari::Sampler>(d, "image2D");
  // Same asymmetric image, scaled by radiance 10 in the material.
  anari::setParameter(d, sampler, "image", image);
  anari::setParameter(d, sampler, "inAttribute", "attribute0");
  anari::setParameter(d, sampler, "filter", "linear");
  anari::setParameter(d, sampler, "wrapMode1", "clampToEdge");
  anari::setParameter(d, sampler, "wrapMode2", "clampToEdge");
  using mat4 = std::array<float, 16>;
  const mat4 transform{10, 0, 0, 0, 0, 10, 0, 0, 0, 0, 10, 0, 0, 0, 0, 1};
  anariSetParameter(
      d, sampler, "outTransform", ANARI_FLOAT32_MAT4, transform.data());
  anari::commitParameters(d, sampler);
  auto em = anari::newObject<anari::Material>(d, "physicallyBased");
  anari::setParameter(d, em, "baseColor", vec3{0, 0, 0});
  anari::setParameter(d, em, "emissive", sampler);
  anari::commitParameters(d, em);
  auto es = anari::newObject<anari::Surface>(d);
  anari::setAndReleaseParameter(d, es, "geometry", eg);
  anari::setAndReleaseParameter(d, es, "material", em);
  anari::commitParameters(d, es);
  const std::array<anari::Surface, 2> surfaces = {surface, es};
  anari::setParameterArray1D(d, world, "surface", surfaces.data(), 2);
  anari::unsetParameter(d, world, "light");
  anari::commitParameters(d, world);
  const double mesh = mean(render());
  std::printf(
      "quad shaping=%d shaped=%g white=%g, quality analytic=%g emissive=%g\n",
      int(testShaping),
      shaped,
      shapedWhite,
      analytic,
      mesh);
  ok &= analytic > 0 && std::abs(mesh - analytic) < .04 * analytic;
  anari::release(d, es);
  anari::release(d, sampler);
  anari::release(d, quality);
  anari::release(d, replacement);
  anari::release(d, image);
  anari::release(d, frame);
  anari::release(d, renderer);
  anari::release(d, camera);
  anari::release(d, world);
  anari::release(d, surface);
  anari::release(d, light);
  anari::release(d, d);
  if (!ok || g_errors) {
    std::fprintf(stderr, "FAIL: textured quad emission/update/visibility\n");
    return 1;
  }
  std::puts("textured quad checks passed");
  return 0;
}
