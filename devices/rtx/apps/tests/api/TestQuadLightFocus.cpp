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
  if (extensions) {
    for (const char **entry = extensions; *entry; ++entry)
      advertised |= std::strcmp(*entry, "ANARI_VISRTX_LIGHT_QUAD_FOCUS") == 0;
  }
  if (!advertised) {
    std::fprintf(stderr, "quad focus extension is missing\n");
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
  const auto focus = [&](float value, vec3 tint = {0.f, 0.f, 0.f}) {
    anari::setParameter(d, light, "focus", value);
    anari::setParameter(d, light, "focusTint", tint);
    anari::commitParameters(d, light);
  };
  const auto wide = render();
  focus(20.f);
  const auto tight = render();
  focus(20.f, vec3{1.f, 1.f, 1.f});
  const auto white = render();
  anari::unsetParameter(d, light, "focus");
  anari::unsetParameter(d, light, "focusTint");
  anari::commitParameters(d, light);
  const auto restored = render();
  bool ok = mean(wide) > 1e-3 && mean(tight) > 1e-4
      && mean(tight, true) < 0.5 * mean(wide, true)
      && std::abs(mean(white) - mean(wide)) < 0.05 * mean(wide)
      && std::abs(mean(restored) - mean(wide)) < 0.05 * mean(wide);
  std::printf("focus floor wide=%g tight=%g white=%g restored=%g\n",
      mean(wide),
      mean(tight),
      mean(white),
      mean(restored));

  // View the emitter 45 degrees off-axis: cosine^2 is approximately one half.
  anari::setParameter(d, camera, "position", vec3{1.f, 0.5f, 0.f});
  anari::setParameter(d, camera, "direction", vec3{-1.f, 1.f, 0.f});
  anari::setParameter(d, camera, "up", vec3{0.f, 0.f, 1.f});
  anari::setParameter(d, camera, "height", 0.6f);
  anari::commitParameters(d, camera);
  for (const char *mode : {"interactive", "quality"}) {
    auto proxyRenderer = anari::newObject<anari::Renderer>(d, mode);
    anari::setParameter(d, proxyRenderer, "pixelSamples", 16);
    anari::setParameter(
        d, proxyRenderer, "background", vec4{0.f, 0.f, 0.f, 1.f});
    anari::setParameter(d, proxyRenderer, "ambientRadiance", 0.f);
    anari::setParameter(d, proxyRenderer, "fireflyFilter", "none");
    anari::commitParameters(d, proxyRenderer);
    anari::setParameter(d, frame, "renderer", proxyRenderer);
    anari::commitParameters(d, frame);
    focus(0.f);
    const auto proxyWide = render();
    focus(2.f);
    const auto proxyHalf = render();
    focus(20.f);
    const auto proxyTight = render();
    focus(20.f, vec3{1.f, 1.f, 1.f});
    const auto proxyWhite = render();
    ok &= mean(proxyWide) > 0.1 && mean(proxyHalf) > 0.35 * mean(proxyWide)
        && mean(proxyHalf) < 0.65 * mean(proxyWide)
        && mean(proxyTight) < 0.02 * mean(proxyWide)
        && std::abs(mean(proxyWhite) - mean(proxyWide))
            < 0.01 * mean(proxyWide);
    std::printf("%s focus proxy wide=%g half=%g tight=%g white=%g\n",
        mode,
        mean(proxyWide),
        mean(proxyHalf),
        mean(proxyTight),
        mean(proxyWhite));
    anari::release(d, proxyRenderer);
  }
  anari::release(d, frame);
  anari::release(d, renderer);
  anari::release(d, camera);
  anari::release(d, world);
  anari::release(d, light);
  anari::release(d, surface);
  anari::release(d, d);
  if (!ok || g_errors) {
    std::fprintf(
        stderr, "FAIL: quad focus/default restoration/proxy behavior\n");
    return 1;
  }
  std::puts("quad focus, retained edits and proxy consistency passed");
  return 0;
}
