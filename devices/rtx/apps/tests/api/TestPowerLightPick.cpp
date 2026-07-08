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

// Power-proportional Light Pick must stay unbiased. Light transport is linear:
// the converged render of a scene lit by two lights must equal the sum of the
// two single-light renders. Power-proportional picking folds each light's pick
// probability into the sampled pdf; if that fold is wrong (dropped, doubled, or
// mismatched between NEE and MIS), the dim light's contribution is mis-scaled
// and additivity breaks. The two lights here have a 20:1 power ratio so a
// broken fold shifts the dim light's weight far outside tolerance. Rendered
// with 'quality' (NEE + MIS) into a linear float buffer at high sample count so
// the additivity residual is Monte-Carlo noise only.

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
#include <utility>
#include <vector>

using uvec2 = std::array<unsigned int, 2>;
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
static constexpr int PIXEL_SAMPLES = 256;

// Render a ground sphere lit by an arbitrary subset of directional lights and
// return the linear-RGB framebuffer.
static std::vector<vec4> renderLit(ANARIDevice device,
    const std::vector<std::pair<vec3, float>> &lights)
{
  auto geometry = anari::newObject<anari::Geometry>(device, "sphere");
  const vec3 center = {0.f, -1000.f, 0.f};
  anari::setParameterArray1D(device, geometry, "vertex.position", &center, 1);
  anari::setParameter(device, geometry, "radius", 1000.f);
  anari::commitParameters(device, geometry);

  auto material = anari::newObject<anari::Material>(device, "matte");
  anari::setParameter(device, material, "color", vec3{0.8f, 0.8f, 0.8f});
  anari::commitParameters(device, material);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geometry);
  anari::setAndReleaseParameter(device, surface, "material", material);
  anari::commitParameters(device, surface);

  std::vector<anari::Light> lightHandles;
  for (const auto &[dir, irradiance] : lights) {
    auto light = anari::newObject<anari::Light>(device, "directional");
    anari::setParameter(device, light, "direction", dir);
    anari::setParameter(device, light, "irradiance", irradiance);
    anari::commitParameters(device, light);
    lightHandles.push_back(light);
  }

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "surface", &surface, 1);
  if (!lightHandles.empty()) {
    anari::setParameterArray1D(
        device, world, "light", lightHandles.data(), lightHandles.size());
  }
  anari::release(device, surface);
  for (auto light : lightHandles)
    anari::release(device, light);
  anari::commitParameters(device, world);

  auto camera = anari::newObject<anari::Camera>(device, "perspective");
  anari::setParameter(device, camera, "position", vec3{0.f, 2.f, 0.f});
  anari::setParameter(device, camera, "direction", vec3{0.f, -0.25f, 1.f});
  anari::setParameter(device, camera, "up", vec3{0.f, 1.f, 0.f});
  anari::setParameter(
      device, camera, "aspect", IMAGE_SIZE[0] / float(IMAGE_SIZE[1]));
  anari::commitParameters(device, camera);

  auto renderer = anari::newObject<anari::Renderer>(device, "quality");
  anari::setParameter(device, renderer, "background", vec4{0.f, 0.f, 0.f, 1.f});
  anari::setParameter(device, renderer, "ambientRadiance", 0.f);
  anari::setParameter(device, renderer, "pixelSamples", PIXEL_SAMPLES);
  // Read linear radiance: the default 'tonemap' firefly filter is nonlinear and
  // would break the additivity invariant this test relies on.
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

// Mean linear luminance over the ground region (bottom half, framebuffer row 0
// is the image bottom).
static double regionMeanLuminance(const std::vector<vec4> &fb)
{
  double sum = 0.0;
  uint64_t n = 0;
  for (uint32_t y = IMAGE_SIZE[1] / 8; y < IMAGE_SIZE[1] / 2; ++y) {
    for (uint32_t x = IMAGE_SIZE[0] / 8; x < 7 * IMAGE_SIZE[0] / 8; ++x) {
      const vec4 &p = fb[y * IMAGE_SIZE[0] + x];
      sum += 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
      ++n;
    }
  }
  return n ? sum / double(n) : 0.0;
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);

  // 20:1 power ratio between the two lights.
  const std::pair<vec3, float> bright = {vec3{0.3f, -1.f, 0.2f}, 3.0f};
  const std::pair<vec3, float> dim = {vec3{-0.4f, -1.f, -0.3f}, 0.15f};

  const double both = regionMeanLuminance(renderLit(device, {bright, dim}));
  const double onlyBright = regionMeanLuminance(renderLit(device, {bright}));
  const double onlyDim = regionMeanLuminance(renderLit(device, {dim}));

  anari::release(device, device);

  const double sum = onlyBright + onlyDim;
  const double relErr = std::abs(both - sum) / sum;

  printf("both=%f  bright=%f  dim=%f  sum=%f  relErr=%f\n",
      both,
      onlyBright,
      onlyDim,
      sum,
      relErr);

  // The dim light carries ~5% of the total; a dropped or doubled pick-pdf fold
  // moves the sum by far more than this tolerance. Residual is MC noise at 256
  // spp averaged over the region.
  constexpr double TOLERANCE = 0.02;
  // Negated so a NaN relErr (black or NaN render) FAILS instead of passing open.
  if (!(relErr <= TOLERANCE)) {
    fprintf(stderr,
        "FAIL: two-light render not additive (relErr=%f, tol %f): light pick "
        "pdf fold is biased\n",
        relErr,
        TOLERANCE);
    return 1;
  }
  printf("power light pick additivity passed\n");
  return 0;
}
