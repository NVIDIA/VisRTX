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

// GPU regression for phantom self-shadowing on large analytic spheres
// (concentric acne rings): an RTOW-scale ground sphere (r=1000) lit by a
// directional light, with no occluders, must render uniformly lit. Shadow rays
// start on the sphere's own surface and graze it, so any false self-hit shows
// up as dark rings. Renders with the 'interactive' renderer (light-sample
// shadow rays, AO off) and asserts that no pixel in the central region is
// significantly darker than the region's brightest pixel.

// anari_cpp
#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp/ext/std.h>
#include <anari/anari_cpp.hpp>
// VisRTX
#include <anari/ext/visrtx/makeVisRTXDevice.h>
// std
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
// stb_image
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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

int main()
{
  auto device = makeVisRTXDevice(statusFunc);

  const uvec2 imageSize = {512, 256};

  // RTOW ground sphere: r=1000 centered at (0,-1000,0).
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

  auto light = anari::newObject<anari::Light>(device, "directional");
  anari::setParameter(device, light, "direction", vec3{0.3f, -1.f, 0.2f});
  anari::setParameter(device, light, "irradiance", 3.f);
  anari::commitParameters(device, light);

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "surface", &surface, 1);
  anari::setParameterArray1D(device, world, "light", &light, 1);
  anari::release(device, surface);
  anari::release(device, light);
  anari::commitParameters(device, world);

  // Camera above the ground, looking out at a shallow angle so the visible
  // ground spans near-to-far grazing shadow-ray geometry (the ring zone).
  auto camera = anari::newObject<anari::Camera>(device, "perspective");
  anari::setParameter(device, camera, "position", vec3{0.f, 2.f, 0.f});
  anari::setParameter(device, camera, "direction", vec3{0.f, -0.25f, 1.f});
  anari::setParameter(device, camera, "up", vec3{0.f, 1.f, 0.f});
  anari::setParameter(
      device, camera, "aspect", imageSize[0] / float(imageSize[1]));
  anari::commitParameters(device, camera);

  auto renderer = anari::newObject<anari::Renderer>(device, "interactive");
  anari::setParameter(device, renderer, "background", vec4{0.f, 0.f, 0.f, 1.f});
  anari::setParameter(device, renderer, "ambientSamples", 0);
  anari::setParameter(device, renderer, "ambientRadiance", 0.f);
  anari::setParameter(device, renderer, "pixelSamples", 16);
  anari::commitParameters(device, renderer);

  auto frame = anari::newObject<anari::Frame>(device);
  anari::setParameter(device, frame, "size", imageSize);
  anari::setParameter(device, frame, "channel.color", ANARI_UFIXED8_RGBA_SRGB);
  anari::setAndReleaseParameter(device, frame, "world", world);
  anari::setAndReleaseParameter(device, frame, "camera", camera);
  anari::setAndReleaseParameter(device, frame, "renderer", renderer);
  anari::commitParameters(device, frame);

  anari::render(device, frame);
  anari::wait(device, frame);

  auto fb = anari::map<uint32_t>(device, frame, "channel.color");

  stbi_flip_vertically_on_write(1);
  stbi_write_png("testApp_groundSphereShadows.png",
      fb.width,
      fb.height,
      4,
      fb.data,
      4 * fb.width);

  // Bottom half only: all ground, no sky/silhouette (framebuffer row 0 is the
  // image bottom). Lighting varies smoothly over the giant sphere, so a dark
  // pixel == a phantom self-shadow ring.
  uint32_t minR = 255, maxR = 0;
  uint64_t dark = 0, total = 0;
  for (uint32_t y = imageSize[1] / 8; y < imageSize[1] / 2; ++y) {
    for (uint32_t x = imageSize[0] / 8; x < 7 * imageSize[0] / 8; ++x) {
      const uint32_t r = fb.data[y * fb.width + x] & 0xFFu;
      minR = std::min(minR, r);
      maxR = std::max(maxR, r);
      ++total;
      if (r < 128u)
        ++dark;
    }
  }
  anari::unmap(device, frame, "channel.color");
  anari::release(device, frame);
  anari::release(device, device);

  printf("ground region: minR=%u maxR=%u darkFraction=%f\n",
      minR,
      maxR,
      double(dark) / double(total));

  // Fully lit gray ground under a bright directional light: every pixel must
  // be bright. Phantom rings pull pixels toward black.
  if (maxR < 128u) {
    fprintf(stderr, "FAIL: ground unexpectedly dark overall (bad scene?)\n");
    return 1;
  }
  if (dark) {
    fprintf(stderr,
        "FAIL: %llu/%llu ground pixels shadowed with no occluder (acne)\n",
        (unsigned long long)dark,
        (unsigned long long)total);
    return 1;
  }
  printf("ground sphere shadow acne regression passed\n");
  return 0;
}
