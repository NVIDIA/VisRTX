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

// anari_cpp
#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp.hpp>
// C++ std anari_cpp type inference (VEC types from std::array<>)
#include <anari/anari_cpp/ext/std.h>
// VisRTX
#include <anari/ext/visrtx/makeVisRTXDevice.h>
// std
#include <algorithm>
#include <array>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <random>
#include <utility>
// stb_image
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// These aliases function as our vec types, which the header above enables
// inferring ANARIDataType enum values from their C++ type.
using uvec2 = std::array<unsigned int, 2>;
using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;

std::vector<anari::Object> g_objectPool; // for releasing at the end

template <typename T, typename... Args>
static T createObjectInPool(Args &&...args)
{
  auto o = anari::newObject<T>(std::forward<Args>(args)...);
  g_objectPool.push_back(o);
  return o;
}

template <typename... Args>
static anari::Array1D createArrayInPool(Args &&...args)
{
  auto a = anari::newArray1D(std::forward<Args>(args)...);
  g_objectPool.push_back(a);
  return a;
}

anari::Frame setupFrame(anari::Device device, anari::World world)
{
  // Create camera //

  auto camera = createObjectInPool<anari::Camera>(device, "perspective");

  const vec3 eye = {0.f, 0.f, -2.f};
  const vec3 dir = {0.f, 0.f, 1.f};
  const vec3 up = {0.f, 1.f, 0.f};

  anari::setParameter(device, camera, "position", eye);
  anari::setParameter(device, camera, "direction", dir);
  anari::setParameter(device, camera, "up", up);

  uvec2 imageSize = {1200, 800};
  anari::setParameter(
      device, camera, "aspect", imageSize[0] / float(imageSize[1]));

  anari::commitParameters(device, camera);

  // Create renderer //

  auto renderer = createObjectInPool<anari::Renderer>(device, "debug");
  const vec4 backgroundColor = {0.1f, 0.1f, 0.1f, 1.f};
  anari::setParameter(device, renderer, "background", backgroundColor);
  anari::setParameter(device, renderer, "method", "baseColor");
  anari::commitParameters(device, renderer);

  // Create frame (top-level object) //

  auto frame = createObjectInPool<anari::Frame>(device);
  anari::setParameter(device, frame, "size", imageSize);
  anari::setParameter(device, frame, "channel.color", ANARI_UFIXED8_RGBA_SRGB);
  anari::setParameter(device, frame, "camera", camera);
  anari::setParameter(device, frame, "world", world);
  anari::setParameter(device, frame, "renderer", renderer);
  anari::commitParameters(device, frame);

  return frame;
}

anari::Surface generateSurface(anari::Device device)
{
  const uint32_t numSpheres = 10000;
  const float radius = .015f;

  std::mt19937 rng;
  rng.seed(0);
  std::normal_distribution<float> vert_dist(0.f, 0.25f);

  // Create + fill position and color arrays with randomized values //

  auto positionsArray =
      createArrayInPool(device, ANARI_FLOAT32_VEC3, numSpheres);
  auto distanceArray = createArrayInPool(device, ANARI_FLOAT32, numSpheres);
  {
    auto *positions = anari::map<vec3>(device, positionsArray);
    std::for_each(positions, positions + numSpheres, [&](auto &p) {
      p[0] = vert_dist(rng);
      p[1] = vert_dist(rng);
      p[2] = vert_dist(rng);
    });
    anari::unmap(device, positionsArray);
  }

  // Create and parameterize geometry //

  auto geometry = createObjectInPool<anari::Geometry>(device, "sphere");
  anari::setParameter(device, geometry, "vertex.position", positionsArray);
  anari::setParameter(device, geometry, "radius", radius);
  anari::commitParameters(device, geometry);

  // Create and parameterize material //

  auto material = createObjectInPool<anari::Material>(device, "matte");

  // Create and parameterize surface //

  auto surface = createObjectInPool<anari::Surface>(device);
  anari::setParameter(device, surface, "geometry", geometry);
  anari::setParameter(device, surface, "material", material);
  anari::commitParameters(device, surface);

  return surface;
}

static void statusFunc(const void * /*userData*/,
    ANARIDevice /*device*/,
    ANARIObject source,
    ANARIDataType /*sourceType*/,
    ANARIStatusSeverity severity,
    ANARIStatusCode /*code*/,
    const char *message)
{
  if (severity == ANARI_SEVERITY_FATAL_ERROR) {
    fprintf(stderr, "[FATAL][%p] %s\n", source, message);
    std::exit(1);
  } else if (severity == ANARI_SEVERITY_ERROR) {
    fprintf(stderr, "[ERROR][%p] %s\n", source, message);
  } else if (severity == ANARI_SEVERITY_WARNING) {
    fprintf(stderr, "[WARN ][%p] %s\n", source, message);
  } else if (severity == ANARI_SEVERITY_PERFORMANCE_WARNING) {
    fprintf(stderr, "[PERF ][%p] %s\n", source, message);
  } else if (severity == ANARI_SEVERITY_INFO) {
    fprintf(stderr, "[INFO ][%p] %s\n", source, message);
  } else if (severity == ANARI_SEVERITY_DEBUG) {
    fprintf(stderr, "[DEBUG][%p] %s\n", source, message);
  }
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);
  auto surface = generateSurface(device);

  auto surfaceArray = createArrayInPool(device, ANARI_SURFACE, 1);
  auto *surfaces = anari::map<anari::Surface>(device, surfaceArray);
  surfaces[0] = surface;
  anari::unmap(device, surfaceArray);

  auto world = createObjectInPool<anari::World>(device);
  anari::setParameter(device, world, "surface", surfaceArray);
  anari::release(device, surface);
  anari::commitParameters(device, world);

  auto frame = setupFrame(device, world);

  // Render frame and print out duration property //

  anari::render(device, frame);
  anari::wait(device, frame);

  float duration = 0.f;
  anari::getProperty(device, frame, "duration", duration, ANARI_NO_WAIT);

  printf("+++++++++++++++++++++++++rendered frame in %fms\n", duration * 1000);

  // Create new surface array, use it, and render again //

  surfaceArray = createArrayInPool(device, ANARI_SURFACE, 1);
  surfaces = anari::map<anari::Surface>(device, surfaceArray);
  surfaces[0] = surface;
  anari::unmap(device, surfaceArray);

  anari::setParameter(device, world, "surface", surfaceArray);
  anari::commitParameters(device, world);

  anari::render(device, frame);
  anari::wait(device, frame);

  duration = 0.f;
  anari::getProperty(device, frame, "duration", duration, ANARI_NO_WAIT);

  printf("+++++++++++++++++++++++++rendered frame in %fms\n", duration * 1000);

  // Output image //

  stbi_flip_vertically_on_write(1);
  auto fb = anari::map<uint32_t>(device, frame, "channel.color");
  stbi_write_png("testApp_evictGPUArray.png",
      fb.width,
      fb.height,
      4,
      fb.data,
      4 * fb.width);
  anari::unmap(device, frame, "channel.color");

  // Cleanup remaining ANARI objets //

  for (auto o : g_objectPool)
    anari::release(device, o);
  anari::release(device, device);

  return 0;
}
