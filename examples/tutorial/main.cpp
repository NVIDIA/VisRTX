/*
 * Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <anari/anari_cpp.hpp>
// C++ std anari_cpp type inference (VEC types from std::array<>)
#include <anari/anari_cpp/ext/std.h>
// std
#include <array>
#include <cstdio>
#include <random>
// stb_image
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// These aliases function as our vec types, which the header above enables
// inferring ANARIDataType enum values from their C++ type.
using uvec2 = std::array<unsigned int, 2>;
using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;

anari::World generateScene(anari::Device device)
{
  const int numSpheres = 10000;
  const float radius = .025f;

  std::mt19937 rng;
  rng.seed(0);
  std::normal_distribution<float> vert_dist(0.5f, 0.5f);

  // Create + fill position and color arrays with randomized values //

  auto positionsArray =
      anari::newArray1D(device, ANARI_FLOAT32_VEC3, numSpheres);
  {
    auto *positions = (vec3 *)anari::map(device, positionsArray);
    for (int i = 0; i < numSpheres; i++) {
      positions[i][0] = vert_dist(rng);
      positions[i][1] = vert_dist(rng);
      positions[i][2] = vert_dist(rng);
    }
    anari::unmap(device, positionsArray);
  }

  auto colorArray = anari::newArray1D(device, ANARI_FLOAT32_VEC4, numSpheres);
  {
    auto *colors = (vec4 *)anari::map(device, colorArray);
    for (int i = 0; i < numSpheres; i++) {
      colors[i][0] = vert_dist(rng);
      colors[i][1] = vert_dist(rng);
      colors[i][2] = vert_dist(rng);
      colors[i][3] = 1.f;
    }
    anari::unmap(device, colorArray);
  }

  // Create and parameterize geometry //

  auto geom = anari::newObject<anari::Geometry>(device, "sphere");
  anari::setAndReleaseParameter(
      device, geom, "vertex.position", positionsArray);
  anari::setAndReleaseParameter(device, geom, "vertex.color", colorArray);
  anari::setParameter(device, geom, "radius", radius);
  anari::commit(device, geom);

  // Create and parameterize material //

  auto mat = anari::newObject<anari::Material>(device, "matte");
  anari::setParameter(
      device, mat, "color", "color"); // draw values from "vertex.color"
  anari::commit(device, mat);

  // Create and parameterize surface //

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commit(device, surface);

  // Create and parameterize world //

  auto world = anari::newObject<anari::World>(device);
  anari::setAndReleaseParameter(
      device, world, "surface", anari::newArray1D(device, &surface));
  anari::release(device, surface);
  anari::commit(device, world);

  return world;
}

static void statusFunc(void * /*userData*/,
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
  }
  // Ignore INFO/DEBUG messages
}

int main()
{
  // Setup ANARI device //

  auto library = anari::loadLibrary("visrtx", statusFunc, nullptr);
  auto device = anari::newDevice(library, "default");

  // Create world from a helper function //

  auto world = generateScene(device);

  // Create camera //

  auto camera = anari::newObject<anari::Camera>(device, "perspective");

  const vec3 eye = {0.399189f, 0.398530f, -3.703001f};
  const vec3 dir = {0.f, 0.f, 1.f};
  const vec3 up = {0.f, 1.f, 0.f};

  anari::setParameter(device, camera, "position", eye);
  anari::setParameter(device, camera, "direction", dir);
  anari::setParameter(device, camera, "up", up);

  uvec2 imageSize = {1200, 800};
  anari::setParameter(
      device, camera, "aspect", imageSize[0] / float(imageSize[1]));

  anari::commit(device, camera);

  // Create renderer //

  auto renderer = anari::newObject<anari::Renderer>(device, "raycast");
  const vec4 backgroundColor = {0.1f, 0.1f, 0.1f, 1.f};
  anari::setParameter(device, renderer, "backgroundColor", backgroundColor);
  anari::commit(device, renderer);

  // Create frame (top-level object) //

  auto frame = anari::newObject<anari::Frame>(device);

  anari::setParameter(device, frame, "size", imageSize);
  anari::setParameter(device, frame, "color", ANARI_UFIXED8_RGBA_SRGB);

  anari::setParameter(device, frame, "world", world);
  anari::setParameter(device, frame, "camera", camera);
  anari::setParameter(device, frame, "renderer", renderer);

  anari::commit(device, frame);

  // Render frame and print out duration property //

  anari::render(device, frame);
  anari::wait(device, frame);

  float duration = 0.f;
  anari::getProperty(device, frame, "duration", duration, ANARI_NO_WAIT);

  printf("rendered frame in %fms\n", duration * 1000);

  stbi_flip_vertically_on_write(1);
  auto *fb = (uint32_t *)anari::map(device, frame, "color");
  stbi_write_png(
      "tutorial.png", imageSize[0], imageSize[1], 4, fb, 4 * imageSize[0]);
  anari::unmap(device, frame, "color");

  // Cleanup remaining ANARI objets //

  anari::release(device, camera);
  anari::release(device, renderer);
  anari::release(device, world);
  anari::release(device, frame);
  anari::release(device, device);
  anari::unloadLibrary(library);

  return 0;
}
