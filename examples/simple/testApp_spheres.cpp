/*
 * Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <anari/ext/visrtx/visrtx.h>
// std
#include <algorithm>
#include <array>
#include <cstdio>
#include <iostream>
#include <numeric>
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
  const uint32_t numSpheres = 10000;
  const float radius = .015f;

  std::mt19937 rng;
  rng.seed(0);
  std::normal_distribution<float> vert_dist(0.f, 0.25f);

  // Create + fill position and color arrays with randomized values //

  auto indicesArray = anari::newArray1D(device, ANARI_UINT32, numSpheres);
  auto positionsArray =
      anari::newArray1D(device, ANARI_FLOAT32_VEC3, numSpheres);
  auto distanceArray = anari::newArray1D(device, ANARI_FLOAT32, numSpheres);
  {
    auto *positions = anari::map<vec3>(device, positionsArray);
    auto *distances = anari::map<float>(device, distanceArray);
    for (uint32_t i = 0; i < numSpheres; i++) {
      const auto a = positions[i][0] = vert_dist(rng);
      const auto b = positions[i][1] = vert_dist(rng);
      const auto c = positions[i][2] = vert_dist(rng);
      distances[i] = std::sqrt(a * a + b * b + c * c); // will be roughly 0-1
    }
    anari::unmap(device, positionsArray);
    anari::unmap(device, distanceArray);

    auto *indicesBegin = anari::map<uint32_t>(device, indicesArray);
    auto *indicesEnd = indicesBegin + numSpheres;
    std::iota(indicesBegin, indicesEnd, 0);
    std::shuffle(indicesBegin, indicesEnd, rng);
    anari::unmap(device, indicesArray);
  }

  // Create and parameterize geometry //

  auto geometry = anari::newObject<anari::Geometry>(device, "sphere");
  anari::setParameter(
      device, geometry, "vertex.color", vec4{1.f, 0.f, 0.f, 1.f});
  anari::setAndReleaseParameter(
      device, geometry, "primitive.index", indicesArray);
  anari::setAndReleaseParameter(
      device, geometry, "vertex.position", positionsArray);
  anari::setAndReleaseParameter(
      device, geometry, "vertex.attribute0", distanceArray);
  anari::setParameter(device, geometry, "radius", radius);
  anari::commitParameters(device, geometry);

  // Create color map texture //

  auto texelArray = anari::newArray1D(device, ANARI_FLOAT32_VEC3, 2);
  {
    auto *texels = anari::map<vec3>(device, texelArray);
    texels[0][0] = 1.f;
    texels[0][1] = 0.f;
    texels[0][2] = 0.f;
    texels[1][0] = 0.f;
    texels[1][1] = 1.f;
    texels[1][2] = 0.f;
    anari::unmap(device, texelArray);
  }

  auto texture = anari::newObject<anari::Sampler>(device, "image1D");
  anari::setAndReleaseParameter(device, texture, "image", texelArray);
  anari::setParameter(device, texture, "filter", "linear");
  anari::commitParameters(device, texture);

  // Create and parameterize material //

  auto material = anari::newObject<anari::Material>(device, "matte");
  anari::setAndReleaseParameter(device, material, "color", texture);
  anari::commitParameters(device, material);

  // Create and parameterize surface //

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geometry);
  anari::setAndReleaseParameter(device, surface, "material", material);
  anari::commitParameters(device, surface);

  // Create and parameterize world //

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "surface", &surface, 1);
  anari::release(device, surface);
  anari::commitParameters(device, world);

  return world;
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
  }
  // Ignore INFO/DEBUG messages
}

int main()
{
  // Setup ANARI device //

  auto device = makeVisRTXDevice(statusFunc);

  anari::Extensions extensions =
      anari::extension::getInstanceExtensionStruct(device, device);

  if (!extensions.ANARI_KHR_GEOMETRY_SPHERE)
    printf("WARNING: device doesn't support ANARI_KHR_GEOMETRY_SPHERE\n");
  if (!extensions.ANARI_KHR_CAMERA_PERSPECTIVE)
    printf("WARNING: device doesn't support ANARI_KHR_CAMERA_PERSPECTIVE\n");
  if (!extensions.ANARI_KHR_LIGHT_DIRECTIONAL)
    printf("WARNING: device doesn't support ANARI_KHR_LIGHT_DIRECTIONAL\n");
  if (!extensions.ANARI_KHR_MATERIAL_MATTE)
    printf("WARNING: device doesn't support ANARI_KHR_MATERIAL_MATTE\n");

  // Create world from a helper function //

  auto world = generateScene(device);

  // Create camera //

  auto camera = anari::newObject<anari::Camera>(device, "perspective");

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

  auto renderer = anari::newObject<anari::Renderer>(device, "raycast");
  const vec4 backgroundColor = {0.1f, 0.1f, 0.1f, 1.f};
  anari::setParameter(device, renderer, "background", backgroundColor);
  anari::commitParameters(device, renderer);

  // Create frame (top-level object) //

  auto frame = anari::newObject<anari::Frame>(device);

  anari::setParameter(device, frame, "size", imageSize);
  anari::setParameter(device, frame, "channel.color", ANARI_UFIXED8_RGBA_SRGB);

  anari::setParameter(device, frame, "world", world);
  anari::setParameter(device, frame, "camera", camera);
  anari::setParameter(device, frame, "renderer", renderer);

  auto fccb = [](const void *uptr, anari::Device, anari::Frame) {
    printf("callback notifying frame completion! user ptr: %p\n", uptr);
  };

  anari::setParameter(device,
      frame,
      "frameCompletionCallback",
      (anari::FrameCompletionCallback)fccb);
  anari::setParameter(
      device, frame, "frameCompletionCallbackUserData", (void *)device);

  anari::commitParameters(device, frame);

  // Render frame and print out duration property //

  anari::render(device, frame);
  anari::wait(device, frame);

  float duration = 0.f;
  anari::getProperty(device, frame, "duration", duration, ANARI_NO_WAIT);

  printf("rendered frame in %fms\n", duration * 1000);

  stbi_flip_vertically_on_write(1);
  auto fb = anari::map<uint32_t>(device, frame, "channel.color");
  stbi_write_png(
      "testApp_spheres.png", fb.width, fb.height, 4, fb.data, 4 * fb.width);
  anari::unmap(device, frame, "channel.color");

  // Cleanup remaining ANARI objets //

  anari::release(device, camera);
  anari::release(device, renderer);
  anari::release(device, world);
  anari::release(device, frame);
  anari::release(device, device);

  return 0;
}
