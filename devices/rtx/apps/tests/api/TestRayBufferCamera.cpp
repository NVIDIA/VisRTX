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

// rayBuffer camera (VISRTX_CAMERA_RAY_BUFFER). Rays built on the host to match
// a perspective camera's pixel-center rays must reproduce its depth and
// primitive IDs; per-pixel tmin/tmax must clip rays; and buffers whose size
// does not match the frame must make every ray miss.

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

static constexpr uvec2 IMAGE_SIZE = {64, 48};
static constexpr unsigned NUM_PIXELS = IMAGE_SIZE[0] * IMAGE_SIZE[1];
static constexpr unsigned MISS = ~0u;

static constexpr vec3 CAM_POS = {0.f, 0.f, 5.f};
static constexpr vec3 CAM_DIR = {0.f, 0.f, -1.f};
static constexpr vec3 CAM_UP = {0.f, 1.f, 0.f};
static constexpr float CAM_FOVY = 0.8f;

static vec3 operator+(vec3 a, vec3 b)
{
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
static vec3 operator*(float s, vec3 a)
{
  return {s * a[0], s * a[1], s * a[2]};
}
static vec3 cross(vec3 a, vec3 b)
{
  return {a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0]};
}
static vec3 normalize(vec3 a)
{
  return (1.f / std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])) * a;
}

struct Image
{
  std::vector<float> depth;
  std::vector<unsigned> primID;
};

static anari::World makeWorld(ANARIDevice device)
{
  const std::array<vec3, 3> centers = {
      vec3{-1.2f, 0.f, 0.f}, vec3{0.f, 0.3f, -0.5f}, vec3{1.2f, -0.2f, 0.f}};

  auto geom = anari::newObject<anari::Geometry>(device, "sphere");
  anari::setParameterArray1D(
      device, geom, "vertex.position", centers.data(), centers.size());
  anari::setParameter(device, geom, "radius", 0.6f);
  anari::commitParameters(device, geom);

  auto mat = anari::newObject<anari::Material>(device, "matte");
  anari::commitParameters(device, mat);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "surface", &surface, 1);
  anari::release(device, surface);
  anari::commitParameters(device, world);
  return world;
}

static Image render(ANARIDevice device, anari::World world, anari::Camera cam)
{
  auto renderer = anari::newObject<anari::Renderer>(device, "default");
  anari::setParameter(device, renderer, "pixelSamples", 1);
  anari::commitParameters(device, renderer);

  auto frame = anari::newObject<anari::Frame>(device);
  anari::setParameter(device, frame, "size", IMAGE_SIZE);
  anari::setParameter(device, frame, "channel.color", ANARI_UFIXED8_RGBA_SRGB);
  anari::setParameter(device, frame, "channel.depth", ANARI_FLOAT32);
  anari::setParameter(device, frame, "channel.primitiveId", ANARI_UINT32);
  anari::setParameter(device, frame, "world", world);
  anari::setParameter(device, frame, "camera", cam);
  anari::setAndReleaseParameter(device, frame, "renderer", renderer);
  anari::commitParameters(device, frame);

  // The first sample of the first frame is pixel-centered for every camera.
  anari::render(device, frame);
  anari::wait(device, frame);

  Image img;
  auto depth = anari::map<float>(device, frame, "channel.depth");
  img.depth.assign(depth.data, depth.data + NUM_PIXELS);
  anari::unmap(device, frame, "channel.depth");
  auto ids = anari::map<uint32_t>(device, frame, "channel.primitiveId");
  img.primID.assign(ids.data, ids.data + NUM_PIXELS);
  anari::unmap(device, frame, "channel.primitiveId");

  anari::release(device, frame);
  return img;
}

static anari::Camera makePerspective(ANARIDevice device)
{
  auto cam = anari::newObject<anari::Camera>(device, "perspective");
  anari::setParameter(device, cam, "position", CAM_POS);
  anari::setParameter(device, cam, "direction", CAM_DIR);
  anari::setParameter(device, cam, "up", CAM_UP);
  anari::setParameter(device, cam, "fovy", CAM_FOVY);
  anari::setParameter(
      device, cam, "aspect", IMAGE_SIZE[0] / float(IMAGE_SIZE[1]));
  anari::commitParameters(device, cam);
  return cam;
}

// Pixel-center ray directions of the perspective camera above, row-major.
static std::vector<vec3> perspectiveDirections(uvec2 size)
{
  const float h = 2.f * std::tan(0.5f * CAM_FOVY);
  const float w = h * IMAGE_SIZE[0] / float(IMAGE_SIZE[1]);
  const vec3 du = w * normalize(cross(CAM_DIR, CAM_UP));
  const vec3 dv = h * normalize(cross(du, CAM_DIR));
  const vec3 d00 = CAM_DIR + -0.5f * du + -0.5f * dv;

  std::vector<vec3> dirs(size_t(size[0]) * size[1]);
  for (unsigned y = 0; y < size[1]; y++) {
    for (unsigned x = 0; x < size[0]; x++) {
      const float sx = (x + 0.5f) / size[0];
      const float sy = (y + 0.5f) / size[1];
      dirs[y * size[0] + x] = d00 + sx * du + sy * dv;
    }
  }
  return dirs;
}

template <typename T>
static void setBuffer(ANARIDevice device,
    anari::Camera cam,
    const char *name,
    const std::vector<T> &data,
    uvec2 size)
{
  auto array = anari::newArray2D(device, data.data(), size[0], size[1]);
  anari::setAndReleaseParameter(device, cam, name, array);
}

// Ray directions only; origins fall back to the camera 'position'.
static anari::Camera makeRayBuffer(ANARIDevice device, uvec2 size)
{
  auto cam = anari::newObject<anari::Camera>(device, "rayBuffer");
  anari::setParameter(device, cam, "position", CAM_POS);
  setBuffer(device, cam, "ray.dir", perspectiveDirections(size), size);
  return cam;
}

static bool check(bool cond, const char *what)
{
  printf("%-55s %s\n", what, cond ? "ok" : "FAIL");
  return cond;
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);
  auto world = makeWorld(device);

  bool ok = true;

  auto persp = makePerspective(device);
  const Image ref = render(device, world, persp);
  anari::release(device, persp);

  unsigned refHits = 0;
  for (unsigned id : ref.primID)
    refHits += id != MISS;
  ok &= check(refHits > NUM_PIXELS / 10 && refHits < NUM_PIXELS,
      "reference image has both hits and misses");

  // 1. Parity with the perspective camera (silhouettes may differ by
  //    floating-point rounding, so allow a handful of mismatched pixels).
  {
    auto cam = makeRayBuffer(device, IMAGE_SIZE);
    anari::commitParameters(device, cam);
    const Image img = render(device, world, cam);
    anari::release(device, cam);

    unsigned idMismatch = 0;
    unsigned depthMismatch = 0;
    for (unsigned i = 0; i < NUM_PIXELS; i++) {
      if (img.primID[i] != ref.primID[i])
        idMismatch++;
      else if (ref.primID[i] != MISS
          && std::fabs(img.depth[i] - ref.depth[i]) > 1e-3f)
        depthMismatch++;
    }
    printf("  parity: %u primID / %u depth mismatches\n",
        idMismatch,
        depthMismatch);
    ok &= check(idMismatch <= 4 && depthMismatch == 0,
        "rayBuffer matches perspective camera");
  }

  // 2. Per-pixel interval: left half tmax before the spheres, right half tmin
  //    past them; every pixel must miss.
  {
    std::vector<float> tmin(NUM_PIXELS, 0.f);
    std::vector<float> tmax(NUM_PIXELS, 1e30f);
    for (unsigned y = 0; y < IMAGE_SIZE[1]; y++) {
      for (unsigned x = 0; x < IMAGE_SIZE[0]; x++) {
        if (x < IMAGE_SIZE[0] / 2)
          tmax[y * IMAGE_SIZE[0] + x] = 1.f;
        else
          tmin[y * IMAGE_SIZE[0] + x] = 100.f;
      }
    }

    auto cam = makeRayBuffer(device, IMAGE_SIZE);
    setBuffer(device, cam, "ray.tmin", tmin, IMAGE_SIZE);
    setBuffer(device, cam, "ray.tmax", tmax, IMAGE_SIZE);
    anari::commitParameters(device, cam);
    const Image img = render(device, world, cam);
    anari::release(device, cam);

    unsigned hits = 0;
    for (unsigned id : img.primID)
      hits += id != MISS;
    ok &= check(hits == 0, "ray.tmin / ray.tmax clip every ray");
  }

  // 3. Explicit origins: moving each origin 1 unit along its own ray
  //    shortens every hit distance by exactly 1.
  {
    const auto dirs = perspectiveDirections(IMAGE_SIZE);
    std::vector<vec3> org(NUM_PIXELS);
    for (unsigned i = 0; i < NUM_PIXELS; i++)
      org[i] = CAM_POS + normalize(dirs[i]);

    auto cam = makeRayBuffer(device, IMAGE_SIZE);
    setBuffer(device, cam, "ray.org", org, IMAGE_SIZE);
    anari::commitParameters(device, cam);
    const Image img = render(device, world, cam);
    anari::release(device, cam);

    unsigned compared = 0;
    unsigned bad = 0;
    for (unsigned i = 0; i < NUM_PIXELS; i++) {
      if (img.primID[i] == MISS || img.primID[i] != ref.primID[i])
        continue;
      compared++;
      bad += std::fabs(ref.depth[i] - img.depth[i] - 1.f) > 1e-3f;
    }
    ok &= check(compared > 0 && bad == 0, "ray.org offsets hit distances");
  }

  // 4. Buffer size not matching the frame: everything misses, no crash.
  {
    const uvec2 small = {IMAGE_SIZE[0] / 2, IMAGE_SIZE[1] / 2};
    auto cam = makeRayBuffer(device, small);
    anari::commitParameters(device, cam);
    const Image img = render(device, world, cam);
    anari::release(device, cam);

    unsigned hits = 0;
    for (unsigned id : img.primID)
      hits += id != MISS;
    ok &= check(hits == 0, "mismatched buffer size misses everywhere");
  }

  anari::release(device, world);
  anari::release(device, device);

  if (!ok) {
    fprintf(stderr, "FAIL: rayBuffer camera\n");
    return 1;
  }
  printf("rayBuffer camera test passed\n");
  return 0;
}
