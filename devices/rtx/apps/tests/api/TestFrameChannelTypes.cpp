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

// KHR_FRAME_CHANNEL_ALBEDO / KHR_FRAME_CHANNEL_NORMAL element types. A single
// camera-facing matte quad of known color fills the frame; each spec-listed
// channel type is mapped and the center pixel compared against the expected
// encoding of that albedo / normal. Unsupported types must not map.

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

using uvec2 = std::array<unsigned int, 2>;
using vec3 = std::array<float, 3>;

static void statusFunc(const void *,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity == ANARI_SEVERITY_FATAL_ERROR) {
    fprintf(stderr, "[FATAL] %s\n", message);
    std::exit(1);
  } else if (severity == ANARI_SEVERITY_ERROR) {
    fprintf(stderr, "[ERROR] %s\n", message);
  }
}

static constexpr uvec2 kSize = {32, 32};
static constexpr vec3 kAlbedo = {0.2f, 0.5f, 0.8f};

static int g_failures = 0;

static void check(bool ok, const char *what)
{
  if (!ok) {
    fprintf(stderr, "FAIL: %s\n", what);
    g_failures++;
  }
}

static float srgbEncode(float c)
{
  return c <= 0.0031308f ? c * 12.92f
                         : 1.055f * std::pow(c, 1.f / 2.4f) - 0.055f;
}

static anari::World makeWorld(anari::Device d)
{
  // Quad in the z=0 plane, facing +z toward the camera at z=2.
  const vec3 positions[4] = {
      {-1.f, -1.f, 0.f}, {1.f, -1.f, 0.f}, {1.f, 1.f, 0.f}, {-1.f, 1.f, 0.f}};
  auto geom = anari::newObject<anari::Geometry>(d, "quad");
  anari::setParameterArray1D(d, geom, "vertex.position", positions, 4);
  anari::commitParameters(d, geom);

  auto mat = anari::newObject<anari::Material>(d, "matte");
  anari::setParameter(d, mat, "color", kAlbedo);
  anari::commitParameters(d, mat);

  auto surf = anari::newObject<anari::Surface>(d);
  anari::setAndReleaseParameter(d, surf, "geometry", geom);
  anari::setAndReleaseParameter(d, surf, "material", mat);
  anari::commitParameters(d, surf);

  auto world = anari::newObject<anari::World>(d);
  anari::setParameterArray1D(d, world, "surface", &surf, 1);
  anari::release(d, surf);
  anari::commitParameters(d, world);
  return world;
}

// Render one frame with the given albedo/normal channel types and return the
// frame (caller releases).
static anari::Frame renderWith(anari::Device d,
    anari::World world,
    anari::Camera camera,
    anari::Renderer renderer,
    ANARIDataType albedoType,
    ANARIDataType normalType)
{
  auto frame = anari::newObject<anari::Frame>(d);
  anari::setParameter(d, frame, "size", kSize);
  anari::setParameter(d, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setParameter(d, frame, "channel.albedo", albedoType);
  anari::setParameter(d, frame, "channel.normal", normalType);
  anari::setParameter(d, frame, "world", world);
  anari::setParameter(d, frame, "camera", camera);
  anari::setParameter(d, frame, "renderer", renderer);
  anari::commitParameters(d, frame);
  anari::render(d, frame);
  anari::wait(d, frame);
  return frame;
}

static size_t centerIndex()
{
  return size_t(kSize[1] / 2) * kSize[0] + kSize[0] / 2;
}

int main()
{
  auto d = makeVisRTXDevice(statusFunc);

  anari::Extensions ext = anari::extension::getInstanceExtensionStruct(d, d);
  check(ext.ANARI_KHR_FRAME_CHANNEL_ALBEDO,
      "advertises KHR_FRAME_CHANNEL_ALBEDO");
  check(ext.ANARI_KHR_FRAME_CHANNEL_NORMAL,
      "advertises KHR_FRAME_CHANNEL_NORMAL");

  auto world = makeWorld(d);

  auto camera = anari::newObject<anari::Camera>(d, "orthographic");
  anari::setParameter(d, camera, "position", vec3{0.f, 0.f, 2.f});
  anari::setParameter(d, camera, "direction", vec3{0.f, 0.f, -1.f});
  anari::setParameter(d, camera, "up", vec3{0.f, 1.f, 0.f});
  anari::setParameter(d, camera, "height", 1.f);
  anari::setParameter(d, camera, "aspect", 1.f);
  anari::commitParameters(d, camera);

  auto renderer = anari::newObject<anari::Renderer>(d, "quality");
  anari::setParameter(d, renderer, "pixelSamples", 1);
  anari::commitParameters(d, renderer);

  const size_t c = centerIndex();

  // FLOAT32_VEC3 albedo + normal.
  {
    auto frame = renderWith(
        d, world, camera, renderer, ANARI_FLOAT32_VEC3, ANARI_FLOAT32_VEC3);
    auto a = anari::map<vec3>(d, frame, "channel.albedo");
    check(a.data && a.pixelType == ANARI_FLOAT32_VEC3, "albedo f32 maps");
    if (a.data) {
      for (int i = 0; i < 3; i++)
        check(std::fabs(a.data[c][i] - kAlbedo[i]) < 1e-3f, "albedo f32 value");
    }
    anari::unmap(d, frame, "channel.albedo");

    auto n = anari::map<vec3>(d, frame, "channel.normal");
    check(n.data && n.pixelType == ANARI_FLOAT32_VEC3, "normal f32 maps");
    if (n.data)
      check(std::fabs(std::fabs(n.data[c][2]) - 1.f) < 1e-3f,
          "normal f32 is +-z");
    anari::unmap(d, frame, "channel.normal");
    anari::release(d, frame);
  }

  // UFIXED8_VEC3 albedo + FIXED16_VEC3 normal.
  {
    auto frame = renderWith(
        d, world, camera, renderer, ANARI_UFIXED8_VEC3, ANARI_FIXED16_VEC3);
    auto a = anari::map<uint8_t>(d, frame, "channel.albedo");
    check(a.data && a.pixelType == ANARI_UFIXED8_VEC3, "albedo u8 maps");
    if (a.data) {
      for (int i = 0; i < 3; i++) {
        const int expected = int(std::round(kAlbedo[i] * 255.f));
        check(std::abs(int(a.data[3 * c + i]) - expected) <= 1,
            "albedo u8 value");
      }
    }
    anari::unmap(d, frame, "channel.albedo");

    auto n = anari::map<int16_t>(d, frame, "channel.normal");
    check(n.data && n.pixelType == ANARI_FIXED16_VEC3, "normal s16 maps");
    if (n.data) {
      check(std::abs(int(n.data[3 * c + 0])) <= 1, "normal s16 x ~ 0");
      check(std::abs(int(n.data[3 * c + 1])) <= 1, "normal s16 y ~ 0");
      check(std::abs(std::abs(int(n.data[3 * c + 2])) - 32767) <= 1,
          "normal s16 z ~ +-1");
    }
    anari::unmap(d, frame, "channel.normal");
    anari::release(d, frame);
  }

  // UFIXED8_RGB_SRGB albedo.
  {
    auto frame = renderWith(
        d, world, camera, renderer, ANARI_UFIXED8_RGB_SRGB, ANARI_UNKNOWN);
    auto a = anari::map<uint8_t>(d, frame, "channel.albedo");
    check(a.data && a.pixelType == ANARI_UFIXED8_RGB_SRGB, "albedo srgb maps");
    if (a.data) {
      for (int i = 0; i < 3; i++) {
        const int expected = int(std::round(srgbEncode(kAlbedo[i]) * 255.f));
        check(std::abs(int(a.data[3 * c + i]) - expected) <= 1,
            "albedo srgb value");
      }
    }
    anari::unmap(d, frame, "channel.albedo");

    auto n = anari::map<uint8_t>(d, frame, "channel.normal");
    check(!n.data && n.pixelType == ANARI_UNKNOWN, "normal off doesn't map");
    anari::unmap(d, frame, "channel.normal");
    anari::release(d, frame);
  }

  // Types outside the spec's list are rejected.
  {
    auto frame = renderWith(
        d, world, camera, renderer, ANARI_FLOAT32_VEC4, ANARI_UFIXED8_VEC3);
    auto a = anari::map<uint8_t>(d, frame, "channel.albedo");
    check(!a.data, "albedo f32x4 rejected");
    anari::unmap(d, frame, "channel.albedo");
    auto n = anari::map<uint8_t>(d, frame, "channel.normal");
    check(!n.data, "normal u8 rejected");
    anari::unmap(d, frame, "channel.normal");
    anari::release(d, frame);
  }

  anari::release(d, renderer);
  anari::release(d, camera);
  anari::release(d, world);
  anari::release(d, d);

  if (g_failures) {
    fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  printf("TestFrameChannelTypes passed\n");
  return 0;
}
