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

// Sampler channel-fill: CUDA texture fetches zero-fill channels absent from the
// array's format, so a 1-channel FLOAT32 color texture reads back (r,0,0,0).
// ANARI requires missing color -> 0 and missing alpha -> 1 (as the attribute
// path already does). A matte surface folds its color.alpha into opacity
// (opacity = color.w * opacity), so under alphaMode=blend the wrong alpha=0
// makes the surface fully transparent — it vanishes. Pins: a blend matte with a
// 1-channel color texture renders identically to the same surface made opaque
// (both fully visible), and NOT as a transparent (dark) frame.

// anari_cpp
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
static constexpr int PIXEL_SAMPLES = 16;
static constexpr float TEX_VALUE = 0.6f; // constant texel, all channels

enum class ColorChannels
{
  ONE, // 1-channel FLOAT32 texture: alpha absent -> must default to 1
  FOUR, // 4-channel texture with alpha=1: control, always visible
};

// A view-filling quad in the z=0 plane, colored by a constant image2D sampler.
// The image is constant, so no texcoords are needed — every fetch hits the same
// texel; the only variable under test is how the missing alpha channel is
// filled.
static anari::Surface makeTexturedQuad(
    ANARIDevice device, ColorChannels channels, const char *alphaMode)
{
  const std::array<vec3, 4> pos = {vec3{-2.f, -2.f, 0.f},
      vec3{2.f, -2.f, 0.f},
      vec3{2.f, 2.f, 0.f},
      vec3{-2.f, 2.f, 0.f}};
  const std::array<std::array<unsigned, 3>, 2> idx = {
      std::array<unsigned, 3>{0, 1, 2}, std::array<unsigned, 3>{0, 2, 3}};

  auto geom = anari::newObject<anari::Geometry>(device, "triangle");
  anari::setParameterArray1D(device, geom, "vertex.position", pos.data(), 4);
  anari::setParameterArray1D(device, geom, "primitive.index", idx.data(), 2);
  anari::commitParameters(device, geom);

  auto sampler = anari::newObject<anari::Sampler>(device, "image2D");
  anari::setParameter(device, sampler, "inAttribute", "attribute0");
  anari::setParameter(device, sampler, "filter", "nearest");
  if (channels == ColorChannels::ONE) {
    const std::array<float, 4> img = {
        TEX_VALUE, TEX_VALUE, TEX_VALUE, TEX_VALUE};
    anari::setParameterArray2D(device, sampler, "image", img.data(), 2, 2);
  } else {
    const std::array<vec4, 4> img = {vec4{TEX_VALUE, TEX_VALUE, TEX_VALUE, 1.f},
        vec4{TEX_VALUE, TEX_VALUE, TEX_VALUE, 1.f},
        vec4{TEX_VALUE, TEX_VALUE, TEX_VALUE, 1.f},
        vec4{TEX_VALUE, TEX_VALUE, TEX_VALUE, 1.f}};
    anari::setParameterArray2D(device, sampler, "image", img.data(), 2, 2);
  }
  anari::commitParameters(device, sampler);

  auto mat = anari::newObject<anari::Material>(device, "matte");
  anari::setAndReleaseParameter(device, mat, "color", sampler);
  anari::setParameter(device, mat, "opacity", 1.f);
  anari::setParameter(device, mat, "alphaMode", alphaMode);
  anari::commitParameters(device, mat);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);
  return surface;
}

// Mean luminance over the whole (quad-filling) frame. A visible lit quad reads
// bright; a transparent quad reveals the black background and reads ~0.
static double frameMean(ANARIDevice device,
    ColorChannels channels,
    const char *alphaMode,
    const char *rendererType)
{
  auto surface = makeTexturedQuad(device, channels, alphaMode);
  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "surface", &surface, 1);
  anari::release(device, surface);

  auto light = anari::newObject<anari::Light>(device, "directional");
  anari::setParameter(device, light, "direction", vec3{0.f, 0.f, 1.f});
  anari::setParameter(device, light, "irradiance", 1.f);
  anari::commitParameters(device, light);
  anari::setParameterArray1D(device, world, "light", &light, 1);
  anari::release(device, light);
  anari::commitParameters(device, world);

  auto camera = anari::newObject<anari::Camera>(device, "perspective");
  anari::setParameter(device, camera, "position", vec3{0.f, 0.f, -3.f});
  anari::setParameter(device, camera, "direction", vec3{0.f, 0.f, 1.f});
  anari::setParameter(device, camera, "up", vec3{0.f, 1.f, 0.f});
  anari::setParameter(
      device, camera, "aspect", IMAGE_SIZE[0] / float(IMAGE_SIZE[1]));
  anari::commitParameters(device, camera);

  auto renderer = anari::newObject<anari::Renderer>(device, rendererType);
  anari::setParameter(device, renderer, "background", vec4{0.f, 0.f, 0.f, 1.f});
  anari::setParameter(device, renderer, "ambientRadiance", 0.f);
  anari::setParameter(device, renderer, "pixelSamples", PIXEL_SAMPLES);
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
  double sum = 0.0;
  uint64_t n = 0;
  for (uint32_t i = 0; i < IMAGE_SIZE[0] * IMAGE_SIZE[1]; ++i) {
    const vec4 &p = fb.data[i];
    sum += 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
    ++n;
  }
  anari::unmap(device, frame, "channel.color");
  anari::release(device, frame);
  return n ? sum / double(n) : 0.0;
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);

  bool ok = true;
  // Within a fixed channel count, a blend surface must render identically to
  // the opaque one: opacity = color.w * opacity, and correct alpha-fill makes
  // color.w == 1 regardless of how many channels the texture has. The bug drops
  // color.w to 0 for the 1-channel case only, so blend1 collapses to black
  // while opaque1 stays lit. The 4-channel case (real alpha=1) is the control
  // that isolates the missing-channel path.
  struct Case
  {
    const char *label;
    ColorChannels channels;
  };
  const std::array<Case, 2> cases = {
      Case{"1ch", ColorChannels::ONE}, Case{"4ch", ColorChannels::FOUR}};

  for (const char *renderer : {"quality", "interactive"}) {
    for (const auto &c : cases) {
      const double opaque = frameMean(device, c.channels, "opaque", renderer);
      const double blend = frameMean(device, c.channels, "blend", renderer);
      const double relErr =
          opaque > 0.0 ? std::abs(blend - opaque) / opaque : (blend > 0.0 ? 1.0 : 0.0);
      printf("%s/%s: opaque=%f blend=%f relErr=%f\n",
          renderer,
          c.label,
          opaque,
          blend,
          relErr);

      if (!std::isfinite(opaque) || opaque <= 0.01) {
        fprintf(stderr,
            "FAIL[%s/%s]: opaque control is dark (%f)\n",
            renderer,
            c.label,
            opaque);
        ok = false;
        continue;
      }
      if (blend <= 0.01 || relErr > 0.02) {
        fprintf(stderr,
            "FAIL[%s/%s]: blend disagrees with opaque (blend=%f opaque=%f "
            "relErr=%f) — sampler alpha defaulted to 0\n",
            renderer,
            c.label,
            blend,
            opaque,
            relErr);
        ok = false;
      }
    }
  }

  anari::release(device, device);

  if (!ok)
    return 1;
  printf("sampler alpha-fill semantics passed\n");
  return 0;
}
