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

#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp.hpp>
#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <array>
#include <cstdio>
#include <string>

using vec2 = std::array<float, 2>;
using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;
using uvec3 = std::array<unsigned int, 3>;
using uvec2 = std::array<unsigned int, 2>;

static void statusFunc(const void *, ANARIDevice, ANARIObject, ANARIDataType,
    ANARIStatusSeverity s, ANARIStatusCode, const char *m)
{ if (s <= ANARI_SEVERITY_WARNING) std::fprintf(stderr, "[anari] %s\n", m); }

// A 1x1 green image2D sampler (linear float, so no colorspace decode involved).
static anari::Sampler makeGreenSampler(anari::Device d)
{
  auto img = anari::newArray2D(d, ANARI_FLOAT32_VEC3, 1, 1);
  *anari::map<vec3>(d, img) = vec3{0.f, 1.f, 0.f};
  anari::unmap(d, img);
  auto s = anari::newObject<anari::Sampler>(d, "image2D");
  anari::setAndReleaseParameter(d, s, "image", img);
  anari::setParameter(d, s, "inAttribute", "attribute0");
  anari::setParameter(d, s, "wrapMode1", "clampToEdge");
  anari::setParameter(d, s, "wrapMode2", "clampToEdge");
  anari::commitParameters(d, s);
  return s;
}

static bool centerIsGreen(anari::Device d, anari::Frame frame, uvec2 size)
{
  anari::render(d, frame); anari::wait(d, frame);
  auto fb = anari::map<float>(d, frame, "channel.color");
  const float *px = fb.data + 4 * (size[0] * (size[1] / 2) + size[0] / 2);
  bool g = px[1] > 0.3f && px[1] > px[0] + 0.1f && px[1] > px[2] + 0.1f;
  if (!g) std::printf("not green: (%.2f,%.2f,%.2f)\n", px[0], px[1], px[2]);
  anari::unmap(d, frame, "channel.color");
  return g;
}
static bool centerIsRed(anari::Device d, anari::Frame frame, uvec2 size)
{
  anari::render(d, frame); anari::wait(d, frame);
  auto fb = anari::map<float>(d, frame, "channel.color");
  const float *px = fb.data + 4 * (size[0] * (size[1] / 2) + size[0] / 2);
  bool r = px[0] > 0.3f && px[0] > px[1] + 0.1f && px[0] > px[2] + 0.1f;
  if (!r) std::printf("not red: (%.2f,%.2f,%.2f)\n", px[0], px[1], px[2]);
  anari::unmap(d, frame, "channel.color");
  return r;
}

int main()
{
  auto d = anari::Device(makeVisRTXDevice(statusFunc));

  auto mat = anari::newObject<anari::Material>(d, "materialx");
  anari::setParameter(d, mat, "source", std::string("visrtx::standard_surface"));
  anari::setParameter(d, mat, "materialName", std::string("StandardSurface"));
  auto green = makeGreenSampler(d);
  anari::setParameter(d, mat, "base_color", green); // sampler on a clean input
  anari::commitParameters(d, mat);

  std::array<vec3, 4> pos = {vec3{-1,-1,0}, {1,-1,0}, {1,1,0}, {-1,1,0}};
  std::array<vec2, 4> uv = {vec2{0,0}, {1,0}, {1,1}, {0,1}};
  std::array<uvec3, 2> idx = {uvec3{0,1,2}, uvec3{0,2,3}};
  auto geom = anari::newObject<anari::Geometry>(d, "triangle");
  anari::setParameterArray1D(d, geom, "vertex.position", pos.data(), pos.size());
  anari::setParameterArray1D(d, geom, "vertex.attribute0", uv.data(), uv.size());
  anari::setParameterArray1D(d, geom, "primitive.index", idx.data(), idx.size());
  anari::commitParameters(d, geom);

  auto surf = anari::newObject<anari::Surface>(d);
  anari::setAndReleaseParameter(d, surf, "geometry", geom);
  anari::setParameter(d, surf, "material", mat);
  anari::commitParameters(d, surf);
  auto world = anari::newObject<anari::World>(d);
  anari::setParameterArray1D(d, world, "surface", &surf, 1);
  anari::commitParameters(d, world);
  anari::release(d, surf);

  auto cam = anari::newObject<anari::Camera>(d, "perspective");
  anari::setParameter(d, cam, "position", vec3{0, 0, 3});
  anari::setParameter(d, cam, "direction", vec3{0, 0, -1});
  anari::setParameter(d, cam, "up", vec3{0, 1, 0});
  anari::commitParameters(d, cam);
  auto rnd = anari::newObject<anari::Renderer>(d, "default");
  anari::setParameter(d, rnd, "ambientRadiance", 1.f);
  anari::commitParameters(d, rnd);

  auto frame = anari::newObject<anari::Frame>(d);
  uvec2 size = {64, 64};
  anari::setParameter(d, frame, "size", size);
  anari::setParameter(d, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setParameter(d, frame, "world", world);
  anari::setParameter(d, frame, "camera", cam);
  anari::setParameter(d, frame, "renderer", rnd);
  anari::commitParameters(d, frame);

  const bool sampled = centerIsGreen(d, frame, size);

  // Re-commit without re-supplying base_color (as a host re-staging the
  // material would): the textured topology must persist, not silently revert.
  anari::commitParameters(d, mat);
  const bool sampledAgain = centerIsGreen(d, frame, size);

  // Topology round-trip: switch base_color back to a constant red.
  anari::setParameter(d, mat, "base_color", vec3{1.f, 0.f, 0.f});
  anari::commitParameters(d, mat);
  const bool constAfter = centerIsRed(d, frame, size);

  anari::release(d, green);
  anari::release(d, mat); anari::release(d, world);
  anari::release(d, cam); anari::release(d, rnd); anari::release(d, frame);
  anari::release(d, d);

  if (!sampled || !sampledAgain || !constAfter) {
    std::printf("FAIL: sampled=%d sampledAgain=%d constAfter=%d\n",
        sampled, sampledAgain, constAfter);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
