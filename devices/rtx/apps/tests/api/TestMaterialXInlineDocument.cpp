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

using vec3 = std::array<float, 3>;
using uvec3 = std::array<unsigned int, 3>;
using uvec2 = std::array<unsigned int, 2>;

static bool g_sawUnknownSourceType = false;
static void statusFunc(const void *, ANARIDevice, ANARIObject, ANARIDataType,
    ANARIStatusSeverity s, ANARIStatusCode, const char *m)
{
  if (std::string(m).find("unknown sourceType") != std::string::npos)
    g_sawUnknownSourceType = true;
  if (s <= ANARI_SEVERITY_WARNING) std::fprintf(stderr, "[anari] %s\n", m);
}

int main()
{
  auto d = anari::Device(makeVisRTXDevice(statusFunc));

  // A complete standard-surface material as inline .mtlx text (no file).
  const std::string xml =
      "<?xml version=\"1.0\"?>\n"
      "<materialx version=\"1.39\">\n"
      "  <standard_surface name=\"srf\" type=\"surfaceshader\">\n"
      "    <input name=\"base_color\" type=\"color3\" value=\"0.0, 1.0, 0.0\"/>\n"
      "  </standard_surface>\n"
      "  <surfacematerial name=\"M\" type=\"material\">\n"
      "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"srf\"/>\n"
      "  </surfacematerial>\n"
      "</materialx>\n";

  auto mat = anari::newObject<anari::Material>(d, "materialx");
  anari::setParameter(d, mat, "sourceType", std::string("documentInline"));
  anari::setParameter(d, mat, "source", xml);
  anari::commitParameters(d, mat);

  std::array<vec3, 4> pos = {vec3{-1,-1,0}, {1,-1,0}, {1,1,0}, {-1,1,0}};
  std::array<uvec3, 2> idx = {uvec3{0,1,2}, uvec3{0,2,3}};
  auto geom = anari::newObject<anari::Geometry>(d, "triangle");
  anari::setParameterArray1D(d, geom, "vertex.position", pos.data(), pos.size());
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
  anari::setAndReleaseParameter(d, frame, "world", world);
  anari::setAndReleaseParameter(d, frame, "camera", cam);
  anari::setAndReleaseParameter(d, frame, "renderer", rnd);
  anari::commitParameters(d, frame);

  auto centerChannelExceeds = [&](int channel) {
    anari::render(d, frame); anari::wait(d, frame);
    auto fb = anari::map<float>(d, frame, "channel.color");
    const float *px = fb.data + 4 * (size[0] * (size[1] / 2) + size[0] / 2);
    const int a = (channel + 1) % 3, b = (channel + 2) % 3;
    bool ok = px[channel] > 0.3f && px[channel] > px[a] + 0.1f
        && px[channel] > px[b] + 0.1f;
    if (!ok)
      std::printf("center wrong: (%.2f,%.2f,%.2f) expected channel %d dominant\n",
          px[0], px[1], px[2], channel);
    anari::unmap(d, frame, "channel.color");
    return ok;
  };
  const bool green = centerChannelExceeds(1);

  // Edit the inline document (re-set `source` only, NOT `sourceType`, as a host
  // editing the text would): the documentInline scheme must persist across our
  // sourceType="code" handoff, so the new red document renders red.
  const std::string xmlRed =
      "<?xml version=\"1.0\"?>\n"
      "<materialx version=\"1.39\">\n"
      "  <standard_surface name=\"srf\" type=\"surfaceshader\">\n"
      "    <input name=\"base_color\" type=\"color3\" value=\"1.0, 0.0, 0.0\"/>\n"
      "  </standard_surface>\n"
      "  <surfacematerial name=\"M\" type=\"material\">\n"
      "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"srf\"/>\n"
      "  </surfacematerial>\n"
      "</materialx>\n";
  anari::setParameter(d, mat, "source", xmlRed);
  anari::commitParameters(d, mat);
  const bool redAfterEdit = centerChannelExceeds(0);

  anari::release(d, mat);
  anari::release(d, frame);
  anari::release(d, d);

  if (g_sawUnknownSourceType) {
    std::printf("FAIL: spurious 'unknown sourceType' warning\n");
    return 1;
  }
  if (!green || !redAfterEdit) {
    std::printf("FAIL: green=%d redAfterEdit=%d\n", green, redAfterEdit);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
