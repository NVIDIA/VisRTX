// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: BSD-3-Clause

// Application-mediated INeuray handoff: the app reads the SDK instance from
// one device's `ineuray` property and passes it to another device's `ineuray`
// parameter. This is the supported path for sharing MDL across devices from
// different vendors, which cannot see each other's internals.

#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <anari/anari_cpp.hpp>
#include <array>
#include <cstdio>
#include <string>
using Vec3 = std::array<float, 3>;
using Vec4 = std::array<float, 4>;
static int errs = 0;
static void st(const void *,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity s,
    ANARIStatusCode,
    const char *m)
{
  if (s == ANARI_SEVERITY_ERROR || s == ANARI_SEVERITY_FATAL_ERROR) {
    ++errs;
    std::fprintf(stderr, "ERR: %s\n", m);
  }
}
static std::string src()
{
  return "mdl 1.6; import ::df::*; export material emitter()=material(surface: material_surface("
         "emission: material_emission(emission: df::diffuse_edf(), intensity: color(4.0))));";
}
static anari::Frame scene(anari::Device d)
{
  const std::array<Vec3, 4> v = {
      Vec3{-1, -1, 0}, Vec3{1, -1, 0}, Vec3{1, 1, 0}, Vec3{-1, 1, 0}};
  const std::array<std::array<unsigned, 3>, 2> i = {
      std::array<unsigned, 3>{0, 1, 2}, std::array<unsigned, 3>{0, 2, 3}};
  auto g = anari::newObject<anari::Geometry>(d, "triangle");
  anari::setParameterArray1D(d, g, "vertex.position", v.data(), v.size());
  anari::setParameterArray1D(d, g, "primitive.index", i.data(), i.size());
  anari::commitParameters(d, g);
  auto m = anari::newObject<anari::Material>(d, "mdl");
  anari::setParameter(d, m, "sourceType", "code");
  anari::setParameter(d, m, "source", src());
  anari::setParameter(d, m, "materialName", "emitter");
  anari::commitParameters(d, m);
  auto s = anari::newObject<anari::Surface>(d);
  anari::setAndReleaseParameter(d, s, "geometry", g);
  anari::setAndReleaseParameter(d, s, "material", m);
  anari::commitParameters(d, s);
  auto w = anari::newObject<anari::World>(d);
  anari::setParameterArray1D(d, w, "surface", &s, 1);
  anari::commitParameters(d, w);
  anari::release(d, s);
  auto c = anari::newObject<anari::Camera>(d, "orthographic");
  anari::setParameter(d, c, "position", Vec3{0, 0, 3});
  anari::setParameter(d, c, "direction", Vec3{0, 0, -1});
  anari::setParameter(d, c, "up", Vec3{0, 1, 0});
  anari::setParameter(d, c, "height", 2.f);
  anari::setParameter(d, c, "aspect", 1.f);
  anari::commitParameters(d, c);
  auto r = anari::newObject<anari::Renderer>(d, "quality");
  anari::setParameter(d, r, "pixelSamples", 1);
  anari::setParameter(d, r, "background", Vec4{0, 0, 0, 1});
  anari::setParameter(d, r, "ambientRadiance", 0.f);
  anari::commitParameters(d, r);
  auto f = anari::newObject<anari::Frame>(d);
  anari::setParameter(d, f, "size", std::array<unsigned, 2>{32, 32});
  anari::setParameter(d, f, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setAndReleaseParameter(d, f, "camera", c);
  anari::setAndReleaseParameter(d, f, "renderer", r);
  anari::setAndReleaseParameter(d, f, "world", w);
  anari::commitParameters(d, f);
  return f;
}
int main()
{
  auto A = makeVisRTXDevice(st);
  anari::setParameter(A, A, "forceInit", true);
  anari::commitParameters(A, A);
  // Can the app READ the instance back out of device A?
  void *p = nullptr;
  int ok = anariGetProperty(
      A, A, "ineuray", ANARI_VOID_POINTER, &p, sizeof(p), ANARI_WAIT);
  std::printf("getProperty(\"ineuray\") -> returned=%d ptr=%p\n", ok, p);
  if (!ok || !p) {
    std::fprintf(
        stderr, "FAIL: `ineuray` property unreadable; handoff impossible\n");
    return 1;
  }
  {
    auto B = makeVisRTXDevice(st);
    anari::setParameter(B, B, "ineuray", p);
    anari::setParameter(B, B, "forceInit", true);
    anari::commitParameters(B, B);
    auto f = scene(B);
    anari::render(B, f);
    anari::wait(B, f);
    auto px = anari::map<Vec4>(B, f, "channel.color");
    const float r = px.data[16 * 32 + 16][0];
    std::printf("handoff render pixel r=%.3f errors=%d\n", r, errs);
    if (r <= 0.1f) {
      std::fprintf(stderr, "FAIL: adopted instance did not render MDL\n");
      return 1;
    }
    anari::unmap(B, f, "channel.color");
    anari::release(B, f);
    anari::release(B, B);
  }
  anari::release(A, A);
  if (errs) {
    std::fprintf(stderr, "FAIL: %d error(s) during handoff\n", errs);
    return 1;
  }
  std::puts("INeuray handoff between devices succeeded");
  return 0;
}
