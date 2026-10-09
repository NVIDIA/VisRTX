// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "RendererFogSupport.h"

#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <anari/ext/visrtx/visrtx_extensions.h>

#include <cstdio>
#include <set>
#include <string>

namespace {

using namespace visrtx::fogtest;

constexpr const char *FOG = "ANARI_VISRTX_RENDERER_FOG";

std::set<std::string> strings(const void *value)
{
  std::set<std::string> result;
  auto list = static_cast<const char *const *>(value);
  if (list)
    for (; *list; ++list)
      result.insert(*list);
  return result;
}

bool check(bool ok, const char *subtype, const char *label)
{
  if (!ok)
    fprintf(stderr, "%s: %s\n", subtype, label);
  return ok;
}

int testDiscovery(anari::Device d)
{
  int failures = 0;
  auto device = strings(anariGetObjectInfo(
      d, ANARI_DEVICE, nullptr, "extension", ANARI_STRING_LIST));
  failures += !check(device.count(FOG), "device", "missing fog capability");
  const char **property = nullptr;
  failures += !anariGetProperty(d,
      d,
      "extension",
      ANARI_STRING_LIST,
      &property,
      sizeof(property),
      ANARI_WAIT);
  failures +=
      !check(strings(property) == device, "device", "property/query disagree");
  failures += !check(
      visrtx::getObjectExtensions(d, ANARI_DEVICE, nullptr).VISRTX_RENDERER_FOG,
      "device",
      "utility misses fog");
  failures += !check(visrtx::getInstanceExtensions(d, d).VISRTX_RENDERER_FOG,
      "device",
      "property utility misses fog");
  failures += !check(!visrtx::getObjectExtensions(d, ANARI_MATERIAL, "matte")
                         .VISRTX_RENDERER_FOG,
      "material",
      "utility leaks renderer fog");
  const auto subtypes = strings(anariGetObjectSubtypes(d, ANARI_RENDERER));
  auto checkedSubtypes = subtypes;
  for (const char *s : {"fast",
           "interactive",
           "default",
           "quality",
           "debug",
           "test",
           "debug_Ng",
           "unknown"})
    checkedSubtypes.insert(s);
  for (const auto &subtype : checkedSubtypes) {
    const char *s = subtype.c_str();
    const bool supported = std::string(s) == "fast"
        || std::string(s) == "interactive" || std::string(s) == "default"
        || std::string(s) == "quality";
    if (supported)
      failures += !check(subtypes.count(s), s, "promised subtype missing");
    const auto extensions = strings(anariGetObjectInfo(
        d, ANARI_RENDERER, s, "extension", ANARI_STRING_LIST));
    failures += !check(bool(extensions.count(FOG)) == supported,
        s,
        "incorrect fog capability");
    failures += !check(bool(visrtx::getObjectExtensions(d, ANARI_RENDERER, s)
                               .VISRTX_RENDERER_FOG)
            == supported,
        s,
        "utility subtype support");
    for (const auto &e : extensions)
      if (e.find("FOG") != std::string::npos)
        failures +=
            !check(e == FOG && supported, s, "unintended fog capability");
    const auto *params = static_cast<const ANARIParameter *>(anariGetObjectInfo(
        d, ANARI_RENDERER, s, "parameter", ANARI_PARAMETER_LIST));
    unsigned count = 0;
    if (params)
      for (auto p = params; p->name; ++p)
        if (std::string(p->name).find("fog") == 0)
          ++count;
    failures += !check(count == (supported ? 7u : 0u),
        s,
        "expected exactly seven fog parameters only on supported subtypes");
    if (!supported)
      continue;
    for (const char *name : {"fogMode",
             "fogDistanceMetric",
             "fogColorSource",
             "fogColor",
             "fogStart",
             "fogEnd",
             "fogDensity"}) {
      const std::string n(name);
      const bool selector =
          n == "fogMode" || n == "fogDistanceMetric" || n == "fogColorSource";
      const auto type = selector ? ANARI_STRING
          : n == "fogColor"      ? ANARI_FLOAT32_VEC3
                                 : ANARI_FLOAT32;
      unsigned matches = 0;
      if (params)
        for (auto p = params; p->name; ++p)
          matches += n == p->name && type == p->type;
      failures += !check(matches == 1, s, name);
      auto info = [&](const char *key, ANARIDataType t) {
        return anariGetParameterInfo(d, ANARI_RENDERER, s, name, type, key, t);
      };
      const auto *source =
          static_cast<const char *>(info("sourceExtension", ANARI_STRING));
      failures += !check(source && std::string(source) == "VISRTX_RENDERER_FOG",
          s,
          "schema identity");
      const auto *description =
          static_cast<const char *>(info("description", ANARI_STRING));
      failures += !check(description && *description, s, "missing description");
      const auto *required =
          static_cast<const int *>(info("required", ANARI_BOOL));
      failures += !check(required && !*required, s, "fog must be optional");
      const void *value = info("default", type);
      bool validDefault = value != nullptr;
      if (value && selector) {
        const std::string expected = n == "fogMode" ? "none"
            : n == "fogDistanceMetric"              ? "viewDepth"
                                                    : "constant";
        validDefault =
            std::string(static_cast<const char *>(value)) == expected;
        const std::set<std::string> allowed = n == "fogMode"
            ? std::set<std::string>{"none", "linear", "exp", "exp2"}
            : n == "fogDistanceMetric"
            ? std::set<std::string>{"viewDepth", "rayDistance"}
            : std::set<std::string>{"constant", "background"};
        failures += !check(strings(info("value", ANARI_STRING_LIST)) == allowed,
            s,
            "accepted selectors");
      } else if (value) {
        const auto *numbers = static_cast<const float *>(value);
        const auto *minimum = static_cast<const float *>(info("minimum", type));
        bool validMinimum = minimum != nullptr;
        for (int c = 0; c < (n == "fogColor" ? 3 : 1); ++c) {
          validDefault &= numbers[c] == (n == "fogStart" ? 0.f : 1.f);
          validMinimum &= minimum && minimum[c] == 0.f;
        }
        failures += !check(validMinimum, s, "nonnegative numeric domain");
        failures +=
            !check(!info("maximum", type), s, "no artificial numeric/HDR cap");
        const std::string desc = description ? description : "";
        failures += !check(desc.find(n == "fogColor" ? "linear RGB" : "world")
                != std::string::npos,
            s,
            "units in metadata");
        failures += !check(desc.find(n == "fogColor"   ? "constant"
                                   : n == "fogDensity" ? "exp"
                                                       : "linear")
                != std::string::npos,
            s,
            "applicability in metadata");
      }
      failures += !check(validDefault, s, "incorrect default");
    }
    printf("discovery %s: seven core parameters checked\n", s);
  }
  for (const auto &e : device)
    if (e.find("FOG") != std::string::npos)
      failures += !check(e == FOG,
          "device",
          "unintended KHR/height/volumetric fog capability");
  return failures;
}

} // namespace

int main()
{
  auto d = makeVisRTXDevice();
  const ObjectOwner deviceOwner(d, d);
  if (!d)
    return 1;
  anari::commitParameters(d, d);
  const int failures = testDiscovery(d);
  printf("Fog discovery: %d failures\n", failures);
  return failures ? 1 : 0;
}
