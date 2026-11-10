// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "material/shaders/physicallyBasedSampling.h"

#include <cstdio>
#include <vector>

using namespace visrtx;

namespace {
PhysicallyBasedShadingState material()
{
  PhysicallyBasedShadingState s{};
  s.baseColor = vec3(1.f);
  s.viewAdjustedNormal = s.viewAdjustedClearcoatNormal = vec3(0, 0, 1);
  s.roughness = s.clearcoatRoughness = .3f;
  s.eta = 1.f / 1.5f;
  s.occlusion = s.specular = 1.f;
  s.specularColor = s.attenuationColor = vec3(1.f);
  return s;
}

struct SampleCase
{
  const char *name;
  PhysicallyBasedShadingState state;
  vec3 outgoing;
  NextRayQmc qmc;
  vec3 direction;
  int fallbackDraws;
  uint32_t flags = NEXT_RAY_NONE;
  bool dead = false;
};

bool check(const SampleCase &test, const RandState &initial)
{
  auto random = initial;
  auto expectedRandom = initial;
  for (int i = 0; i < test.fallbackDraws; ++i)
    (void)pcg_uniform(&expectedRandom);
  const auto ray =
      samplePhysicallyBased(&test.state, test.outgoing, &random, &test.qmc);
  const bool output = test.dead
      ? ray.pdf == 0.f && glm::length(ray.contributionWeight) == 0.f
      : glm::length(ray.direction - test.direction) < 1e-6f
          && ray.flags == test.flags;
  const bool ok = output && random.state == expectedRandom.state
      && random.inc == expectedRandom.inc;
  if (!ok)
    std::fprintf(stderr, "FAIL: %s\n", test.name);
  return ok;
}

// Only component ordering belongs here. The exhaustive n/cursor/fallback matrix
// is tested separately by SamplerContracts on both host and CUDA.
bool checkMdlComponents(const RandState &initial, int supplied)
{
  NextRayQmc qmc{{.125f, .25f, .5f, .75f}, supplied};
  auto random = initial;
  auto expectedRandom = initial;
  float expected[4];
  for (int i = 0; i < 4; ++i)
    expected[i] = i < supplied ? qmc.u[i] : pcg_uniform(&expectedRandom);
  const auto xi = materialUniform4(random, &qmc);
  const bool ok = xi.x == expected[0] && xi.y == expected[1]
      && xi.z == expected[2] && xi.w == expected[3]
      && random.state == expectedRandom.state
      && random.inc == expectedRandom.inc;
  if (!ok)
    std::fprintf(stderr, "FAIL: MDL component order (%d supplied)\n", supplied);
  return ok;
}
} // namespace

int main()
{
  RandState initial;
  pcg_init(&initial, 42, 7);
  const vec3 outgoing(0, 0, 1);
  // Independent analytic fixtures at normal incidence. Asymmetric u/v makes
  // swapping the radius and azimuth observable without duplicating the BSDF.
  const vec3 diffuse(
      .5f * std::cos(kPi / 4), .5f * std::sin(kPi / 4), std::sqrt(.75f));
  const vec3 h = glm::normalize(vec3(.09f * .5f * std::cos(kPi / 4),
      .09f * .5f * std::sin(kPi / 4),
      std::sqrt(.75f)));
  const vec3 reflected(2.f * h.z * h.x, 2.f * h.z * h.y, 2.f * h.z * h.z - 1.f);
  auto coat = material();
  coat.clearcoat = 1.f;
  auto transmission = coat;
  transmission.transmission = 1.f;
  const float eta = transmission.eta;
  const float cosT = std::sqrt(1.f - eta * eta * (1.f - h.z * h.z));
  const vec3 refracted = glm::normalize(vec3((eta * h.z - cosT) * h.x,
      (eta * h.z - cosT) * h.y,
      -eta + (eta * h.z - cosT) * h.z));

  std::vector<SampleCase> cases = {{"no-coat diffuse",
                                       material(),
                                       outgoing,
                                       {{.9f, .25f, .125f, 0.f}, 3},
                                       diffuse,
                                       0},
      {"selected coat",
          coat,
          outgoing,
          {{.01f, .25f, .125f, 0.f}, 3},
          reflected,
          0},
      {"rejected coat diffuse",
          coat,
          outgoing,
          {{.99f, .9f, .25f, .125f}, 4},
          diffuse,
          0},
      {"rejected coat specular fifth draw",
          coat,
          outgoing,
          {{.99f, 0.f, .25f, .125f}, 4},
          reflected,
          1},
      {"no-coat specular",
          material(),
          outgoing,
          {{0.f, .25f, .125f, .5f}, 4},
          reflected,
          0},
      {"transmission fifth draw",
          transmission,
          outgoing,
          {{.99f, 0.f, .25f, .125f}, 4},
          refracted,
          1,
          NEXT_RAY_CONTINUES_THROUGH_SURFACE}};

  // Set n to the allowed prefix: an eager request past any early exit would
  // advance PCG, so the same checks catch it without observing a private
  // cursor.
  auto zero = material();
  zero.baseColor = vec3(0.f);
  zero.specular = 0.f;
  auto zeroCoat = zero;
  zeroCoat.clearcoat = 1.f;
  auto roughCoat = coat;
  roughCoat.clearcoatRoughness = 1.f;
  auto rough = material();
  rough.roughness = 1.f;
  const SampleCase dead[] = {{"invalid base view",
                                 material(),
                                 -outgoing,
                                 {{}, 0},
                                 {},
                                 0,
                                 NEXT_RAY_NONE,
                                 true},
      {"invalid coat view",
          coat,
          -outgoing,
          {{0.f}, 1},
          {},
          0,
          NEXT_RAY_NONE,
          true},
      {"zero base energy", zero, outgoing, {{}, 0}, {}, 0, NEXT_RAY_NONE, true},
      {"rejected coat zero energy",
          zeroCoat,
          outgoing,
          {{.99f}, 1},
          {},
          0,
          NEXT_RAY_NONE,
          true},
      {"coat below-surface reflection",
          roughCoat,
          outgoing,
          {{0.f, .9f, .125f, .5f}, 3},
          {},
          0,
          NEXT_RAY_NONE,
          true},
      {"base below-surface reflection",
          rough,
          outgoing,
          {{0.f, .9f, .125f, .5f}, 4},
          {},
          0,
          NEXT_RAY_NONE,
          true}};
  cases.insert(cases.end(), std::begin(dead), std::end(dead));
  bool ok = true;
  for (const auto &test : cases)
    ok = check(test, initial) && ok;
  ok = checkMdlComponents(initial, 4) && ok;
  ok = checkMdlComponents(initial, 2) && ok;
  return ok ? 0 : 1;
}
