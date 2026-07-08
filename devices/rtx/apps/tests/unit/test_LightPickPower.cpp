/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Host unit tests for lightPickPower — the Pick Power estimate that drives the
// power-proportional Light Pick. These lock the *relative* ordering the pick
// depends on (brighter/larger/closer-to-both-sided lights outweigh dimmer ones,
// infinite lights grow with the scene), which the API additivity test cannot
// see because any unbiased pick converges to the same image. Absolute constants
// are deliberately not asserted: Pick Power only needs to be proportional.

#include "gpu/lightPickPower.h"

#include <glm/gtc/matrix_transform.hpp>

#include <cstdio>

using namespace visrtx;

static int g_failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);             \
      ++g_failures;                                                            \
    }                                                                          \
  } while (0)

static LightGPUData pointLight(float intensity, const vec3 &color = vec3(1.f))
{
  LightGPUData ld{};
  ld.type = LightType::POINT;
  ld.color = color;
  ld.point.intensity = intensity;
  return ld;
}

static LightGPUData rectLight(float intensity, float oneOverArea, bool front, bool back)
{
  LightGPUData ld{};
  ld.type = LightType::RECT;
  ld.color = vec3(1.f);
  ld.rect.intensity = intensity;
  ld.rect.oneOverArea = oneOverArea;
  ld.rect.side.front = front ? 1 : 0;
  ld.rect.side.back = back ? 1 : 0;
  return ld;
}

static LightGPUData directionalLight(float irradiance)
{
  LightGPUData ld{};
  ld.type = LightType::DIRECTIONAL;
  ld.color = vec3(1.f);
  ld.distant.irradiance = irradiance;
  return ld;
}

int main()
{
  const mat4 identity(1.f);
  const float radius = 3.f;

  // Monotonic in intensity, and proportional (2x intensity => 2x power).
  {
    const float p1 = lightPickPower(pointLight(1.f), identity, radius);
    const float p2 = lightPickPower(pointLight(2.f), identity, radius);
    CHECK(p1 > 0.f);
    CHECK(p2 > 2.f * p1 - 1e-3f && p2 < 2.f * p1 + 1e-3f);
  }

  // A dark light contributes no pick weight (positivity boundary).
  CHECK(lightPickPower(pointLight(0.f), identity, radius) == 0.f);

  // Luminance-weighted: a green light outweighs an equal-intensity blue one.
  {
    const float green =
        lightPickPower(pointLight(1.f, vec3(0.f, 1.f, 0.f)), identity, radius);
    const float blue =
        lightPickPower(pointLight(1.f, vec3(0.f, 0.f, 1.f)), identity, radius);
    CHECK(green > blue);
  }

  // A rectangle emitting from both sides has twice the power of one side.
  {
    const float oneSide =
        lightPickPower(rectLight(1.f, 0.25f, true, false), identity, radius);
    const float bothSides =
        lightPickPower(rectLight(1.f, 0.25f, true, true), identity, radius);
    CHECK(bothSides > 2.f * oneSide - 1e-3f
        && bothSides < 2.f * oneSide + 1e-3f);
    // Larger area (smaller oneOverArea) => more power.
    const float bigger =
        lightPickPower(rectLight(1.f, 0.1f, true, false), identity, radius);
    CHECK(bigger > oneSide);
  }

  // Non-uniform instance scale enlarges an area light's Pick Power (area grows).
  {
    const auto rect = rectLight(1.f, 0.25f, true, false);
    const float base = lightPickPower(rect, identity, radius);
    const mat4 scaled = glm::scale(identity, vec3(2.f, 2.f, 2.f));
    CHECK(lightPickPower(rect, scaled, radius) > base);
  }

  // Infinite lights grow with the scene cross-section (~radius^2).
  {
    const auto sun = directionalLight(1.f);
    const float small = lightPickPower(sun, identity, 1.f);
    const float big = lightPickPower(sun, identity, 2.f);
    CHECK(big > 3.9f * small && big < 4.1f * small); // 2^2 = 4x
  }

  // Sphere (a point light with radius > 0): power grows with intensity and area.
  {
    LightGPUData dim{}, bright{}, bigger{};
    dim.type = bright.type = bigger.type = LightType::SPHERE;
    dim.color = bright.color = bigger.color = vec3(1.f);
    dim.sphere.intensity = 1.f;
    dim.sphere.radius = 1.f;
    bright.sphere.intensity = 2.f;
    bright.sphere.radius = 1.f;
    bigger.sphere.intensity = 1.f;
    bigger.sphere.radius = 2.f;
    CHECK(lightPickPower(dim, identity, radius) > 0.f);
    CHECK(lightPickPower(bright, identity, radius)
        > lightPickPower(dim, identity, radius));
    CHECK(lightPickPower(bigger, identity, radius)
        > lightPickPower(dim, identity, radius));
  }

  // Spot: power grows with intensity and with a wider cone (smaller cosOuter).
  {
    LightGPUData narrow{}, wide{}, bright{};
    narrow.type = wide.type = bright.type = LightType::SPOT;
    narrow.color = wide.color = bright.color = vec3(1.f);
    narrow.spot.intensity = 1.f;
    narrow.spot.cosOuterAngle = 0.9f;
    wide.spot.intensity = 1.f;
    wide.spot.cosOuterAngle = 0.1f; // wider cone
    bright.spot.intensity = 3.f;
    bright.spot.cosOuterAngle = 0.9f;
    CHECK(lightPickPower(narrow, identity, radius) > 0.f);
    CHECK(lightPickPower(wide, identity, radius)
        > lightPickPower(narrow, identity, radius));
    CHECK(lightPickPower(bright, identity, radius)
        > lightPickPower(narrow, identity, radius));
  }

  // Ring: power grows with intensity and with area (smaller oneOverArea).
  {
    LightGPUData small{}, big{}, bright{};
    small.type = big.type = bright.type = LightType::RING;
    small.color = big.color = bright.color = vec3(1.f);
    small.ring.intensity = 1.f;
    small.ring.oneOverArea = 1.f;
    big.ring.intensity = 1.f;
    big.ring.oneOverArea = 0.25f; // 4x area
    bright.ring.intensity = 2.f;
    bright.ring.oneOverArea = 1.f;
    CHECK(lightPickPower(small, identity, radius) > 0.f);
    CHECK(lightPickPower(big, identity, radius)
        > lightPickPower(small, identity, radius));
    CHECK(lightPickPower(bright, identity, radius)
        > lightPickPower(small, identity, radius));
  }

  // HDRI: an infinite light — power grows with scale and with radius^2.
  {
    LightGPUData env{};
    env.type = LightType::HDRI;
    env.color = vec3(1.f);
    env.hdri.scale = 1.f;
    LightGPUData brighter = env;
    brighter.hdri.scale = 2.f;
    CHECK(lightPickPower(env, identity, radius) > 0.f);
    CHECK(lightPickPower(brighter, identity, radius)
        > lightPickPower(env, identity, radius));
    CHECK(lightPickPower(env, identity, 2.f) > lightPickPower(env, identity, 1.f));
  }

  if (g_failures) {
    std::printf("%d lightPickPower checks failed\n", g_failures);
    return 1;
  }
  std::printf("lightPickPower unit tests passed\n");
  return 0;
}
