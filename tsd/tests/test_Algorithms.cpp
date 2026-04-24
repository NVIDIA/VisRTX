// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd_algorithms
#include "tsd/algorithms/cpu/autoExposure.hpp"
#include "tsd/algorithms/cpu/clearBuffers.hpp"
#include "tsd/algorithms/cpu/convertColorBuffer.hpp"
#include "tsd/algorithms/cpu/depthCompositeFrame.hpp"
#include "tsd/algorithms/cpu/outline.hpp"
#include "tsd/algorithms/cpu/outputTransform.hpp"
#include "tsd/algorithms/cpu/toneMap.hpp"
#include "tsd/algorithms/cpu/visualizeAOV.hpp"
// tsd_core
#include "tsd/core/TSDMath.hpp"
// helium
#include <helium/helium_math.h>
// std
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace cpu = tsd::algorithms::cpu;
namespace math = tsd::math;

namespace {

uint32_t packColor(const math::float4 &c)
{
  return helium::cvt_color_to_uint32(c);
}

uint32_t packSrgbColor(const math::float4 &c)
{
  return helium::cvt_color_to_uint32_srgb(c);
}

math::float3 linearToGamma(math::float3 c, float invGamma)
{
  c.x = std::pow(std::clamp(c.x, 0.f, 1.f), invGamma);
  c.y = std::pow(std::clamp(c.y, 0.f, 1.f), invGamma);
  c.z = std::pow(std::clamp(c.z, 0.f, 1.f), invGamma);
  return c;
}

uint32_t shadePixel(uint32_t c)
{
  auto c_in = helium::cvt_color_to_float4(c);
  auto c_out = math::lerp(c_in, math::float4(1.f, 0.5f, 0.f, 1.f), 0.8f);
  return helium::cvt_color_to_uint32(c_out);
}

} // namespace

SCENARIO("Host buffer helpers apply fills and float-to-byte conversion",
    "[Algorithms]")
{
  GIVEN("Simple host buffers")
  {
    std::vector<float> src{-1.f, 0.f, 0.5f, 1.f, 2.f};
    std::vector<uint8_t> converted(src.size(), 0u);
    std::vector<uint32_t> uintBuf(4, 0u);
    std::vector<float> floatBuf(3, 0.f);

    WHEN("The CPU helpers are applied")
    {
      cpu::convertFloatToUint8(src.data(), converted.data(), converted.size());
      cpu::fill(uintBuf.data(), uint32_t(uintBuf.size()), 0xDEADBEEFu);
      cpu::fill(floatBuf.data(), uint32_t(floatBuf.size()), 3.5f);

      THEN("The conversion clamps to [0,255]")
      {
        REQUIRE(converted == std::vector<uint8_t>{0u, 0u, 127u, 255u, 255u});
      }

      THEN("The fills write every element")
      {
        REQUIRE(std::all_of(uintBuf.begin(), uintBuf.end(), [](uint32_t v) {
          return v == 0xDEADBEEFu;
        }));
        REQUIRE(std::all_of(floatBuf.begin(), floatBuf.end(), [](float v) {
          return v == Approx(3.5f);
        }));
      }
    }
  }
}

SCENARIO(
    "Host auto exposure computes strided log luminance sums", "[Algorithms]")
{
  GIVEN("An HDR buffer with known luminance samples")
  {
    const std::array<float, 16> hdr{1.f,
        1.f,
        1.f,
        1.f,
        9.f,
        9.f,
        9.f,
        1.f,
        0.25f,
        0.25f,
        0.25f,
        1.f,
        16.f,
        16.f,
        16.f,
        1.f};

    WHEN("Sampling every other pixel")
    {
      const float sum = cpu::sumLogLuminance(hdr.data(), 2u, 2u);

      THEN("Only the strided samples contribute")
      {
        REQUIRE(sum == Approx(-2.f));
      }
    }
  }

  GIVEN("A black HDR sample")
  {
    const std::array<float, 4> hdr{0.f, 0.f, 0.f, 1.f};

    WHEN("The luminance is accumulated")
    {
      const float sum = cpu::sumLogLuminance(hdr.data(), 1u, 1u);

      THEN("The minimum luminance clamp is respected")
      {
        REQUIRE(sum == Approx(std::log2(1e-4f)));
      }
    }
  }
}

SCENARIO(
    "Host tone mapping and output transforms preserve the expected color math",
    "[Algorithms]")
{
  GIVEN("A simple HDR pixel")
  {
    std::array<float, 4> hdr{0.25f, 0.5f, 1.f, 0.3f};

    WHEN("Tone mapping is disabled")
    {
      cpu::toneMap(hdr.data(), 1u, 2.f, tsd::algorithms::ToneMapOperator::NONE);

      THEN("Exposure scaling still applies and alpha is preserved")
      {
        REQUIRE(hdr[0] == Approx(0.5f));
        REQUIRE(hdr[1] == Approx(1.f));
        REQUIRE(hdr[2] == Approx(2.f));
        REQUIRE(hdr[3] == Approx(0.3f));
      }
    }
  }

  GIVEN("A pixel with negative and bright values")
  {
    std::array<float, 4> hdr{-1.f, 1.f, 3.f, 0.8f};

    WHEN("Reinhard tonemapping is applied")
    {
      cpu::toneMap(
          hdr.data(), 1u, 1.f, tsd::algorithms::ToneMapOperator::REINHARD);

      THEN("Negative values are clamped before the curve")
      {
        REQUIRE(hdr[0] == Approx(0.f));
        REQUIRE(hdr[1] == Approx(0.5f));
        REQUIRE(hdr[2] == Approx(0.75f));
        REQUIRE(hdr[3] == Approx(0.8f));
      }
    }
  }

  GIVEN("A float HDR source buffer")
  {
    const std::array<float, 4> hdr{0.25f, 1.f, 4.f, 0.5f};
    uint32_t out = 0u;

    WHEN("The output transform reads float input")
    {
      cpu::outputTransform(
          hdr.data(), nullptr, &out, 1u, 1.f, ANARI_FLOAT32_VEC4);

      THEN("The output is clamped and packed")
      {
        REQUIRE(out == packColor({0.25f, 1.f, 1.f, 0.5f}));
      }
    }
  }

  GIVEN("A packed color buffer")
  {
    const uint32_t in = packColor({0.25f, 0.5f, 1.f, 0.75f});
    uint32_t out = 0u;

    WHEN("The output transform reads packed input")
    {
      cpu::outputTransform(
          nullptr, &in, &out, 1u, 0.5f, ANARI_UFIXED8_RGBA_SRGB);

      THEN("Gamma encoding uses the packed source colors")
      {
        const auto unpacked = helium::cvt_color_to_float4(in);
        const auto encoded = linearToGamma(
            math::float3(unpacked.x, unpacked.y, unpacked.z), 0.5f);
        REQUIRE(
            out == packColor({encoded.x, encoded.y, encoded.z, unpacked.w}));
      }
    }
  }
}

SCENARIO("Host AOV visualization converts buffers into display colors",
    "[Algorithms]")
{
  GIVEN("A depth buffer")
  {
    const std::array<float, 3> depth{1.f, 2.f, 3.f};
    std::array<uint32_t, 3> color{};

    WHEN("Depth is visualized over a known range")
    {
      cpu::visualizeDepth(depth.data(), color.data(), 1.f, 3.f, 3u, 1u);

      THEN("The values are normalized into grayscale")
      {
        REQUIRE(color[0] == packColor({0.f, 0.f, 0.f, 1.f}));
        REQUIRE(color[1] == packColor({0.5f, 0.5f, 0.5f, 1.f}));
        REQUIRE(color[2] == packColor({1.f, 1.f, 1.f, 1.f}));
      }
    }
  }

  GIVEN("Normal and albedo AOVs")
  {
    const std::array<math::float3, 1> normal{math::float3(-1.f, 0.f, 1.f)};
    const std::array<math::float3, 1> albedo{math::float3(0.25f, 0.5f, 1.f)};
    std::array<uint32_t, 1> normalColor{};
    std::array<uint32_t, 1> albedoColor{};

    WHEN("They are visualized")
    {
      cpu::visualizeNormal(normal.data(), normalColor.data(), 1u, 1u);
      cpu::visualizeAlbedo(albedo.data(), albedoColor.data(), 1u, 1u);

      THEN("Normals are remapped and albedo uses sRGB packing")
      {
        REQUIRE(normalColor[0] == packColor({0.f, 0.5f, 1.f, 1.f}));
        REQUIRE(albedoColor[0] == packSrgbColor({0.25f, 0.5f, 1.f, 1.f}));
      }
    }
  }

  GIVEN("Object IDs containing edges and background")
  {
    const std::array<uint32_t, 9> objectId{
        ~0u, ~0u, ~0u, ~0u, 7u, 7u, ~0u, 7u, 7u};
    std::array<uint32_t, 9> edgeColor{};
    std::array<uint32_t, 4> idColor{};
    const std::array<uint32_t, 4> idInput{11u, ~0u, 11u, 12u};

    WHEN("Edges and IDs are visualized")
    {
      cpu::visualizeEdges(objectId.data(), edgeColor.data(), false, 3u, 3u);
      cpu::visualizeId(idInput.data(), idColor.data(), 2u, 2u);

      THEN("Background remains black and duplicate IDs map identically")
      {
        REQUIRE(edgeColor[0] == packColor({0.f, 0.f, 0.f, 1.f}));
        REQUIRE(edgeColor[4] == packColor({1.f, 1.f, 1.f, 1.f}));
        REQUIRE(idColor[1] == packColor({0.f, 0.f, 0.f, 1.f}));
        REQUIRE(idColor[0] == idColor[2]);
        REQUIRE(idColor[0] != idColor[3]);
      }
    }
  }
}

SCENARIO("Host compositing and outlining update only the intended pixels",
    "[Algorithms]")
{
  GIVEN("Two partially overlapping frames")
  {
    std::array<uint32_t, 3> outColor{1u, 2u, 3u};
    std::array<float, 3> outDepth{5.f, 2.f, 9.f};
    std::array<uint32_t, 3> outObjectId{10u, 20u, 30u};
    const std::array<uint32_t, 3> inColor{100u, 200u, 300u};
    const std::array<float, 3> inDepth{1.f, 4.f, 8.f};
    const std::array<uint32_t, 3> inObjectId{11u, 22u, 33u};

    WHEN("Depth compositing runs after the first pass")
    {
      cpu::depthCompositeFrame(outColor.data(),
          outDepth.data(),
          outObjectId.data(),
          inColor.data(),
          inDepth.data(),
          inObjectId.data(),
          3u,
          false);

      THEN("Only the nearer pixels overwrite the destination")
      {
        REQUIRE(outColor[0] == 100u);
        REQUIRE(outColor[1] == 2u);
        REQUIRE(outColor[2] == 300u);
        REQUIRE(outDepth[0] == Approx(1.f));
        REQUIRE(outDepth[1] == Approx(2.f));
        REQUIRE(outDepth[2] == Approx(8.f));
        REQUIRE(outObjectId[0] == 11u);
        REQUIRE(outObjectId[1] == 20u);
        REQUIRE(outObjectId[2] == 33u);
      }
    }
  }

  GIVEN("A destination frame and a first-pass source without object IDs")
  {
    std::array<uint32_t, 2> outColor{1u, 2u};
    std::array<float, 2> outDepth{9.f, 9.f};
    std::array<uint32_t, 2> outObjectId{44u, 55u};
    const std::array<uint32_t, 2> inColor{10u, 20u};
    const std::array<float, 2> inDepth{3.f, 4.f};

    WHEN("The first pass is composited")
    {
      cpu::depthCompositeFrame(outColor.data(),
          outDepth.data(),
          outObjectId.data(),
          inColor.data(),
          inDepth.data(),
          nullptr,
          2u,
          true);

      THEN("Color and depth overwrite while IDs stay untouched")
      {
        REQUIRE(outColor[0] == 10u);
        REQUIRE(outColor[1] == 20u);
        REQUIRE(outDepth[0] == Approx(3.f));
        REQUIRE(outDepth[1] == Approx(4.f));
        REQUIRE(outObjectId[0] == 44u);
        REQUIRE(outObjectId[1] == 55u);
      }
    }
  }

  GIVEN("A selected object fully inside the image")
  {
    const std::array<uint32_t, 16> objectId{~0u,
        ~0u,
        ~0u,
        ~0u,
        ~0u,
        5u,
        5u,
        ~0u,
        ~0u,
        5u,
        5u,
        ~0u,
        ~0u,
        ~0u,
        ~0u,
        ~0u};
    const uint32_t baseColor = packColor({0.2f, 0.4f, 0.6f, 1.f});
    std::array<uint32_t, 16> color{};
    color.fill(baseColor);

    WHEN("Outline shading is applied")
    {
      cpu::outlineObject(objectId.data(), color.data(), 5u, 4u, 4u);

      THEN("Boundary-adjacent pixels are shaded while distant pixels are not")
      {
        REQUIRE(color[5] == shadePixel(baseColor));
        REQUIRE(color[15] == baseColor);
      }
    }
  }
}
