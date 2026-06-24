// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/scene/objects/Material.hpp"

using tsd::scene::Material;

SCENARIO("tsd::scene::Material materialx", "[Material]")
{
  GIVEN("A materialx material")
  {
    Material obj(tsd::scene::tokens::material::materialx);
    THEN("It exposes source and materialName parameters")
    {
      REQUIRE(obj.parameter("source") != nullptr);
      REQUIRE(obj.parameter("materialName") != nullptr);
    }
  }
}

SCENARIO("tsd::scene::applyMaterialXStandardSurfacePreset", "[Material]")
{
  GIVEN("A materialx material with the StandardSurface preset")
  {
    Material obj(tsd::scene::tokens::material::materialx);
    tsd::scene::applyMaterialXStandardSurfacePreset(obj);
    THEN("source is the builtin and curated params have defaults")
    {
      REQUIRE(obj.parameter("source")->value().getString()
          == std::string("visrtx::standard_surface"));
      REQUIRE(obj.parameter("materialName")->value().getString()
          == std::string("StandardSurface"));
      REQUIRE(obj.parameter("base_color") != nullptr);
      REQUIRE(obj.parameter("base_color")->value().get<tsd::core::float3>()
          == tsd::core::float3(0.8f, 0.8f, 0.8f));
      REQUIRE(obj.parameter("specular_IOR")->value().get<float>() == 1.5f);
    }
  }
}

SCENARIO("tsd::scene::Material interface", "[Material]")
{
  GIVEN("A default constructed Material")
  {
    Material obj;

    THEN("The object value type is correct")
    {
      REQUIRE(obj.type() == ANARI_MATERIAL);
    }
  }
}
