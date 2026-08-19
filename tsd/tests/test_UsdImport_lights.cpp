// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// Lights and cameras.

#if TSD_USE_USD

// catch
#include "catch.hpp"
// helium
#include <helium/helium_math.h>
// tsd_tests
#include "UsdTestFixtures.h"
// std
#include <cmath>

SCENARIO(
    "Light exposure and normalization reach the emitted light", "[UsdImport]")
{
  GIVEN("A sphere light with exposure and normalization set")
  {
    // intensity 4, exposure 2 -> 4 * 2^2 = 16; normalize divides by the
    // sphere's area, 4*pi*r^2 with r = 2 -> 16 / (16*pi).
    ImportedStage stage("tsd_test_usd_light_radiometry.usda", R"(#usda 1.0

def SphereLight "Lamp"
{
    float inputs:intensity = 4
    float inputs:exposure = 2
    bool inputs:normalize = true
    float inputs:radius = 2
    color3f inputs:color = (1, 1, 1)
}
)");

    WHEN("The Stage is imported")
    {
      THEN("The light's intensity accounts for both")
      {
        REQUIRE(stage.scene.numberOfObjects(ANARI_LIGHT) == 1);
        auto light = stage.scene.getObject<tsd::scene::Light>(0);
        REQUIRE(light);
        REQUIRE(light->subtype() == tsd::scene::tokens::light::point);

        const auto intensity = light->parameterValueAs<float>("intensity");
        REQUIRE(intensity.has_value());
        const float expected = 16.f / (4.f * float(M_PI) * 4.f);
        REQUIRE(*intensity == Approx(expected));
        REQUIRE(stage.report.skipped.empty());
      }
    }
  }
}

SCENARIO("A shaped sphere light becomes a spot light", "[UsdImport]")
{
  GIVEN("A sphere light carrying shaping attributes")
  {
    ImportedStage stage("tsd_test_usd_spot.usda", R"(#usda 1.0

def SphereLight "Spot" (
    prepend apiSchemas = ["ShapingAPI"]
)
{
    float inputs:intensity = 1
    float inputs:radius = 0.5
    float inputs:shaping:cone:angle = 30
    float inputs:shaping:cone:softness = 0.5
}
)");

    WHEN("The Stage is imported")
    {
      THEN("Spot lighting survives the import")
      {
        auto light = stage.scene.getObject<tsd::scene::Light>(0);
        REQUIRE(light);
        REQUIRE(light->subtype() == tsd::scene::tokens::light::spot);

        const auto opening = light->parameterValueAs<float>("openingAngle");
        REQUIRE(opening.has_value());
        REQUIRE(*opening == Approx(2.f * 30.f * float(M_PI) / 180.f));

        const auto falloff = light->parameterValueAs<float>("falloffAngle");
        REQUIRE(falloff.has_value());
        REQUIRE(*falloff == Approx(0.5f * 0.5f * *opening));
      }
    }
  }
}

// Each light type puts its brightness on a different ANARI parameter, and
// which one is decided per branch. A converter that reached for the wrong name
// would still produce a light of the right subtype, so subtype alone does not
// hold this.
SCENARIO("A distant light's brightness lands on irradiance", "[UsdImport]")
{
  GIVEN("A distant light with an authored intensity and colour")
  {
    ImportedStage stage("tsd_test_usd_distant.usda", R"(#usda 1.0

def DistantLight "Sun"
{
    float inputs:intensity = 3
    bool inputs:normalize = true
    color3f inputs:color = (0.25, 0.5, 1)
}
)");

    WHEN("The Stage is imported")
    {
      THEN("It becomes a directional light carrying irradiance, not intensity")
      {
        REQUIRE(stage.scene.numberOfObjects(ANARI_LIGHT) == 1);
        auto light = stage.scene.getObject<tsd::scene::Light>(0);
        REQUIRE(light);
        REQUIRE(light->subtype() == tsd::scene::tokens::light::directional);

        const auto irradiance = light->parameterValueAs<float>("irradiance");
        REQUIRE(irradiance.has_value());
        // A distant light subtends no area, so `normalize` has nothing to
        // divide by and must leave the authored intensity alone.
        REQUIRE(*irradiance == Approx(3.f));
        REQUIRE(!light->parameterValueAs<float>("intensity").has_value());

        auto color = light->parameterValueAs<tsd::math::float3>("color");
        REQUIRE(color.has_value());
        REQUIRE(color->x == Approx(0.25f));
        REQUIRE(color->y == Approx(0.5f));
        REQUIRE(color->z == Approx(1.f));
        REQUIRE(stage.report.skipped.empty());
      }
    }
  }
}

SCENARIO("A rect light becomes a quad spanned by its own width and height",
    "[UsdImport]")
{
  GIVEN("A rect light 4 wide and 2 high, normalized")
  {
    ImportedStage stage("tsd_test_usd_rect.usda", R"(#usda 1.0

def RectLight "Panel"
{
    float inputs:intensity = 8
    bool inputs:normalize = true
    float inputs:width = 4
    float inputs:height = 2
}
)");

    WHEN("The Stage is imported")
    {
      THEN("Its corner and edges describe the authored rectangle")
      {
        REQUIRE(stage.scene.numberOfObjects(ANARI_LIGHT) == 1);
        auto light = stage.scene.getObject<tsd::scene::Light>(0);
        REQUIRE(light);
        REQUIRE(light->subtype() == tsd::scene::tokens::light::quad);

        // normalize divides by the rectangle's area, 4 * 2.
        const auto intensity = light->parameterValueAs<float>("intensity");
        REQUIRE(intensity.has_value());
        REQUIRE(*intensity == Approx(1.f));

        // ANARI's quad is a corner plus two edge vectors; USD's is centred on
        // the prim's origin, so the corner is half of each extent back.
        auto position = light->parameterValueAs<tsd::math::float3>("position");
        REQUIRE(position.has_value());
        REQUIRE(position->x == Approx(-2.f));
        REQUIRE(position->y == Approx(-1.f));
        REQUIRE(position->z == Approx(0.f));

        auto edge1 = light->parameterValueAs<tsd::math::float3>("edge1");
        REQUIRE(edge1.has_value());
        REQUIRE(edge1->x == Approx(4.f));
        REQUIRE(edge1->y == Approx(0.f));

        auto edge2 = light->parameterValueAs<tsd::math::float3>("edge2");
        REQUIRE(edge2.has_value());
        REQUIRE(edge2->x == Approx(0.f));
        REQUIRE(edge2->y == Approx(2.f));
        REQUIRE(stage.report.skipped.empty());
      }
    }
  }
}

// The dome light is the one branch that cannot go through the shared helper:
// its colour is baked into the radiance it maps over the sphere, so a `color`
// parameter would be applied twice.
SCENARIO("An untextured dome light still lights the scene", "[UsdImport]")
{
  GIVEN("A dome light with a colour and an intensity but no texture")
  {
    ImportedStage stage("tsd_test_usd_dome.usda", R"(#usda 1.0

def DomeLight "Sky"
{
    float inputs:intensity = 2
    color3f inputs:color = (0.5, 0.25, 0)
}
)");

    WHEN("The Stage is imported")
    {
      THEN("Its colour arrives baked into a constant radiance, not as color")
      {
        REQUIRE(stage.scene.numberOfObjects(ANARI_LIGHT) == 1);
        auto light = stage.scene.getObject<tsd::scene::Light>(0);
        REQUIRE(light);
        REQUIRE(light->subtype() == tsd::scene::tokens::light::hdri);

        // Brightness rides on `scale` here rather than on `intensity`, and
        // `color` must stay unset: the texels below already carry it.
        const auto scale = light->parameterValueAs<float>("scale");
        REQUIRE(scale.has_value());
        REQUIRE(*scale == Approx(2.f));
        REQUIRE(
            !light->parameterValueAs<tsd::math::float3>("color").has_value());

        // Devices require radiance to be set, so an untextured dome gets a
        // synthesized single texel rather than nothing.
        auto *radiance =
            light->parameterValueAsObject<tsd::scene::Array>("radiance");
        REQUIRE(radiance != nullptr);
        REQUIRE(radiance->dim(0) == 1);
        REQUIRE(radiance->dim(1) == 1);

        const auto texel = helium::readAsAttributeValueFlat(
            radiance->data(), radiance->elementType(), 0);
        REQUIRE(texel.x == Approx(1.f));
        REQUIRE(texel.y == Approx(0.5f));
        REQUIRE(texel.z == Approx(0.f));
        REQUIRE(stage.report.skipped.empty());
      }
    }
  }
}

SCENARIO("Cameras from a Stage arrive in the camera pool", "[UsdImport]")
{
  GIVEN("A Stage with an animated camera rig")
  {
    ImportedStage stage("tsd_test_usd_camera.usda", R"(#usda 1.0
(
    startTimeCode = 0
    endTimeCode = 2
)

def Xform "Rig"
{
    double3 xformOp:translate.timeSamples = {
        0: (0, 0, 0),
        2: (0, 0, 10),
    }
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Camera "Shot"
    {
        float focalLength = 50
        float horizontalAperture = 36
        float verticalAperture = 24
    }
}
)");

    // A Scene starts out with a camera of its own, so the count to compare
    // against is an empty Scene's rather than this one's after the import.
    const auto camerasBefore =
        tsd::scene::Scene().numberOfObjects(ANARI_CAMERA);

    WHEN("The Stage is imported")
    {
      THEN("The authored viewpoint is available and animated")
      {
        REQUIRE(stage.scene.numberOfObjects(ANARI_CAMERA) == camerasBefore + 1);

        bool hasCameraAnimation = false;
        for (const auto &animation : stage.animMgr.animations()) {
          if (!animation.objectParameterBindings().empty())
            hasCameraAnimation = true;
        }
        REQUIRE(hasCameraAnimation);
      }
    }
  }
}

#endif // TSD_USE_USD
