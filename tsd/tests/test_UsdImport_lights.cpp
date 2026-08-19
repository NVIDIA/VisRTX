// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// Lights and cameras.

#if TSD_USE_USD

// catch
#include "catch.hpp"
// tsd_tests
#include "UsdTestFixtures.h"

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
