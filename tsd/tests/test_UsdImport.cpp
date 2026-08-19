// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/core/DataTree.hpp"
#include "tsd/io/UsdImport.hpp"
// std
#include <string>

// Options and report are plain value types compiled regardless of whether the
// build has OpenUSD, so their tests are unguarded.

SCENARIO("USD import options round-trip through a data tree", "[UsdImport]")
{
  GIVEN("Options differing from their defaults in every field")
  {
    tsd::io::UsdImportOptions options;
    options.purposes.defaultPurpose = false;
    options.purposes.render = false;
    options.purposes.proxy = true;
    options.purposes.guide = true;
    options.renderContexts = {"mtlx", "mdl"};
    options.materialMode = tsd::io::UsdMaterialMode::MATERIALX;
    options.refinementLevel = 4;
    options.primPath = "/World/Asset";

    WHEN("They are written to a data tree and read back")
    {
      tsd::core::DataTree tree;
      options.toDataNode(tree.root());

      tsd::io::UsdImportOptions restored;
      restored.fromDataNode(tree.root());

      THEN("Every field survives the round-trip")
      {
        REQUIRE(restored.purposes.defaultPurpose == false);
        REQUIRE(restored.purposes.render == false);
        REQUIRE(restored.purposes.proxy == true);
        REQUIRE(restored.purposes.guide == true);
        REQUIRE(restored.renderContexts == options.renderContexts);
        REQUIRE(restored.materialMode == tsd::io::UsdMaterialMode::MATERIALX);
        REQUIRE(restored.refinementLevel == 4);
        REQUIRE(restored.primPath == "/World/Asset");
      }
    }
  }
}

SCENARIO("Import report counts skipped prims by reason", "[UsdImport]")
{
  tsd::io::UsdImportReport report;
  report.stageOpened = true;
  report.convertedPrims = 3;
  report.skipped.push_back(
      {"/a", "Mesh", tsd::io::UsdSkipReason::PURPOSE_EXCLUDED, ""});
  report.skipped.push_back(
      {"/b", "Mesh", tsd::io::UsdSkipReason::PURPOSE_EXCLUDED, ""});
  report.skipped.push_back({"/c",
      "PhysicsScene",
      tsd::io::UsdSkipReason::UNSUPPORTED_PRIM_TYPE,
      ""});

  THEN("Counts are reported per reason")
  {
    REQUIRE(report.countOf(tsd::io::UsdSkipReason::PURPOSE_EXCLUDED) == 2);
    REQUIRE(report.countOf(tsd::io::UsdSkipReason::UNSUPPORTED_PRIM_TYPE) == 1);
    REQUIRE(report.countOf(tsd::io::UsdSkipReason::TEXTURE_LOAD_FAILED) == 0);
    REQUIRE(report.contains(tsd::io::UsdSkipReason::PURPOSE_EXCLUDED));
    REQUIRE_FALSE(report.contains(tsd::io::UsdSkipReason::TEXTURE_LOAD_FAILED));
  }
}

#if TSD_USE_USD

// tsd_tests
#include "UsdTestFixtures.h"

SCENARIO("A USD Stage's prim hierarchy is mirrored in the Layer", "[UsdImport]")
{
  GIVEN("A Stage nesting a mesh two Xforms deep")
  {
    ImportedStage stage("tsd_test_usd_hierarchy.usda",
        std::string(R"(#usda 1.0

def Xform "World"
{
    double3 xformOp:translate = (1, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Xform "Group"
    {
        double3 xformOp:translate = (0, 2, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        def Mesh "Quad"
        {
)") + QUAD_MESH_BODY
            + R"(
        }
    }
}
)");

    WHEN("The Stage is imported")
    {
      auto *layer = stage.scene.defaultLayer();

      THEN("Each prim's name is findable at its own level")
      {
        auto world = findNode(layer, "World");
        auto group = findNode(layer, "Group");
        auto quad = findNode(layer, "Quad");
        REQUIRE(world);
        REQUIRE(group);
        REQUIRE(quad);
        REQUIRE(layer->isAncestorOf(world, group));
        REQUIRE(layer->isAncestorOf(group, quad));
      }

      THEN("Transforms are left nested rather than flattened")
      {
        auto world = findNode(layer, "World");
        auto group = findNode(layer, "Group");
        REQUIRE((*world)->getTransform()[3].x == Approx(1.0f));
        REQUIRE((*world)->getTransform()[3].y == Approx(0.0f));
        REQUIRE((*group)->getTransform()[3].x == Approx(0.0f));
        REQUIRE((*group)->getTransform()[3].y == Approx(2.0f));
      }

      THEN("Nothing is silently lost")
      {
        REQUIRE(stage.report.skipped.empty());
      }
    }
  }
}

SCENARIO(
    "Guide and proxy Purpose content is excluded by default", "[UsdImport]")
{
  GIVEN("A Stage with one mesh per Purpose")
  {
    const std::string purposeStage = std::string(R"(#usda 1.0

def Xform "World"
{
    def Mesh "Real"
    {
        uniform token purpose = "default"
)") + QUAD_MESH_BODY
            + R"(
    }

    def Mesh "Rendered"
    {
        uniform token purpose = "render"
)" + QUAD_MESH_BODY
            + R"(
    }

    def Mesh "Stand_In"
    {
        uniform token purpose = "proxy"
)" + QUAD_MESH_BODY
            + R"(
    }

    def Mesh "Helper"
    {
        uniform token purpose = "guide"
)" + QUAD_MESH_BODY
            + R"(
    }
}
)";

    WHEN("The Stage is imported with default options")
    {
      ImportedStage stage("tsd_test_usd_purpose.usda", purposeStage);

      THEN("Only default and render Purpose content arrives")
      {
        REQUIRE(stage.scene.numberOfObjects(ANARI_SURFACE) == 2);
      }

      THEN("Each excluded prim is reported")
      {
        REQUIRE(stage.report.countOf(tsd::io::UsdSkipReason::PURPOSE_EXCLUDED)
            == 2);
      }

      THEN("Excluded prims leave a disabled Placeholder Node")
      {
        auto *layer = stage.scene.defaultLayer();
        auto helper = findNode(layer, "Helper");
        REQUIRE(helper);
        REQUIRE((*helper)->isEmpty());
        REQUIRE_FALSE((*helper)->isEnabled());
      }
    }

    WHEN("The Stage is imported asking for proxy Purpose as well")
    {
      tsd::io::UsdImportOptions options;
      options.purposes.proxy = true;
      ImportedStage stage("tsd_test_usd_purpose.usda", purposeStage, options);

      THEN("Proxy content arrives too")
      {
        REQUIRE(stage.scene.numberOfObjects(ANARI_SURFACE) == 3);
        REQUIRE(stage.report.countOf(tsd::io::UsdSkipReason::PURPOSE_EXCLUDED)
            == 1);
      }
    }
  }
}

SCENARIO("Prims resolving to invisible import as disabled nodes", "[UsdImport]")
{
  GIVEN("A Stage with an invisible mesh")
  {
    ImportedStage stage("tsd_test_usd_invisible.usda",
        std::string(R"(#usda 1.0

def Xform "World"
{
    def Mesh "Hidden"
    {
        token visibility = "invisible"
)") + QUAD_MESH_BODY
            + R"(
    }
}
)");

    WHEN("The Stage is imported")
    {
      THEN("The content still arrives so it can be toggled on")
      {
        REQUIRE(stage.scene.numberOfObjects(ANARI_SURFACE) == 1);
      }

      THEN("Its node is disabled and the reason is reported")
      {
        auto *layer = stage.scene.defaultLayer();
        auto hidden = findNode(layer, "Hidden");
        REQUIRE(hidden);
        REQUIRE_FALSE((*hidden)->isEnabled());
        REQUIRE(stage.report.countOf(
                    tsd::io::UsdSkipReason::RESOLVED_INVISIBLE)
            == 1);
      }
    }
  }
}

SCENARIO(
    "Prim types TSD cannot represent become Placeholder Nodes", "[UsdImport]")
{
  GIVEN("A Stage containing a prim type with no TSD equivalent")
  {
    ImportedStage stage("tsd_test_usd_unsupported.usda",
        std::string(R"(#usda 1.0

def Xform "World"
{
    def CylinderLight "Tube"
    {
        float inputs:radius = 0.5
        float inputs:length = 2
    }

    def Mesh "Quad"
    {
)") + QUAD_MESH_BODY
            + R"(
    }
}
)");

    WHEN("The Stage is imported")
    {
      THEN("The supported content still arrives")
      {
        REQUIRE(stage.scene.numberOfObjects(ANARI_SURFACE) == 1);
        REQUIRE(stage.scene.numberOfObjects(ANARI_LIGHT) == 0);
      }

      THEN("The unsupported prim is named in the report")
      {
        REQUIRE(stage.report.skipped.size() == 1);
        REQUIRE(stage.report.skipped[0].primPath == "/World/Tube");
        REQUIRE(stage.report.skipped[0].reason
            == tsd::io::UsdSkipReason::UNSUPPORTED_LIGHT_TYPE);
      }

      THEN("It leaves a disabled Placeholder Node where it belongs")
      {
        auto *layer = stage.scene.defaultLayer();
        auto tube = findNode(layer, "Tube");
        REQUIRE(tube);
        REQUIRE((*tube)->isEmpty());
        REQUIRE_FALSE((*tube)->isEnabled());
      }
    }
  }
}

SCENARIO("An import can be restricted to one prim subtree", "[UsdImport]")
{
  GIVEN("A Stage with two sibling assets")
  {
    const std::string siblingAssets = std::string(R"(#usda 1.0

def Xform "AssetA"
{
    def Mesh "Quad"
    {
)") + QUAD_MESH_BODY
            + R"(
    }
}

def Xform "AssetB"
{
    def Mesh "Quad"
    {
)" + QUAD_MESH_BODY
            + R"(
    }
}
)";

    WHEN("The import is pointed at one subtree")
    {
      tsd::io::UsdImportOptions options;
      options.primPath = "/AssetB";
      ImportedStage stage("tsd_test_usd_subtree.usda", siblingAssets, options);

      THEN("Only that subtree arrives")
      {
        REQUIRE(stage.scene.numberOfObjects(ANARI_SURFACE) == 1);
        REQUIRE(stage.report.skipped.empty());
        REQUIRE_FALSE(findNode(stage.scene.defaultLayer(), "AssetA"));
        REQUIRE(findNode(stage.scene.defaultLayer(), "AssetB"));
      }
    }
  }
}

SCENARIO(
    "Stage framing metadata is recorded for the application", "[UsdImport]")
{
  GIVEN("A Z-up Stage authored in centimetres")
  {
    ImportedStage stage("tsd_test_usd_framing.usda",
        std::string(R"(#usda 1.0
(
    upAxis = "Z"
    metersPerUnit = 0.01
)

def Mesh "Quad"
{
)") + QUAD_MESH_BODY
            + R"(
}
)");

    WHEN("The Stage is imported")
    {
      THEN("Up-axis and unit scale are recorded on the import root node")
      {
        auto *layer = stage.scene.defaultLayer();
        auto root = findNode(layer, stage.path().c_str());
        REQUIRE(root);
        const auto &params = (*root)->getInstanceParameters();
        const auto *upAxis = params.at("usd:upAxis");
        const auto *scale = params.at("usd:metersPerUnit");
        REQUIRE(upAxis != nullptr);
        REQUIRE(scale != nullptr);
        REQUIRE(upAxis->getString() == "Z");
        REQUIRE(scale->get<float>() == Approx(0.01f));
      }

      THEN("Geometry coordinates are left exactly as authored")
      {
        auto geometry = stage.scene.getObject<tsd::scene::Geometry>(0);
        auto *position = geometry->parameterValueAsObject<tsd::scene::Array>(
            "vertex.position");
        REQUIRE(position != nullptr);
        const auto *p = position->dataAs<tsd::math::float3>();
        REQUIRE(p[1].x == Approx(1.0f));
        REQUIRE(p[1].y == Approx(0.0f));
        REQUIRE(p[1].z == Approx(0.0f));
      }
    }
  }
}

SCENARIO("A prim that resets the transform stack ignores its ancestors",
    "[UsdImport]")
{
  GIVEN("A child that resets the transform stack under a moved parent")
  {
    ImportedStage stage("tsd_test_usd_xform_reset.usda", R"(#usda 1.0

def Xform "Parent"
{
    double3 xformOp:translate = (10, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Mesh "Detached"
    {
        double3 xformOp:translate = (1, 2, 3)
        uniform token[] xformOpOrder = ["!resetXformStack!", "xformOp:translate"]
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    }
}
)");

    WHEN("The Stage is imported")
    {
      THEN("Composing parent and child lands where USD puts the child")
      {
        auto *layer = stage.scene.defaultLayer();
        auto parent = findNode(layer, "Parent");
        auto detached = findNode(layer, "Detached");
        REQUIRE(parent);
        REQUIRE(detached);

        const auto composed = tsd::math::mul(
            (*parent)->getTransform(), (*detached)->getTransform());
        REQUIRE(composed[3].x == Approx(1.f));
        REQUIRE(composed[3].y == Approx(2.f));
        REQUIRE(composed[3].z == Approx(3.f));
      }
    }
  }
}

SCENARIO("Time-varying visibility is reported rather than lost", "[UsdImport]")
{
  GIVEN("A mesh whose visibility is animated")
  {
    ImportedStage stage("tsd_test_usd_animated_visibility.usda", R"(#usda 1.0
(
    startTimeCode = 0
    endTimeCode = 2
)

def Mesh "Blinker"
{
    token visibility.timeSamples = {
        0: "inherited",
        1: "invisible",
    }
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
}
)");

    WHEN("The Stage is imported")
    {
      THEN("The caller is told the animation was not represented")
      {
        REQUIRE(stage.report.countOf(
                    tsd::io::UsdSkipReason::TIME_VARYING_VALUE_DROPPED)
            == 1);
      }
    }
  }
}

#endif // TSD_USE_USD
