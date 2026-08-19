// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// Animated Stages: sample times, deforming geometry, dialect prims.

#if TSD_USE_USD

// catch
#include "catch.hpp"
// tsd_tests
#include "UsdTestFixtures.h"
// tsd
#include "tsd/core/DataTree.hpp"
#include "tsd/io/animation/UsdGeometryFileBinding.hpp"
#include "tsd/io/archives/AnimationManagerArchive.hpp"
// std
#include <cstring>
#include <fstream>
#include <string>

SCENARIO("Animation is captured at the times actually authored", "[UsdImport]")
{
  GIVEN("A Stage with transforms keyed on a non-uniform time base")
  {
    ImportedStage stage("tsd_test_usd_time_base.usda", R"(#usda 1.0
(
    startTimeCode = 0
    endTimeCode = 10
)

def Xform "Mover"
{
    double3 xformOp:translate.timeSamples = {
        0: (0, 0, 0),
        1: (1, 0, 0),
        10: (10, 0, 0),
    }
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Mesh "Quad"
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    }
}
)");

    WHEN("The Stage is imported")
    {
      THEN("The binding's time base mirrors the authored sample spacing")
      {
        REQUIRE(stage.animMgr.animations().size() == 1);
        const auto &bindings =
            stage.animMgr.animations()[0].transformBindings();
        REQUIRE(bindings.size() == 1);
        const auto &timeBase = bindings[0].timeBase();
        REQUIRE(timeBase.size() == 3);
        REQUIRE(timeBase[0] == Approx(0.0f));
        REQUIRE(timeBase[1] == Approx(0.1f));
        REQUIRE(timeBase[2] == Approx(1.0f));
      }
    }
  }
}

SCENARIO("A full turn authored with two keys does not collapse", "[UsdImport]")
{
  GIVEN("A prim rotating 360 degrees between two keyframes")
  {
    ImportedStage stage("tsd_test_usd_full_turn.usda", R"(#usda 1.0
(
    startTimeCode = 0
    endTimeCode = 1
)

def Xform "Spinner"
{
    float3 xformOp:rotateXYZ.timeSamples = {
        0: (0, 0, 0),
        1: (0, 360, 0),
    }
    uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]

    def Mesh "Quad"
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    }
}
)");

    WHEN("The Stage is imported")
    {
      THEN("Extra samples are inserted so the rotation still animates")
      {
        REQUIRE(stage.animMgr.animations().size() == 1);
        const auto &bindings =
            stage.animMgr.animations()[0].transformBindings();
        REQUIRE(bindings.size() == 1);
        REQUIRE(bindings[0].sampleCount() > 2);
      }
    }
  }

  GIVEN("A prim translating between two keyframes")
  {
    ImportedStage stage("tsd_test_usd_small_move.usda", R"(#usda 1.0
(
    startTimeCode = 0
    endTimeCode = 1
)

def Xform "Slider"
{
    double3 xformOp:translate.timeSamples = {
        0: (0, 0, 0),
        1: (5, 0, 0),
    }
    uniform token[] xformOpOrder = ["xformOp:translate"]
}
)");

    WHEN("The Stage is imported")
    {
      THEN("No extra samples are invented")
      {
        const auto &bindings =
            stage.animMgr.animations()[0].transformBindings();
        REQUIRE(bindings[0].sampleCount() == 2);
      }
    }
  }
}

SCENARIO(
    "Lazily-bound deforming geometry survives save and reload", "[UsdImport]")
{
  GIVEN("A Stage whose mesh points are time-sampled")
  {
    ImportedStage stage("tsd_test_usd_deforming.usda", R"(#usda 1.0
(
    startTimeCode = 0
    endTimeCode = 2
)

def Mesh "Blob"
{
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points.timeSamples = {
        0: [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
        1: [(0, 0, 0), (2, 0, 0), (0, 2, 0)],
        2: [(0, 0, 0), (3, 0, 0), (0, 3, 0)],
    }
}
)");

    WHEN("The Stage is imported")
    {
      THEN("Only one frame is eager; the rest is bound to the Stage")
      {
        REQUIRE(stage.animMgr.animations().size() == 1);
        REQUIRE(stage.animMgr.animations()[0].fileBindings().size() == 1);
        REQUIRE(stage.animMgr.animations()[0].fileBindings()[0]->kind()
            == "usdGeometry");

        auto geometry = stage.scene.getObject<tsd::scene::Geometry>(0);
        auto *positions = geometry->parameterValueAsObject<tsd::scene::Array>(
            "vertex.position");
        REQUIRE(positions != nullptr);
        REQUIRE(positions->size() == 3);
      }

      THEN("Scrubbing pulls a later frame from the Stage")
      {
        stage.animMgr.setAnimationTime(1.0f);

        auto geometry = stage.scene.getObject<tsd::scene::Geometry>(0);
        auto *positions = geometry->parameterValueAsObject<tsd::scene::Array>(
            "vertex.position");
        REQUIRE(positions != nullptr);
        REQUIRE(positions->dataAs<tsd::math::float3>()[1].x == Approx(3.f));
      }

      THEN("The binding reconstructs from an Archive")
      {
        tsd::core::DataTree tree;
        REQUIRE(tsd::io::serialize_AnimationManagerArchive(
            stage.animMgr, tree.root()));

        tsd::animation::AnimationManager restored(&stage.scene);
        REQUIRE(tsd::io::deserialize_AnimationManagerArchive(
            restored, tree.root()));
        REQUIRE(restored.animations().size() == 1);
        REQUIRE(restored.animations()[0].fileBindings().size() == 1);
        REQUIRE(restored.animations()[0].fileBindings()[0]->kind()
            == "usdGeometry");

        restored.setAnimationTime(1.0f);
        auto geometry = stage.scene.getObject<tsd::scene::Geometry>(0);
        auto *positions = geometry->parameterValueAsObject<tsd::scene::Array>(
            "vertex.position");
        REQUIRE(positions->dataAs<tsd::math::float3>()[1].x == Approx(3.f));
      }
    }
  }
}

SCENARIO("Claimed dialect prims are handled once and only once", "[UsdImport]")
{
  GIVEN("A Stage whose carrier prims are claimed by the TSD dialect")
  {
    // The EnSight carrier marker is customData on the carrier's children; the
    // claim-and-prune pre-pass must keep the generic path from converting
    // them into meaningless geometry.
    ImportedStage stage("tsd_test_usd_dialect.usda", R"(#usda 1.0

def Scope "Dataset"
{
    def Mesh "part_one" (
        customData = {
            dictionary ensight = {
                string partName = "part_one"
            }
        }
    )
    {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    }
}

def Mesh "Real"
{
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
}
)");

    WHEN("The Stage is imported")
    {
      THEN("The carrier prim does not also arrive as generic geometry")
      {
        // Only the ordinary mesh converts: the claimed subtree is pruned from
        // the resolved scene entirely.
        REQUIRE(stage.scene.numberOfObjects(ANARI_SURFACE) == 1);
        REQUIRE_FALSE(findNode(stage.scene.defaultLayer(), "part_one"));
      }
    }
  }
}

// EnSight Gold writes its strings as fixed 80-byte records and its numbers
// raw, so both files below are laid out with these two.
void writeRecord(std::ofstream &out, const char *text)
{
  char buffer[80] = {};
  std::strncpy(buffer, text, sizeof(buffer) - 1);
  out.write(buffer, sizeof(buffer));
}

void writeInteger(std::ofstream &out, int32_t value)
{
  out.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

// A minimal EnSight Gold dataset -- two single-triangle parts -- written into
// the shared fixture directory in the binary geometry format import_ENSIGHT
// accepts. The files live for the lifetime of the test binary along with every
// other fixture in that directory.
//
// With `withScalarField`, the dataset also carries one node-centered scalar,
// which is what makes import_ENSIGHT synthesize a colormap material for a part
// that has no material bound to it.
std::string writeEnSightDataset(const char *baseName, bool withScalarField)
{
  const auto geoName = std::string(baseName) + ".geo";
  const auto scalarName = std::string(baseName) + ".scl";
  const auto casePath = fixtureDirectory() / (std::string(baseName) + ".case");
  const auto geoPath = fixtureDirectory() / geoName;

  {
    std::ofstream caseFile(casePath);
    caseFile << "FORMAT\n"
             << "type: ensight gold\n"
             << "GEOMETRY\n"
             << "model: " << geoName << "\n";
    if (withScalarField) {
      caseFile << "VARIABLE\n"
               << "scalar per node: density " << scalarName << "\n";
    }
  }

  const int32_t partIds[2] = {1, 2};
  const char *partDescriptions[2] = {"part_one", "part_two"};
  constexpr int numNodes = 3;

  {
    std::ofstream geo(geoPath, std::ios::binary);
    writeRecord(geo, "C Binary");
    writeRecord(geo, "TSD test dataset");
    writeRecord(geo, "two single-triangle parts");
    writeRecord(geo, "node id off");
    writeRecord(geo, "element id off");

    for (int i = 0; i < 2; ++i) {
      writeRecord(geo, "part");
      writeInteger(geo, partIds[i]);
      writeRecord(geo, partDescriptions[i]);
      writeRecord(geo, "coordinates");
      writeInteger(geo, numNodes);
      // clang-format off
      const float coordinates[9] = {
          0.f, 1.f, 0.f, // x
          0.f, 0.f, 1.f, // y
          0.f, 0.f, 0.f}; // z
      // clang-format on
      geo.write(
          reinterpret_cast<const char *>(coordinates), sizeof(coordinates));
      writeRecord(geo, "tria3");
      writeInteger(geo, 1);
      writeInteger(geo, 1);
      writeInteger(geo, 2);
      writeInteger(geo, 3);
    }
  }

  if (withScalarField) {
    std::ofstream scalar(fixtureDirectory() / scalarName, std::ios::binary);
    writeRecord(scalar, "per node scalar values");
    for (int i = 0; i < 2; ++i) {
      writeRecord(scalar, "part");
      writeInteger(scalar, partIds[i]);
      writeRecord(scalar, "coordinates");
      // The values only have to span a range for a colormap to be built over.
      const float values[numNodes] = {0.f, 0.5f, 1.f};
      scalar.write(reinterpret_cast<const char *>(values), sizeof(values));
    }
  }

  return casePath.string();
}

SCENARIO(
    "EnSight parts take the materials their carrier prims bind", "[UsdImport]")
{
  GIVEN("A carrier scope and one of its parts each binding a material")
  {
    // Claimed Prims never reach the resolved traversal, so the materials they
    // bind only convert if the dialect importer asks for them itself.
    const auto caseFile =
        writeEnSightDataset("tsd_test_ensight_materials", false);
    ImportedStage stage("tsd_test_usd_ensight_materials.usda",
        R"(#usda 1.0
(
    customLayerData = {
        dictionary ensight = {
            string caseFile = ")"
            + caseFile + R"("
        }
    }
)

def Scope "Looks"
{
    def Material "Shared"
    {
        token outputs:surface.connect = </Looks/Shared/PBR.outputs:surface>

        def Shader "PBR"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (1, 0, 0)
            token outputs:surface
        }
    }

    def Material "PartOne"
    {
        token outputs:surface.connect = </Looks/PartOne/PBR.outputs:surface>

        def Shader "PBR"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0, 1, 0)
            token outputs:surface
        }
    }
}

def Scope "Dataset" (
    prepend apiSchemas = ["MaterialBindingAPI"]
)
{
    rel material:binding = </Looks/Shared>

    def Mesh "part_one" (
        prepend apiSchemas = ["MaterialBindingAPI"]
        customData = {
            dictionary ensight = {
                string partName = "part_one"
            }
        }
    )
    {
        rel material:binding = </Looks/PartOne>
)" + std::string(QUAD_MESH_BODY)
            + R"(    }

    def Mesh "part_two" (
        customData = {
            dictionary ensight = {
                string partName = "part_two"
            }
        }
    )
    {
)" + std::string(QUAD_MESH_BODY)
            + R"(    }
}
)");

    // Materials are named for the prim path they came from, so the name says
    // which Material prim the part ended up bound to.
    auto materialNameOfPart = [&](const char *partName) {
      auto surface =
          findObject<tsd::scene::Surface>(stage.scene, ANARI_SURFACE, partName);
      REQUIRE(surface);
      auto *material = surface->parameterValueAsObject<tsd::scene::Material>(
          tsd::scene::tokens::surface::material);
      REQUIRE(material != nullptr);
      return material->name();
    };

    WHEN("The Stage is imported")
    {
      THEN("The part with its own binding gets that material")
      {
        REQUIRE(materialNameOfPart("part_one") == "/Looks/PartOne");
      }

      THEN("The part without one falls back to the carrier's material")
      {
        REQUIRE(materialNameOfPart("part_two") == "/Looks/Shared");
      }
    }
  }
}

SCENARIO("A material bound on an EnSight carrier outranks the scalar colormap",
    "[UsdImport]")
{
  // import_ENSIGHT picks a part's material as
  //   per-part binding > carrier binding > scalar colormap > default,
  // but until carrier bindings converted at all, neither binding rung could
  // ever win. Both halves below read the same dataset, so the only thing that
  // differs is whether the carrier binds anything.
  //
  // The carriers map a field explicitly. The dialect always hands
  // import_ENSIGHT a field list, so an unmapped variable is loaded by nobody
  // and there would be no colormap for a binding to outrank.
  const auto caseFile = writeEnSightDataset("tsd_test_ensight_colormap", true);

  auto materialOfPart = [](ImportedStage &stage, const char *partName) {
    auto surface =
        findObject<tsd::scene::Surface>(stage.scene, ANARI_SURFACE, partName);
    REQUIRE(surface);
    auto *material = surface->parameterValueAsObject<tsd::scene::Material>(
        tsd::scene::tokens::surface::material);
    REQUIRE(material != nullptr);
    return material;
  };

  const auto stageBody = [&](const char *carrierBinding) {
    return R"(#usda 1.0
(
    customLayerData = {
        dictionary ensight = {
            string caseFile = ")"
        + caseFile + R"("
        }
    }
)

def Scope "Looks"
{
    def Material "Shared"
    {
        token outputs:surface.connect = </Looks/Shared/PBR.outputs:surface>

        def Shader "PBR"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (1, 0, 0)
            token outputs:surface
        }
    }
}

def Scope "Dataset" (
    prepend apiSchemas = ["MaterialBindingAPI"]
)
{
    custom string ensight:fieldMapping:attribute0 = "density"
)" + std::string(carrierBinding)
        + R"(
    def Mesh "part_one" (
        customData = {
            dictionary ensight = {
                string partName = "part_one"
            }
        }
    )
    {
)" + std::string(QUAD_MESH_BODY)
        + R"(    }
}
)";
  };

  GIVEN("A carrier that binds no material over a dataset with a scalar field")
  {
    ImportedStage stage(
        "tsd_test_usd_ensight_colormap_unbound.usda", stageBody(""));

    WHEN("The Stage is imported")
    {
      THEN("The part takes the colormap built from that field")
      {
        // The colormap material is synthesized rather than converted from a
        // prim, so it carries no name -- unlike the Scene's own default, which
        // is a matte one called "default".
        auto *material = materialOfPart(stage, "part_one");
        REQUIRE(material->name().empty());
        REQUIRE(material->subtype()
            == tsd::scene::tokens::material::physicallyBased);
      }
    }
  }

  GIVEN("The same dataset under a carrier that does bind one")
  {
    ImportedStage stage("tsd_test_usd_ensight_colormap_bound.usda",
        stageBody("    rel material:binding = </Looks/Shared>"));

    WHEN("The Stage is imported")
    {
      THEN("The bound material wins and no colormap is built")
      {
        REQUIRE(materialOfPart(stage, "part_one")->name() == "/Looks/Shared");
      }
    }
  }
}

SCENARIO(
    "Constant-valued time samples are not reported as a loss", "[UsdImport]")
{
  GIVEN("A mesh whose visibility is authored at every frame but never changes")
  {
    // What a simulation exporter writes: every attribute re-authored at every
    // frame regardless of whether it moved.
    ImportedStage stage("tsd_test_usd_constant_visibility.usda", R"(#usda 1.0
(
    startTimeCode = 0
    endTimeCode = 2
)

def Mesh "Steady"
{
    token visibility.timeSamples = {
        0: "inherited",
        1: "inherited",
        2: "inherited",
    }
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
}
)");

    WHEN("The Stage is imported")
    {
      THEN("Nothing is reported as dropped")
      {
        REQUIRE(stage.report.countOf(
                    tsd::io::UsdSkipReason::TIME_VARYING_VALUE_DROPPED)
            == 0);
      }
    }
  }
}

SCENARIO("One import is one Animation", "[UsdImport]")
{
  GIVEN("A Stage animating two prims that share a leaf name")
  {
    ImportedStage stage("tsd_test_usd_one_animation.usda", R"(#usda 1.0
(
    startTimeCode = 0
    endTimeCode = 2
)

def Xform "A"
{
    def Xform "Mover"
    {
        double3 xformOp:translate.timeSamples = {
            0: (0, 0, 0),
            2: (2, 0, 0),
        }
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}

def Xform "B"
{
    def Xform "Mover"
    {
        double3 xformOp:translate.timeSamples = {
            0: (0, 0, 0),
            2: (0, 5, 0),
        }
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}
)");

    WHEN("The Stage is imported")
    {
      THEN("Both prims land in one Animation named for the file")
      {
        REQUIRE(stage.animMgr.animations().size() == 1);
        REQUIRE(stage.animMgr.animations()[0].name() == stage.path());
        REQUIRE(stage.animMgr.animations()[0].transformBindings().size() == 2);
      }

      THEN("The Report counts them in place of the lost per-prim entries")
      {
        REQUIRE(stage.report.animatedPrims == 2);
      }
    }
  }
}

SCENARIO("An old-format geometry binding still reconstructs", "[UsdImport]")
{
  GIVEN("An Archive node carrying the dropped sampleTimes and timeBase fields")
  {
    ImportedStage stage("tsd_test_usd_legacy_binding.usda", R"(#usda 1.0
(
    startTimeCode = 0
    endTimeCode = 2
)

def Mesh "Blob"
{
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points.timeSamples = {
        0: [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
        2: [(0, 0, 0), (3, 0, 0), (0, 3, 0)],
    }
}
)");

    stage.animMgr.removeAllAnimations();

    // Written the way an Archive from before continuous Time Code evaluation
    // was: the derived sample cache is present and must simply be ignored.
    tsd::core::DataTree tree;
    auto &node = tree.root();
    node["targetIndex"] = size_t(0);
    node["stageFile"] = stage.path();
    node["primPath"] = std::string("/Blob");
    node["sampleTimes"].append() = 0.f;
    node["sampleTimes"].append() = 2.f;
    node["timeBase"].append() = 0.f;
    node["timeBase"].append() = 1.f;

    WHEN("It is read back")
    {
      auto &anim = stage.animMgr.addAnimation("legacy");
      REQUIRE(tsd::io::UsdGeometryFileBinding::addToAnimation(
                  anim, stage.scene, node)
          != nullptr);

      THEN("It scrubs from the Stage's own clock")
      {
        stage.animMgr.setAnimationTime(1.0f);

        auto geometry = stage.scene.getObject<tsd::scene::Geometry>(0);
        auto *positions = geometry->parameterValueAsObject<tsd::scene::Array>(
            "vertex.position");
        REQUIRE(positions != nullptr);
        REQUIRE(positions->dataAs<tsd::math::float3>()[1].x == Approx(3.f));
      }
    }
  }
}

SCENARIO(
    "A mesh whose topology changes re-pulls a consistent set", "[UsdImport]")
{
  GIVEN("A mesh whose points, indices and primvars all change together")
  {
    // The case a binding that only re-pulls points cannot serve: writing new
    // positions without new indices would describe a mesh that never existed.
    ImportedStage stage("tsd_test_usd_morphing_mesh.usda", R"(#usda 1.0
(
    startTimeCode = 0
    endTimeCode = 2
)

def Mesh "Morphing"
{
    int[] faceVertexCounts.timeSamples = {
        0: [3],
        2: [3, 3],
    }
    int[] faceVertexIndices.timeSamples = {
        0: [0, 1, 2],
        2: [0, 1, 2, 0, 2, 3],
    }
    point3f[] points.timeSamples = {
        0: [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
        2: [(0, 0, 0), (1, 0, 0), (0, 1, 0), (-1, 1, 0)],
    }
    color3f[] primvars:displayColor (
        interpolation = "vertex"
    )
    color3f[] primvars:displayColor.timeSamples = {
        0: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        2: [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)],
    }
}
)");

    WHEN("The Stage is imported")
    {
      auto geometry = stage.scene.getObject<tsd::scene::Geometry>(0);
      REQUIRE(geometry);

      auto arraySize = [&](const char *parameter) -> size_t {
        auto *array =
            geometry->parameterValueAsObject<tsd::scene::Array>(parameter);
        return array ? array->size() : 0;
      };

      THEN("The first frame is one triangle over three vertices")
      {
        REQUIRE(arraySize("vertex.position") == 3);
        REQUIRE(arraySize("primitive.index") == 1);
        REQUIRE(arraySize("vertex.color") == 3);
      }

      THEN("Scrubbing re-pulls points, indices and primvars together")
      {
        stage.animMgr.setAnimationTime(1.0f);

        REQUIRE(arraySize("vertex.position") == 4);
        REQUIRE(arraySize("primitive.index") == 2);
        REQUIRE(arraySize("vertex.color") == 4);

        // Every index has to address the positions that arrived with it.
        auto *indices = geometry->parameterValueAsObject<tsd::scene::Array>(
            "primitive.index");
        const auto *triangles = indices->dataAs<tsd::math::uint3>();
        for (size_t i = 0; i < indices->size(); ++i) {
          REQUIRE(triangles[i].x < 4);
          REQUIRE(triangles[i].y < 4);
          REQUIRE(triangles[i].z < 4);
        }
      }

      THEN("The Surface and its Geometry keep their identity across the scrub")
      {
        const auto surfacesBefore = stage.scene.numberOfObjects(ANARI_SURFACE);
        const auto geometriesBefore =
            stage.scene.numberOfObjects(ANARI_GEOMETRY);
        const auto materialsBefore =
            stage.scene.numberOfObjects(ANARI_MATERIAL);

        stage.animMgr.setAnimationTime(1.0f);

        // Re-running conversion would have built new ones, forcing the render
        // index to tear down and recreate ANARI handles (ADR 0022).
        REQUIRE(stage.scene.numberOfObjects(ANARI_SURFACE) == surfacesBefore);
        REQUIRE(
            stage.scene.numberOfObjects(ANARI_GEOMETRY) == geometriesBefore);
        REQUIRE(stage.scene.numberOfObjects(ANARI_MATERIAL) == materialsBefore);
      }
    }
  }
}

SCENARIO("Parts keep sharing one position Array across a resize", "[UsdImport]")
{
  GIVEN("A mesh divided into subsets whose vertex count changes")
  {
    ImportedStage stage("tsd_test_usd_shared_resize.usda", R"(#usda 1.0
(
    startTimeCode = 0
    endTimeCode = 2
)

def Mesh "Split"
{
    int[] faceVertexCounts.timeSamples = {
        0: [3, 3],
        2: [3, 3, 3],
    }
    int[] faceVertexIndices.timeSamples = {
        0: [0, 1, 2, 0, 2, 3],
        2: [0, 1, 2, 0, 2, 3, 0, 3, 4],
    }
    point3f[] points.timeSamples = {
        0: [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
        2: [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (-1, 1, 0)],
    }

    def GeomSubset "A"
    {
        uniform token elementType = "face"
        uniform token familyName = "materialBind"
        int[] indices = [0]
    }

    def GeomSubset "B"
    {
        uniform token elementType = "face"
        uniform token familyName = "materialBind"
        int[] indices = [1]
    }
}
)");

    WHEN("The Stage is imported")
    {
      auto positionArrayOf = [&](size_t i) {
        auto geometry = stage.scene.getObject<tsd::scene::Geometry>(i);
        return geometry ? geometry->parameterValueAsObject<tsd::scene::Array>(
                              "vertex.position")
                        : nullptr;
      };

      const auto parts = stage.scene.numberOfObjects(ANARI_GEOMETRY);
      REQUIRE(parts > 1); // the subsets, plus any unclaimed remainder

      THEN("Every Part shares one position Array on import")
      {
        auto *first = positionArrayOf(0);
        REQUIRE(first != nullptr);
        for (size_t i = 1; i < parts; ++i)
          REQUIRE(positionArrayOf(i) == first);
      }

      THEN("They still share one after a resize, not a copy each")
      {
        stage.animMgr.setAnimationTime(1.0f);

        auto *first = positionArrayOf(0);
        REQUIRE(first != nullptr);
        REQUIRE(first->size() == 5);
        for (size_t i = 1; i < parts; ++i)
          REQUIRE(positionArrayOf(i) == first);
      }
    }
  }
}

#endif // TSD_USE_USD
