// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// Geometry conversion: meshes, quadrics, curves and subdivision.

#if TSD_USE_USD

// catch
#include "catch.hpp"
// tsd_tests
#include "UsdTestFixtures.h"
// std
#include <cmath>

SCENARIO("A USD Stage's meshes arrive as surfaces", "[UsdImport]")
{
  GIVEN("A Stage with a single quad mesh")
  {
    ImportedStage stage("tsd_test_usd_single_mesh.usda",
        std::string(R"(#usda 1.0

def Xform "World"
{
    def Mesh "Quad"
    {
)") + QUAD_MESH_BODY
            + R"(
    }
}
)");

    WHEN("The Stage is imported")
    {
      THEN("The mesh becomes one triangle-geometry surface")
      {
        REQUIRE(stage.report.stageOpened);
        REQUIRE(stage.scene.numberOfObjects(ANARI_SURFACE) == 1);
        REQUIRE(stage.scene.numberOfObjects(ANARI_GEOMETRY) == 1);

        auto geometry = stage.scene.getObject<tsd::scene::Geometry>(0);
        REQUIRE(geometry);
        REQUIRE(geometry->subtype() == tsd::scene::tokens::geometry::triangle);

        auto *index = geometry->parameterValueAsObject<tsd::scene::Array>(
            "primitive.index");
        REQUIRE(index != nullptr);
        REQUIRE(index->size() == 2); // a quad tessellates to two triangles

        auto *position = geometry->parameterValueAsObject<tsd::scene::Array>(
            "vertex.position");
        REQUIRE(position != nullptr);
        REQUIRE(position->size() == 4);
      }

      THEN("Nothing is silently lost")
      {
        REQUIRE(stage.report.skipped.empty());
        REQUIRE(stage.report.convertedPrims == 1);
      }
    }
  }
}

SCENARIO("Analytic quadrics stay analytic", "[UsdImport]")
{
  GIVEN("A Stage with a sphere and a cylinder")
  {
    ImportedStage stage("tsd_test_usd_quadrics.usda", R"(#usda 1.0

def Sphere "Ball"
{
    double radius = 2
}

def Cylinder "Tube"
{
    double radius = 0.5
    double height = 4
    uniform token axis = "Y"
}
)");

    WHEN("The Stage is imported")
    {
      THEN("They map onto TSD's native quadric geometry, not meshes")
      {
        REQUIRE(stage.scene.numberOfObjects(ANARI_GEOMETRY) == 2);

        auto ball = stage.scene.getObject<tsd::scene::Geometry>(0);
        REQUIRE(ball->subtype() == tsd::scene::tokens::geometry::sphere);
        REQUIRE(ball->parameterValueAs<float>("radius").value() == Approx(2.f));

        auto tube = stage.scene.getObject<tsd::scene::Geometry>(1);
        REQUIRE(tube->subtype() == tsd::scene::tokens::geometry::cylinder);

        // The spine axis is folded into the endpoints rather than a transform.
        auto *positions =
            tube->parameterValueAsObject<tsd::scene::Array>("vertex.position");
        REQUIRE(positions != nullptr);
        REQUIRE(positions->size() == 2);
        const auto *p = positions->dataAs<tsd::math::float3>();
        REQUIRE(p[0].y == Approx(-2.f));
        REQUIRE(p[1].y == Approx(2.f));
        REQUIRE(stage.report.skipped.empty());
      }
    }
  }
}

SCENARIO(
    "A non-convex polygon tessellates without spurious geometry", "[UsdImport]")
{
  GIVEN("A mesh with one concave five-sided face")
  {
    ImportedStage stage("tsd_test_usd_nonconvex.usda", R"(#usda 1.0

def Mesh "Arrow"
{
    int[] faceVertexCounts = [5]
    int[] faceVertexIndices = [0, 1, 2, 3, 4]
    point3f[] points = [(0, 0, 0), (2, 0, 0), (2, 2, 0), (1, 1, 0), (0, 2, 0)]
}
)");

    WHEN("The Stage is imported")
    {
      THEN("It becomes exactly n-2 triangles")
      {
        auto geometry = stage.scene.getObject<tsd::scene::Geometry>(0);
        REQUIRE(geometry);
        auto *index = geometry->parameterValueAsObject<tsd::scene::Array>(
            "primitive.index");
        REQUIRE(index != nullptr);
        REQUIRE(index->size() == 3);
      }
    }
  }
}

SCENARIO("Subdivision surfaces are refined by default", "[UsdImport]")
{
  GIVEN("A cube that explicitly declares a subdivision scheme")
  {
    const std::string subdivCube = R"(#usda 1.0

def Mesh "SubdivCube"
{
    uniform token subdivisionScheme = "catmullClark"
    int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
    int[] faceVertexIndices = [0, 1, 2, 3,  4, 7, 6, 5,  0, 4, 5, 1,
                               1, 5, 6, 2,  2, 6, 7, 3,  3, 7, 4, 0]
    point3f[] points = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                        (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
}

def Mesh "PolygonCube"
{
    uniform token subdivisionScheme = "none"
    int[] faceVertexCounts = [4]
    int[] faceVertexIndices = [0, 1, 2, 3]
    point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
}
)";

    WHEN("The Stage is imported at the default refinement level")
    {
      ImportedStage stage("tsd_test_usd_subdiv.usda", subdivCube);

      THEN("The subdivision mesh gains vertices and the polygon mesh does not")
      {
        auto subdiv = stage.scene.getObject<tsd::scene::Geometry>(0);
        auto polygon = stage.scene.getObject<tsd::scene::Geometry>(1);
        REQUIRE(subdiv);
        REQUIRE(polygon);

        auto vertexCount = [](auto geometry) {
          auto *p =
              geometry->template parameterValueAsObject<tsd::scene::Array>(
                  "vertex.position");
          return p ? p->size() : size_t(0);
        };

        // Two levels of Catmull-Clark on a cube: 8 -> 26 -> 98 vertices.
        REQUIRE(vertexCount(subdiv) == 98);
        REQUIRE(vertexCount(polygon) == 4);
      }
    }

    WHEN("Refinement is turned off")
    {
      tsd::io::UsdImportOptions options;
      options.refinementLevel = 0;
      ImportedStage stage("tsd_test_usd_subdiv.usda", subdivCube, options);

      THEN("The subdivision mesh arrives at its authored resolution")
      {
        auto subdiv = stage.scene.getObject<tsd::scene::Geometry>(0);
        auto *p = subdiv->parameterValueAsObject<tsd::scene::Array>(
            "vertex.position");
        REQUIRE(p != nullptr);
        REQUIRE(p->size() == 8);
      }
    }
  }
}

SCENARIO(
    "Refinement carries face-varying primvars with the surface", "[UsdImport]")
{
  GIVEN("A subdivision mesh whose UVs are authored per face corner")
  {
    ImportedStage stage("tsd_test_usd_subdiv_uvs.usda", R"(#usda 1.0

def Mesh "SubdivQuad"
{
    uniform token subdivisionScheme = "catmullClark"
    int[] faceVertexCounts = [4]
    int[] faceVertexIndices = [0, 1, 2, 3]
    point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
        interpolation = "faceVarying"
    )
}
)");

    WHEN("The Stage is imported")
    {
      THEN("The UVs survive refinement rather than being dropped")
      {
        auto geometry = stage.scene.getObject<tsd::scene::Geometry>(0);
        REQUIRE(geometry);
        auto *uvs = geometry->parameterValueAsObject<tsd::scene::Array>(
            "faceVarying.attribute0");
        REQUIRE(uvs != nullptr);
        REQUIRE(uvs->size() > 4);
      }

      THEN("Nothing is reported as lost")
      {
        REQUIRE(stage.report.skipped.empty());
      }
    }
  }
}

SCENARIO(
    "Width-less curves and points get a bounds-scaled radius", "[UsdImport]")
{
  // Blender hair exports commonly omit widths; without an explicit radius the
  // ANARI default of 1 world unit dwarfs most scenes.
  GIVEN("A Stage with a widthless curve, a widthed curve, and widthless points")
  {
    ImportedStage stage("tsd_test_usd_widthless_curves.usda", R"(#usda 1.0

def Xform "World"
{
    def BasisCurves "Hair"
    {
        uniform token type = "linear"
        int[] curveVertexCounts = [4]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)]
    }

    def BasisCurves "Rope"
    {
        uniform token type = "linear"
        int[] curveVertexCounts = [2]
        point3f[] points = [(0, 0, 0), (1, 0, 0)]
        float[] widths = [0.2, 0.2] (interpolation = "vertex")
    }

    def Points "Sprinkles"
    {
        point3f[] points = [(0, 0, 0), (2, 0, 0)]
    }
}
)");

    WHEN("The Stage is imported")
    {
      THEN("The widthless curve's radius scales with its bounds")
      {
        auto geometry = findGeometry(stage.scene, "/World/Hair");
        REQUIRE(geometry);
        REQUIRE(geometry->subtype() == tsd::scene::tokens::geometry::curve);
        REQUIRE(
            geometry->parameterValueAsObject<tsd::scene::Array>("vertex.radius")
            == nullptr);

        const auto radius = geometry->parameterValueAs<float>("radius");
        REQUIRE(radius.has_value());
        REQUIRE(*radius == Approx(1e-3f * std::sqrt(3.f)));
      }

      THEN("Authored widths still become per-vertex radii")
      {
        auto geometry = findGeometry(stage.scene, "/World/Rope");
        REQUIRE(geometry);

        auto *radii = geometry->parameterValueAsObject<tsd::scene::Array>(
            "vertex.radius");
        REQUIRE(radii != nullptr);
        REQUIRE(radii->size() == 2);
        REQUIRE(radii->dataAs<float>()[0] == Approx(0.1f));
        REQUIRE_FALSE(geometry->parameterValueAs<float>("radius").has_value());
      }

      THEN("Widthless points scale the same way")
      {
        auto geometry = findGeometry(stage.scene, "/World/Sprinkles");
        REQUIRE(geometry);
        REQUIRE(geometry->subtype() == tsd::scene::tokens::geometry::sphere);

        const auto radius = geometry->parameterValueAs<float>("radius");
        REQUIRE(radius.has_value());
        REQUIRE(*radius == Approx(2e-3f));
      }
    }
  }
}

SCENARIO("Conversion leaves nothing behind for geometry it does not emit",
    "[UsdImport]")
{
  GIVEN("A mesh with no points that nonetheless binds a material")
  {
    ImportedStage stage("tsd_test_usd_empty_mesh.usda", R"(#usda 1.0

def Material "Orphan"
{
    token outputs:surface.connect = </Orphan/Shader.outputs:surface>

    def Shader "Shader"
    {
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor = (1, 0, 0)
        token outputs:surface
    }
}

def Mesh "Empty" (
    prepend apiSchemas = ["MaterialBindingAPI"]
)
{
    rel material:binding = </Orphan>
    int[] faceVertexCounts = []
    int[] faceVertexIndices = []
    point3f[] points = []
}
)");

    // A Scene creates one default Material of its own, so the count to
    // compare against is an empty Scene's rather than this one's after the
    // import.
    const auto materialsBefore =
        tsd::scene::Scene().numberOfObjects(ANARI_MATERIAL);

    WHEN("The Stage is imported")
    {
      THEN("No Surface, Geometry or Material is created for it")
      {
        REQUIRE(stage.scene.numberOfObjects(ANARI_SURFACE) == 0);
        REQUIRE(stage.scene.numberOfObjects(ANARI_GEOMETRY) == 0);
        REQUIRE(stage.scene.numberOfObjects(ANARI_MATERIAL) == materialsBefore);
      }
    }
  }

  GIVEN("A mesh whose subset claims no faces but binds its own material")
  {
    ImportedStage stage("tsd_test_usd_empty_subset.usda", R"(#usda 1.0

def Material "Used"
{
    token outputs:surface.connect = </Used/Shader.outputs:surface>

    def Shader "Shader"
    {
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor = (0, 1, 0)
        token outputs:surface
    }
}

def Material "Unused"
{
    token outputs:surface.connect = </Unused/Shader.outputs:surface>

    def Shader "Shader"
    {
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor = (0, 0, 1)
        token outputs:surface
    }
}

def Mesh "Quad"
{
    int[] faceVertexCounts = [3, 3]
    int[] faceVertexIndices = [0, 1, 2, 0, 2, 3]
    point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]

    def GeomSubset "Drawn" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        uniform token elementType = "face"
        uniform token familyName = "materialBind"
        int[] indices = [0, 1]
        rel material:binding = </Used>
    }

    def GeomSubset "Empty" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        uniform token elementType = "face"
        uniform token familyName = "materialBind"
        int[] indices = []
        rel material:binding = </Unused>
    }
}
)");

    WHEN("The Stage is imported")
    {
      THEN("Only the material the drawn subset uses is created")
      {
        REQUIRE(stage.scene.numberOfObjects(ANARI_SURFACE) == 1);

        bool sawUnused = false;
        const auto numMaterials = stage.scene.numberOfObjects(ANARI_MATERIAL);
        for (size_t i = 0; i < numMaterials; ++i) {
          auto material = stage.scene.getObject<tsd::scene::Material>(i);
          if (material && material->name().find("Unused") != std::string::npos)
            sawUnused = true;
        }
        REQUIRE_FALSE(sawUnused);
      }
    }
  }
}

#endif // TSD_USE_USD
