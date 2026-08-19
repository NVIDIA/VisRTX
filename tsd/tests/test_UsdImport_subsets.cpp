// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// Material subsets and the per-subset attributes that follow them.

#if TSD_USE_USD

// catch
#include "catch.hpp"
// tsd_tests
#include "UsdTestFixtures.h"
// std
#include <vector>

SCENARIO("Per-face material subsets become several Surfaces", "[UsdImport]")
{
  GIVEN("A two-face mesh with one face bound to its own material")
  {
    ImportedStage stage("tsd_test_usd_subsets.usda", R"(#usda 1.0

def Xform "World"
{
    def Scope "Looks"
    {
        def Material "Red"
        {
            token outputs:surface.connect = </World/Looks/Red/PBR.outputs:surface>
            def Shader "PBR"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (1, 0, 0)
                token outputs:surface
            }
        }

        def Material "Blue"
        {
            token outputs:surface.connect = </World/Looks/Blue/PBR.outputs:surface>
            def Shader "PBR"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0, 0, 1)
                token outputs:surface
            }
        }
    }

    def Mesh "Strip" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        int[] faceVertexCounts = [4, 4]
        int[] faceVertexIndices = [0, 1, 2, 3, 4, 5, 6, 7]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                            (2, 0, 0), (3, 0, 0), (3, 1, 0), (2, 1, 0)]
        rel material:binding = </World/Looks/Red>

        def GeomSubset "Second" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {
            uniform token elementType = "face"
            uniform token familyName = "materialBind"
            int[] indices = [1]
            rel material:binding = </World/Looks/Blue>
        }
    }
}
)");

    WHEN("The Stage is imported")
    {
      THEN("There is one Surface per subset")
      {
        REQUIRE(stage.scene.numberOfObjects(ANARI_SURFACE) >= 1);
      }

      THEN("The faces no subset claims are still drawn, by the mesh's material")
      {
        // The first face belongs to no subset, so it stays with the mesh's own
        // binding instead of vanishing with the un-surfaced parent geometry.
        auto leftover = findGeometry(stage.scene, "/World/Strip");
        REQUIRE(leftover);
        auto *index = leftover->parameterValueAsObject<tsd::scene::Array>(
            "primitive.index");
        REQUIRE(index != nullptr);
        REQUIRE(index->size() == 2);
        REQUIRE(stage.scene.numberOfObjects(ANARI_SURFACE) == 2);
      }

      THEN("The subset Surfaces share the mesh's vertex positions")
      {
        // Every geometry produced for this mesh points at the same
        // vertex.position Array; only the index arrays differ.
        std::vector<const tsd::scene::Array *> positions;
        const auto numGeometries = stage.scene.numberOfObjects(ANARI_GEOMETRY);
        for (size_t i = 0; i < numGeometries; ++i) {
          auto geometry = stage.scene.getObject<tsd::scene::Geometry>(i);
          if (!geometry)
            continue;
          if (auto *p = geometry->parameterValueAsObject<tsd::scene::Array>(
                  "vertex.position"))
            positions.push_back(p);
        }
        REQUIRE(positions.size() >= 2);
        for (size_t i = 1; i < positions.size(); ++i)
          REQUIRE(positions[i] == positions[0]);
      }
    }
  }
}

SCENARIO("Face-varying UVs follow each material subset", "[UsdImport]")
{
  GIVEN("A two-face mesh with per-corner UVs and a subset over each face")
  {
    ImportedStage stage("tsd_test_usd_subset_facevarying.usda", R"(#usda 1.0

def Xform "World"
{
    def Mesh "Quad"
    {
        int[] faceVertexCounts = [4, 4]
        int[] faceVertexIndices = [0, 1, 2, 3, 4, 5, 6, 7]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                            (2, 0, 0), (3, 0, 0), (3, 1, 0), (2, 1, 0)]
        texCoord2f[] primvars:st = [(0, 0), (0.25, 0), (0.25, 0.25), (0, 0.25),
                                    (0.5, 0), (0.75, 0), (0.75, 0.75),
                                    (0.5, 0.75)] (
            interpolation = "faceVarying"
        )
        normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1),
                              (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (
            interpolation = "vertex"
        )

        def GeomSubset "Left"
        {
            uniform token elementType = "face"
            uniform token familyName = "materialBind"
            int[] indices = [0]
        }

        def GeomSubset "Right"
        {
            uniform token elementType = "face"
            uniform token familyName = "materialBind"
            int[] indices = [1]
        }
    }
}
)");

    WHEN("The Stage is imported")
    {
      THEN("Each subset carries the corners of the faces it selected")
      {
        // Face-varying data is indexed by triangle corner, so a subset cannot
        // share the parent array the way vertex data can -- it must gather the
        // corners of its own triangles. Each quad triangulates to two
        // triangles, hence six corners, and the two faces' UVs are authored
        // into disjoint halves of the unit square so a mis-gather shows up.
        auto left = findGeometry(stage.scene, "/World/Quad/Left");
        auto right = findGeometry(stage.scene, "/World/Quad/Right");
        REQUIRE(left);
        REQUIRE(right);

        auto *leftUVs = left->parameterValueAsObject<tsd::scene::Array>(
            "faceVarying.attribute0");
        auto *rightUVs = right->parameterValueAsObject<tsd::scene::Array>(
            "faceVarying.attribute0");
        REQUIRE(leftUVs != nullptr);
        REQUIRE(rightUVs != nullptr);
        REQUIRE(leftUVs->size() == 6);
        REQUIRE(rightUVs->size() == 6);

        const auto *l = leftUVs->dataAs<tsd::math::float2>();
        const auto *r = rightUVs->dataAs<tsd::math::float2>();
        for (size_t i = 0; i < 6; ++i) {
          REQUIRE(l[i].x < 0.5f);
          REQUIRE(r[i].x >= 0.5f);
        }
      }

      THEN("Vertex-interpolated attributes are still shared, not copied")
      {
        // Only per-corner and per-triangle data has to be gathered; vertex
        // data is indexed by the indices each subset already carries, so one
        // Array serves every Surface.
        auto left = findGeometry(stage.scene, "/World/Quad/Left");
        auto right = findGeometry(stage.scene, "/World/Quad/Right");
        REQUIRE(left);
        REQUIRE(right);

        auto *leftNormals =
            left->parameterValueAsObject<tsd::scene::Array>("vertex.normal");
        REQUIRE(leftNormals != nullptr);
        REQUIRE(leftNormals
            == right->parameterValueAsObject<tsd::scene::Array>(
                "vertex.normal"));
      }
    }
  }
}

SCENARIO(
    "Face-varying primvars survive an already-triangulated mesh", "[UsdImport]")
{
  GIVEN("An all-triangle mesh with indexed face-varying UVs and normals")
  {
    // Hydra's triangulator reports this topology as Unchanged rather than
    // producing a copy of the input, a distinct result from Success that the
    // conversion must not mistake for failure -- pre-triangulated exports
    // carry every face-varying primvar down this path.
    ImportedStage stage(
        "tsd_test_usd_triangulated_facevarying.usda", R"(#usda 1.0

def Xform "World"
{
    def Mesh "Triangles"
    {
        int[] faceVertexCounts = [3, 3]
        int[] faceVertexIndices = [0, 1, 2, 0, 2, 3]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
            interpolation = "faceVarying"
        )
        int[] primvars:st:indices = [0, 1, 2, 0, 2, 3]
        normal3f[] primvars:normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1),
                                       (0, 0, 1), (0, 0, 1), (0, 0, 1)] (
            interpolation = "faceVarying"
        )
    }
}
)");

    WHEN("The Stage is imported")
    {
      THEN("The UVs arrive flattened, one value per triangle corner")
      {
        auto geometry = findGeometry(stage.scene, "/World/Triangles");
        REQUIRE(geometry);

        auto *uvs = geometry->parameterValueAsObject<tsd::scene::Array>(
            "faceVarying.attribute0");
        REQUIRE(uvs != nullptr);
        REQUIRE(uvs->size() == 6);

        const auto *uv = uvs->dataAs<tsd::math::float2>();
        REQUIRE(uv[3].x == Approx(0.0f)); // second triangle's first corner
        REQUIRE(uv[4].x == Approx(1.0f));
        REQUIRE(uv[5].x == Approx(0.0f));
        // `v` arrives reversed out of USD's v-up `st` into ANARI's, which runs
        // down the image: the authored (0, 0) and (0, 1) become 1 and 0.
        REQUIRE(uv[3].y == Approx(1.0f));
        REQUIRE(uv[5].y == Approx(0.0f));
      }

      THEN("The normals arrive too")
      {
        auto geometry = findGeometry(stage.scene, "/World/Triangles");
        REQUIRE(geometry);

        auto *normals = geometry->parameterValueAsObject<tsd::scene::Array>(
            "faceVarying.normal");
        REQUIRE(normals != nullptr);
        REQUIRE(normals->size() == 6);
      }
    }
  }
}

SCENARIO("A subset binds the UV primvar its own material reads", "[UsdImport]")
{
  GIVEN("Two subsets whose materials read differently named UV primvars")
  {
    ImportedStage stage("tsd_test_usd_subset_uv_primvar.usda", R"(#usda 1.0

def Xform "World"
{
    def Scope "Looks"
    {
        def Material "ReadsMapOne"
        {
            token outputs:surface.connect = </World/Looks/ReadsMapOne/PBR.outputs:surface>

            def Shader "PBR"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = </World/Looks/ReadsMapOne/Tex.outputs:rgb>
                token outputs:surface
            }

            def Shader "Tex"
            {
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @missing_texture.png@
                float2 inputs:st.connect = </World/Looks/ReadsMapOne/Reader.outputs:result>
                float3 outputs:rgb
            }

            def Shader "Reader"
            {
                uniform token info:id = "UsdPrimvarReader_float2"
                token inputs:varname = "map1"
                float2 outputs:result
            }
        }

        def Material "ReadsMapTwo"
        {
            token outputs:surface.connect = </World/Looks/ReadsMapTwo/PBR.outputs:surface>

            def Shader "PBR"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = </World/Looks/ReadsMapTwo/Tex.outputs:rgb>
                token outputs:surface
            }

            def Shader "Tex"
            {
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @missing_texture.png@
                float2 inputs:st.connect = </World/Looks/ReadsMapTwo/Reader.outputs:result>
                float3 outputs:rgb
            }

            def Shader "Reader"
            {
                uniform token info:id = "UsdPrimvarReader_float2"
                token inputs:varname = "map2"
                float2 outputs:result
            }
        }
    }

    def Mesh "Quad"
    {
        int[] faceVertexCounts = [4, 4]
        int[] faceVertexIndices = [0, 1, 2, 3, 4, 5, 6, 7]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                            (2, 0, 0), (3, 0, 0), (3, 1, 0), (2, 1, 0)]
        texCoord2f[] primvars:map1 = [(0.25, 0), (0.25, 0), (0.25, 0),
                                      (0.25, 0), (0.25, 0), (0.25, 0),
                                      (0.25, 0), (0.25, 0)] (
            interpolation = "faceVarying"
        )
        texCoord2f[] primvars:map2 = [(0.75, 0), (0.75, 0), (0.75, 0),
                                      (0.75, 0), (0.75, 0), (0.75, 0),
                                      (0.75, 0), (0.75, 0)] (
            interpolation = "faceVarying"
        )

        def GeomSubset "Left" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {
            uniform token elementType = "face"
            uniform token familyName = "materialBind"
            int[] indices = [0]
            rel material:binding = </World/Looks/ReadsMapOne>
        }

        def GeomSubset "Right" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {
            uniform token elementType = "face"
            uniform token familyName = "materialBind"
            int[] indices = [1]
            rel material:binding = </World/Looks/ReadsMapTwo>
        }
    }
}
)");

    WHEN("The Stage is imported")
    {
      THEN("Each subset's own primvar lands on its first attribute")
      {
        // The mesh itself binds no material, so the UV name cannot be decided
        // once for the whole mesh: each subset's material names its own.
        auto left = findGeometry(stage.scene, "/World/Quad/Left");
        auto right = findGeometry(stage.scene, "/World/Quad/Right");
        REQUIRE(left);
        REQUIRE(right);

        auto *leftUVs = left->parameterValueAsObject<tsd::scene::Array>(
            "faceVarying.attribute0");
        auto *rightUVs = right->parameterValueAsObject<tsd::scene::Array>(
            "faceVarying.attribute0");
        REQUIRE(leftUVs != nullptr);
        REQUIRE(rightUVs != nullptr);
        REQUIRE(leftUVs->dataAs<tsd::math::float2>()[0].x == Approx(0.25f));
        REQUIRE(rightUVs->dataAs<tsd::math::float2>()[0].x == Approx(0.75f));
      }
    }
  }
}

#endif // TSD_USE_USD
