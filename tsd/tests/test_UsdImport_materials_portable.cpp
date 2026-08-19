// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// Materials that land on TSD's portable material.

#if TSD_USE_USD

// catch
#include "catch.hpp"
// tsd_tests
#include "UsdTestFixtures.h"

SCENARIO("An unconventionally named UV primvar is still found", "[UsdImport]")
{
  GIVEN("A material whose reader node asks for a primvar not called 'st'")
  {
    ImportedStage stage("tsd_test_usd_uv_primvar.usda", R"(#usda 1.0

def Xform "World"
{
    def Material "Textured"
    {
        token outputs:surface.connect = </World/Textured/PBR.outputs:surface>

        def Shader "PBR"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor.connect = </World/Textured/Tex.outputs:rgb>
            token outputs:surface
        }

        def Shader "Tex"
        {
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @missing_texture.png@
            float2 inputs:st.connect = </World/Textured/Reader.outputs:result>
            float3 outputs:rgb
        }

        def Shader "Reader"
        {
            uniform token info:id = "UsdPrimvarReader_float2"
            token inputs:varname = "map1"
            float2 outputs:result
        }
    }

    def Mesh "Quad" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        texCoord2f[] primvars:map1 = [(0, 0), (1, 0), (1, 1), (0, 1)] (
            interpolation = "vertex"
        )
        rel material:binding = </World/Textured>
    }
}
)");

    WHEN("The Stage is imported")
    {
      THEN("The named primvar lands on the geometry's first attribute")
      {
        auto geometry = stage.scene.getObject<tsd::scene::Geometry>(0);
        REQUIRE(geometry);
        auto *uvs = geometry->parameterValueAsObject<tsd::scene::Array>(
            "vertex.attribute0");
        REQUIRE(uvs != nullptr);
        REQUIRE(uvs->size() == 4);
      }
    }
  }
}

// A material resolves once and is cached, including when it does not resolve
// at all. Without a negative entry every prim bound to the same broken
// material re-runs the resolve and files its own Import Report entry, which
// makes the counts the report prints scale with the binding count.
SCENARIO(
    "An unresolvable material is reported once per material rather than"
    " once per binding",
    "[UsdImport]")
{
  GIVEN("A Stage where three meshes bind one material with no network")
  {
    ImportedStage stage("tsd_test_usd_unresolvable_material.usda", R"(#usda 1.0

def Xform "World"
{
    def Material "Broken"
    {
    }

    def Mesh "QuadA" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        rel material:binding = </World/Broken>
    }

    def Mesh "QuadB" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(2, 0, 0), (3, 0, 0), (3, 1, 0), (2, 1, 0)]
        rel material:binding = </World/Broken>
    }

    def Mesh "QuadC" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(4, 0, 0), (5, 0, 0), (5, 1, 0), (4, 1, 0)]
        rel material:binding = </World/Broken>
    }
}
)");

    WHEN("The Stage is imported")
    {
      THEN("The Import Report names the material exactly once")
      {
        REQUIRE(stage.report.countOf(
                    tsd::io::UsdSkipReason::MATERIAL_RESOLUTION_FAILED)
            == 1);
      }

      THEN("Every mesh still arrives")
      {
        REQUIRE(stage.scene.numberOfObjects(ANARI_SURFACE) == 3);
      }
    }
  }
}

SCENARIO("An OmniPBR material maps onto the portable material", "[UsdImport]")
{
  GIVEN("A Stage whose material is an OmniPBR MDL shader")
  {
    const std::string omniPbrStage = R"(#usda 1.0

def Xform "World"
{
    def Material "OmniPBR"
    {
        token outputs:mdl:surface.connect = </World/OmniPBR/Shader.outputs:out>
        def Shader "Shader"
        {
            uniform token info:implementationSource = "sourceAsset"
            uniform asset info:mdl:sourceAsset = @OmniPBR.mdl@
            uniform token info:mdl:sourceAsset:subIdentifier = "OmniPBR"
            color3f inputs:diffuse_color_constant = (0.9, 0.1, 0.2)
            float inputs:metallic_constant = 0.75
            float inputs:reflection_roughness_constant = 0.25
            float inputs:ior_constant = 1.4
            bool inputs:enable_emission = 1
            color3f inputs:emissive_color = (0, 0.5, 0)
            float inputs:emissive_intensity = 2
            token outputs:out
        }
    }

    def Mesh "Quad" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        rel material:binding = </World/OmniPBR>
    }
}
)";

    WHEN("The default material mode is used")
    {
      ImportedStage stage("tsd_test_usd_omnipbr.usda", omniPbrStage);
      auto *material = boundMaterial(stage.scene);

      THEN("Its authored inputs arrive, not the preview-surface defaults")
      {
        REQUIRE(material->subtype()
            == tsd::scene::tokens::material::physicallyBased);

        auto color = material->parameterValueAs<tsd::math::float3>("baseColor");
        REQUIRE(color.has_value());
        REQUIRE(color->x == Approx(0.9f));
        REQUIRE(color->y == Approx(0.1f));
        REQUIRE(color->z == Approx(0.2f));

        REQUIRE(
            *material->parameterValueAs<float>("metallic") == Approx(0.75f));
        REQUIRE(
            *material->parameterValueAs<float>("roughness") == Approx(0.25f));
        REQUIRE(*material->parameterValueAs<float>("ior") == Approx(1.4f));

        auto emissive =
            material->parameterValueAs<tsd::math::float3>("emissive");
        REQUIRE(emissive.has_value());
        REQUIRE(emissive->y == Approx(1.0f));
      }

      THEN("Nothing claims a richer material was left on the table")
      {
        REQUIRE(!stage.report.contains(
            tsd::io::UsdSkipReason::RICHER_MATERIAL_AVAILABLE));
      }
    }

    WHEN("MDL passthrough is asked for instead")
    {
      tsd::io::UsdImportOptions options;
      options.materialMode = tsd::io::UsdMaterialMode::MDL;
      ImportedStage stage("tsd_test_usd_omnipbr.usda", omniPbrStage, options);

      THEN("The native shader still wins over the portable mapping")
      {
        REQUIRE(boundMaterial(stage.scene)->subtype()
            == tsd::scene::tokens::material::mdl);
      }
    }
  }

  GIVEN("An OmniPBR material reading textures and cutting out on opacity")
  {
    TextureFixture diffuse("tsd_test_usd_omnipbr_diffuse.tga");

    ImportedStage stage("tsd_test_usd_omnipbr_textured.usda", R"(#usda 1.0

def Xform "World"
{
    def Material "OmniPBR"
    {
        token outputs:mdl:surface.connect = </World/OmniPBR/Shader.outputs:out>
        def Shader "Shader"
        {
            uniform token info:implementationSource = "sourceAsset"
            uniform asset info:mdl:sourceAsset = @OmniPBR.mdl@
            uniform token info:mdl:sourceAsset:subIdentifier = "OmniPBR"
            asset inputs:diffuse_texture = @tsd_test_usd_omnipbr_diffuse.tga@
            bool inputs:enable_opacity = 1
            float inputs:opacity_threshold = 0.3
            token outputs:out
        }
    }

    def Mesh "Quad" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        rel material:binding = </World/OmniPBR>
    }
}
)");

    WHEN("The default material mode is used")
    {
      auto *material = boundMaterial(stage.scene);

      THEN("The texture named on the shader input is bound")
      {
        REQUIRE(
            material->parameterValueAsObject<tsd::scene::Sampler>("baseColor")
            != nullptr);
        REQUIRE(!stage.report.contains(
            tsd::io::UsdSkipReason::TEXTURE_LOAD_FAILED));
      }

      THEN("An authored threshold becomes a cutout rather than a blend")
      {
        REQUIRE(material->parameterValueAs<std::string>("alphaMode")
            == std::string("mask"));
        REQUIRE(
            *material->parameterValueAs<float>("alphaCutoff") == Approx(0.3f));

        // The mode is a string selection, so the index has to agree with the
        // value wherever the selection is what gets read.
        auto *alphaMode = material->parameter("alphaMode");
        REQUIRE(alphaMode->stringValues()[alphaMode->stringSelection()]
            == std::string("mask"));
      }
    }
  }

  GIVEN("A material whose MDL module only looks like OmniPBR")
  {
    ImportedStage stage("tsd_test_usd_omnipbr_lookalike.usda", R"(#usda 1.0

def Xform "World"
{
    def Material "Lookalike"
    {
        token outputs:mdl:surface.connect = </World/Lookalike/Mdl.outputs:out>
        token outputs:surface.connect = </World/Lookalike/PBR.outputs:surface>

        def Shader "Mdl"
        {
            uniform token info:implementationSource = "sourceAsset"
            uniform asset info:mdl:sourceAsset = @OmniPBRBase.mdl@
            uniform token info:mdl:sourceAsset:subIdentifier = "OmniPBRBase"
            color3f inputs:diffuse_color_constant = (0.9, 0.1, 0.2)
            token outputs:out
        }

        def Shader "PBR"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.1, 0.2, 0.9)
            token outputs:surface
        }
    }

    def Mesh "Quad" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        rel material:binding = </World/Lookalike>
    }
}
)");

    WHEN("The default material mode is used")
    {
      THEN("A module that merely starts with the name is not mapped as one")
      {
        // The authored preview surface is what this material actually says;
        // OmniPBRBase is a different shader with input semantics of its own.
        auto color = boundMaterial(stage.scene)
                         ->parameterValueAs<tsd::math::float3>("baseColor");
        REQUIRE(color.has_value());
        REQUIRE(color->z == Approx(0.9f));
      }
    }
  }
}

SCENARIO(
    "A prim with no bound material takes its display colour", "[UsdImport]")
{
  GIVEN("A mesh with display colour and opacity but no material")
  {
    ImportedStage stage("tsd_test_usd_display_color.usda", R"(#usda 1.0

def Mesh "Quad"
{
    int[] faceVertexCounts = [4]
    int[] faceVertexIndices = [0, 1, 2, 3]
    point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    color3f[] primvars:displayColor = [(0.25, 0.5, 0.75)] (
        interpolation = "constant"
    )
    float[] primvars:displayOpacity = [0.5] (
        interpolation = "constant"
    )
}
)");

    WHEN("The Stage is imported")
    {
      THEN("The Surface's material carries the display values")
      {
        auto surface = stage.scene.getObject<tsd::scene::Surface>(0);
        REQUIRE(surface);
        auto *material = surface->parameterValueAsObject<tsd::scene::Material>(
            tsd::scene::tokens::surface::material);
        REQUIRE(material != nullptr);

        const auto color =
            material->parameterValueAs<tsd::math::float3>("color");
        REQUIRE(color.has_value());
        REQUIRE(color->x == Approx(0.25f));
        REQUIRE(color->y == Approx(0.5f));
        REQUIRE(color->z == Approx(0.75f));

        const auto opacity = material->parameterValueAs<float>("opacity");
        REQUIRE(opacity.has_value());
        REQUIRE(*opacity == Approx(0.5f));
      }
    }
  }
}

SCENARIO("Analytic prims without a material take their display colour",
    "[UsdImport]")
{
  GIVEN("A point cloud carrying only display colour")
  {
    ImportedStage stage("tsd_test_usd_points_display_color.usda", R"(#usda 1.0

def Points "Cloud"
{
    point3f[] points = [(0, 0, 0), (1, 0, 0)]
    float[] widths = [0.2, 0.4]
    color3f[] primvars:displayColor = [(1, 0, 0)] (
        interpolation = "constant"
    )
}
)");

    WHEN("The Stage is imported")
    {
      THEN("Its material carries the display colour, not TSD's default")
      {
        auto surface = stage.scene.getObject<tsd::scene::Surface>(0);
        REQUIRE(surface);
        auto *material = surface->parameterValueAsObject<tsd::scene::Material>(
            tsd::scene::tokens::surface::material);
        REQUIRE(material != nullptr);
        REQUIRE(material != stage.scene.defaultMaterial().data());

        const auto color =
            material->parameterValueAs<tsd::math::float3>("color");
        REQUIRE(color.has_value());
        REQUIRE(color->x == Approx(1.f));
        REQUIRE(color->y == Approx(0.f));
      }

      THEN("Authored widths become per-point radii")
      {
        auto geometry = stage.scene.getObject<tsd::scene::Geometry>(0);
        auto *radii = geometry->parameterValueAsObject<tsd::scene::Array>(
            "vertex.radius");
        REQUIRE(radii != nullptr);
        REQUIRE(radii->size() == 2);
        REQUIRE(radii->dataAs<float>()[1] == Approx(0.2f));
      }
    }
  }
}

#endif // TSD_USE_USD
