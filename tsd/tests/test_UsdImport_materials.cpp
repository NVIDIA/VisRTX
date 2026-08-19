// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// Native material passthrough: which Render Context wins, and when.

#if TSD_USE_USD

// catch
#include "catch.hpp"
// tsd_tests
#include "UsdTestFixtures.h"

SCENARIO("Native material passthrough is opt-in", "[UsdImport]")
{
  auto stringParameter = [](tsd::scene::Material *material, const char *name) {
    auto *p = material->parameter(name);
    return p ? p->value().getString() : std::string();
  };

  GIVEN("A Stage whose material is an ordinary preview surface")
  {
    const std::string previewSurface = R"(#usda 1.0

def Xform "World"
{
    def Material "Surface"
    {
        token outputs:surface.connect = </World/Surface/PBR.outputs:surface>
        def Shader "PBR"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.8, 0.2, 0.1)
            float inputs:roughness = 0.4
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
        rel material:binding = </World/Surface>
    }
}
)";

    WHEN("The default material mode is used")
    {
      ImportedStage stage("tsd_test_usd_preview_material.usda", previewSurface);

      THEN("A portable physically-based material is emitted")
      {
        REQUIRE(boundMaterial(stage.scene)->subtype()
            == tsd::scene::tokens::material::physicallyBased);
        REQUIRE(stage.report.skipped.empty());
      }
    }

    WHEN("A native passthrough is asked for that this material cannot give")
    {
      tsd::io::UsdImportOptions options;
      options.materialMode = tsd::io::UsdMaterialMode::MDL;
      ImportedStage stage(
          "tsd_test_usd_preview_material.usda", previewSurface, options);

      THEN("The fallback to a portable mapping is reported, not silent")
      {
        REQUIRE(boundMaterial(stage.scene)->subtype()
            == tsd::scene::tokens::material::physicallyBased);
        REQUIRE(stage.report.countOf(
                    tsd::io::UsdSkipReason::RICHER_MATERIAL_AVAILABLE)
            == 1);
      }
    }

#if TSD_USD_HAS_MATERIALX
    WHEN("MaterialX emission is asked for")
    {
      tsd::io::UsdImportOptions options;
      options.materialMode = tsd::io::UsdMaterialMode::MATERIALX;
      ImportedStage stage(
          "tsd_test_usd_preview_material.usda", previewSurface, options);

      THEN("A preview surface falls back rather than emitting a bad document")
      {
        // MaterialX has no node definition for UsdPreviewSurface, so there is
        // nothing to pass through; the portable mapping is used and said so.
        REQUIRE(boundMaterial(stage.scene)->subtype()
            == tsd::scene::tokens::material::physicallyBased);
        REQUIRE(stage.report.countOf(
                    tsd::io::UsdSkipReason::RICHER_MATERIAL_AVAILABLE)
            == 1);
      }
    }
#endif
  }

#if TSD_USD_HAS_MATERIALX
  GIVEN("A Stage with an authored MaterialX network")
  {
    const std::string materialxNetwork = R"(#usda 1.0

def Xform "World"
{
    def Material "Surface"
    {
        token outputs:mtlx:surface.connect = </World/Surface/Standard.outputs:surface>

        def Shader "Standard"
        {
            uniform token info:id = "ND_standard_surface_surfaceshader"
            color3f inputs:base_color = (0.8, 0.2, 0.1)
            float inputs:specular_roughness = 0.4
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
        rel material:binding = </World/Surface>
    }
}
)";

    WHEN("MaterialX emission is asked for")
    {
      tsd::io::UsdImportOptions options;
      options.materialMode = tsd::io::UsdMaterialMode::MATERIALX;
      ImportedStage stage(
          "tsd_test_usd_materialx.usda", materialxNetwork, options);

      THEN("The network passes through as an inline MaterialX document")
      {
        auto *material = boundMaterial(stage.scene);
        REQUIRE(material->subtype() == tsd::scene::tokens::material::materialx);
        REQUIRE(stringParameter(material, "sourceType") == "documentInline");

        const auto source = stringParameter(material, "source");
        REQUIRE(source.find("<materialx") != std::string::npos);
        REQUIRE(source.find("standard_surface") != std::string::npos);

        const auto materialName = stringParameter(material, "materialName");
        REQUIRE_FALSE(materialName.empty());
        REQUIRE(source.find(materialName) != std::string::npos);
      }

      THEN("Nothing is reported as lost")
      {
        REQUIRE(stage.report.skipped.empty());
      }
    }

    WHEN("The default material mode is used")
    {
      ImportedStage stage("tsd_test_usd_materialx.usda", materialxNetwork);

      THEN("The portable mapping is still what arrives")
      {
        REQUIRE(boundMaterial(stage.scene)->subtype()
            != tsd::scene::tokens::material::materialx);
      }
    }

    // The importer's MaterialX mode is only reachable from an application
    // through an Importer Type, so the dispatch is worth pinning separately
    // from the option it sets.
    WHEN("The file is imported through the USD_MTLX Importer Type")
    {
      StageFixture stage("tsd_test_usd_materialx.usda", materialxNetwork);
      tsd::scene::Scene scene;
      tsd::animation::AnimationManager animMgr(&scene);
      tsd::io::import_file(
          scene, animMgr, {tsd::io::ImporterType::USD_MTLX, stage.path()});

      THEN("MaterialX materials arrive without asking for options")
      {
        REQUIRE(boundMaterial(scene)->subtype()
            == tsd::scene::tokens::material::materialx);
      }
    }

    WHEN("The file is imported through the plain USD Importer Type")
    {
      StageFixture stage("tsd_test_usd_materialx.usda", materialxNetwork);
      tsd::scene::Scene scene;
      tsd::animation::AnimationManager animMgr(&scene);
      tsd::io::import_file(
          scene, animMgr, {tsd::io::ImporterType::USD, stage.path()});

      THEN("The portable mapping is what arrives")
      {
        REQUIRE(boundMaterial(scene)->subtype()
            != tsd::scene::tokens::material::materialx);
      }
    }
  }

  // An inline document has no file of its own for a relative path to be
  // relative to, so a texture that stays relative is a texture the device
  // cannot open.
  GIVEN("A MaterialX network reading textures by relative path")
  {
    TextureFixture present("tsd_test_usd_mtlx_present.tga");

    const std::string texturedNetwork = R"(#usda 1.0

def Xform "World"
{
    def Material "Surface"
    {
        token outputs:mtlx:surface.connect = </World/Surface/Standard.outputs:surface>

        def Shader "Present"
        {
            uniform token info:id = "ND_image_color3"
            asset inputs:file = @tsd_test_usd_mtlx_present.tga@
            color3f outputs:out
        }

        def Shader "Tiled"
        {
            uniform token info:id = "ND_image_color3"
            asset inputs:file = @tiles/tsd_test_tile.<UDIM>.png@
            color3f outputs:out
        }

        def Shader "Standard"
        {
            uniform token info:id = "ND_standard_surface_surfaceshader"
            color3f inputs:base_color.connect = </World/Surface/Present.outputs:out>
            color3f inputs:coat_color.connect = </World/Surface/Tiled.outputs:out>
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
        rel material:binding = </World/Surface>
    }
}
)";

    WHEN("MaterialX emission is asked for")
    {
      tsd::io::UsdImportOptions options;
      options.materialMode = tsd::io::UsdMaterialMode::MATERIALX;
      ImportedStage stage(
          "tsd_test_usd_materialx_textures.usda", texturedNetwork, options);

      const auto source = stringParameter(boundMaterial(stage.scene), "source");

      THEN("Texture paths leave as absolute paths")
      {
        REQUIRE(source.find(present.path()) != std::string::npos);
        REQUIRE(source.find("\"tsd_test_usd_mtlx_present.tga\"")
            == std::string::npos);
      }

      THEN("A tile set is anchored without losing its token")
      {
        const auto tiled = (fixtureDirectory() / "tiles").string();
        REQUIRE(source.find(tiled) != std::string::npos);
        REQUIRE(source.find("<UDIM>") != std::string::npos);
      }

      THEN("The texture that exists is not reported as missing")
      {
        for (const auto &skip : stage.report.skipped) {
          const bool missedThisOne =
              skip.reason == tsd::io::UsdSkipReason::TEXTURE_LOAD_FAILED
              && skip.detail == present.path();
          REQUIRE_FALSE(missedThisOne);
        }
      }

      // The device reads texels from samplers bound to the document's
      // `filename` inputs by their document path, not by opening the files
      // itself, so a material without them renders untextured however correct
      // its paths are.
      THEN("A sampler is bound to the input by its document path")
      {
        REQUIRE(stage.scene.numberOfObjects(ANARI_SAMPLER) == 1);

        // The name is the contract: the device publishes each textured input
        // under its MaterialX element path.
        auto *material = boundMaterial(stage.scene);
        std::string boundName;
        for (size_t i = 0; i < material->numParameters(); i++) {
          if (material->parameterAt(i).value().type() == ANARI_SAMPLER)
            boundName = material->parameterNameAt(i);
        }
        REQUIRE_FALSE(boundName.empty());
        // The document path, node graph included -- the same string the
        // device's shader generator reports as the port's path.
        REQUIRE(boundName == "_/Present/file");
      }

      THEN("A tile set binds nothing, and says so")
      {
        REQUIRE(
            stage.report.contains(tsd::io::UsdSkipReason::TEXTURE_LOAD_FAILED));
        // Only the one loadable texture became a sampler.
        REQUIRE(stage.scene.numberOfObjects(ANARI_SAMPLER) == 1);
      }
    }
  }

  GIVEN("A MaterialX network naming a texture that is not there")
  {
    const std::string missingTexture = R"(#usda 1.0

def Xform "World"
{
    def Material "Surface"
    {
        token outputs:mtlx:surface.connect = </World/Surface/Standard.outputs:surface>

        def Shader "Missing"
        {
            uniform token info:id = "ND_image_color3"
            asset inputs:file = @tsd_test_usd_absent.png@
            color3f outputs:out
        }

        def Shader "Standard"
        {
            uniform token info:id = "ND_standard_surface_surfaceshader"
            color3f inputs:base_color.connect = </World/Surface/Missing.outputs:out>
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
        rel material:binding = </World/Surface>
    }
}
)";

    WHEN("MaterialX emission is asked for")
    {
      tsd::io::UsdImportOptions options;
      options.materialMode = tsd::io::UsdMaterialMode::MATERIALX;
      ImportedStage stage(
          "tsd_test_usd_materialx_missing.usda", missingTexture, options);

      THEN("The Import Report names it rather than leaving it to the device")
      {
        REQUIRE(
            stage.report.contains(tsd::io::UsdSkipReason::TEXTURE_LOAD_FAILED));
      }
    }
  }

  // MaterialX matches a node to its definition on the exact set of inputs, so
  // a connection between mismatched types leaves the surface node resolving to
  // nothing. Emitting it anyway puts the failure inside the device, where it
  // reads as `Could not find a nodedef for node 'Surface'` and the prim
  // silently renders with the default material.
  GIVEN("A MaterialX network connecting a color3 output to a float input")
  {
    const std::string mistypedNetwork = R"(#usda 1.0

def Xform "World"
{
    def Material "Surface"
    {
        token outputs:mtlx:surface.connect = </World/Surface/Standard.outputs:surface>

        def Shader "Tint"
        {
            uniform token info:id = "ND_constant_color3"
            color3f inputs:value = (0.25, 0.5, 0.75)
            color3f outputs:out
        }

        def Shader "Standard"
        {
            uniform token info:id = "ND_standard_surface_surfaceshader"
            float inputs:specular_roughness.connect = </World/Surface/Tint.outputs:out>
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
        rel material:binding = </World/Surface>
    }
}
)";

    WHEN("MaterialX emission is asked for")
    {
      tsd::io::UsdImportOptions options;
      options.materialMode = tsd::io::UsdMaterialMode::MATERIALX;
      ImportedStage stage(
          "tsd_test_usd_materialx_mistyped.usda", mistypedNetwork, options);

      THEN("The Import Report names it rather than leaving it to the device")
      {
        REQUIRE(stage.report.contains(
            tsd::io::UsdSkipReason::MATERIAL_RESOLUTION_FAILED));
      }

      THEN("The portable mapping is what arrives, not a document")
      {
        REQUIRE(boundMaterial(stage.scene)->subtype()
            != tsd::scene::tokens::material::materialx);
      }
    }
  }
#endif
}

#endif // TSD_USE_USD
