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

#include "tsd/animation/AnimationManager.hpp"
#include "tsd/io/archives/AnimationManagerArchive.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/scene/Scene.hpp"
// std
#include <filesystem>
#include <fstream>
#include <vector>

namespace {

// Writes a text-format Stage to a temporary path for the lifetime of one
// scenario. USD's text format keeps every fixture readable next to the
// assertion it supports and keeps binary assets out of the repository.
struct StageFixture
{
  StageFixture(const char *name, const std::string &contents);
  ~StageFixture();

  std::string path() const;

 private:
  std::filesystem::path m_path;
};

// Inlined definitions ////////////////////////////////////////////////////////

inline StageFixture::StageFixture(const char *name, const std::string &contents)
    : m_path(std::filesystem::temp_directory_path() / name)
{
  std::ofstream file(m_path);
  file << contents;
}

inline StageFixture::~StageFixture()
{
  std::error_code ec;
  std::filesystem::remove(m_path, ec);
}

inline std::string StageFixture::path() const
{
  return m_path.string();
}

// A real, decodable texture for the lifetime of one scenario. The import binds
// samplers by loading these, so a stand-in file with arbitrary bytes would not
// exercise the path -- this is a 1x1 uncompressed true-colour TGA, the
// smallest thing the image loader accepts that can be written by hand.
struct TextureFixture
{
  explicit TextureFixture(const char *name);
  ~TextureFixture();

  std::string path() const;

 private:
  std::filesystem::path m_path;
};

inline TextureFixture::TextureFixture(const char *name)
    : m_path(std::filesystem::temp_directory_path() / name)
{
  const unsigned char tga[] = {
      0, // no image ID
      0, // no colour map
      2, // uncompressed true-colour
      0, 0, 0, 0, 0, // empty colour map spec
      0, 0, 0, 0, // origin
      1, 0, // width
      1, 0, // height
      24, // bits per pixel
      0, // descriptor
      0x20, 0x40, 0x60 // one BGR pixel
  };
  std::ofstream file(m_path, std::ios::binary);
  file.write(reinterpret_cast<const char *>(tga), sizeof(tga));
}

inline TextureFixture::~TextureFixture()
{
  std::error_code ec;
  std::filesystem::remove(m_path, ec);
}

inline std::string TextureFixture::path() const
{
  return m_path.string();
}

// Depth-first search for the first node whose name matches.
tsd::scene::LayerNodeRef findNode(tsd::scene::Layer *layer, const char *name)
{
  tsd::scene::LayerNodeRef found;
  layer->traverse(layer->root(), [&](auto &node, int) {
    if (!found && node->name() == name)
      found = layer->at(node.index());
    return true;
  });
  return found;
}

// The converted object a prim produced, found by the prim path the importer
// names it after.
template <typename T>
tsd::core::ObjectPoolRef<T> findObject(
    tsd::scene::Scene &scene, anari::DataType type, const char *name)
{
  for (size_t i = 0; i < scene.numberOfObjects(type); ++i) {
    auto object = scene.getObject<T>(i);
    if (object && object->name() == name)
      return object;
  }
  return {};
}

tsd::scene::GeometryRef findGeometry(tsd::scene::Scene &scene, const char *name)
{
  return findObject<tsd::scene::Geometry>(scene, ANARI_GEOMETRY, name);
}

constexpr const char *QUAD_MESH_BODY = R"(
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
)";

} // namespace

SCENARIO("A USD Stage's meshes arrive as surfaces", "[UsdImport]")
{
  GIVEN("A Stage with a single quad mesh")
  {
    StageFixture stage("tsd_test_usd_single_mesh.usda",
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

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("The Stage is imported")
    {
      auto report = tsd::io::import_USD(scene, animMgr, stage.path().c_str());

      THEN("The mesh becomes one triangle-geometry surface")
      {
        REQUIRE(report.stageOpened);
        REQUIRE(scene.numberOfObjects(ANARI_SURFACE) == 1);
        REQUIRE(scene.numberOfObjects(ANARI_GEOMETRY) == 1);

        auto geometry = scene.getObject<tsd::scene::Geometry>(0);
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
        REQUIRE(report.skipped.empty());
        REQUIRE(report.convertedPrims == 1);
      }
    }
  }
}

SCENARIO("A USD Stage's prim hierarchy is mirrored in the Layer", "[UsdImport]")
{
  GIVEN("A Stage nesting a mesh two Xforms deep")
  {
    StageFixture stage("tsd_test_usd_hierarchy.usda",
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

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("The Stage is imported")
    {
      auto report = tsd::io::import_USD(scene, animMgr, stage.path().c_str());
      auto *layer = scene.defaultLayer();

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
        REQUIRE(report.skipped.empty());
      }
    }
  }
}

SCENARIO(
    "Guide and proxy Purpose content is excluded by default", "[UsdImport]")
{
  GIVEN("A Stage with one mesh per Purpose")
  {
    StageFixture stage("tsd_test_usd_purpose.usda",
        std::string(R"(#usda 1.0

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
)");

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("The Stage is imported with default options")
    {
      auto report = tsd::io::import_USD(scene, animMgr, stage.path().c_str());

      THEN("Only default and render Purpose content arrives")
      {
        REQUIRE(scene.numberOfObjects(ANARI_SURFACE) == 2);
      }

      THEN("Each excluded prim is reported")
      {
        REQUIRE(report.countOf(tsd::io::UsdSkipReason::PURPOSE_EXCLUDED) == 2);
      }

      THEN("Excluded prims leave a disabled Placeholder Node")
      {
        auto *layer = scene.defaultLayer();
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
      auto report = tsd::io::import_USD(
          scene, animMgr, stage.path().c_str(), {}, options);

      THEN("Proxy content arrives too")
      {
        REQUIRE(scene.numberOfObjects(ANARI_SURFACE) == 3);
        REQUIRE(report.countOf(tsd::io::UsdSkipReason::PURPOSE_EXCLUDED) == 1);
      }
    }
  }
}

SCENARIO("Prims resolving to invisible import as disabled nodes", "[UsdImport]")
{
  GIVEN("A Stage with an invisible mesh")
  {
    StageFixture stage("tsd_test_usd_invisible.usda",
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

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("The Stage is imported")
    {
      auto report = tsd::io::import_USD(scene, animMgr, stage.path().c_str());

      THEN("The content still arrives so it can be toggled on")
      {
        REQUIRE(scene.numberOfObjects(ANARI_SURFACE) == 1);
      }

      THEN("Its node is disabled and the reason is reported")
      {
        auto *layer = scene.defaultLayer();
        auto hidden = findNode(layer, "Hidden");
        REQUIRE(hidden);
        REQUIRE_FALSE((*hidden)->isEnabled());
        REQUIRE(
            report.countOf(tsd::io::UsdSkipReason::RESOLVED_INVISIBLE) == 1);
      }
    }
  }
}

SCENARIO(
    "Prim types TSD cannot represent become Placeholder Nodes", "[UsdImport]")
{
  GIVEN("A Stage containing a prim type with no TSD equivalent")
  {
    StageFixture stage("tsd_test_usd_unsupported.usda",
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

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("The Stage is imported")
    {
      auto report = tsd::io::import_USD(scene, animMgr, stage.path().c_str());

      THEN("The supported content still arrives")
      {
        REQUIRE(scene.numberOfObjects(ANARI_SURFACE) == 1);
        REQUIRE(scene.numberOfObjects(ANARI_LIGHT) == 0);
      }

      THEN("The unsupported prim is named in the report")
      {
        REQUIRE(report.skipped.size() == 1);
        REQUIRE(report.skipped[0].primPath == "/World/Tube");
        REQUIRE(report.skipped[0].reason
            == tsd::io::UsdSkipReason::UNSUPPORTED_LIGHT_TYPE);
      }

      THEN("It leaves a disabled Placeholder Node where it belongs")
      {
        auto *layer = scene.defaultLayer();
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
    StageFixture stage("tsd_test_usd_subtree.usda",
        std::string(R"(#usda 1.0

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
)");

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("The import is pointed at one subtree")
    {
      tsd::io::UsdImportOptions options;
      options.primPath = "/AssetB";
      auto report = tsd::io::import_USD(
          scene, animMgr, stage.path().c_str(), {}, options);

      THEN("Only that subtree arrives")
      {
        REQUIRE(scene.numberOfObjects(ANARI_SURFACE) == 1);
        REQUIRE(report.skipped.empty());
        REQUIRE_FALSE(findNode(scene.defaultLayer(), "AssetA"));
        REQUIRE(findNode(scene.defaultLayer(), "AssetB"));
      }
    }
  }
}

SCENARIO(
    "Stage framing metadata is recorded for the application", "[UsdImport]")
{
  GIVEN("A Z-up Stage authored in centimetres")
  {
    StageFixture stage("tsd_test_usd_framing.usda",
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

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("The Stage is imported")
    {
      auto report = tsd::io::import_USD(scene, animMgr, stage.path().c_str());

      THEN("Up-axis and unit scale are recorded on the import root node")
      {
        auto *layer = scene.defaultLayer();
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
        auto geometry = scene.getObject<tsd::scene::Geometry>(0);
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

SCENARIO("A point instancer shares one set of Prototype objects", "[UsdImport]")
{
  GIVEN("A Stage scattering one Prototype three times, one of them hidden")
  {
    StageFixture stage("tsd_test_usd_point_instancer.usda", R"(#usda 1.0

def PointInstancer "Scatter"
{
    point3f[] positions = [(0, 0, 0), (2, 0, 0), (4, 0, 0)]
    int64[] ids = [0, 1, 2]
    int[] protoIndices = [0, 0, 0]
    int64[] invisibleIds = [1]
    rel prototypes = [</Scatter/Proto>]

    def Mesh "Proto"
    {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    }
}
)");

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("The Stage is imported")
    {
      auto report = tsd::io::import_USD(scene, animMgr, stage.path().c_str());

      THEN("The Prototype is imported once, not once per placement")
      {
        REQUIRE(scene.numberOfObjects(ANARI_SURFACE) == 1);
        REQUIRE(scene.numberOfObjects(ANARI_GEOMETRY) == 1);
      }

      THEN("The placements become a single transform-array node")
      {
        auto *layer = scene.defaultLayer();
        auto scatter = findNode(layer, "Scatter");
        REQUIRE(scatter);

        // The array node is the instancer's own child, holding the visible
        // placements only.
        tsd::scene::Array *transforms = nullptr;
        layer->traverse(scatter, [&](auto &node, int) {
          if (!transforms && node->type() == ANARI_ARRAY1D)
            transforms = node->getTransformArray();
          return true;
        });
        REQUIRE(transforms != nullptr);
        REQUIRE(transforms->size() == 2); // the invisible placement is omitted
      }

      THEN("Nothing is silently lost")
      {
        REQUIRE(report.skipped.empty());
      }
    }
  }
}

SCENARIO("USD Instances share objects across placements", "[UsdImport]")
{
  GIVEN("A Stage referencing one Prototype from two instanceable prims")
  {
    StageFixture stage("tsd_test_usd_native_instance.usda", R"(#usda 1.0

def Xform "Protos"
{
    def Xform "Asset"
    {
        def Mesh "Quad"
        {
            int[] faceVertexCounts = [3]
            int[] faceVertexIndices = [0, 1, 2]
            point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        }
    }
}

def Xform "InstanceA" (
    instanceable = true
    prepend references = </Protos/Asset>
)
{
    double3 xformOp:translate = (5, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]
}

def Xform "InstanceB" (
    instanceable = true
    prepend references = </Protos/Asset>
)
{
    double3 xformOp:translate = (9, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]
}
)");

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("The Stage is imported")
    {
      tsd::io::import_USD(scene, animMgr, stage.path().c_str());

      THEN("The Prototype geometry exists once, plus the un-instanced source")
      {
        // /Protos/Asset/Quad imports as ordinary content; the two placements
        // share a single converted Prototype rather than copying it.
        REQUIRE(scene.numberOfObjects(ANARI_SURFACE) == 2);
      }

      THEN("Each placement is a node referencing the shared objects")
      {
        auto *layer = scene.defaultLayer();
        auto a = findNode(layer, "InstanceA");
        auto b = findNode(layer, "InstanceB");
        REQUIRE(a);
        REQUIRE(b);

        auto sharedObjectUnder = [&](tsd::scene::LayerNodeRef parent) {
          size_t index = tsd::core::INVALID_INDEX;
          layer->traverse(parent, [&](auto &node, int) {
            if (index == tsd::core::INVALID_INDEX && node->isObject())
              index = node->getObjectIndex();
            return true;
          });
          return index;
        };

        const auto indexA = sharedObjectUnder(a);
        const auto indexB = sharedObjectUnder(b);
        REQUIRE(indexA != tsd::core::INVALID_INDEX);
        REQUIRE(indexA == indexB);
      }
    }
  }
}

SCENARIO("Per-face material subsets become several Surfaces", "[UsdImport]")
{
  GIVEN("A two-face mesh with one face bound to its own material")
  {
    StageFixture stage("tsd_test_usd_subsets.usda", R"(#usda 1.0

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

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("The Stage is imported")
    {
      tsd::io::import_USD(scene, animMgr, stage.path().c_str());

      THEN("There is one Surface per subset")
      {
        REQUIRE(scene.numberOfObjects(ANARI_SURFACE) >= 1);
      }

      THEN("The faces no subset claims are still drawn, by the mesh's material")
      {
        // The first face belongs to no subset, so it stays with the mesh's own
        // binding instead of vanishing with the un-surfaced parent geometry.
        auto leftover = findGeometry(scene, "/World/Strip");
        REQUIRE(leftover);
        auto *index = leftover->parameterValueAsObject<tsd::scene::Array>(
            "primitive.index");
        REQUIRE(index != nullptr);
        REQUIRE(index->size() == 2);
        REQUIRE(scene.numberOfObjects(ANARI_SURFACE) == 2);
      }

      THEN("The subset Surfaces share the mesh's vertex positions")
      {
        // Every geometry produced for this mesh points at the same
        // vertex.position Array; only the index arrays differ.
        std::vector<const tsd::scene::Array *> positions;
        for (size_t i = 0; i < scene.numberOfObjects(ANARI_GEOMETRY); ++i) {
          auto geometry = scene.getObject<tsd::scene::Geometry>(i);
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
    StageFixture stage("tsd_test_usd_subset_facevarying.usda", R"(#usda 1.0

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

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("The Stage is imported")
    {
      tsd::io::import_USD(scene, animMgr, stage.path().c_str());

      THEN("Each subset carries the corners of the faces it selected")
      {
        // Face-varying data is indexed by triangle corner, so a subset cannot
        // share the parent array the way vertex data can -- it must gather the
        // corners of its own triangles. Each quad triangulates to two
        // triangles, hence six corners, and the two faces' UVs are authored
        // into disjoint halves of the unit square so a mis-gather shows up.
        auto left = findGeometry(scene, "/World/Quad/Left");
        auto right = findGeometry(scene, "/World/Quad/Right");
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
        auto left = findGeometry(scene, "/World/Quad/Left");
        auto right = findGeometry(scene, "/World/Quad/Right");
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
    StageFixture stage("tsd_test_usd_triangulated_facevarying.usda", R"(#usda 1.0

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

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("The Stage is imported")
    {
      tsd::io::import_USD(scene, animMgr, stage.path().c_str());

      THEN("The UVs arrive flattened, one value per triangle corner")
      {
        auto geometry = findGeometry(scene, "/World/Triangles");
        REQUIRE(geometry);

        auto *uvs = geometry->parameterValueAsObject<tsd::scene::Array>(
            "faceVarying.attribute0");
        REQUIRE(uvs != nullptr);
        REQUIRE(uvs->size() == 6);

        const auto *uv = uvs->dataAs<tsd::math::float2>();
        REQUIRE(uv[3].x == Approx(0.0f)); // second triangle's first corner
        REQUIRE(uv[4].x == Approx(1.0f));
        REQUIRE(uv[5].x == Approx(0.0f));
        REQUIRE(uv[5].y == Approx(1.0f));
      }

      THEN("The normals arrive too")
      {
        auto geometry = findGeometry(scene, "/World/Triangles");
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
    StageFixture stage("tsd_test_usd_subset_uv_primvar.usda", R"(#usda 1.0

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

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("The Stage is imported")
    {
      tsd::io::import_USD(scene, animMgr, stage.path().c_str());

      THEN("Each subset's own primvar lands on its first attribute")
      {
        // The mesh itself binds no material, so the UV name cannot be decided
        // once for the whole mesh: each subset's material names its own.
        auto left = findGeometry(scene, "/World/Quad/Left");
        auto right = findGeometry(scene, "/World/Quad/Right");
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

SCENARIO("An unconventionally named UV primvar is still found", "[UsdImport]")
{
  GIVEN("A material whose reader node asks for a primvar not called 'st'")
  {
    StageFixture stage("tsd_test_usd_uv_primvar.usda", R"(#usda 1.0

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

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("The Stage is imported")
    {
      tsd::io::import_USD(scene, animMgr, stage.path().c_str());

      THEN("The named primvar lands on the geometry's first attribute")
      {
        auto geometry = scene.getObject<tsd::scene::Geometry>(0);
        REQUIRE(geometry);
        auto *uvs = geometry->parameterValueAsObject<tsd::scene::Array>(
            "vertex.attribute0");
        REQUIRE(uvs != nullptr);
        REQUIRE(uvs->size() == 4);
      }
    }
  }
}

SCENARIO("Native material passthrough is opt-in", "[UsdImport]")
{
  // The material a Surface actually uses, rather than whatever happens to sit
  // at index 0 of the pool (which is the Scene's own default material).
  auto boundMaterial = [](tsd::scene::Scene &scene) {
    auto surface = scene.getObject<tsd::scene::Surface>(0);
    REQUIRE(surface);
    auto *material = surface->parameterValueAsObject<tsd::scene::Material>(
        tsd::scene::tokens::surface::material);
    REQUIRE(material != nullptr);
    return material;
  };

  auto stringParameter = [](tsd::scene::Material *material, const char *name) {
    auto *p = material->parameter(name);
    return p ? p->value().getString() : std::string();
  };

  GIVEN("A Stage whose material is an ordinary preview surface")
  {
    StageFixture stage("tsd_test_usd_preview_material.usda", R"(#usda 1.0

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
)");

    WHEN("The default material mode is used")
    {
      tsd::scene::Scene scene;
      tsd::animation::AnimationManager animMgr(&scene);
      auto report = tsd::io::import_USD(scene, animMgr, stage.path().c_str());

      THEN("A portable physically-based material is emitted")
      {
        REQUIRE(boundMaterial(scene)->subtype()
            == tsd::scene::tokens::material::physicallyBased);
        REQUIRE(report.skipped.empty());
      }
    }

    WHEN("A native passthrough is asked for that this material cannot give")
    {
      tsd::scene::Scene scene;
      tsd::animation::AnimationManager animMgr(&scene);
      tsd::io::UsdImportOptions options;
      options.materialMode = tsd::io::UsdMaterialMode::MDL;
      auto report = tsd::io::import_USD(
          scene, animMgr, stage.path().c_str(), {}, options);

      THEN("The fallback to a portable mapping is reported, not silent")
      {
        REQUIRE(boundMaterial(scene)->subtype()
            == tsd::scene::tokens::material::physicallyBased);
        REQUIRE(
            report.countOf(tsd::io::UsdSkipReason::RICHER_MATERIAL_AVAILABLE)
            == 1);
      }
    }

#if TSD_USD_HAS_MATERIALX
    WHEN("MaterialX emission is asked for")
    {
      tsd::scene::Scene scene;
      tsd::animation::AnimationManager animMgr(&scene);
      tsd::io::UsdImportOptions options;
      options.materialMode = tsd::io::UsdMaterialMode::MATERIALX;
      auto report = tsd::io::import_USD(
          scene, animMgr, stage.path().c_str(), {}, options);

      THEN("A preview surface falls back rather than emitting a bad document")
      {
        // MaterialX has no node definition for UsdPreviewSurface, so there is
        // nothing to pass through; the portable mapping is used and said so.
        REQUIRE(boundMaterial(scene)->subtype()
            == tsd::scene::tokens::material::physicallyBased);
        REQUIRE(
            report.countOf(tsd::io::UsdSkipReason::RICHER_MATERIAL_AVAILABLE)
            == 1);
      }
    }
#endif
  }

#if TSD_USD_HAS_MATERIALX
  GIVEN("A Stage with an authored MaterialX network")
  {
    StageFixture stage("tsd_test_usd_materialx.usda", R"(#usda 1.0

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
)");

    WHEN("MaterialX emission is asked for")
    {
      tsd::scene::Scene scene;
      tsd::animation::AnimationManager animMgr(&scene);
      tsd::io::UsdImportOptions options;
      options.materialMode = tsd::io::UsdMaterialMode::MATERIALX;
      auto report = tsd::io::import_USD(
          scene, animMgr, stage.path().c_str(), {}, options);

      THEN("The network passes through as an inline MaterialX document")
      {
        auto *material = boundMaterial(scene);
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
        REQUIRE(report.skipped.empty());
      }
    }

    WHEN("The default material mode is used")
    {
      tsd::scene::Scene scene;
      tsd::animation::AnimationManager animMgr(&scene);
      tsd::io::import_USD(scene, animMgr, stage.path().c_str());

      THEN("The portable mapping is still what arrives")
      {
        REQUIRE(boundMaterial(scene)->subtype()
            != tsd::scene::tokens::material::materialx);
      }
    }

    // The importer's MaterialX mode is only reachable from an application
    // through an Importer Type, so the dispatch is worth pinning separately
    // from the option it sets.
    WHEN("The file is imported through the USD_MATX Importer Type")
    {
      tsd::scene::Scene scene;
      tsd::animation::AnimationManager animMgr(&scene);
      tsd::io::import_file(scene,
          animMgr,
          {tsd::io::ImporterType::USD_MATX, stage.path()});

      THEN("MaterialX materials arrive without asking for options")
      {
        REQUIRE(boundMaterial(scene)->subtype()
            == tsd::scene::tokens::material::materialx);
      }
    }

    WHEN("The file is imported through the plain USD Importer Type")
    {
      tsd::scene::Scene scene;
      tsd::animation::AnimationManager animMgr(&scene);
      tsd::io::import_file(scene,
          animMgr,
          {tsd::io::ImporterType::USD, stage.path()});

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

    StageFixture stage("tsd_test_usd_materialx_textures.usda", R"(#usda 1.0

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
)");

    WHEN("MaterialX emission is asked for")
    {
      tsd::scene::Scene scene;
      tsd::animation::AnimationManager animMgr(&scene);
      tsd::io::UsdImportOptions options;
      options.materialMode = tsd::io::UsdMaterialMode::MATERIALX;
      auto report = tsd::io::import_USD(
          scene, animMgr, stage.path().c_str(), {}, options);

      const auto source =
          stringParameter(boundMaterial(scene), "source");

      THEN("Texture paths leave as absolute paths")
      {
        REQUIRE(source.find(present.path()) != std::string::npos);
        REQUIRE(source.find("\"tsd_test_usd_mtlx_present.tga\"")
            == std::string::npos);
      }

      THEN("A tile set is anchored without losing its token")
      {
        const auto tiled =
            (std::filesystem::temp_directory_path() / "tiles").string();
        REQUIRE(source.find(tiled) != std::string::npos);
        REQUIRE(source.find("<UDIM>") != std::string::npos);
      }

      THEN("The texture that exists is not reported as missing")
      {
        for (const auto &skip : report.skipped) {
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
        REQUIRE(scene.numberOfObjects(ANARI_SAMPLER) == 1);

        // The name is the contract: the device publishes each textured input
        // under its MaterialX element path.
        auto *material = boundMaterial(scene);
        std::string boundName;
        for (size_t i = 0; i < material->numParameters(); i++) {
          if (material->parameterAt(i).value().type() == ANARI_SAMPLER)
            boundName = material->parameterNameAt(i);
        }
        REQUIRE_FALSE(boundName.empty());
        // The document path, node graph included -- the same string the device's
        // shader generator reports as the port's path.
        REQUIRE(boundName == "_/Present/file");
      }

      THEN("A tile set binds nothing, and says so")
      {
        REQUIRE(report.contains(tsd::io::UsdSkipReason::TEXTURE_LOAD_FAILED));
        // Only the one loadable texture became a sampler.
        REQUIRE(scene.numberOfObjects(ANARI_SAMPLER) == 1);
      }
    }
  }

  GIVEN("A MaterialX network naming a texture that is not there")
  {
    StageFixture stage("tsd_test_usd_materialx_missing.usda", R"(#usda 1.0

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
)");

    WHEN("MaterialX emission is asked for")
    {
      tsd::scene::Scene scene;
      tsd::animation::AnimationManager animMgr(&scene);
      tsd::io::UsdImportOptions options;
      options.materialMode = tsd::io::UsdMaterialMode::MATERIALX;
      auto report = tsd::io::import_USD(
          scene, animMgr, stage.path().c_str(), {}, options);

      THEN("The Import Report names it rather than leaving it to the device")
      {
        REQUIRE(report.contains(tsd::io::UsdSkipReason::TEXTURE_LOAD_FAILED));
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
    StageFixture stage("tsd_test_usd_materialx_mistyped.usda", R"(#usda 1.0

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
)");

    WHEN("MaterialX emission is asked for")
    {
      tsd::scene::Scene scene;
      tsd::animation::AnimationManager animMgr(&scene);
      tsd::io::UsdImportOptions options;
      options.materialMode = tsd::io::UsdMaterialMode::MATERIALX;
      auto report = tsd::io::import_USD(
          scene, animMgr, stage.path().c_str(), {}, options);

      THEN("The Import Report names it rather than leaving it to the device")
      {
        REQUIRE(
            report.contains(tsd::io::UsdSkipReason::MATERIAL_RESOLUTION_FAILED));
      }

      THEN("The portable mapping is what arrives, not a document")
      {
        REQUIRE(boundMaterial(scene)->subtype()
            != tsd::scene::tokens::material::materialx);
      }
    }
  }
#endif
}

SCENARIO(
    "A prim with no bound material takes its display colour", "[UsdImport]")
{
  GIVEN("A mesh with display colour and opacity but no material")
  {
    StageFixture stage("tsd_test_usd_display_color.usda", R"(#usda 1.0

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

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("The Stage is imported")
    {
      tsd::io::import_USD(scene, animMgr, stage.path().c_str());

      THEN("The Surface's material carries the display values")
      {
        auto surface = scene.getObject<tsd::scene::Surface>(0);
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

SCENARIO(
    "Light exposure and normalization reach the emitted light", "[UsdImport]")
{
  GIVEN("A sphere light with exposure and normalization set")
  {
    // intensity 4, exposure 2 -> 4 * 2^2 = 16; normalize divides by the
    // sphere's area, 4*pi*r^2 with r = 2 -> 16 / (16*pi).
    StageFixture stage("tsd_test_usd_light_radiometry.usda", R"(#usda 1.0

def SphereLight "Lamp"
{
    float inputs:intensity = 4
    float inputs:exposure = 2
    bool inputs:normalize = true
    float inputs:radius = 2
    color3f inputs:color = (1, 1, 1)
}
)");

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("The Stage is imported")
    {
      auto report = tsd::io::import_USD(scene, animMgr, stage.path().c_str());

      THEN("The light's intensity accounts for both")
      {
        REQUIRE(scene.numberOfObjects(ANARI_LIGHT) == 1);
        auto light = scene.getObject<tsd::scene::Light>(0);
        REQUIRE(light);
        REQUIRE(light->subtype() == tsd::scene::tokens::light::point);

        const auto intensity = light->parameterValueAs<float>("intensity");
        REQUIRE(intensity.has_value());
        const float expected = 16.f / (4.f * float(M_PI) * 4.f);
        REQUIRE(*intensity == Approx(expected));
        REQUIRE(report.skipped.empty());
      }
    }
  }
}

SCENARIO("A shaped sphere light becomes a spot light", "[UsdImport]")
{
  GIVEN("A sphere light carrying shaping attributes")
  {
    StageFixture stage("tsd_test_usd_spot.usda", R"(#usda 1.0

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

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("The Stage is imported")
    {
      tsd::io::import_USD(scene, animMgr, stage.path().c_str());

      THEN("Spot lighting survives the import")
      {
        auto light = scene.getObject<tsd::scene::Light>(0);
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

SCENARIO("Analytic quadrics stay analytic", "[UsdImport]")
{
  GIVEN("A Stage with a sphere and a cylinder")
  {
    StageFixture stage("tsd_test_usd_quadrics.usda", R"(#usda 1.0

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

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("The Stage is imported")
    {
      auto report = tsd::io::import_USD(scene, animMgr, stage.path().c_str());

      THEN("They map onto TSD's native quadric geometry, not meshes")
      {
        REQUIRE(scene.numberOfObjects(ANARI_GEOMETRY) == 2);

        auto ball = scene.getObject<tsd::scene::Geometry>(0);
        REQUIRE(ball->subtype() == tsd::scene::tokens::geometry::sphere);
        REQUIRE(ball->parameterValueAs<float>("radius").value() == Approx(2.f));

        auto tube = scene.getObject<tsd::scene::Geometry>(1);
        REQUIRE(tube->subtype() == tsd::scene::tokens::geometry::cylinder);

        // The spine axis is folded into the endpoints rather than a transform.
        auto *positions =
            tube->parameterValueAsObject<tsd::scene::Array>("vertex.position");
        REQUIRE(positions != nullptr);
        REQUIRE(positions->size() == 2);
        const auto *p = positions->dataAs<tsd::math::float3>();
        REQUIRE(p[0].y == Approx(-2.f));
        REQUIRE(p[1].y == Approx(2.f));
        REQUIRE(report.skipped.empty());
      }
    }
  }
}

SCENARIO(
    "A non-convex polygon tessellates without spurious geometry", "[UsdImport]")
{
  GIVEN("A mesh with one concave five-sided face")
  {
    StageFixture stage("tsd_test_usd_nonconvex.usda", R"(#usda 1.0

def Mesh "Arrow"
{
    int[] faceVertexCounts = [5]
    int[] faceVertexIndices = [0, 1, 2, 3, 4]
    point3f[] points = [(0, 0, 0), (2, 0, 0), (2, 2, 0), (1, 1, 0), (0, 2, 0)]
}
)");

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("The Stage is imported")
    {
      tsd::io::import_USD(scene, animMgr, stage.path().c_str());

      THEN("It becomes exactly n-2 triangles")
      {
        auto geometry = scene.getObject<tsd::scene::Geometry>(0);
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
    StageFixture stage("tsd_test_usd_subdiv.usda", R"(#usda 1.0

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
)");

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("The Stage is imported at the default refinement level")
    {
      tsd::io::import_USD(scene, animMgr, stage.path().c_str());

      THEN("The subdivision mesh gains vertices and the polygon mesh does not")
      {
        auto subdiv = scene.getObject<tsd::scene::Geometry>(0);
        auto polygon = scene.getObject<tsd::scene::Geometry>(1);
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
      tsd::io::import_USD(scene, animMgr, stage.path().c_str(), {}, options);

      THEN("The subdivision mesh arrives at its authored resolution")
      {
        auto subdiv = scene.getObject<tsd::scene::Geometry>(0);
        auto *p = subdiv->parameterValueAsObject<tsd::scene::Array>(
            "vertex.position");
        REQUIRE(p != nullptr);
        REQUIRE(p->size() == 8);
      }
    }
  }
}

SCENARIO("Animation is captured at the times actually authored", "[UsdImport]")
{
  GIVEN("A Stage with transforms keyed on a non-uniform time base")
  {
    StageFixture stage("tsd_test_usd_time_base.usda", R"(#usda 1.0
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

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("The Stage is imported")
    {
      tsd::io::import_USD(scene, animMgr, stage.path().c_str());

      THEN("The binding's time base mirrors the authored sample spacing")
      {
        REQUIRE(animMgr.animations().size() == 1);
        const auto &bindings = animMgr.animations()[0].transformBindings();
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
    StageFixture stage("tsd_test_usd_full_turn.usda", R"(#usda 1.0
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

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("The Stage is imported")
    {
      tsd::io::import_USD(scene, animMgr, stage.path().c_str());

      THEN("Extra samples are inserted so the rotation still animates")
      {
        REQUIRE(animMgr.animations().size() == 1);
        const auto &bindings = animMgr.animations()[0].transformBindings();
        REQUIRE(bindings.size() == 1);
        REQUIRE(bindings[0].sampleCount() > 2);
      }
    }
  }

  GIVEN("A prim translating between two keyframes")
  {
    StageFixture stage("tsd_test_usd_small_move.usda", R"(#usda 1.0
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

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("The Stage is imported")
    {
      tsd::io::import_USD(scene, animMgr, stage.path().c_str());

      THEN("No extra samples are invented")
      {
        const auto &bindings = animMgr.animations()[0].transformBindings();
        REQUIRE(bindings[0].sampleCount() == 2);
      }
    }
  }
}

SCENARIO("Cameras from a Stage arrive in the camera pool", "[UsdImport]")
{
  GIVEN("A Stage with an animated camera rig")
  {
    StageFixture stage("tsd_test_usd_camera.usda", R"(#usda 1.0
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

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);
    const auto camerasBefore = scene.numberOfObjects(ANARI_CAMERA);

    WHEN("The Stage is imported")
    {
      tsd::io::import_USD(scene, animMgr, stage.path().c_str());

      THEN("The authored viewpoint is available and animated")
      {
        REQUIRE(scene.numberOfObjects(ANARI_CAMERA) == camerasBefore + 1);

        bool hasCameraAnimation = false;
        for (const auto &animation : animMgr.animations()) {
          if (!animation.objectParameterBindings().empty())
            hasCameraAnimation = true;
        }
        REQUIRE(hasCameraAnimation);
      }
    }
  }
}

SCENARIO(
    "Refinement carries face-varying primvars with the surface", "[UsdImport]")
{
  GIVEN("A subdivision mesh whose UVs are authored per face corner")
  {
    StageFixture stage("tsd_test_usd_subdiv_uvs.usda", R"(#usda 1.0

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

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("The Stage is imported")
    {
      auto report = tsd::io::import_USD(scene, animMgr, stage.path().c_str());

      THEN("The UVs survive refinement rather than being dropped")
      {
        auto geometry = scene.getObject<tsd::scene::Geometry>(0);
        REQUIRE(geometry);
        auto *uvs = geometry->parameterValueAsObject<tsd::scene::Array>(
            "faceVarying.attribute0");
        REQUIRE(uvs != nullptr);
        REQUIRE(uvs->size() > 4);
      }

      THEN("Nothing is reported as lost")
      {
        REQUIRE(report.skipped.empty());
      }
    }
  }
}

SCENARIO("Analytic prims without a material take their display colour",
    "[UsdImport]")
{
  GIVEN("A point cloud carrying only display colour")
  {
    StageFixture stage("tsd_test_usd_points_display_color.usda", R"(#usda 1.0

def Points "Cloud"
{
    point3f[] points = [(0, 0, 0), (1, 0, 0)]
    float[] widths = [0.2, 0.4]
    color3f[] primvars:displayColor = [(1, 0, 0)] (
        interpolation = "constant"
    )
}
)");

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("The Stage is imported")
    {
      tsd::io::import_USD(scene, animMgr, stage.path().c_str());

      THEN("Its material carries the display colour, not TSD's default")
      {
        auto surface = scene.getObject<tsd::scene::Surface>(0);
        REQUIRE(surface);
        auto *material = surface->parameterValueAsObject<tsd::scene::Material>(
            tsd::scene::tokens::surface::material);
        REQUIRE(material != nullptr);
        REQUIRE(material != scene.defaultMaterial().data());

        const auto color =
            material->parameterValueAs<tsd::math::float3>("color");
        REQUIRE(color.has_value());
        REQUIRE(color->x == Approx(1.f));
        REQUIRE(color->y == Approx(0.f));
      }

      THEN("Authored widths become per-point radii")
      {
        auto geometry = scene.getObject<tsd::scene::Geometry>(0);
        auto *radii = geometry->parameterValueAsObject<tsd::scene::Array>(
            "vertex.radius");
        REQUIRE(radii != nullptr);
        REQUIRE(radii->size() == 2);
        REQUIRE(radii->dataAs<float>()[1] == Approx(0.2f));
      }
    }
  }
}

SCENARIO("A prim that resets the transform stack ignores its ancestors",
    "[UsdImport]")
{
  GIVEN("A child that resets the transform stack under a moved parent")
  {
    StageFixture stage("tsd_test_usd_xform_reset.usda", R"(#usda 1.0

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

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("The Stage is imported")
    {
      tsd::io::import_USD(scene, animMgr, stage.path().c_str());

      THEN("Composing parent and child lands where USD puts the child")
      {
        auto *layer = scene.defaultLayer();
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
    StageFixture stage("tsd_test_usd_animated_visibility.usda", R"(#usda 1.0
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

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("The Stage is imported")
    {
      auto report = tsd::io::import_USD(scene, animMgr, stage.path().c_str());

      THEN("The caller is told the animation was not represented")
      {
        REQUIRE(
            report.countOf(tsd::io::UsdSkipReason::TIME_VARYING_VALUE_DROPPED)
            == 1);
      }
    }
  }
}

SCENARIO(
    "Lazily-bound deforming geometry survives save and reload", "[UsdImport]")
{
  GIVEN("A Stage whose mesh points are time-sampled")
  {
    StageFixture stage("tsd_test_usd_deforming.usda", R"(#usda 1.0
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

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("The Stage is imported")
    {
      tsd::io::import_USD(scene, animMgr, stage.path().c_str());

      THEN("Only one frame is eager; the rest is bound to the Stage")
      {
        REQUIRE(animMgr.animations().size() == 1);
        REQUIRE(animMgr.animations()[0].fileBindings().size() == 1);
        REQUIRE(
            animMgr.animations()[0].fileBindings()[0]->kind() == "usdGeometry");

        auto geometry = scene.getObject<tsd::scene::Geometry>(0);
        auto *positions = geometry->parameterValueAsObject<tsd::scene::Array>(
            "vertex.position");
        REQUIRE(positions != nullptr);
        REQUIRE(positions->size() == 3);
      }

      THEN("Scrubbing pulls a later frame from the Stage")
      {
        animMgr.setAnimationTime(1.0f);

        auto geometry = scene.getObject<tsd::scene::Geometry>(0);
        auto *positions = geometry->parameterValueAsObject<tsd::scene::Array>(
            "vertex.position");
        REQUIRE(positions != nullptr);
        REQUIRE(positions->dataAs<tsd::math::float3>()[1].x == Approx(3.f));
      }

      THEN("The binding reconstructs from an Archive")
      {
        tsd::core::DataTree tree;
        REQUIRE(
            tsd::io::serialize_AnimationManagerArchive(animMgr, tree.root()));

        tsd::animation::AnimationManager restored(&scene);
        REQUIRE(tsd::io::deserialize_AnimationManagerArchive(
            restored, tree.root()));
        REQUIRE(restored.animations().size() == 1);
        REQUIRE(restored.animations()[0].fileBindings().size() == 1);
        REQUIRE(restored.animations()[0].fileBindings()[0]->kind()
            == "usdGeometry");

        restored.setAnimationTime(1.0f);
        auto geometry = scene.getObject<tsd::scene::Geometry>(0);
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
    StageFixture stage("tsd_test_usd_dialect.usda", R"(#usda 1.0

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

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("The Stage is imported")
    {
      tsd::io::import_USD(scene, animMgr, stage.path().c_str());

      THEN("The carrier prim does not also arrive as generic geometry")
      {
        // Only the ordinary mesh converts: the claimed subtree is pruned from
        // the resolved scene entirely.
        REQUIRE(scene.numberOfObjects(ANARI_SURFACE) == 1);
        REQUIRE_FALSE(findNode(scene.defaultLayer(), "part_one"));
      }
    }
  }
}

#endif // TSD_USE_USD

