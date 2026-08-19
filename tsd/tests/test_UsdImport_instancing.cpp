// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// Point instancers and native USD instances, static and animated.

#if TSD_USE_USD

// catch
#include "catch.hpp"
// tsd_tests
#include "UsdTestFixtures.h"
// tsd
#include "tsd/core/DataTree.hpp"
#include "tsd/io/archives/AnimationManagerArchive.hpp"

SCENARIO("A point instancer shares one set of Prototype objects", "[UsdImport]")
{
  GIVEN("A Stage scattering one Prototype three times, one of them hidden")
  {
    ImportedStage stage("tsd_test_usd_point_instancer.usda", R"(#usda 1.0

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

    WHEN("The Stage is imported")
    {
      THEN("The Prototype is imported once, not once per placement")
      {
        REQUIRE(stage.scene.numberOfObjects(ANARI_SURFACE) == 1);
        REQUIRE(stage.scene.numberOfObjects(ANARI_GEOMETRY) == 1);
      }

      THEN("The placements become a single transform-array node")
      {
        auto *layer = stage.scene.defaultLayer();
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
        REQUIRE(stage.report.skipped.empty());
      }
    }
  }
}

SCENARIO("USD Instances share objects across placements", "[UsdImport]")
{
  GIVEN("A Stage referencing one Prototype from two instanceable prims")
  {
    ImportedStage stage("tsd_test_usd_native_instance.usda", R"(#usda 1.0

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

    WHEN("The Stage is imported")
    {
      THEN("The Prototype geometry exists once, plus the un-instanced source")
      {
        // /Protos/Asset/Quad imports as ordinary content; the two placements
        // share a single converted Prototype rather than copying it.
        REQUIRE(stage.scene.numberOfObjects(ANARI_SURFACE) == 2);
      }

      THEN("Each placement is a node referencing the shared objects")
      {
        auto *layer = stage.scene.defaultLayer();
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

namespace {

// The transform Array of the first transform-array node beneath `name`.
tsd::scene::Array *findTransformArray(
    tsd::scene::Layer *layer, const char *name)
{
  auto parent = findNode(layer, name);
  if (!parent)
    return nullptr;
  tsd::scene::Array *found = nullptr;
  layer->traverse(parent, [&](auto &node, int) {
    if (!found && node->type() == ANARI_ARRAY1D)
      found = node->getTransformArray();
    return true;
  });
  return found;
}

} // namespace

SCENARIO("A point instancer's placements follow the Stage clock", "[UsdImport]")
{
  GIVEN("A PointInstancer whose positions and scales are time-sampled")
  {
    ImportedStage stage("tsd_test_usd_animated_instancer.usda", R"(#usda 1.0
(
    startTimeCode = 0
    endTimeCode = 2
)

def PointInstancer "Swarm"
{
    point3f[] positions.timeSamples = {
        0: [(0, 0, 0), (1, 0, 0), (2, 0, 0)],
        2: [(0, 0, 0), (10, 0, 0), (20, 0, 0)],
    }
    float3[] scales.timeSamples = {
        0: [(1, 1, 1), (1, 1, 1), (1, 1, 1)],
        2: [(2, 2, 2), (2, 2, 2), (2, 2, 2)],
    }
    int[] protoIndices = [0, 0, 0]
    rel prototypes = [</Swarm/Proto>]

    def Mesh "Proto"
    {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    }
}
)");

    WHEN("The Stage is imported")
    {
      THEN("One Animation holds the instancer's binding")
      {
        REQUIRE(stage.animMgr.animations().size() == 1);
        REQUIRE(stage.animMgr.animations()[0].fileBindings().size() == 1);
        REQUIRE(stage.animMgr.animations()[0].fileBindings()[0]->kind()
            == "usdInstancer");
      }

      THEN("The Stage's frame range and rate are reported, not applied")
      {
        REQUIRE(stage.report.animatedPrims == 1);
        REQUIRE(stage.report.sampleCount == 2);
        REQUIRE(stage.report.timeCodesPerSecond == Approx(24.f));
        REQUIRE(stage.animMgr.getAnimationTotalFrames() == 100); // untouched
      }

      THEN("The imported placements are the Stage's first frame")
      {
        auto *transforms =
            findTransformArray(stage.scene.defaultLayer(), "Swarm");
        REQUIRE(transforms != nullptr);
        REQUIRE(transforms->size() == 3);
        const auto *m = transforms->dataAs<tsd::math::mat4>();
        REQUIRE(m[2][3].x == Approx(2.f));
        REQUIRE(m[2][0].x == Approx(1.f));
      }

      THEN("Scrubbing re-fills the same Array in place")
      {
        auto *before = findTransformArray(stage.scene.defaultLayer(), "Swarm");
        REQUIRE(before != nullptr);

        stage.animMgr.setAnimationTime(1.0f);

        auto *after = findTransformArray(stage.scene.defaultLayer(), "Swarm");
        REQUIRE(after == before); // no reallocation on a constant count
        const auto *m = after->dataAs<tsd::math::mat4>();
        REQUIRE(m[2][3].x == Approx(20.f));
        REQUIRE(m[2][0].x == Approx(2.f));
      }

      THEN("A time between authored samples is interpolated, not snapped")
      {
        stage.animMgr.setAnimationTime(0.5f);

        auto *transforms =
            findTransformArray(stage.scene.defaultLayer(), "Swarm");
        REQUIRE(transforms != nullptr);
        const auto *m = transforms->dataAs<tsd::math::mat4>();
        REQUIRE(m[2][3].x == Approx(11.f));
      }
    }
  }
}

SCENARIO(
    "An instancer whose placement count changes reallocates", "[UsdImport]")
{
  GIVEN("A PointInstancer that gains a placement mid-sequence")
  {
    ImportedStage stage("tsd_test_usd_growing_instancer.usda", R"(#usda 1.0
(
    startTimeCode = 0
    endTimeCode = 2
)

def PointInstancer "Growing"
{
    point3f[] positions.timeSamples = {
        0: [(0, 0, 0), (1, 0, 0)],
        2: [(0, 0, 0), (1, 0, 0), (2, 0, 0)],
    }
    int[] protoIndices.timeSamples = {
        0: [0, 0],
        2: [0, 0, 0],
    }
    rel prototypes = [</Growing/Proto>]

    def Mesh "Proto"
    {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    }
}
)");

    WHEN("The Stage is imported and scrubbed past the change")
    {
      auto *before = findTransformArray(stage.scene.defaultLayer(), "Growing");
      REQUIRE(before != nullptr);
      REQUIRE(before->size() == 2);

      stage.animMgr.setAnimationTime(1.0f);

      THEN("The node is re-pointed at a right-sized Array")
      {
        auto *after = findTransformArray(stage.scene.defaultLayer(), "Growing");
        REQUIRE(after != nullptr);
        REQUIRE(after->size() == 3);
        REQUIRE(after->dataAs<tsd::math::mat4>()[2][3].x == Approx(2.f));
      }
    }
  }
}

SCENARIO("Instancer bindings survive save and reload", "[UsdImport]")
{
  GIVEN("An imported Stage with an animated PointInstancer")
  {
    ImportedStage stage("tsd_test_usd_instancer_archive.usda", R"(#usda 1.0
(
    startTimeCode = 0
    endTimeCode = 2
)

def PointInstancer "Swarm"
{
    point3f[] positions.timeSamples = {
        0: [(0, 0, 0), (1, 0, 0)],
        2: [(0, 0, 0), (9, 0, 0)],
    }
    int[] protoIndices = [0, 0]
    rel prototypes = [</Swarm/Proto>]

    def Mesh "Proto"
    {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    }
}
)");

    WHEN("The animation manager round-trips through an Archive")
    {
      tsd::core::DataTree tree;
      REQUIRE(tsd::io::serialize_AnimationManagerArchive(
          stage.animMgr, tree.root()));

      tsd::animation::AnimationManager restored(&stage.scene);
      REQUIRE(
          tsd::io::deserialize_AnimationManagerArchive(restored, tree.root()));

      THEN("The reconstructed binding scrubs the same Array")
      {
        REQUIRE(restored.animations().size() == 1);
        REQUIRE(restored.animations()[0].fileBindings().size() == 1);
        REQUIRE(restored.animations()[0].fileBindings()[0]->kind()
            == "usdInstancer");

        restored.setAnimationTime(1.0f);

        auto *transforms =
            findTransformArray(stage.scene.defaultLayer(), "Swarm");
        REQUIRE(transforms != nullptr);
        REQUIRE(transforms->dataAs<tsd::math::mat4>()[1][3].x == Approx(9.f));
      }
    }
  }
}

SCENARIO(
    "A Stage that authored no time-code range still animates", "[UsdImport]")
{
  GIVEN("A PointInstancer with time samples but no startTimeCode")
  {
    // Nothing forces a Stage to declare its own range, and USD reports 0 for
    // both ends when it does not. Without a fallback every animation time
    // would map onto one Time Code and the placements would never move.
    ImportedStage stage("tsd_test_usd_unranged_instancer.usda", R"(#usda 1.0

def PointInstancer "Drifting"
{
    point3f[] positions.timeSamples = {
        5: [(0, 0, 0), (1, 0, 0)],
        9: [(0, 0, 0), (7, 0, 0)],
    }
    int[] protoIndices = [0, 0]
    rel prototypes = [</Drifting/Proto>]

    def Mesh "Proto"
    {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    }
}
)");

    WHEN("The Stage is imported and scrubbed to the end")
    {
      stage.animMgr.setAnimationTime(1.0f);

      THEN("The authored samples define the range that time maps onto")
      {
        auto *transforms =
            findTransformArray(stage.scene.defaultLayer(), "Drifting");
        REQUIRE(transforms != nullptr);
        REQUIRE(transforms->dataAs<tsd::math::mat4>()[1][3].x == Approx(7.f));
      }
    }
  }

  GIVEN("A deforming mesh with time samples but no startTimeCode")
  {
    ImportedStage stage("tsd_test_usd_unranged_mesh.usda", R"(#usda 1.0

def Mesh "Blob"
{
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points.timeSamples = {
        5: [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
        9: [(0, 0, 0), (4, 0, 0), (0, 4, 0)],
    }
}
)");

    WHEN("The Stage is imported and scrubbed to the end")
    {
      stage.animMgr.setAnimationTime(1.0f);

      THEN("The authored samples define the range that time maps onto")
      {
        auto geometry = stage.scene.getObject<tsd::scene::Geometry>(0);
        auto *positions = geometry->parameterValueAsObject<tsd::scene::Array>(
            "vertex.position");
        REQUIRE(positions != nullptr);
        REQUIRE(positions->dataAs<tsd::math::float3>()[1].x == Approx(4.f));
      }
    }
  }
}

#endif // TSD_USE_USD
