// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/scene/Scene.hpp"
#include "tsd/scene/UpdateDelegate.hpp"
// std
#include <cmath>
#include <string>

namespace {

void requireMat4Near(const tsd::math::mat4 &actual,
    const tsd::math::mat4 &expected,
    float eps = 1e-4f)
{
  for (int c = 0; c < 4; c++) {
    for (int r = 0; r < 4; r++) {
      CAPTURE(c, r, actual[c][r], expected[c][r]);
      REQUIRE(std::abs(actual[c][r] - expected[c][r]) <= eps);
    }
  }
}

struct CountingDelegate : public tsd::scene::EmptyUpdateDelegate
{
  CountingDelegate(int *objectAddedCount) : m_objectAddedCount(objectAddedCount)
  {}

  void signalObjectAdded(const tsd::scene::Object *) override
  {
    if (m_objectAddedCount)
      (*m_objectAddedCount)++;
  }

  int *m_objectAddedCount{nullptr};
};

} // namespace

static tsd::scene::LayerNodeRef findDirectChild(
    tsd::scene::LayerNodeRef parent, const std::string &name)
{
  if (!parent)
    return {};

  auto child = parent->next();
  while (child && child != parent) {
    if ((*child)->name() == name)
      return child;
    child = child->sibling();
  }

  return {};
}

SCENARIO("tsd::scene::Scene owns an intrinsic update delegate root", "[Scene]")
{
  GIVEN("A scene")
  {
    tsd::scene::Scene scene;
    auto &delegate = scene.updateDelegate();

    THEN("The scene starts with an empty MultiUpdateDelegate root")
    {
      REQUIRE(delegate.size() == 0);
    }

    THEN("Const and non-const accessors return the same delegate root")
    {
      const auto &constScene = scene;
      REQUIRE(&constScene.updateDelegate() == &delegate);
    }

    THEN("Registering a child delegate observes new scene-owned objects and arrays")
    {
      int objectAddedCount = 0;
      auto *countingDelegate = scene.updateDelegate().emplace<CountingDelegate>(
          &objectAddedCount);
      auto geometry = scene.createObject<tsd::scene::Geometry>(
          tsd::scene::tokens::geometry::sphere);
      auto array = scene.createArray(ANARI_FLOAT32, 4);

      REQUIRE(geometry);
      REQUIRE(array);
      REQUIRE(countingDelegate != nullptr);
      REQUIRE(objectAddedCount == 2);
    }
  }
}

SCENARIO("tsd::scene::Scene deep-clones layer subtrees", "[Scene]")
{
  GIVEN("A scene with a layer subtree containing object references")
  {
    tsd::scene::Scene scene;
    auto *layer = scene.defaultLayer();
    REQUIRE(layer != nullptr);

    auto material = scene.createObject<tsd::scene::Material>(
        tsd::scene::tokens::material::matte);
    material->setName("linkedMaterial");

    auto light = scene.createObject<tsd::scene::Light>(
        tsd::scene::tokens::light::directional);
    light->setName("sourceLight");
    light->setParameter("irradiance", 2.f);
    light->setParameterObject("linkedMaterial", *material);

    auto group = scene.insertChildTransformNode(
        layer->root(), tsd::math::IDENTITY_MAT4, "group");
    group->value().setInstanceParameter(
        "linkedMaterial", {material->type(), material->index()});
    scene.insertChildObjectNode(group, light, "mainLight");
    scene.insertChildTransformNode(
        layer->root(), tsd::math::IDENTITY_MAT4, "capacityFiller");

    WHEN("The subtree is cloned with default object sharing")
    {
      auto cloneGroup = scene.cloneLayerSubtree(group, layer->root());

      THEN("The clone references the same objects")
      {
        REQUIRE(cloneGroup);
        auto cloneLightNode = findDirectChild(cloneGroup, "mainLight");
        REQUIRE(cloneLightNode);
        REQUIRE((*cloneLightNode)->getObject() == light.data());

        const auto *instanceValue =
            cloneGroup->value().getInstanceParameters().at("linkedMaterial");
        REQUIRE(instanceValue != nullptr);
        REQUIRE(instanceValue->holdsObject());
        REQUIRE(instanceValue->type() == material->type());
        REQUIRE(instanceValue->getAsObjectIndex() == material->index());
      }
    }

    WHEN("The subtree is cloned with object references cloned")
    {
      const bool cloneObjectReferences = true;
      auto cloneGroup = scene.cloneLayerSubtree(
          group, layer->root(), cloneObjectReferences);

      THEN("The clone has distinct objects and remapped object parameters")
      {
        REQUIRE(cloneGroup);
        REQUIRE(cloneGroup != group);

        auto cloneLightNode = findDirectChild(cloneGroup, "mainLight");
        REQUIRE(cloneLightNode);
        auto *cloneLight = dynamic_cast<tsd::scene::Light *>(
            (*cloneLightNode)->getObject());
        REQUIRE(cloneLight != nullptr);
        REQUIRE(cloneLight != light.data());
        REQUIRE(cloneLight->parameterValueAs<float>("irradiance").value()
            == Approx(2.f));

        auto *cloneMaterial =
            cloneLight->parameterValueAsObject<tsd::scene::Material>(
                "linkedMaterial");
        REQUIRE(cloneMaterial != nullptr);
        REQUIRE(cloneMaterial != material.data());

        const auto *instanceValue =
            cloneGroup->value().getInstanceParameters().at("linkedMaterial");
        REQUIRE(instanceValue != nullptr);
        REQUIRE(instanceValue->holdsObject());
        REQUIRE(instanceValue->type() == cloneMaterial->type());
        REQUIRE(instanceValue->getAsObjectIndex() == cloneMaterial->index());

        cloneLight->setParameter("irradiance", 7.f);
        REQUIRE(light->parameterValueAs<float>("irradiance").value()
            == Approx(2.f));
      }
    }
  }
}

SCENARIO("tsd::scene::Scene clears layer references when removing an object",
    "[Scene]")
{
  GIVEN("An object referenced by nodes across multiple layers")
  {
    tsd::scene::Scene scene;
    auto *layerA = scene.defaultLayer();
    auto *layerB = scene.addLayer(tsd::core::Token("other"));

    auto light = scene.createObject<tsd::scene::Light>(
        tsd::scene::tokens::light::directional);
    light->setName("sharedLight");

    scene.insertChildObjectNode(layerA->root(), light, "instA");
    auto groupB = scene.insertChildTransformNode(
        layerB->root(), tsd::math::IDENTITY_MAT4, "groupB");
    scene.insertChildObjectNode(groupB, light, "instB");

    REQUIRE(light->useCount(tsd::scene::Object::UseKind::LAYER) == 2);

    WHEN("The object is removed")
    {
      const size_t lightIndex = light->index();
      scene.removeObject(light.data());

      THEN("All referencing layer nodes are removed, leaving no dangling refs")
      {
        REQUIRE(scene.getObject(ANARI_LIGHT, lightIndex) == nullptr);

        // No node in any layer still references the removed object.
        auto countReferencingNodes = [&](tsd::scene::Layer &layer) {
          int count = 0;
          layer.traverse(layer.root(), [&](auto &node, int) {
            if (node.value().isObject()
                && node.value().getObjectIndex() == lightIndex)
              count++;
            return true;
          });
          return count;
        };
        REQUIRE(countReferencingNodes(*layerA) == 0);
        REQUIRE(countReferencingNodes(*layerB) == 0);

        // The "instA" node is gone entirely (not left as an empty placeholder).
        REQUIRE_FALSE(findDirectChild(layerA->root(), "instA"));

        // Surrounding structure (the unrelated transform group) is preserved.
        REQUIRE(groupB.valid());
        REQUIRE((*groupB)->isTransform());
        REQUIRE_FALSE(findDirectChild(groupB, "instB"));
      }
    }
  }

  GIVEN("An object that is not referenced by any layer")
  {
    tsd::scene::Scene scene;
    auto material = scene.createObject<tsd::scene::Material>(
        tsd::scene::tokens::material::matte);
    REQUIRE(material->useCount(tsd::scene::Object::UseKind::LAYER) == 0);

    WHEN("The object is removed")
    {
      const size_t idx = material->index();
      scene.removeObject(material.data());

      THEN("It is removed without issue")
      {
        REQUIRE(scene.getObject(ANARI_MATERIAL, idx) == nullptr);
      }
    }
  }
}

SCENARIO(
    "tsd::scene::LayerNodeData preserves singular SRT transforms", "[Scene]")
{
  GIVEN("A transform with elevation 90 and roll 270")
  {
    tsd::math::mat3 srt;
    srt[0] = tsd::math::float3(1.f, 1.f, 1.f);
    srt[1] = tsd::math::float3(0.f, 90.f, 270.f);
    srt[2] = tsd::math::float3(0.f, 0.f, 0.f);

    tsd::scene::LayerNodeData source(nullptr, srt);
    tsd::scene::LayerNodeData node(nullptr, source.getTransform());

    WHEN("The transform is exposed as UI SRT and applied back")
    {
      auto uiSrt = node.getTransformSRT();
      node.setAsTransform(uiSrt);

      THEN("The UI SRT keeps the roll and the matrix does not move")
      {
        REQUIRE(tsd::math::neql(uiSrt[1].x, 0.f, 1e-3f));
        REQUIRE(tsd::math::neql(uiSrt[1].y, 90.f, 1e-3f));
        REQUIRE(tsd::math::neql(uiSrt[1].z, 270.f, 1e-3f));
        requireMat4Near(node.getTransform(), source.getTransform());
      }
    }
  }
}

SCENARIO("tsd::scene::Scene delegate registration controls live signaling",
    "[Scene]")
{
  GIVEN("A scene with an explicitly registered child delegate")
  {
    tsd::scene::Scene scene;
    int objectAddedCount = 0;
    auto *delegate = scene.updateDelegate().emplace<CountingDelegate>(
        &objectAddedCount);

    REQUIRE(scene.updateDelegate().size() == 1);

    WHEN("A new scene object is created while the delegate is registered")
    {
      auto geometry = scene.createObject<tsd::scene::Geometry>(
          tsd::scene::tokens::geometry::sphere);

      THEN("The delegate receives the object-added signal")
      {
        REQUIRE(geometry);
        REQUIRE(objectAddedCount == 1);
      }
    }

    WHEN("The delegate is erased before more scene changes occur")
    {
      scene.updateDelegate().erase(delegate);
      auto geometry = scene.createObject<tsd::scene::Geometry>(
          tsd::scene::tokens::geometry::sphere);

      THEN("The root becomes empty and no further signals are delivered")
      {
        REQUIRE(geometry);
        REQUIRE(scene.updateDelegate().size() == 0);
        REQUIRE(objectAddedCount == 0);
      }
    }
  }
}
