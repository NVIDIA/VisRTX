// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/rendering/index/RenderIndexAllLayers.hpp"
#include "tsd/scene/Scene.hpp"
// anari
#include <anari/anari_cpp.hpp>
// std
#include <cmath>

namespace {

using namespace tsd::core;
using namespace tsd::scene;

// A device the tests can drive without a display or a GPU. Absent builds skip
// rather than fail: what is under test is TSD's bookkeeping, not the device.
struct TestDevice
{
  TestDevice()
  {
    library = anari::loadLibrary("helide", [](const void *,
                                               ANARIDevice,
                                               ANARIObject,
                                               anari::DataType,
                                               ANARIStatusSeverity,
                                               ANARIStatusCode,
                                               const char *) {});
    if (library)
      device = anari::newDevice(library, "default");
  }

  ~TestDevice()
  {
    if (device)
      anari::release(device, device);
    if (library)
      anari::unloadLibrary(library);
  }

  explicit operator bool() const
  {
    return device != nullptr;
  }

  anari::Library library{nullptr};
  anari::Device device{nullptr};
};

// The world's bounds are the cheapest thing a device will tell us about where
// the geometry it was given actually ended up.
tsd::math::float3 worldBoundsMax(
    anari::Device d, tsd::rendering::RenderIndexAllLayers &index)
{
  tsd::math::float3 bounds[2] = {{-1.f, -1.f, -1.f}, {1.f, 1.f, 1.f}};
  anariGetProperty(d,
      index.world(),
      "bounds",
      ANARI_FLOAT32_BOX3,
      &bounds[0],
      sizeof(bounds),
      ANARI_WAIT);
  return bounds[1];
}

// One unit sphere at the origin, placed by a single-element transform array so
// that the only thing that can move it is the contents of that Array.
struct InstancedSphereScene
{
  InstancedSphereScene()
  {
    auto geometry = scene.createObject<Geometry>(tokens::geometry::sphere);
    auto positions = scene.createArray(ANARI_FLOAT32_VEC3, 1);
    const tsd::math::float3 origin(0.f, 0.f, 0.f);
    positions->setData(&origin, 1);
    geometry->setParameterObject("vertex.position", *positions);
    geometry->setParameter("radius", 1.f);

    auto material = scene.createObject<Material>(tokens::material::matte);
    auto surface = scene.createSurface("sphere", geometry, material);

    transforms = scene.createArray(ANARI_FLOAT32_MAT4, 1);
    const auto identity = tsd::math::IDENTITY_MAT4;
    transforms->setData(&identity, 1);

    auto node = scene.insertChildTransformArrayNode(
        scene.defaultLayer()->root(), transforms.data(), "instances");
    scene.insertChildObjectNode(node, surface, "sphere");
  }

  Scene scene;
  ArrayRef transforms;
};

tsd::math::mat4 translation(float x)
{
  auto retval = tsd::math::IDENTITY_MAT4;
  retval[3] = tsd::math::float4(x, 0.f, 0.f, 1.f);
  return retval;
}

} // namespace

SCENARIO("Rewriting a transform array moves its instances", "[RenderIndex]")
{
  TestDevice anariDevice;
  if (!anariDevice) {
    WARN("helide unavailable, skipping");
    return;
  }

  GIVEN("A populated render index over a transform-array node")
  {
    InstancedSphereScene content;
    auto *index =
        content.scene.updateDelegate()
            .emplace<tsd::rendering::RenderIndexAllLayers>(
                content.scene, Token("helide"), anariDevice.device);
    index->populate();

    REQUIRE(worldBoundsMax(anariDevice.device, *index).x == Approx(1.f));

    WHEN("The transform array is rewritten in place")
    {
      const auto moved = translation(50.f);
      content.transforms->setData(&moved, 1);

      // A layer's node transforms are copied into its ANARI instances rather
      // than referenced, so an Array rewritten behind the render index's back
      // has to make that copy happen again. Rebuilding the world does not.
      THEN("The instance moves with it")
      {
        REQUIRE(worldBoundsMax(anariDevice.device, *index).x == Approx(51.f));
      }
    }

    WHEN("The transform array is rewritten inside an update batch")
    {
      content.scene.beginUpdateBatch();
      const auto moved = translation(50.f);
      content.transforms->setData(&moved, 1);
      content.scene.endUpdateBatch();

      THEN("The instance has moved by the time the batch ends")
      {
        REQUIRE(worldBoundsMax(anariDevice.device, *index).x == Approx(51.f));
      }
    }
  }
}
