// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef TSD_USE_CUDA
#define TSD_USE_CUDA 1
#endif

#include "tsd/io/serialization/Object.hpp"
#include "tsd/io/serialization/Parameter.hpp"
// tsd_core
#include "tsd/core/Logging.hpp"
// std
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#if TSD_USE_CUDA
// cuda
#include <cuda_runtime.h>
#endif

namespace tsd::io {

namespace {

void serializeArray(
    const scene::Array &array, core::DataNode &node, bool forceArraysAsProxies)
{
  node["arrayDim"] = tsd::math::uint3{
      uint32_t(array.dim(0)), uint32_t(array.dim(1)), uint32_t(array.dim(2))};

  auto &arrayData = node.append("arrayData");

  const bool isProxy =
      forceArraysAsProxies || array.kind() == scene::Array::MemoryKind::PROXY;
  if (isProxy) {
    arrayData = static_cast<int>(array.elementType());
    return;
  }

  const void *memory = array.data();
#if TSD_USE_CUDA
  if (array.kind() == scene::Array::MemoryKind::CUDA) {
    const size_t numBytes = array.size() * array.elementSize();
    std::vector<uint8_t> hostBuffer(numBytes);
    cudaMemcpy(hostBuffer.data(), memory, numBytes, cudaMemcpyDeviceToHost);
    arrayData.setValueAsArray(
        array.elementType(), hostBuffer.data(), array.size());
  } else
#endif
    arrayData.setValueAsExternalArray(
        array.elementType(), memory, array.size());
}

} // namespace

void serialize_Object(const scene::Object &object,
    core::DataNode &node,
    bool forceArraysAsProxies)
{
  node["name"] = object.name();
  node["self"] = core::Any(object.type(), object.index());
  node["subtype"] = object.subtype().c_str();

  if (object.type() == ANARI_RENDERER && object.rendererDeviceName())
    node["rendererDeviceName"] = object.rendererDeviceName().c_str();

  if (object.numParameters() > 0) {
    auto &parameters = node["parameters"];
    for (size_t i = 0; i < object.numParameters(); i++) {
      const auto &parameter = object.parameterAt(i);
      serialize_Parameter(
          parameter, parameters.append(parameter.name().c_str()));
    }
  }

  if (object.numMetadata() > 0) {
    auto &metadata = node["metadata"];
    for (size_t i = 0; i < object.numMetadata(); i++) {
      const std::string name = object.getMetadataName(i);
      anari::DataType type = ANARI_UNKNOWN;
      const void *data = nullptr;
      size_t size = 0;
      object.getMetadataArray(name, &type, &data, &size);
      if (type != ANARI_UNKNOWN)
        metadata[name].setValueAsExternalArray(type, data, size);
      else if (auto value = object.getMetadataValue(name); value.valid())
        metadata[name] = value;
    }
  }

  if (anari::isArray(object.type())) {
    serializeArray(
        static_cast<const scene::Array &>(object), node, forceArraysAsProxies);
  }
}

namespace {

void deserializeObjectParameters(core::DataNode &node, scene::Object &object)
{
  node.foreach_child([&](core::DataNode &parameterNode) {
    const core::Token parameterName(parameterNode.name().c_str());
    auto &parameter = object.addParameter(parameterName);
    deserialize_Parameter(parameterNode, parameter);
  });
}

void deserializeObjectMetadata(core::DataNode &node, scene::Object &object)
{
  node.foreach_child([&](core::DataNode &metadataNode) {
    if (metadataNode.holdsArray()) {
      anari::DataType type = ANARI_UNKNOWN;
      const void *data = nullptr;
      size_t size = 0;
      metadataNode.getValueAsArray(&type, &data, &size);
      object.setMetadataArray(metadataNode.name(), type, data, size);
    } else {
      object.setMetadataValue(metadataNode.name(), metadataNode.getValue());
    }
  });
}

} // namespace

void deserialize_Object(core::DataNode &node, scene::Object &object)
{
  if (auto *child = node.child("name"); child != nullptr)
    object.setName(child->getValueAs<std::string>().c_str());

  if (auto *child = node.child("parameters"); child != nullptr)
    deserializeObjectParameters(*child, object);

  if (auto *child = node.child("metadata"); child != nullptr)
    deserializeObjectMetadata(*child, object);
}

void deserialize_Object(scene::Scene &scene, core::DataNode &node)
{
  const core::Any self = node["self"].getValue();
  const auto type = self.type();
  const size_t index = self.getAsObjectIndex();
  const core::Token subtype(node["subtype"].getValueAs<std::string>());

  if (!anari::isObject(type)) {
    core::logError("[deserialize_Object] parsed invalid object type '%s'",
        anari::toString(type));
    return;
  }

  scene::Object *object = nullptr;
  switch (type) {
  case ANARI_ARRAY:
  case ANARI_ARRAY1D:
  case ANARI_ARRAY2D:
  case ANARI_ARRAY3D: {
    auto &arrayData = node["arrayData"];
    auto &arrayDim = node["arrayDim"];
    const bool isProxy = !arrayData.holdsArray();

    const auto dim = arrayDim.getValueAs<tsd::math::uint3>();

    const bool is2D = type == ANARI_ARRAY2D;
    const bool is3D = type == ANARI_ARRAY3D;
    const size_t dimX = dim[0];
    const size_t dimY = is2D || is3D ? dim[1] : size_t(0);
    const size_t dimZ = is3D ? dim[2] : size_t(0);

    anari::DataType elementType = ANARI_UNKNOWN;
    const void *arrayDataPointer = nullptr;
    size_t arraySize = 0;

    if (isProxy) {
      elementType = static_cast<anari::DataType>(arrayData.getValueAs<int>());
    } else {
      arrayData.getValueAsArray(&elementType, &arrayDataPointer, &arraySize);
    }

    auto array = isProxy ? scene.createArrayProxy(elementType, dimX, dimY, dimZ)
                         : scene.createArray(elementType, dimX, dimY, dimZ);

    if (array) {
      object = array.data();
      if (!isProxy) {
        auto *destination = array->map();
        std::memcpy(destination,
            arrayDataPointer,
            array->size() * array->elementSize());
        array->unmap();
      }
    }
  } break;
  case ANARI_GEOMETRY:
    object = scene.createObject<scene::Geometry>(subtype).data();
    break;
  case ANARI_MATERIAL:
    object = scene.createObject<scene::Material>(subtype).data();
    break;
  case ANARI_SAMPLER:
    object = scene.createObject<scene::Sampler>(subtype).data();
    break;
  case ANARI_SURFACE:
    object = scene.createSurface().data();
    break;
  case ANARI_SPATIAL_FIELD:
    object = scene.createObject<scene::SpatialField>(subtype).data();
    break;
  case ANARI_VOLUME:
    object = scene.createObject<scene::Volume>(subtype).data();
    break;
  case ANARI_LIGHT:
    object = scene.createObject<scene::Light>(subtype).data();
    break;
  case ANARI_CAMERA:
    object = scene.createObject<scene::Camera>(subtype).data();
    break;
  case ANARI_RENDERER: {
    std::string rendererDeviceName;
    if (auto *child = node.child("rendererDeviceName"); child != nullptr)
      rendererDeviceName = child->getValueAs<std::string>();
    if (!rendererDeviceName.empty()) {
      object = scene.createRenderer(rendererDeviceName, subtype).get();
    }
  } break;
  default:
    break;
  }

  if (!object) {
    core::logError(
        "[deserialize_Object] unable to create object from DataNode");
    return;
  }

  if (object->index() != index) {
    core::logError("[deserialize_Object] object (%s) index mismatch: %zu | %zu",
        anari::toString(type),
        object->index(),
        index);
  }

  object->removeAllParameters();
  deserialize_Object(node, *object);
}

} // namespace tsd::io
