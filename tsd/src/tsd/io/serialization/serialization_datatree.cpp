// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef TSD_USE_CUDA
#define TSD_USE_CUDA 1
#endif

#include "tsd/core/DataTree.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/io/serialization.hpp"
// std
#include <stack>
#include <stdexcept>
#include <type_traits>
#if TSD_USE_CUDA
// cuda
#include <cuda_runtime.h>
#endif

namespace tsd::io {

// Parameters /////////////////////////////////////////////////////////////////

void parameterToNode(const Parameter &p, core::DataNode &node)
{
  node["value"] = p.value();
  if (!p.description().empty())
    node["description"] = p.description();
  if (p.usage() != ParameterUsageHint::NONE)
    node["usage"] = static_cast<int>(p.usage());
  if (p.min())
    node["min"] = p.min();
  if (p.max())
    node["max"] = p.max();

  if (!p.stringValues().empty()) {
    auto &stringValues = node["stringValues"];
    for (const auto &sv : p.stringValues())
      stringValues.append() = sv;
    node["stringSelection"] = p.stringSelection();
  }
}

void nodeToParameter(core::DataNode &node, Parameter &p)
{
  if (auto *c = node.child("description"); c != nullptr)
    p.setDescription(c->getValueAs<std::string>().c_str());

  if (auto *c = node.child("usage"); c != nullptr)
    p.setUsage(static_cast<ParameterUsageHint>(c->getValueAs<int>()));

  if (auto *c = node.child("min"); c != nullptr)
    p.setMin(c->getValue());

  if (auto *c = node.child("max"); c != nullptr)
    p.setMax(c->getValue());

  if (auto *c = node.child("stringValues"); c != nullptr) {
    std::vector<std::string> stringValues;
    c->foreach_child([&](core::DataNode &child) {
      stringValues.push_back(child.getValueAs<std::string>());
    });
    p.setStringValues(stringValues);
    p.setStringSelection(node["stringSelection"].getValueAs<int>());
  }

  p.setValue(node["value"].getValue());
}

void nodeToObjectParameters(core::DataNode &node, Object &obj)
{
  node.foreach_child([&](core::DataNode &parameterNode) {
    const Token parameterName(parameterNode.name().c_str());
    auto &p = obj.addParameter(parameterName);
    nodeToParameter(parameterNode, p);
  });
}

// Objects ////////////////////////////////////////////////////////////////////

void objectToNode(const Object &obj, core::DataNode &node)
{
  node["name"] = obj.name();
  node["self"] = Any(obj.type(), obj.index());
  node["subtype"] = obj.subtype().c_str();

  if (obj.numParameters() > 0) {
    auto &params = node["parameters"];
    for (size_t i = 0; i < obj.numParameters(); i++) {
      const auto &p = obj.parameterAt(i);
      parameterToNode(p, params.append(p.name().c_str()));
    }
  }

  if (obj.numMetadata() > 0) {
    auto &metadata = node["metadata"];
    for (size_t i = 0; i < obj.numMetadata(); i++) {
      std::string n = obj.getMetadataName(i);
      anari::DataType type = ANARI_UNKNOWN;
      const void *ptr = nullptr;
      size_t size = 0;
      obj.getMetadataArray(n, &type, &ptr, &size);
      if (type != ANARI_UNKNOWN)
        metadata[n].setValueAsExternalArray(type, ptr, size);
      else if (auto v = obj.getMetadataValue(n); v.valid())
        metadata[n] = v;
    }
  }
}

void nodeToObject(core::DataNode &node, Object &obj)
{
  if (auto *c = node.child("name"); c != nullptr)
    obj.setName(c->getValueAs<std::string>().c_str());

  if (auto *c = node.child("parameters"); c != nullptr)
    nodeToObjectParameters(*c, obj);

  if (auto *c = node.child("metadata"); c != nullptr)
    nodeToObjectMetadata(*c, obj);
}

void nodeToObjectMetadata(core::DataNode &node, Object &obj)
{
  node.foreach_child([&](core::DataNode &n) {
    if (n.holdsArray()) {
      anari::DataType type = ANARI_UNKNOWN;
      const void *ptr = nullptr;
      size_t size = 0;
      n.getValueAsArray(&type, &ptr, &size);
      obj.setMetadataArray(n.name(), type, ptr, size);
    } else {
      obj.setMetadataValue(n.name(), n.getValue());
    }
  });
}

void nodeToNewObject(Scene &scene, core::DataNode &node)
{
  const Any self = node["self"].getValue();
  const auto type = self.type();
  const size_t index = self.getAsObjectIndex();
  const Token subtype(node["subtype"].getValueAs<std::string>().c_str());

  if (!anari::isObject(type)) {
    logError("[nodeToObject] parsed invalid object type '%s'",
        anari::toString(type));
    return;
  }

  Object *obj = nullptr;
  switch (type) {
  case ANARI_ARRAY:
  case ANARI_ARRAY1D:
  case ANARI_ARRAY2D:
  case ANARI_ARRAY3D: {
    auto &arrayData = node["arrayData"];
    auto &arrayDim = node["arrayDim"];

    auto dim = arrayDim.getValueAs<tsd::math::uint3>();

    anari::DataType arrayElementType = ANARI_UNKNOWN;
    const void *arrayPtr = nullptr;
    size_t arraySize = 0;

    arrayData.getValueAsArray(&arrayElementType, &arrayPtr, &arraySize);

    const bool is2D = type == ANARI_ARRAY2D;
    const bool is3D = type == ANARI_ARRAY3D;
    const size_t dim_x = dim[0];
    const size_t dim_y = is2D || is3D ? dim[1] : size_t(0);
    const size_t dim_z = is3D ? dim[2] : size_t(0);
    auto arr = scene.createArray(arrayElementType, dim_x, dim_y, dim_z);
    if (arr) {
      auto *memOut = arr->map();
      std::memcpy(memOut, arrayPtr, arr->size() * arr->elementSize());
      arr->unmap();
      obj = arr.data();
    }
  } break;
  case ANARI_GEOMETRY:
    obj = scene.createObject<Geometry>(subtype).data();
    break;
  case ANARI_MATERIAL:
    obj = scene.createObject<Material>(subtype).data();
    break;
  case ANARI_SAMPLER:
    obj = scene.createObject<Sampler>(subtype).data();
    break;
  case ANARI_SURFACE:
    obj = scene.createObject<Surface>().data();
    break;
  case ANARI_SPATIAL_FIELD:
    obj = scene.createObject<SpatialField>(subtype).data();
    break;
  case ANARI_VOLUME:
    obj = scene.createObject<Volume>(subtype).data();
    break;
  case ANARI_LIGHT:
    obj = scene.createObject<Light>(subtype).data();
    break;
  case ANARI_CAMERA:
    obj = scene.createObject<Camera>(subtype).data();
    break;
  default:
    break;
  }

  if (!obj) {
    logError("[nodeToObject] unable to create object from DataNode");
    return;
  }

  if (obj->index() != index) {
    logError("[nodeToObject] object (%s) index mismatch on import: %zu | %zu",
        anari::toString(type),
        obj->index(),
        index);
  }

  nodeToObject(node, *obj);
}

// Camera poses ///////////////////////////////////////////////////////////////

void cameraPoseToNode(const rendering::CameraPose &p, core::DataNode &node)
{
  node["name"] = p.name;
  node["lookat"] = p.lookat;
  node["azeldist"] = p.azeldist;
  node["fixedDist"] = p.fixedDist;
  node["upAxis"] = p.upAxis;
}

void nodeToCameraPose(core::DataNode &node, rendering::CameraPose &pose)
{
  node["name"].getValue(ANARI_STRING, &pose.name);
  node["lookat"].getValue(ANARI_FLOAT32_VEC3, &pose.lookat);
  node["azeldist"].getValue(ANARI_FLOAT32_VEC3, &pose.azeldist);
  node["fixedDist"].getValue(ANARI_FLOAT32, &pose.fixedDist);
  node["upAxis"].getValue(ANARI_INT32, &pose.upAxis);
}

// Arrays /////////////////////////////////////////////////////////////////////

void arrayToNode(const Array &arr, core::DataNode &node)
{
  objectToNode(arr, node);

  node["arrayDim"] = tsd::math::uint3{
      uint32_t(arr.dim(0)), uint32_t(arr.dim(1)), uint32_t(arr.dim(2))};

  const void *mem = arr.data();
#if TSD_USE_CUDA
  if (arr.kind() == Array::MemoryKind::CUDA) {
    const size_t numBytes = arr.size() * arr.elementSize();
    std::vector<uint8_t> hostBuf(numBytes);
    cudaMemcpy(hostBuf.data(), mem, numBytes, cudaMemcpyDeviceToHost);
    node["arrayData"].setValueAsArray(
        arr.elementType(), hostBuf.data(), numBytes);
  } else
#endif
    node["arrayData"].setValueAsExternalArray(
        arr.elementType(), mem, arr.size());
}

// Layers /////////////////////////////////////////////////////////////////////

void layerToNode(Layer &layer, core::DataNode &node)
{
  std::stack<core::DataNode *> nodes;
  core::DataNode *currentParentNode = nullptr;
  core::DataNode *currentNode = &node;
  int currentLevel = -1;
  layer.traverse(layer.root(), [&](LayerNode &tsdNode, int level) {
    if (currentLevel < level) {
      nodes.push(currentNode);
      currentParentNode = currentNode;
    } else if (currentLevel > level) {
      for (int i = 0; i < currentLevel - level; i++)
        nodes.pop();
      currentParentNode = nodes.top();
    }

    currentLevel = level;

    if (level == 0)
      currentNode = &node;
    else
      currentNode = &currentParentNode->child("children")->append();

    currentNode->append("name") = tsdNode->name();
    currentNode->append("value") = tsdNode->getValueRaw();
    if (tsdNode->isTransform())
      currentNode->append("transformSRT") = tsdNode->getTransformSRT();
    currentNode->append("enabled") = tsdNode->isEnabled();
    currentNode->append("children");

    return true;
  });
}

void nodeToLayer(core::DataNode &rootNode, Layer &layer, Scene &scene)
{
  std::stack<LayerNodeRef> tsdNodes;
  LayerNodeRef currentParentNode;
  LayerNodeRef currentNode = layer.root();
  int currentLevel = -1;
  rootNode.traverse([&](core::DataNode &node, int level) {
    if (level & 0x1 || !node.child("children"))
      return true;

    level /= 2;
    if (currentLevel < level) {
      tsdNodes.push(currentNode);
      currentParentNode = currentNode;
    } else if (currentLevel > level) {
      for (int i = 0; i < currentLevel - level; i++)
        tsdNodes.pop();
      currentParentNode = tsdNodes.top();
    }

    currentLevel = level;

    if (level == 0)
      currentNode = layer.root();
    else {
      currentNode = layer.insert_last_child(currentParentNode, {});
      if (auto *c = node.child("transformSRT"); c != nullptr)
        (*currentNode)->setAsTransform(c->getValueAs<math::mat3>());
      else
        (*currentNode)->setValueRaw(node["value"].getValue(), &scene);
      (*currentNode)->setEnabled(node["enabled"].getValueOr(true));
      (*currentNode)->name() = node["name"].getValueAs<std::string>();
    }

    return true;
  });
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void save_Scene(Scene &scene, const char *filename)
{
  tsd::core::logStatus("Saving context to file: %s", filename);
  tsd::core::logStatus("  ...serializing context");
  core::DataTree tree;
  save_Scene(scene, tree.root());
  tsd::core::logStatus("  ...writing file");
  tree.save(filename);
  tsd::core::logStatus("  ...done!");
}

void save_Scene(Scene &scene, core::DataNode &root)
{
  // Layers //

  tsd::core::logStatus("    ...serializing %zu layers", scene.numberOfLayers());

  auto &layersRoot = root["layers"];
  for (auto l : scene.layers()) {
    if (l.second.ptr) {
      auto &layerRoot = layersRoot[l.first.c_str()];
      layerToNode(*l.second.ptr, layerRoot);
      layerRoot["isActive"] = l.second.active;
    }
  }

  // Animations //

  tsd::core::logStatus(
      "    ...serializing %zu animations", scene.numberOfAnimations());

  auto &animationsRoot = root["animation"];

  auto &animationObjects = animationsRoot["objects"];
  for (size_t i = 0; i < scene.numberOfAnimations(); i++)
    scene.animation(i)->serialize(animationObjects.append());

  auto &animationSettings = animationsRoot["settings"];
  animationSettings["time"] = scene.getAnimationTime();
  animationSettings["increment"] = scene.getAnimationIncrement();

  // ObjectDB //

  auto &objectDB = root["objectDB"];
  auto objectArrayToNode = [](core::DataNode &objArrayRoot,
                               const auto &objArray,
                               const char *arrayName) {
    if (objArray.empty())
      return;

    tsd::core::logStatus("    ...serializing %zu %s objects",
        size_t(objArray.size()),
        arrayName);

    auto &childNode = objArrayRoot[arrayName];
    foreach_item_const(objArray, [&](const auto *obj) {
      if (!obj)
        return;
      auto &m = childNode.append();
      if constexpr (std::is_same<decltype(obj), const Array *>::value)
        arrayToNode(*obj, m);
      else
        objectToNode(*obj, m);
    });
  };

  objectArrayToNode(objectDB, scene.m_db.geometry, "geometry");
  objectArrayToNode(objectDB, scene.m_db.sampler, "sampler");
  objectArrayToNode(objectDB, scene.m_db.material, "material");
  objectArrayToNode(objectDB, scene.m_db.surface, "surface");
  objectArrayToNode(objectDB, scene.m_db.field, "spatialfield");
  objectArrayToNode(objectDB, scene.m_db.volume, "volume");
  objectArrayToNode(objectDB, scene.m_db.light, "light");
  objectArrayToNode(objectDB, scene.m_db.camera, "camera");
  objectArrayToNode(objectDB, scene.m_db.array, "array");
}

void load_Scene(Scene &scene, const char *filename)
{
  tsd::core::logStatus("Loading context from file: %s", filename);
  tsd::core::logStatus("  ...loading file");
  core::DataTree tree;
  tree.load(filename);
  auto &root = tree.root();
  if (auto *c = root.child("context"); c != nullptr)
    load_Scene(scene, *c);
  else
    load_Scene(scene, root);
}

void load_Scene(Scene &scene, core::DataNode &root)
{
  // Clear out any existing context contents //

  tsd::core::logStatus("  ...clearing old context");

  scene.removeAllObjects();
  scene.removeAllLayers();

  // Load data from file (objects then layer) //

  // ObjectDB

  tsd::core::logStatus("  ...converting objects");

  auto &objectDB = root["objectDB"];
  auto nodeToObjectArray =
      [](core::DataNode &node, Scene &scene, const char *childNodeName) {
        auto &objectsNode = node[childNodeName];
        objectsNode.foreach_child([&](auto &n) { nodeToNewObject(scene, n); });
      };

  nodeToObjectArray(objectDB, scene, "array");
  nodeToObjectArray(objectDB, scene, "sampler");
  nodeToObjectArray(objectDB, scene, "material");
  nodeToObjectArray(objectDB, scene, "geometry");
  nodeToObjectArray(objectDB, scene, "surface");
  nodeToObjectArray(objectDB, scene, "spatialfield");
  nodeToObjectArray(objectDB, scene, "volume");
  nodeToObjectArray(objectDB, scene, "light");
  nodeToObjectArray(objectDB, scene, "camera");

  // Layers

  tsd::core::logStatus("  ...converting layers");

  auto &layerRoot = root["layers"];
  layerRoot.foreach_child([&](auto &nLayer) {
    tsd::core::Token layerName = nLayer.name().c_str();
    auto &tLayer = *scene.addLayer(layerName);
    nodeToLayer(nLayer, tLayer, scene);
    bool active = true;
    nLayer["isActive"].getValue(ANARI_BOOL, &active);
    scene.setLayerActive(layerName, active);
    scene.signalLayerChange(&tLayer);
  });

  scene.m_numActiveLayers = 0;
  for (auto &ls : scene.layers()) {
    if (ls.second.active)
      scene.m_numActiveLayers++;
  }

  scene.signalActiveLayersChanged();

  // Animations

  if (auto *c = root.child("animation"); c != nullptr) {
    tsd::core::logStatus("  ...converting animations");

    auto &animationRoot = *c;
    auto &animationObjects = animationRoot["objects"];
    animationObjects.foreach_child([&](auto &animationNode) {
      scene.addAnimation()->deserialize(animationNode);
    });

    auto &animationSettings = animationRoot["settings"];
    scene.setAnimationTime(animationSettings["time"].getValueAs<float>());
    scene.setAnimationIncrement(
        animationSettings["increment"].getValueAs<float>());
  } else {
    tsd::core::logStatus("  ...no animations found!");
  }

  tsd::core::logStatus("  ...done!");
}

} // namespace tsd::io
