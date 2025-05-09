// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef TSD_USE_CUDA
#define TSD_USE_CUDA 1
#endif

#include "tsd/authoring/serialization.hpp"
#include "tsd/containers/DataTree.hpp"
#include "tsd/core/Logging.hpp"
// std
#include <stack>
#include <stdexcept>
#include <type_traits>
#if TSD_USE_CUDA
// cuda
#include <cuda_runtime.h>
#endif

namespace tsd {

///////////////////
// Serialization //
///////////////////

static void parameterToNode(const Parameter &p, serialization::DataNode &node)
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

static void objectToNode(const Object &obj, serialization::DataNode &node)
{
  node["name"] = obj.name();
  node["self"] = utility::Any(obj.type(), obj.index());
  node["subtype"] = obj.subtype().c_str();

  if (obj.numParameters() > 0) {
    auto &params = node["parameters"];
    for (size_t i = 0; i < obj.numParameters(); i++) {
      const auto &p = obj.parameterAt(i);
      parameterToNode(p, params.append(p.name().c_str()));
    }
  }
}

static void arrayToNode(const Array &arr, serialization::DataNode &node)
{
  objectToNode(arr, node);

  node["arrayDim"] = tsd::uint3{
      uint32_t(arr.dim(0)), uint32_t(arr.dim(1)), uint32_t(arr.dim(2))};

  const auto *mem = reinterpret_cast<const uint8_t *>(arr.data());
#if TSD_USE_CUDA
  if (arr.kind() == Array::MemoryKind::CUDA) {
    const size_t numBytes = arr.size() * arr.elementSize();
    std::vector<uint8_t> hostBuf(numBytes);
    cudaMemcpy(hostBuf.data(), mem, numBytes, cudaMemcpyDeviceToHost);
    node["arrayData"].setValueAsArray(
        arr.elementType(), hostBuf.data(), numBytes);
  } else
#endif
    node["arrayData"].setValueAsArray(arr.elementType(), mem, arr.size());
}

static void layerToNode(Layer &layer, serialization::DataNode &node)
{
  std::stack<serialization::DataNode *> nodes;
  serialization::DataNode *currentParentNode = nullptr;
  serialization::DataNode *currentNode = &node;
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

    currentNode->append("name") = tsdNode->name;
    currentNode->append("value") = tsdNode->value;
    currentNode->append("children");

    return true;
  });
}

/////////////////////
// Deserialization //
/////////////////////

static void nodeToParameter(serialization::DataNode &node, Parameter &p)
{
  p.setValue(node["value"].getValue());

  if (auto *c = node.child("desription"); c != nullptr)
    p.setDescription(c->getValueAs<std::string>().c_str());

  if (auto *c = node.child("usage"); c != nullptr)
    p.setUsage(static_cast<tsd::ParameterUsageHint>(c->getValueAs<int>()));

  if (auto *c = node.child("min"); c != nullptr)
    p.setMin(c->getValue());

  if (auto *c = node.child("max"); c != nullptr)
    p.setMax(c->getValue());

  if (auto *c = node.child("stringValues"); c != nullptr) {
    std::vector<std::string> stringValues;
    c->foreach_child([&](serialization::DataNode &child) {
      stringValues.push_back(child.getValueAs<std::string>());
    });
    p.setStringValues(stringValues);
    p.setStringSelection(node["stringSelection"].getValueAs<int>());
  }
}

static void nodeToObjectParameters(serialization::DataNode &node, Object &obj)
{
  node.foreach_child([&](serialization::DataNode &parameterNode) {
    const Token parameterName(parameterNode.name().c_str());
    auto &p = obj.addParameter(parameterName);
    nodeToParameter(parameterNode, p);
  });
}

static void nodeToObject(Context &ctx, serialization::DataNode &node)
{
  const utility::Any self = node["self"].getValue();
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

    auto dim = arrayDim.getValueAs<tsd::uint3>();

    anari::DataType arrayElementType = ANARI_UNKNOWN;
    const void *arrayPtr = nullptr;
    size_t arraySize = 0;

    arrayData.getValueAsArray(&arrayElementType, &arrayPtr, &arraySize);

    const bool is2D = type == ANARI_ARRAY2D;
    const bool is3D = type == ANARI_ARRAY3D;
    const size_t dim_x = dim[0];
    const size_t dim_y = is2D || is3D ? dim[1] : size_t(0);
    const size_t dim_z = is3D ? dim[2] : size_t(0);
    auto arr = ctx.createArray(arrayElementType, dim_x, dim_y, dim_z);
    auto *memOut = arr->map();

    std::memcpy(memOut, arrayPtr, arr->size() * arr->elementSize());
    arr->unmap();

    obj = arr.data();
  } break;
  case ANARI_GEOMETRY:
    obj = ctx.createObject<Geometry>(subtype).data();
    break;
  case ANARI_MATERIAL:
    obj = ctx.createObject<Material>(subtype).data();
    break;
  case ANARI_SAMPLER:
    obj = ctx.createObject<Sampler>(subtype).data();
    break;
  case ANARI_SURFACE:
    obj = ctx.createObject<Surface>().data();
    break;
  case ANARI_SPATIAL_FIELD:
    obj = ctx.createObject<SpatialField>(subtype).data();
    break;
  case ANARI_VOLUME:
    obj = ctx.createObject<Volume>(subtype).data();
    break;
  case ANARI_LIGHT:
    obj = ctx.createObject<Light>(subtype).data();
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

  if (auto *c = node.child("name"); c != nullptr)
    obj->setName(c->getValueAs<std::string>().c_str());

  if (auto *c = node.child("parameters"); c != nullptr)
    nodeToObjectParameters(*c, *obj);
}

static void nodeToLayer(serialization::DataNode &rootNode, Layer &layer)
{
  std::stack<LayerNodeRef> tsdNodes;
  LayerNodeRef currentParentNode;
  LayerNodeRef currentNode = layer.root();
  int currentLevel = -1;
  rootNode.traverse([&](serialization::DataNode &node, int level) {
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
      currentNode = layer.insert_last_child(currentParentNode,
          {node["value"].getValue(),
              node["name"].getValueAs<std::string>().c_str()});
    }

    return true;
  });
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void save_Context(Context &ctx, const char *filename)
{
  tsd::logStatus("Saving context to file: %s", filename);

  serialization::DataTree tree;
  auto &root = tree.root();

  tsd::logStatus("  ...converting layers");

  auto &layersRoot = root["layers"];
  for (auto l : ctx.layers()) {
    if (l.second)
      layerToNode(*l.second, layersRoot[l.first.c_str()]);
  }

  tsd::logStatus("  ...converting objects");

  auto &objectDB = root["objectDB"];
  auto objectArrayToNode = [](serialization::DataNode &objArrayRoot,
                               const auto &objArray,
                               const char *arrayName) {
    if (objArray.empty())
      return;

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

  objectArrayToNode(objectDB, ctx.m_db.geometry, "geometry");
  objectArrayToNode(objectDB, ctx.m_db.sampler, "sampler");
  objectArrayToNode(objectDB, ctx.m_db.material, "material");
  objectArrayToNode(objectDB, ctx.m_db.surface, "surface");
  objectArrayToNode(objectDB, ctx.m_db.field, "spatialfield");
  objectArrayToNode(objectDB, ctx.m_db.volume, "volume");
  objectArrayToNode(objectDB, ctx.m_db.light, "light");
  objectArrayToNode(objectDB, ctx.m_db.array, "array");

  tsd::logStatus("  ...writing file");

  tree.save(filename);

  tsd::logStatus("  ...done!");
}

void import_Context(Context &ctx, const char *filename)
{
  tsd::logStatus("Loading context from file: %s", filename);

  // Clear out any existing context contents //

  tsd::logStatus("  ...clearing old context");

  ctx.removeAllObjects();
  ctx.removeAllSecondaryLayers();
  ctx.defaultLayer()->root()->erase_subtree();

  // Load from the conduit file (objects then layer) //

  tsd::logStatus("  ...loading file");

  serialization::DataTree tree;
  tree.load(filename);

  tsd::logStatus("  ...converting objects");

  auto &root = tree.root();
  auto &objectDB = root["objectDB"];
  auto nodeToObjectArray = [](serialization::DataNode &node,
                               Context &ctx,
                               const char *childNodeName) {
    auto &objectsNode = node[childNodeName];
    objectsNode.foreach_child([&](auto &n) { nodeToObject(ctx, n); });
  };

  nodeToObjectArray(objectDB, ctx, "geometry");
  nodeToObjectArray(objectDB, ctx, "sampler");
  nodeToObjectArray(objectDB, ctx, "material");
  nodeToObjectArray(objectDB, ctx, "surface");
  nodeToObjectArray(objectDB, ctx, "spatialfield");
  nodeToObjectArray(objectDB, ctx, "volume");
  nodeToObjectArray(objectDB, ctx, "light");
  nodeToObjectArray(objectDB, ctx, "array");

  tsd::logStatus("  ...converting layers");

  auto &v1LayerRoot = root["objectTree"];
  if (v1LayerRoot.numChildren() > 0)
    nodeToLayer(v1LayerRoot, *ctx.defaultLayer());
  else {
    auto &v2LayerRoot = root["layers"];
    v2LayerRoot.foreach_child([&](auto &nLayer) {
      auto &tLayer = *ctx.addLayer(nLayer.name().c_str());
      nodeToLayer(nLayer, tLayer);
    });
  }

  tsd::logStatus("  ...done!");
}

} // namespace tsd
