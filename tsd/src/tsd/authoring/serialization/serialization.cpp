// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/authoring/serialization.hpp"
#include "tsd/core/Logging.hpp"
#if TSD_ENABLE_SERIALIZATION
// conduit
#include <conduit.hpp>
#endif
// std
#include <stack>
#include <stdexcept>
#include <type_traits>

#ifndef TSD_ENABLE_SERIALIZATION
#define TSD_ENABLE_SERIALIZATION 1
#endif

namespace tsd {

#if TSD_ENABLE_SERIALIZATION
// Helper functions ///////////////////////////////////////////////////////////

// Conduit //

namespace conduit_utility {

using NodeVisitorEntryFunction =
    std::function<bool(const conduit::Node &n, int level)>;

namespace detail {

inline void TraverseNodesImpl(
    const conduit::Node &n, const NodeVisitorEntryFunction &onEntry, int level)
{
  bool traverseChildren = onEntry(n, level);
  if (traverseChildren) {
    for (size_t i = 0; i < n.number_of_children(); i++)
      TraverseNodesImpl(n[i], onEntry, level + 1);
  }
}

} // namespace detail

inline void TraverseNodes(
    const conduit::Node &n, const NodeVisitorEntryFunction &onEntry)
{
  detail::TraverseNodesImpl(n, onEntry, 0);
}

} // namespace conduit_utility

///////////////////
// Serialization //
///////////////////

static void anyToConduit(const Any &a, conduit::Node &node)
{
  node["type"] = static_cast<int>(a.type());
  node["typeStr"] = anari::toString(static_cast<int>(a.type()));
  if (a.is(ANARI_STRING))
    node["data"] = a.getCStr();
  else
    node["data"].set((const uint8_t *)a.data(), anari::sizeOf(a.type()));
}

static void parameterToConduit(const Parameter &p, conduit::Node &node)
{
  node["name"] = p.name().c_str();
  node["description"] = p.description();
  node["type"] = anari::toString(p.value().type());
  node["usage"] = static_cast<int>(p.usage());

  anyToConduit(p.value(), node["value"]);
  anyToConduit(p.min(), node["min"]);
  anyToConduit(p.max(), node["max"]);

  auto &stringValues = node["stringValues"];
  for (const auto &sv : p.stringValues())
    stringValues.append() = sv;

  node["stringSelection"] = p.stringSelection();
}

static void objectToConduit(const Object &obj, conduit::Node &node)
{
  if (!obj.name().empty())
    node["name"] = obj.name();
  node["type"] = static_cast<int>(obj.type());
  node["subtype"] = obj.subtype().c_str();
  node["index"] = obj.index();

  auto &params = node["parameters"];
  for (size_t i = 0; i < obj.numParameters(); i++)
    parameterToConduit(obj.parameterAt(i), params.append());
}

static void arrayToConduit(const Array &arr, conduit::Node &node)
{
  objectToConduit(arr, node);

  auto &arrayData = node["arrayData"];
  arrayData["elementType"] = static_cast<int>(arr.elementType());
  arrayData["size"] = arr.size();
  arrayData["dim"] = {arr.dim(0), arr.dim(1), arr.dim(2)};

  const auto *mem = reinterpret_cast<const uint8_t *>(arr.data());
  arrayData["bytes"].set(mem, arr.size() * arr.elementSize());
}

static void objectTreeToConduit(InstanceTree &tree, conduit::Node &node)
{
  std::stack<conduit::Node *> nodes;
  conduit::Node *currentParentNode = nullptr;
  conduit::Node *currentNode = &node;
  int currentLevel = -1;
  tree.traverse(tree.root(), [&](InstanceNode &tsdNode, int level) {
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
      currentNode = &currentParentNode->child("children").append();
    if (!tsdNode->name.empty())
      currentNode->add_child("name") = tsdNode->name;
    anyToConduit(tsdNode->value, currentNode->add_child("value"));
    currentNode->add_child("level") = level;
    currentNode->add_child("children");

    return true;
  });
}

/////////////////////
// Deserialization //
/////////////////////

static Any conduitToAny(const conduit::Node &node)
{
  const auto type = static_cast<anari::DataType>(node["type"].to_int());
  if (type == ANARI_STRING)
    return Any(type, node["data"].as_char8_str());
  else {
    conduit::Node bytesNode;
    node["data"].to_uint8_array(bytesNode);
    auto bytes = bytesNode.as_uint8_array();
    return Any(type, bytes.data_ptr());
  }
}

static void conduitToParameter(const conduit::Node &node, Parameter &p)
{
  p.setDescription(node["description"].to_string().c_str());
  p.setUsage(static_cast<ParameterUsageHint>(node["usage"].to_int()));
  p.setValue(conduitToAny(node["value"]));
  p.setMin(conduitToAny(node["min"]));
  p.setMax(conduitToAny(node["max"]));

  const auto &stringValuesNode = node["stringValues"];
  std::vector<std::string> stringValues;
  for (size_t i = 0; i < stringValuesNode.number_of_children(); i++)
    stringValues.push_back(stringValuesNode[i].as_string());
  p.setStringValues(stringValues);

  p.setStringSelection(node["stringSelection"].to_int());
}

static void conduitToObjectParameters(const conduit::Node &node, Object &obj)
{
  for (size_t i = 0; i < node.number_of_children(); i++) {
    const auto &parameterNode = node[i];
    const Token parameterName(parameterNode["name"].as_char8_str());
    const auto parameterType =
        static_cast<anari::DataType>(parameterNode["value/type"].to_int());
    auto &p = obj.addParameter(parameterName);
    conduitToParameter(parameterNode, p);
  }
}

static void conduitToObject(Context &ctx, const conduit::Node &node)
{
  const auto type = static_cast<anari::DataType>(node["type"].to_int());
  const Token subtype(node["subtype"].as_char8_str());
  const size_t index = node["index"].to_uint64();

  if (!anari::isObject(type)) {
    logError("[conduitToObject] parsed invalid object type from conduit: '%s'",
        anari::toString(type));
    return;
  }

  Object *obj = nullptr;
  switch (type) {
  case ANARI_ARRAY:
  case ANARI_ARRAY1D:
  case ANARI_ARRAY2D:
  case ANARI_ARRAY3D: {
    const auto &arrayData = node["arrayData"];

    conduit::Node dimNode;
    arrayData["dim"].to_uint64_array(dimNode);
    auto dim = dimNode.as_uint64_array();

    const auto elementType =
        static_cast<anari::DataType>(arrayData["elementType"].as_int());
    const bool is2D = type == ANARI_ARRAY2D;
    const bool is3D = type == ANARI_ARRAY3D;
    const size_t dim_x = dim[0];
    const size_t dim_y = is2D || is3D ? dim[1] : size_t(0);
    const size_t dim_z = is3D ? dim[2] : size_t(0);
    auto arr = ctx.createArray(elementType, dim_x, dim_y, dim_z);
    auto *memOut = arr->map();

    conduit::Node bytesNode;
    arrayData["bytes"].to_uint8_array(bytesNode);
    auto *memIn = bytesNode.as_uint8_array().data_ptr();

    std::memcpy(memOut, memIn, arr->size() * arr->elementSize());
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
    logError("[conduitToObject] unable to create object from Conduit");
    return;
  }

  if (obj->index() != index) {
    logError(
        "[conduitToObject] object (%s) index mismatch on import: %zu | %zu",
        anari::toString(type),
        obj->index(),
        index);
  }

  if (node.has_child("name"))
    obj->setName(node["name"].as_char8_str());
  conduitToObjectParameters(node["parameters"], *obj);
}

static void conduitToObjectTree(conduit::Node &rootNode, InstanceTree &tree)
{
  std::stack<InstanceNode::Ref> tsdNodes;
  InstanceNode::Ref currentParentNode;
  InstanceNode::Ref currentNode = tree.root();
  int currentLevel = -1;
  conduit_utility::TraverseNodes(rootNode, [&](auto &node, int level) {
    if (level & 0x1 || !node.has_child("children"))
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
      currentNode = tree.root();
    else {
      const char *name = "";
      if (node.has_child("name"))
        name = node["name"].as_char8_str();
      currentNode = tree.insert_last_child(
          currentParentNode, {conduitToAny(node["value"]), name});
    }

    return true;
  });
}
#endif

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void save_Context(Context &ctx, const char *filename)
{
#if TSD_ENABLE_SERIALIZATION
  conduit::Node root;

  auto &objectTree = root["objectTree"];
  objectTreeToConduit(ctx.tree, objectTree);

  auto &objectDB = root["objectDB"];
  auto objectArrayToConduit =
      [](conduit::Node &root, const auto &objArray, const char *childNodeName) {
        auto &childNode = root[childNodeName];
        foreach_item_const(objArray, [&](const auto *obj) {
          if (!obj)
            return;
          auto &m = childNode.append();
          if constexpr (std::is_same<decltype(obj), const Array *>::value)
            arrayToConduit(*obj, m);
          else
            objectToConduit(*obj, m);
        });
      };

  objectArrayToConduit(objectDB, ctx.m_db.geometry, "geometry");
  objectArrayToConduit(objectDB, ctx.m_db.sampler, "sampler");
  objectArrayToConduit(objectDB, ctx.m_db.material, "material");
  objectArrayToConduit(objectDB, ctx.m_db.surface, "surface");
  objectArrayToConduit(objectDB, ctx.m_db.field, "spatialfield");
  objectArrayToConduit(objectDB, ctx.m_db.volume, "volume");
  objectArrayToConduit(objectDB, ctx.m_db.light, "light");
  objectArrayToConduit(objectDB, ctx.m_db.array, "array");

  root.save(filename, "conduit_bin");
#else
  logError("[save_Context] serialization not enabled in TSD build.");
#endif
}

void import_Context(Context &ctx, const char *filename)
{
#if TSD_ENABLE_SERIALIZATION
  // Clear out any existing context contents //

  ctx.removeAllObjects();
  ctx.tree.erase_subtree(ctx.tree.root());

  // Load from the conduit file (objects then tree) //

  conduit::Node root;
  root.load(filename, "conduit_bin");
  auto &objectDB = root["objectDB"];
  auto conduitToObjectArray =
      [](const conduit::Node &node, Context &ctx, const char *childNodeName) {
        const auto &objectsNode = node[childNodeName];
        for (size_t i = 0; i < objectsNode.number_of_children(); i++)
          conduitToObject(ctx, objectsNode[i]);
      };

  conduitToObjectArray(objectDB, ctx, "geometry");
  conduitToObjectArray(objectDB, ctx, "sampler");
  conduitToObjectArray(objectDB, ctx, "material");
  conduitToObjectArray(objectDB, ctx, "surface");
  conduitToObjectArray(objectDB, ctx, "spatialfield");
  conduitToObjectArray(objectDB, ctx, "volume");
  conduitToObjectArray(objectDB, ctx, "light");
  conduitToObjectArray(objectDB, ctx, "array");

  conduitToObjectTree(root["objectTree"], ctx.tree);
#else
  logError("[import_Context] serialization not enabled in TSD build.");
#endif
}

} // namespace tsd
