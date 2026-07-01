// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/archives/AnimationArchive.hpp"
// tsd_animation
#include "tsd/animation/Animation.hpp"
#include "tsd/animation/AnimationManager.hpp"
// tsd_core
#include "tsd/core/DataTree.hpp"
#include "tsd/core/Logging.hpp"
// tsd_io
#include "tsd/io/serialization/serialization_internal.hpp"
// tsd_scene
#include "tsd/scene/Scene.hpp"
// std
#include <cstdint>

namespace tsd::io {

namespace {

bool fail(std::string *message, std::string text)
{
  if (message)
    *message = std::move(text);
  return false;
}

bool readInt(core::DataNode &node, const char *name, int &value)
{
  auto *child = node.child(name);
  return child && child->getValue(ANARI_INT32, &value);
}

bool readIndex(core::DataNode &node, const char *name, size_t &value)
{
  auto *child = node.child(name);
  if (!child)
    return false;
  uint64_t stored = 0;
  if (!child->getValue(ANARI_UINT64, &stored))
    return false;
  value = size_t(stored);
  return uint64_t(value) == stored;
}

size_t arraySize(core::DataNode *node, anari::DataType *type = nullptr)
{
  if (!node || !node->holdsArray())
    return 0;
  anari::DataType elementType = ANARI_UNKNOWN;
  const void *data = nullptr;
  size_t size = 0;
  node->getValueAsArray(&elementType, &data, &size);
  if (type)
    *type = elementType;
  return size;
}

bool validateObjectBinding(
    scene::Scene &scene, core::DataNode &binding, std::string *message)
{
  int targetTypeValue = int(ANARI_UNKNOWN);
  size_t targetIndex = core::INVALID_INDEX;
  if (!readInt(binding, "targetType", targetTypeValue)
      || !readIndex(binding, "targetIndex", targetIndex)
      || !binding.child("paramName") || !binding.child("dataType")
      || !binding.child("interp")) {
    return fail(message, "object binding is missing required state");
  }
  const auto targetType = anari::DataType(targetTypeValue);
  if (targetType == ANARI_UNKNOWN || targetIndex == core::INVALID_INDEX
      || !scene.getObject(targetType, targetIndex)) {
    return fail(message, "object binding target is incompatible with Scene");
  }

  int dataTypeValue = int(ANARI_UNKNOWN);
  if (!readInt(binding, "dataType", dataTypeValue))
    return fail(message, "object binding data type is invalid");
  const auto dataType = anari::DataType(dataTypeValue);
  anari::DataType storedType = ANARI_UNKNOWN;
  const auto dataCount = arraySize(binding.child("data"), &storedType);
  const auto timeCount = arraySize(binding.child("timeBase"));
  if (dataCount != 0 && storedType != dataType)
    return fail(message, "object binding data type does not match payload");
  if (dataCount != 0 && timeCount != 0 && dataCount != timeCount)
    return fail(message, "object binding sample arrays have different sizes");

  if (anari::isObject(dataType) && dataCount != 0) {
    const void *data = nullptr;
    size_t count = 0;
    anari::DataType storedDataType = ANARI_UNKNOWN;
    binding.child("data")->getValueAsArray(&storedDataType, &data, &count);
    const auto *indices = static_cast<const size_t *>(data);
    for (size_t i = 0; i < count; ++i) {
      if (indices[i] != core::INVALID_INDEX
          && !scene.getObject(dataType, indices[i])) {
        return fail(
            message, "object binding keyframe is incompatible with Scene");
      }
    }
  }
  return true;
}

bool validateTransformBinding(
    scene::Scene &scene, core::DataNode &binding, std::string *message)
{
  auto *layerNameNode = binding.child("layerName");
  size_t nodeIndex = core::INVALID_INDEX;
  if (!layerNameNode || !readIndex(binding, "nodeIndex", nodeIndex))
    return fail(message, "transform binding is missing its target");
  const auto layerName = layerNameNode->getValueOr<std::string>("");
  auto *layer = scene.layer(layerName.c_str());
  if (!layer || !layer->at(nodeIndex))
    return fail(message, "transform binding target is incompatible with Scene");

  const auto timeCount = arraySize(binding.child("timeBase"));
  const auto rotationCount = arraySize(binding.child("rotation"));
  const auto translationCount = arraySize(binding.child("translation"));
  const auto scaleCount = arraySize(binding.child("scale"));
  if (timeCount == 0)
    return rotationCount == 0 && translationCount == 0 && scaleCount == 0
        ? true
        : fail(message, "transform binding has incomplete sample arrays");
  if (rotationCount != timeCount || translationCount != timeCount
      || scaleCount != timeCount) {
    return fail(
        message, "transform binding sample arrays have different sizes");
  }
  return true;
}

bool validateFileBinding(
    scene::Scene &scene, core::DataNode &binding, std::string *message)
{
  auto *kindNode = binding.child("kind");
  if (!kindNode)
    return fail(message, "file binding is missing its kind");
  const auto kind = kindNode->getValueOr<std::string>("");
  if (kind == "spatialField") {
    size_t targetIndex = core::INVALID_INDEX;
    if (!readIndex(binding, "targetIndex", targetIndex)
        || !scene.getObject(ANARI_VOLUME, targetIndex)) {
      return fail(
          message, "spatial-field binding target is incompatible with Scene");
    }
    return true;
  }
  if (kind == "ensight") {
    auto *parts = binding.child("parts");
    if (!parts || parts->numChildren() == 0)
      return fail(message, "EnSight binding requires geometry targets");
    bool valid = true;
    parts->foreach_child([&](core::DataNode &part) {
      size_t targetIndex = core::INVALID_INDEX;
      valid &= readIndex(part, "targetIndex", targetIndex)
          && scene.getObject(ANARI_GEOMETRY, targetIndex);
    });
    return valid
        ? true
        : fail(message, "EnSight binding target is incompatible with Scene");
  }
  return fail(
      message, "Animation Archive contains an unsupported file binding");
}

} // namespace

bool serialize_AnimationArchive(
    const animation::Animation &animation, core::DataNode &archive)
{
  if (!animation.manager() || !animation.manager()->scene())
    return false;
  if (animation.callbackBindings().size() > animation.fileBindings().size()) {
    core::logError(
        "[serialize_AnimationArchive] callback bindings are not serializable");
    return false;
  }
  archive.reset();
  animationToNode(animation, archive);
  std::string message;
  if (!validate_AnimationArchive(*animation.manager(), archive, &message)) {
    core::logError("[serialize_AnimationArchive] %s", message.c_str());
    archive.reset();
    return false;
  }
  return true;
}

bool validate_AnimationArchive(const animation::AnimationManager &manager,
    core::DataNode &archive,
    std::string *message)
{
  auto *scene = manager.scene();
  if (!scene)
    return fail(message, "Animation Manager has no Scene");
  if (!archive.child("name"))
    return fail(message, "Animation Archive requires a name");

  bool valid = true;
  if (auto *bindings = archive.child("objectBindings")) {
    bindings->foreach_child([&](core::DataNode &binding) {
      if (valid)
        valid = validateObjectBinding(*scene, binding, message);
    });
  }
  if (auto *bindings = archive.child("transformBindings")) {
    bindings->foreach_child([&](core::DataNode &binding) {
      if (valid)
        valid = validateTransformBinding(*scene, binding, message);
    });
  }
  if (auto *bindings = archive.child("fileBindings")) {
    bindings->foreach_child([&](core::DataNode &binding) {
      if (valid)
        valid = validateFileBinding(*scene, binding, message);
    });
  }
  return valid;
}

animation::Animation *deserialize_AnimationArchive(
    animation::AnimationManager &manager, core::DataNode &archive)
{
  std::string message;
  if (!validate_AnimationArchive(manager, archive, &message)) {
    core::logError("[deserialize_AnimationArchive] %s", message.c_str());
    return nullptr;
  }
  const auto index = manager.animations().size();
  try {
    auto &animation = manager.addAnimation();
    nodeToAnimation(archive, animation, *manager.scene());
    return &animation;
  } catch (...) {
    manager.removeAnimation(index);
    return nullptr;
  }
}

bool save_AnimationArchive(
    const animation::Animation &animation, const char *filename)
{
  if (!filename)
    return false;
  core::DataTree tree;
  return serialize_AnimationArchive(animation, tree.root())
      && tree.save(filename);
}

animation::Animation *load_AnimationArchive(
    animation::AnimationManager &manager, const char *filename)
{
  if (!filename)
    return nullptr;
  core::DataTree tree;
  if (!tree.load(filename))
    return nullptr;
  return deserialize_AnimationArchive(manager, tree.root());
}

} // namespace tsd::io
