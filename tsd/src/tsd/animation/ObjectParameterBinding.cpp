// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/animation/ObjectParameterBinding.hpp"

namespace tsd::animation {

ObjectParameterBinding::ObjectParameterBinding(scene::Object *target,
    core::Token paramName,
    anari::DataType type,
    const void *data,
    const float *timeBase,
    size_t count,
    InterpolationRule interp)
    : Binding(target->scene()),
      m_paramName(paramName),
      m_type(type),
      m_interp(interp)
{
  if (target)
    m_target = *target;

  if (data && count > 0) {
    if (anari::isObject(type)) {
      std::vector<size_t> indices(count);
      const auto *objects = static_cast<const scene::Object *const *>(data);
      m_objectRefs.reserve(count);
      for (size_t i = 0; i < count; i++) {
        auto *object = objects[i];
        indices[i] = object ? object->index() : TSD_INVALID_INDEX;
        if (object)
          m_objectRefs.emplace_back(*const_cast<scene::Object *>(object));
        else
          m_objectRefs.emplace_back();
      }
      m_data = core::AnyArray(type, indices.data(), count);
    } else {
      m_data = core::AnyArray(type, data, count);
    }
  }

  if (timeBase && count > 0)
    m_timeBase.assign(timeBase, timeBase + count);
}

scene::Object *ObjectParameterBinding::target() const
{
  return const_cast<scene::Object *>(m_target.get());
}

core::Token ObjectParameterBinding::paramName() const
{
  return m_paramName;
}

anari::DataType ObjectParameterBinding::type() const
{
  return m_type;
}

const core::AnyArray &ObjectParameterBinding::data() const
{
  return m_data;
}

const std::vector<float> &ObjectParameterBinding::timeBase() const
{
  return m_timeBase;
}

InterpolationRule ObjectParameterBinding::interpolation() const
{
  return m_interp;
}

void ObjectParameterBinding::insertKeyframeImpl(float time, const void *value)
{
  size_t elemSize = anari::sizeOf(m_type);
  if (elemSize == 0)
    return;

  size_t count = m_timeBase.size();

  // Find insertion point (maintain time sort)
  size_t insertIdx = count;
  for (size_t i = 0; i < count; i++) {
    if (std::abs(m_timeBase[i] - time) < 1e-4f) {
      auto *dst = static_cast<uint8_t *>(m_data.data());
      std::memcpy(dst + i * elemSize, value, elemSize);
      if (anari::isObject(m_type)) {
        const size_t newIdx =
            *reinterpret_cast<const size_t *>(dst + i * elemSize);
        scene::AnyObjectUsePtr<scene::Object::UseKind::ANIM> newRef;
        if (auto *obj = scene()->getObject(m_type, newIdx))
          newRef = *obj;
        m_objectRefs[i] = std::move(newRef);
      }
      return;
    }
    if (m_timeBase[i] > time) {
      insertIdx = i;
      break;
    }
  }

  m_timeBase.insert(m_timeBase.begin() + insertIdx, time);

  size_t newCount = count + 1;
  core::AnyArray newData(m_type, newCount);
  auto *newVals = static_cast<uint8_t *>(newData.data());
  const auto *oldData = static_cast<const uint8_t *>(m_data.data());

  if (oldData && insertIdx > 0)
    std::memcpy(newVals, oldData, insertIdx * elemSize);

  std::memcpy(newVals + insertIdx * elemSize, value, elemSize);

  if (oldData && insertIdx < count)
    std::memcpy(newVals + (insertIdx + 1) * elemSize,
        oldData + insertIdx * elemSize,
        (count - insertIdx) * elemSize);

  m_data = std::move(newData);

  if (anari::isObject(m_type)) {
    const size_t newIdx = *reinterpret_cast<const size_t *>(
        static_cast<const uint8_t *>(m_data.data()) + insertIdx * elemSize);
    scene::AnyObjectUsePtr<scene::Object::UseKind::ANIM> newRef;
    if (auto *obj = scene()->getObject(m_type, newIdx))
      newRef = *obj;
    m_objectRefs.insert(m_objectRefs.begin() + insertIdx, std::move(newRef));
  }
}

void ObjectParameterBinding::removeKeyframe(size_t i)
{
  size_t count = m_timeBase.size();
  if (i >= count)
    return;

  size_t elemSize = anari::sizeOf(m_type);
  if (count <= 1) {
    m_timeBase.clear();
    m_data.reset();
    m_objectRefs.clear();
    return;
  }

  m_timeBase.erase(m_timeBase.begin() + i);

  size_t newCount = count - 1;
  core::AnyArray newData(m_type, newCount);
  auto *newVals = static_cast<uint8_t *>(newData.data());
  const auto *oldData = static_cast<const uint8_t *>(m_data.data());

  if (i > 0)
    std::memcpy(newVals, oldData, i * elemSize);
  if (i < newCount)
    std::memcpy(newVals + i * elemSize,
        oldData + (i + 1) * elemSize,
        (newCount - i) * elemSize);

  m_data = std::move(newData);

  if (anari::isObject(m_type))
    m_objectRefs.erase(m_objectRefs.begin() + i);
}

void ObjectParameterBinding::updateObjectDefragmentedIndices(
    const scene::IndexRemapper &cb)
{
  if (!m_target || !cb)
    return;
  m_target.updateDefragmentedIndex(cb(m_target->type(), m_target->index()));

  if (anari::isObject(m_type) && !m_objectRefs.empty()) {
    auto *indices = static_cast<size_t *>(m_data.data());
    for (size_t i = 0; i < m_objectRefs.size(); i++) {
      if (!m_objectRefs[i])
        continue;
      size_t newIdx = cb(m_type, m_objectRefs[i]->index());
      m_objectRefs[i].updateDefragmentedIndex(newIdx);
      indices[i] = newIdx;
    }
  }
}

void ObjectParameterBinding::toDataNode(core::DataNode &node) const
{
  node["targetType"] = int(m_target ? m_target->type() : ANARI_UNKNOWN);
  node["targetIndex"] = m_target ? m_target->index() : TSD_INVALID_INDEX;
  node["paramName"] = m_paramName.str();
  node["dataType"] = (int)m_type;
  if (!m_timeBase.empty())
    node["timeBase"].setValueAsArray(m_timeBase);
  if (!m_data.empty())
    node["data"].setValueAsArray(m_type, m_data.data(), m_data.size());
  node["interp"] = (int)m_interp;
}

void ObjectParameterBinding::fromDataNode(core::DataNode &node)
{
  auto targetType = (anari::DataType)node["targetType"].getValueAs<int>();
  auto targetIndex = node["targetIndex"].getValueAs<size_t>();
  scene::Object *target = nullptr;
  if (targetType != ANARI_UNKNOWN && targetIndex != size_t(-1))
    target = this->scene()->getObject(targetType, targetIndex);

  m_paramName =
      core::Token(node["paramName"].getValueAs<std::string>().c_str());
  m_type = (anari::DataType)node["dataType"].getValueAs<int>();
  m_interp = (InterpolationRule)node["interp"].getValueAs<int>();

  const float *tbPtr = nullptr;
  size_t tbCount = 0;
  if (auto *tbNode = node.child("timeBase"))
    tbNode->getValueAsArray(&tbPtr, &tbCount);

  const void *dataPtr = nullptr;
  size_t dataCount = 0;
  anari::DataType dt = ANARI_UNKNOWN;
  if (auto *dataNode = node.child("data"); dataNode && dataNode->holdsArray())
    dataNode->getValueAsArray(&dt, &dataPtr, &dataCount);

  size_t count = dataCount > 0 ? dataCount : tbCount;
  *this = ObjectParameterBinding(
      target, m_paramName, m_type, dataPtr, tbPtr, count, m_interp);
}

} // namespace tsd::animation
