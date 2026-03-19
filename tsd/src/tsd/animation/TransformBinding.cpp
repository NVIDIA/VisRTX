// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/animation/TransformBinding.hpp"
// tsd_scene
#include "tsd/scene/Scene.hpp"

namespace tsd::animation {

TransformBinding::TransformBinding(scene::LayerNodeRef target)
    : Binding((*m_target)->layer()->scene()), m_target(target)
{}

TransformBinding::TransformBinding(scene::LayerNodeRef target,
    const float *timeBase,
    const math::float4 *rotation,
    const math::float3 *translation,
    const math::float3 *scale,
    size_t count)
    : TransformBinding(target)
{
  m_timeBase.assign(timeBase, timeBase + count);
  m_rotation.assign(rotation, rotation + count);
  m_translation.assign(translation, translation + count);
  m_scale.assign(scale, scale + count);
}

scene::LayerNodeRef TransformBinding::target() const
{
  return m_target;
}

const std::vector<float> &TransformBinding::timeBase() const
{
  return m_timeBase;
}

const std::vector<math::float4> &TransformBinding::rotation() const
{
  return m_rotation;
}

const std::vector<math::float3> &TransformBinding::translation() const
{
  return m_translation;
}

const std::vector<math::float3> &TransformBinding::scale() const
{
  return m_scale;
}

size_t TransformBinding::sampleCount() const
{
  return m_timeBase.size();
}

float TransformBinding::timeBase(size_t i) const
{
  return m_timeBase[i];
}

const math::float4 &TransformBinding::rotation(size_t i) const
{
  return m_rotation[i];
}

const math::float3 &TransformBinding::translation(size_t i) const
{
  return m_translation[i];
}

const math::float3 &TransformBinding::scale(size_t i) const
{
  return m_scale[i];
}

void TransformBinding::insertKeyframe(float time, const math::mat4 &m)
{
  // Decompose mat4 -> rotation quaternion, translation, scale
  math::float3 c0 = {m[0][0], m[0][1], m[0][2]};
  math::float3 c1 = {m[1][0], m[1][1], m[1][2]};
  math::float3 c2 = {m[2][0], m[2][1], m[2][2]};

  auto vecLen = [](math::float3 v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
  };
  math::float3 scl = {vecLen(c0), vecLen(c1), vecLen(c2)};
  if (scl.x > 0.f)
    c0 = {c0.x / scl.x, c0.y / scl.x, c0.z / scl.x};
  if (scl.y > 0.f)
    c1 = {c1.x / scl.y, c1.y / scl.y, c1.z / scl.y};
  if (scl.z > 0.f)
    c2 = {c2.x / scl.z, c2.y / scl.z, c2.z / scl.z};

  // Shepperd's method (mat3 -> unit quaternion)
  float trace = c0.x + c1.y + c2.z;
  math::float4 rot;
  if (trace > 0.f) {
    float s = 0.5f / std::sqrt(trace + 1.f);
    rot = {(c1.z - c2.y) * s, (c2.x - c0.z) * s, (c0.y - c1.x) * s, 0.25f / s};
  } else if (c0.x > c1.y && c0.x > c2.z) {
    float s = 0.5f / std::sqrt(1.f + c0.x - c1.y - c2.z);
    rot = {0.25f / s, (c0.y + c1.x) * s, (c2.x + c0.z) * s, (c1.z - c2.y) * s};
  } else if (c1.y > c2.z) {
    float s = 0.5f / std::sqrt(1.f + c1.y - c0.x - c2.z);
    rot = {(c0.y + c1.x) * s, 0.25f / s, (c1.z + c2.y) * s, (c2.x - c0.z) * s};
  } else {
    float s = 0.5f / std::sqrt(1.f + c2.z - c0.x - c1.y);
    rot = {(c2.x + c0.z) * s, (c1.z + c2.y) * s, 0.25f / s, (c0.y - c1.x) * s};
  }
  float qlen =
      std::sqrt(rot.x * rot.x + rot.y * rot.y + rot.z * rot.z + rot.w * rot.w);
  rot = {rot.x / qlen, rot.y / qlen, rot.z / qlen, rot.w / qlen};

  math::float3 trans = {m[3][0], m[3][1], m[3][2]};

  // Find insertion point (maintain time sort)
  size_t count = m_timeBase.size();
  size_t insertIdx = count;
  for (size_t i = 0; i < count; i++) {
    if (std::abs(m_timeBase[i] - time) < 1e-4f) {
      m_rotation[i] = rot;
      m_translation[i] = trans;
      m_scale[i] = scl;
      return;
    }
    if (m_timeBase[i] > time) {
      insertIdx = i;
      break;
    }
  }

  m_timeBase.insert(m_timeBase.begin() + insertIdx, time);
  m_rotation.insert(m_rotation.begin() + insertIdx, rot);
  m_translation.insert(m_translation.begin() + insertIdx, trans);
  m_scale.insert(m_scale.begin() + insertIdx, scl);
}

void TransformBinding::removeKeyframe(size_t i)
{
  if (i < m_timeBase.size()) {
    m_timeBase.erase(m_timeBase.begin() + i);
    m_rotation.erase(m_rotation.begin() + i);
    m_translation.erase(m_translation.begin() + i);
    m_scale.erase(m_scale.begin() + i);
  }
}

void TransformBinding::toDataNode(core::DataNode &node) const
{
  if (m_target) {
    auto *lay = m_target->value().layer();
    node["layerName"] = scene()->getLayerName(lay).str();
    node["nodeIndex"] = m_target->index();
  }
  if (!m_timeBase.empty())
    node["timeBase"].setValueAsArray(m_timeBase);
  if (!m_rotation.empty())
    node["rotation"].setValueAsArray(m_rotation);
  if (!m_translation.empty())
    node["translation"].setValueAsArray(m_translation);
  if (!m_scale.empty())
    node["scale"].setValueAsArray(m_scale);
}

void TransformBinding::fromDataNode(core::DataNode &node)
{
  scene::LayerNodeRef tbTarget;
  std::vector<float> tbTimeBase;
  std::vector<tsd::core::math::float4> tbRotation;
  std::vector<tsd::core::math::float3> tbTranslation;
  std::vector<tsd::core::math::float3> tbScale;

  if (auto *lnNode = node.child("layerName")) {
    auto layerName = core::Token(lnNode->getValueAs<std::string>().c_str());
    auto *lay = scene()->layer(layerName);
    if (lay) {
      size_t idx = node["nodeIndex"].getValueAs<size_t>();
      tbTarget = lay->at(idx);
    }
  }

  auto loadVec = [&](const char *key, auto &out) {
    if (auto *n = node.child(key)) {
      const auto *ptr = out.data();
      size_t count = 0;
      n->getValueAsArray(&ptr, &count);
      if (ptr && count > 0)
        out.assign(ptr, ptr + count);
    }
  };

  loadVec("timeBase", tbTimeBase);
  loadVec("rotation", tbRotation);
  loadVec("translation", tbTranslation);
  loadVec("scale", tbScale);

  *this = TransformBinding(tbTarget,
      tbTimeBase.data(),
      tbRotation.data(),
      tbTranslation.data(),
      tbScale.data(),
      tbTimeBase.size());
}

} // namespace tsd::animation
