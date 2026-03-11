// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_core
#include <anari/anari_cpp/ext/linalg.h>
#include "tsd/core/scene/Layer.hpp"
// tsd_rendering
#include "tsd/rendering/index/RenderIndexFilterFcn.hpp"
// std
#include <algorithm>
#include <anari/anari_cpp.hpp>
#include <cstdint>
#include <iterator>
#include <stack>

namespace tsd::rendering {

///////////////////////////////////////////////////////////////////////////////

struct TransformsToAnariVisitor : public tsd::core::LayerVisitor
{
  TransformsToAnariVisitor(anari::Device d,
      anari::Instance *instances,
      RenderIndexFilterFcn *filter = nullptr);
  ~TransformsToAnariVisitor();

  bool preChildren(tsd::core::LayerNode &n, int level) override;
  void postChildren(tsd::core::LayerNode &n, int level) override;

 private:
  bool isIncludedAfterFiltering(const tsd::core::LayerNode &n) const;

  anari::Device m_device{nullptr};
  anari::Instance *m_currentInstance;
  std::stack<tsd::math::mat4> m_xfms;
  std::stack<bool> m_hasObjects;
  const tsd::core::Array *m_xfmArray{nullptr};
  RenderIndexFilterFcn *m_filter{nullptr};
};

// Inlined definitions ////////////////////////////////////////////////////////

inline TransformsToAnariVisitor::TransformsToAnariVisitor(
    anari::Device d, anari::Instance *instances, RenderIndexFilterFcn *filter)
    : m_device(d), m_currentInstance(instances), m_filter(filter)
{
  anari::retain(d, d);
  m_xfms.emplace(tsd::math::identity);
  m_hasObjects.emplace(false);
}

inline TransformsToAnariVisitor::~TransformsToAnariVisitor()
{
  anari::release(m_device, m_device);
}

inline bool TransformsToAnariVisitor::preChildren(
    tsd::core::LayerNode &n, int level)
{
  if (!n->isEnabled())
    return false;

  auto type = n->type();
  switch (type) {
  case ANARI_SURFACE:
  case ANARI_VOLUME:
    if (isIncludedAfterFiltering(n))
      m_hasObjects.top() = true;
    break;
  case ANARI_LIGHT: {
    m_hasObjects.top() = true;
    break;
  }
  case ANARI_FLOAT32_MAT4:
    m_xfms.push(tsd::math::mul(m_xfms.top(), n->getTransform()));
    m_hasObjects.emplace(false);
    break;
  case ANARI_ARRAY1D: {
    if (auto *a = n->getTransformArray(); a) {
      m_xfmArray = a;
      m_hasObjects.emplace(false);
    }
  }
  default:
    break;
  }

  return true;
}

inline void TransformsToAnariVisitor::postChildren(
    tsd::core::LayerNode &n, int level)
{
  if (!n->isEnabled())
    return;

  switch (n->type()) {
  case ANARI_FLOAT32_MAT4: {
    if (m_hasObjects.top()) {
      anari::Instance instance = *m_currentInstance++;
      anari::setParameter(m_device, instance, "transform", m_xfms.top());
      anari::commitParameters(m_device, instance);
    }
    m_hasObjects.pop();
    m_xfms.pop();
    break;
  }
  case ANARI_ARRAY1D: {
    if (auto *a = n->getTransformArray(); !a)
      break;

    if (m_hasObjects.top()) {
      anari::Instance instance = *m_currentInstance++;
      const auto *xfms_in = m_xfmArray->dataAs<tsd::math::mat4>();

      uint64_t stride = 0;
      auto *xfms_out = (tsd::math::mat4 *)anariMapParameterArray1D(m_device,
          instance,
          "transform",
          ANARI_FLOAT32_MAT4,
          m_xfmArray->size(),
          &stride);

      if (stride == sizeof(tsd::math::mat4)) {
        std::transform(xfms_in,
            xfms_in + m_xfmArray->size(),
            xfms_out,
            [&xfm = m_xfms.top()](
                const tsd::math::mat4 &m) { return tsd::math::mul(xfm, m); });
      } else {
        throw std::runtime_error("render index -- bad transform array stride");
      }

      anariUnmapParameterArray(m_device, instance, "transform");

      m_xfmArray = nullptr;
      anari::commitParameters(m_device, instance);
    }
    m_hasObjects.pop();
    break;
  }
  default:
    // no-op
    break;
  }
}

inline bool TransformsToAnariVisitor::isIncludedAfterFiltering(
    const tsd::core::LayerNode &n) const
{
  if (!m_filter)
    return true;

  auto type = n->type();
  if (!anari::isObject(type))
    return false;

  return (*m_filter)(n->getObject());
}

} // namespace tsd::rendering
