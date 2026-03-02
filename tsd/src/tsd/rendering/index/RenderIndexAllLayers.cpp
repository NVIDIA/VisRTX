// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "RenderIndexAllLayers.hpp"
#include "RenderToAnariObjectsVisitor.hpp"

// tsd_core
#include "tsd/core/Logging.hpp"
#include "tsd/core/ObjectPool.hpp"
#include "tsd/core/TSDMath.hpp"
#include "tsd/core/TSDTypes.hpp"
#include "tsd/core/scene/Layer.hpp"
#include "tsd/core/scene/Object.hpp"
#include "tsd/core/scene/objects/Array.hpp"
#include "tsd/core/scene/objects/Transform.hpp"

// anari
#include <anari/anari_cpp/Traits.h>
#include <anari/anari_cpp/ext/linalg.h>
#include <anari/frontend/anari_enums.h>
#include <helium/utility/TimeStamp.h>

// std
#include <algorithm>
#include <anari/anari_cpp.hpp>
#include <cstring>
#include <variant>

namespace tsd::rendering {

// Helpers

namespace {

// Transform composition

mat4 composeTransform(const mat4 &a, const mat4 &b)
{
  return mul(a, b);
}

std::vector<mat4> composeTransform(
    const mat4 &a, const mat4 *begin, const mat4 *end)
{
  std::vector<mat4> res;
  res.reserve(end - begin);
  for (auto it = begin; it != end; ++it)
    res.push_back(mul(a, *it));
  return res;
}

std::vector<mat4> composeTransform(
    const mat4 *begin, const mat4 *end, const mat4 &b)
{
  std::vector<mat4> res;
  res.reserve(end - begin);
  for (auto it = begin; it != end; ++it)
    res.push_back(mul(*it, b));
  return res;
}

std::vector<mat4> composeTransform(
    const mat4 *beginA, const mat4 *endA, const mat4 *beginB, const mat4 *endB)
{
  std::vector<mat4> res;
  res.reserve((endA - beginA) * (endB - beginB));
  for (auto itA = beginA; itA != endA; ++itA)
    for (auto itB = beginB; itB != endB; ++itB)
      res.push_back(mul(*itA, *itB));

  return res;
}

// Attribute composition

RenderIndexAllLayers::AttributeValue broadcastValue(
    const void *src, ANARIDataType type, size_t n)
{
  auto elementSize = anari::sizeOf(type);
  RenderIndexAllLayers::AttributeValue result;
  result.elementType = type;
  result.data.resize(elementSize * n);
  auto *dst = result.data.data();
  for (size_t i = 0; i < n; ++i)
    std::memcpy(dst + i * elementSize, src, elementSize);
  return result;
}

static RenderIndexAllLayers::AttributeValue repeatBlock(
    const void *src, size_t srcCount, ANARIDataType type, size_t n)
{
  auto es = anari::sizeOf(type);
  auto blockSize = es * srcCount;
  RenderIndexAllLayers::AttributeValue result;
  result.elementType = type;
  result.data.resize(blockSize * n);
  auto *dst = result.data.data();
  for (size_t i = 0; i < n; ++i)
    std::memcpy(dst + i * blockSize, src, blockSize);
  return result;
}

static RenderIndexAllLayers::AttributeValue expandEach(
    const RenderIndexAllLayers::AttributeValue &src, size_t n)
{
  auto es = src.elementSize();
  auto srcCount = src.count();
  RenderIndexAllLayers::AttributeValue result;
  result.elementType = src.elementType;
  result.data.resize(es * srcCount * n);
  auto *dst = result.data.data();
  auto *srcPtr = src.data.data();
  for (size_t i = 0; i < srcCount; ++i) {
    for (size_t j = 0; j < n; ++j)
      std::memcpy(dst + (i * n + j) * es, srcPtr + i * es, es);
  }
  return result;
}

} // namespace

// AttributeValue //////////////////////////////////////////////////////////////

bool RenderIndexAllLayers::AttributeValue::empty() const
{
  return data.empty();
}

size_t RenderIndexAllLayers::AttributeValue::elementSize() const
{
  return anari::sizeOf(elementType);
}

size_t RenderIndexAllLayers::AttributeValue::count() const
{
  auto count = elementSize();
  return count ? data.size() / count : 0;
}

// Attribute byte-level helpers ////////////////////////////////////////////////

static size_t instanceCount(
    const RenderIndexAllLayers::TransformCache::Value &v)
{
  if (std::holds_alternative<math::mat4>(v))
    return 1;
  return std::get<std::vector<math::mat4>>(v).size();
}

// RenderIndexAllLayers definitions ///////////////////////////////////////////

RenderIndexAllLayers::RenderIndexAllLayers(
    Scene &scene, tsd::core::Token deviceName, anari::Device d)
    : RenderIndex(scene, deviceName, d)
{
  m_includedLayers = scene.getActiveLayers();
}

RenderIndexAllLayers::~RenderIndexAllLayers()
{
  for (auto &&[_, instanceCache] : m_layerInstanceCache) {
    for (auto &instance : instanceCache)
      anari::release(device(), instance);
  }
  for (auto &&[_, rootInst] : m_layerRootInstance)
    anari::release(device(), rootInst);
}

bool RenderIndexAllLayers::isFlat() const
{
  return false;
}

void RenderIndexAllLayers::setFilterFunction(RenderIndexFilterFcn f)
{
  m_filter = f;
  m_filterForceUpdate = true;
  signalObjectFilteringChanged();
}

void RenderIndexAllLayers::setIncludedLayers(
    const std::vector<const Layer *> &layers)
{
  m_includedLayers = layers;
  m_customIncludedLayers = !layers.empty();
  signalActiveLayersChanged();
}

void RenderIndexAllLayers::signalArrayUnmapped(const Array *a)
{
  bool hasChanged = false;

  const auto &transforms = m_ctx->objectDB().transform;
  for (auto txIndex = 0; txIndex < transforms.capacity(); ++txIndex) {
    if (const auto &tx = transforms.at(txIndex)) {
      for (auto paramIndex = 0; paramIndex < tx->numParameters();
          ++paramIndex) {
        const auto &param = tx->parameterAt(paramIndex);
        const auto &value = param.value();

        if (value.holdsObject() && m_ctx->getObject(value) == a) {
          hasChanged |= invalidateTransformAtObjectIndex(tx.index());
        }
      }
    }
  }

  if (hasChanged) {
    updateWorld();
  }
  RenderIndex::signalArrayUnmapped(a);
}

void RenderIndexAllLayers::signalParameterUpdated(
    const Object *o, const Parameter *p)
{
  if (o->type() == TSD_TRANSFORM) {
    if (invalidateTransformAtObjectIndex(o->index()))
      updateWorld();
  }

  RenderIndex::signalParameterUpdated(o, p);
}

void RenderIndexAllLayers::signalObjectLayerUseCountZero(const Object *o)
{
  if (o->type() == TSD_TRANSFORM) {
    auto objectIndex = o->index();
    std::vector<const Layer *> layersToClear;
    // Invalidate all layers holding that very object
    for (auto &&[layer, transformDependency] : m_layerTransformDependency) {
      for (const auto &transform : transformDependency) {
        if (transform.transformObjectIndex == objectIndex) {
          layersToClear.push_back(layer);
          break;
        }
      }
    }

    for (auto l : layersToClear) {
      tagDirtyTopology(l);
    }
  }

  updateWorld();
  RenderIndex::signalObjectLayerUseCountZero(o);
}

void RenderIndexAllLayers::signalLayerAdded(const Layer *l)
{
  m_includedLayers.push_back(l);
  updateWorld();
}

void RenderIndexAllLayers::signalLayerUpdated(const Layer *l)
{
  tagDirtyTopology(l);
  updateWorld();
}

void RenderIndexAllLayers::signalLayerRemoved(const Layer *l)
{
  tagDirtyTopology(l);
  m_includedLayers.erase(
      std::remove(m_includedLayers.begin(), m_includedLayers.end(), l),
      m_includedLayers.end());
  updateWorld();
}

void RenderIndexAllLayers::signalActiveLayersChanged()
{
  if (!m_customIncludedLayers)
    m_includedLayers = m_ctx->getActiveLayers();

  signalInvalidateCachedObjects();

  RenderIndex::signalActiveLayersChanged();
}

void RenderIndexAllLayers::signalObjectFilteringChanged()
{
  if (m_filter || m_filterForceUpdate) {
    m_filterForceUpdate = false;
    updateWorld();
  }

  RenderIndex::signalObjectFilteringChanged();
}

void RenderIndexAllLayers::signalAnimationTimeChanged(float)
{
  // Transform/value updates are applied through signalParameterUpdated().
}

void RenderIndexAllLayers::signalRemoveAllObjects()
{
  RenderIndex::signalRemoveAllObjects();
  m_includedLayers.clear();
  m_layerTransformDependency.clear();
  m_layerTransformCache.clear();
  for (auto &&[_, instanceCache] : m_layerInstanceCache) {
    for (auto &instance : instanceCache)
      anari::release(device(), instance);
    instanceCache.clear();
  }
  m_layerInstanceCache.clear();
  for (auto &&[_, rootInst] : m_layerRootInstance)
    anari::release(device(), rootInst);
  m_layerRootInstance.clear();
  m_layerNodeInstanceParams.clear();
  m_layerAttributeCache.clear();

  m_transformLastUpdateTimeStamp = helium::newTimeStamp();
}

void RenderIndexAllLayers::updateWorld()
{
  auto d = device();
  auto w = world();

  auto effectiveLayers = m_includedLayers;

  bool needWorldInstanceRebuild = false;
  for (const auto *l : effectiveLayers) {
    if (!m_layerTransformDependency.contains(l)) {
      needWorldInstanceRebuild = true;
      updateLayerTransformDependency(l);
      RenderToAnariObjectsVisitor visitor(d,
          m_cache,
          m_layerInstanceCache[l],
          m_layerRootInstance[l],
          m_filter);
      const_cast<Layer *>(l)->traverse(l->root(), visitor);
      visitor.finalizeRootObjects();
    }

    bool needInstanceArraysUpdate = updateLayerTransformCache(l);
    updateLayerAttributeCache(l);

    if (needInstanceArraysUpdate)
      writeTransformCacheToInstances(l);
  }
  m_transformLastUpdateTimeStamp = helium::newTimeStamp();

  if (needWorldInstanceRebuild) {
    std::vector<anari::Instance> allInstances;
    for (const auto *l : effectiveLayers) {
      auto *deps = m_layerTransformDependency.at(l);
      auto &instances = m_layerInstanceCache[l];
      for (size_t i = 0; i < instances.size(); i++) {
        if ((*deps)[i].transformObjectIndex != INVALID_INDEX && instances[i])
          allInstances.push_back(instances[i]);
      }
      if (auto *rootInst = m_layerRootInstance.at(l); rootInst && *rootInst)
        allInstances.push_back(*rootInst);
    }

    if (!allInstances.empty()) {
      anari::setParameterArray1D(device(),
          m_world,
          "instance",
          allInstances.data(),
          allInstances.size());
    } else {
      anari::unsetParameter(device(), m_world, "instance");
    }

    anari::commitParameters(d, w);
  }
}

void RenderIndexAllLayers::updateLayerTransformDependency(const Layer *l)
{
  // Dependency/cache/instance vectors are indexed by transform traversal order
  // (entry index), not by transform object pool index.
  // We only use pool capacity here as a conservative reserve upper bound.
  auto xfmCapacity = m_ctx->objectDB().transform.capacity();

  std::vector<TransformDependency> orderedTransforms;
  std::vector<InstanceParameterMap> nodeInstanceParams;
  // Final size equals the number of enabled transform nodes in this layer.
  orderedTransforms.reserve(xfmCapacity);
  nodeInstanceParams.reserve(xfmCapacity);

  auto mutableLayer = const_cast<Layer *>(l);
  size_t currentOrderedIndex = INVALID_INDEX;

  mutableLayer->traverse(
      mutableLayer->root(),
      [&](LayerNode &n, int /*level*/) {
        if (!n->isEnabled())
          return false; // match RenderToAnariObjectsVisitor subtree pruning
        if (!n->isTransform())
          return true;

        auto transformObject = n->getTransformObject();

        const size_t parentOrderedIndex = currentOrderedIndex;
        currentOrderedIndex = orderedTransforms.size();
        orderedTransforms.push_back({n->getObjectIndex(),
            parentOrderedIndex,
            n->name().size() ? n->name() : transformObject->name()});
        nodeInstanceParams.push_back(
            transformObject->getInstanceParameterMap());

        return true;
      },
      [&](LayerNode &n, int /*level*/) {
        if (!n->isEnabled())
          return true;
        if (!n->isTransform())
          return true;

        currentOrderedIndex = currentOrderedIndex == INVALID_INDEX
            ? INVALID_INDEX
            : orderedTransforms[currentOrderedIndex].parentOrderedIndex;
        return true;
      });

  auto xfmCount = orderedTransforms.size();
  m_layerTransformDependency[l] = std::move(orderedTransforms);
  m_layerNodeInstanceParams[l] = std::move(nodeInstanceParams);
  m_layerTransformCache[l].clear();
  m_layerTransformCache[l].resize(
      xfmCount, {math::IDENTITY_MAT4, helium::newTimeStamp()});
  m_layerAttributeCache[l].clear();
  m_layerAttributeCache[l].resize(xfmCount);

  // Release any previously-allocated instances before recreating
  if (auto *oldInstances = m_layerInstanceCache.at(l)) {
    for (auto &inst : *oldInstances)
      anari::release(device(), inst);
  }
  m_layerInstanceCache[l].clear();
  m_layerInstanceCache[l].reserve(orderedTransforms.size());
  std::generate_n(
      back_inserter(m_layerInstanceCache[l]), xfmCount, [d = device()]() {
        return anari::newObject<anari::Instance>(d, "transform");
      });

  if (auto *oldRoot = m_layerRootInstance.at(l))
    anari::release(device(), *oldRoot);
  m_layerRootInstance[l] =
      anari::newObject<anari::Instance>(device(), "transform");
}

bool RenderIndexAllLayers::updateLayerTransformCache(const Layer *l)
{
  auto *orderedTransforms = m_layerTransformDependency.at(l);
  auto *transformCache = m_layerTransformCache.at(l);

  const TransformCache identityCache = {math::IDENTITY_MAT4, 0};

  auto thisTs = helium::newTimeStamp();

  bool hasUpdate = false;

  for (size_t transformOrderedIndex = 0;
      transformOrderedIndex < orderedTransforms->size();
      ++transformOrderedIndex) {
    const auto &t = (*orderedTransforms)[transformOrderedIndex];
    auto transformObjectIndex = t.transformObjectIndex;
    if (transformObjectIndex == INVALID_INDEX)
      continue;

    auto parentOrderedIndex = t.parentOrderedIndex;
    bool hasParent = parentOrderedIndex != INVALID_INDEX;

    if ((*transformCache)[transformOrderedIndex].timestamp
            < m_transformLastUpdateTimeStamp
        && (!hasParent
            || (*transformCache)[parentOrderedIndex].timestamp
                < m_transformLastUpdateTimeStamp)) {
      continue;
    }

    const auto *parentCache =
        hasParent ? &(*transformCache)[parentOrderedIndex] : &identityCache;

    auto thisTransformRef =
        m_ctx->objectDB().transform.at(transformObjectIndex);
    auto *thisTransform = thisTransformRef.data();
    auto thisXfm = thisTransform->getTransformAsAny();
    auto *thisCache = &(*transformCache)[transformOrderedIndex];

    if (thisXfm.is<math::mat4>()) {
      const auto xfm = thisXfm.getAs<math::mat4>();

      if (std::holds_alternative<math::mat4>(parentCache->value)) {
        const auto &parentXfm = std::get<math::mat4>(parentCache->value);
        thisCache->value = composeTransform(parentXfm, xfm);
      } else {
        const auto &parentXfms =
            std::get<std::vector<math::mat4>>(parentCache->value);
        thisCache->value =
            composeTransform(&*cbegin(parentXfms), &*cend(parentXfms), xfm);
      }
    } else if (auto xfmArray =
                   m_ctx->getObject<Array>(thisXfm.getAsObjectIndex());
        xfmArray && xfmArray->elementType() == ANARI_FLOAT32_MAT4) {
      const auto xfmsCount = xfmArray->size();
      const auto *xfms = xfmArray->dataAs<math::mat4>();

      if (std::holds_alternative<math::mat4>(parentCache->value)) {
        const auto &parentXfm = std::get<math::mat4>(parentCache->value);
        thisCache->value = composeTransform(parentXfm, xfms, xfms + xfmsCount);
      } else {
        const auto &parentXfms =
            std::get<std::vector<math::mat4>>(parentCache->value);
        thisCache->value = composeTransform(
            &*cbegin(parentXfms), &*cend(parentXfms), xfms, xfms + xfmsCount);
      }
    } else {
      logWarning("Unexpected transformation value type, ignoring...\n");
      thisCache->value = math::IDENTITY_MAT4;
    }

    hasUpdate = true;
    thisCache->timestamp = thisTs;
  }

  return hasUpdate;
}

void RenderIndexAllLayers::updateLayerAttributeCache(const Layer *l)
{
  auto *orderedTransforms = m_layerTransformDependency.at(l);
  auto *transformCache = m_layerTransformCache.at(l);
  auto *nodeParams = m_layerNodeInstanceParams.at(l);
  auto *attributeCache = m_layerAttributeCache.at(l);

  for (size_t transformOrderedIndex = 0;
      transformOrderedIndex < orderedTransforms->size();
      ++transformOrderedIndex) {
    const auto &t = (*orderedTransforms)[transformOrderedIndex];
    auto transformObjectIndex = t.transformObjectIndex;
    if (transformObjectIndex == INVALID_INDEX)
      continue;

    auto parentOrderedIndex = t.parentOrderedIndex;
    bool hasParent = parentOrderedIndex != INVALID_INDEX;

    if ((*transformCache)[transformOrderedIndex].timestamp
            < m_transformLastUpdateTimeStamp
        && (!hasParent
            || (*transformCache)[parentOrderedIndex].timestamp
                < m_transformLastUpdateTimeStamp)) {
      continue;
    }

    const auto *parentXfmCache =
        hasParent ? &(*transformCache)[parentOrderedIndex] : nullptr;
    auto parentElementCount =
        hasParent ? instanceCount(parentXfmCache->value) : 1;

    auto thisTransformRef =
        m_ctx->objectDB().transform.at(transformObjectIndex);
    auto *thisTransform = thisTransformRef.data();

    auto *thisXfmCache = &(*transformCache)[transformOrderedIndex];
    auto *thisCache = &(*attributeCache)[transformOrderedIndex];
    auto thisElementCount = instanceCount(thisXfmCache->value);

    for (auto &&[name, value] : thisTransform->getInstanceParameterMap()) {
      auto &thisAttributeCache = (*thisCache)[name];
      if (anari::isArray(value.type())) {
        auto array = m_ctx->getObject<Array>(value.getAsObjectIndex());
        thisAttributeCache.elementType = array->elementType();
        if (!array || !array->data())
          continue;
        thisAttributeCache = repeatBlock(array->data(),
            array->size(),
            array->elementType(),
            parentElementCount);
      } else {
        thisAttributeCache = broadcastValue(value.data(), value.type(), 1);
      }
    }

    if (const auto *parentCache =
            hasParent ? &(*attributeCache)[parentOrderedIndex] : nullptr) {
      for (auto &&[name, value] : (*parentCache)) {
        if (thisCache->contains(name))
          continue;

        auto &thisAttributeCache = (*thisCache)[name];

        if (value.data.size() > 1) {
          thisAttributeCache = expandEach(value, thisElementCount);
        } else if (value.data.size() == 1) {
          thisAttributeCache = value;
        } else
          continue;
      }
    }
  }
}

void RenderIndexAllLayers::writeTransformCacheToInstances(const Layer *layer)
{
  auto d = device();

  auto *transformCache = m_layerTransformCache.at(layer);
  auto *transformDependency = m_layerTransformDependency.at(layer);
  auto *attrCacheVec = m_layerAttributeCache.at(layer);
  auto &instanceCache = m_layerInstanceCache[layer];
  for (size_t transformIndex = 0; transformIndex < transformCache->size();
      transformIndex++) {
    if (!transformDependency
        || (*transformDependency)[transformIndex].transformObjectIndex
            == INVALID_INDEX)
      continue;

    const auto &transform = (*transformCache)[transformIndex];
    auto instance = instanceCache[transformIndex];
    if (!instance)
      continue;

    if (std::holds_alternative<std::vector<math::mat4>>(transform.value)) {
      const auto &array = std::get<std::vector<math::mat4>>(transform.value);
      uint64_t stride = 0;
      auto *xfms = (math::mat4 *)anariMapParameterArray1D(
          d, instance, "transform", ANARI_FLOAT32_MAT4, array.size(), &stride);

      if (stride == sizeof(math::mat4))
        std::copy(cbegin(array), cend(array), xfms);

      anariUnmapParameterArray(d, instance, "transform");
    } else {
      anari::setParameter(
          d, instance, "transform", std::get<math::mat4>(transform.value));
    }

    // Write resolved instance attributes
    if (attrCacheVec) {
      const auto &attrs = (*attrCacheVec)[transformIndex];
      for (const auto &[name, av] : attrs) {
        if (av.empty())
          continue;
        if (av.count() == 1) {
          anariSetParameter(
              d, instance, name.c_str(), av.elementType, av.data.data());
        } else {
          uint64_t stride = 0;
          auto *dst = (uint8_t *)anariMapParameterArray1D(
              d, instance, name.c_str(), av.elementType, av.count(), &stride);
          if (stride == av.elementSize())
            std::memcpy(dst, av.data.data(), av.data.size());
          anariUnmapParameterArray(d, instance, name.c_str());
        }
      }
    }

    anari::commitParameters(d, instance);
  }
}

bool RenderIndexAllLayers::invalidateTransformAtObjectIndex(
    size_t objectIndex, const Layer *layer)
{
  bool didInvalidate = false;

  auto &transformDependency = *m_layerTransformDependency.at(layer);
  auto &transformCache = *m_layerTransformCache.at(layer);
  auto &instanceCache = *m_layerInstanceCache.at(layer);
  for (size_t i = 0; i < transformDependency.size(); ++i) {
    if (transformDependency[i].transformObjectIndex == objectIndex) {
      transformCache[i].timestamp = helium::newTimeStamp();
      didInvalidate = true;
    }
  }

  return didInvalidate;
}

bool RenderIndexAllLayers::invalidateTransformAtObjectIndex(size_t index)
{
  bool didInvalidate = false;

  for (auto &&[layer, _] : m_layerTransformDependency) {
    didInvalidate |= invalidateTransformAtObjectIndex(index, layer);
  }

  return didInvalidate;
}

void RenderIndexAllLayers::tagDirtyTopology(const Layer *l)
{
  m_layerTransformDependency.erase(l);
  m_layerTransformCache.erase(l);
  m_layerNodeInstanceParams.erase(l);
  m_layerAttributeCache.erase(l);
  if (auto *instances = m_layerInstanceCache.at(l)) {
    for (auto &inst : *instances)
      anari::release(device(), inst);
  }
  m_layerInstanceCache.erase(l);
  if (auto *rootInst = m_layerRootInstance.at(l)) {
    anari::release(device(), *rootInst);
    m_layerRootInstance.erase(l);
  }
}
} // namespace tsd::rendering
