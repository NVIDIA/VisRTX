// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/scene/objects/Camera.hpp"
#include <utility>
#include <memory>

namespace tsd::core {

class CameraUpdateDelegate : public BaseUpdateDelegate {
public:
  using UpdateToken = size_t;
  using DeletedCallback = std::function<void()>;

  CameraUpdateDelegate(Camera *camera);
  ~CameraUpdateDelegate();

  void signalParameterUpdated(const Object *o, const Parameter *p) override;

  // Empty overrides for all pure virtuals from BaseUpdateDelegate
  void signalObjectAdded(const Object *o) override {}
  void signalParameterRemoved(const Object *o, const Parameter *p) override {}
  void signalArrayMapped(const Array *a) override {}
  void signalArrayUnmapped(const Array *a) override {}
  void signalObjectParameterUseCountZero(const Object *obj) override {}
  void signalObjectLayerUseCountZero(const Object *obj) override {}
  void signalObjectRemoved(const Object *o) override {};
  void signalRemoveAllObjects() override {}
  void signalLayerAdded(const Layer *l) override {}
  void signalLayerUpdated(const Layer *l) override {}
  void signalLayerRemoved(const Layer *l) override {}
  void signalActiveLayersChanged() override {}
  void signalObjectFilteringChanged() override {}
  void signalInvalidateCachedObjects() override {}

  bool hasChanged(UpdateToken &t) const;
  void detach();

  void pushIgnoreChangeScope();
  void popIgnoreChangeScope();

private:
  Camera *m_camera = nullptr;
  UpdateToken m_token = 1;
  int m_ignoreChangesScope = 0;
};

} // namespace tsd::core
