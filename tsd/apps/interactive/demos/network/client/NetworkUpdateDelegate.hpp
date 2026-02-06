// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_core
#include "tsd/core/scene/Scene.hpp"
#include "tsd/core/scene/UpdateDelegate.hpp"
// tsd_network
#include "tsd/network/NetworkChannel.hpp"

namespace tsd::network {

struct NetworkUpdateDelegate : public tsd::core::BaseUpdateDelegate
{
  NetworkUpdateDelegate(
      tsd::core::Scene *scene, tsd::network::NetworkChannel *channel);
  ~NetworkUpdateDelegate() override = default;

  void setEnabled(bool enabled);
  void setNetworkChannel(tsd::network::NetworkChannel *channel);

  // Update signals //

  void signalObjectAdded(const tsd::core::Object *) override;
  void signalParameterUpdated(
      const tsd::core::Object *, const tsd::core::Parameter *) override;
  void signalParameterRemoved(
      const tsd::core::Object *, const tsd::core::Parameter *) override;
  void signalArrayMapped(const tsd::core::Array *) override;
  void signalArrayUnmapped(const tsd::core::Array *) override;
  void signalObjectParameterUseCountZero(const tsd::core::Object *obj) override;
  void signalObjectLayerUseCountZero(const tsd::core::Object *obj) override;
  void signalObjectRemoved(const tsd::core::Object *) override;
  void signalRemoveAllObjects() override;
  void signalLayerAdded(const tsd::core::Layer *) override;
  void signalLayerUpdated(const tsd::core::Layer *) override;
  void signalLayerRemoved(const tsd::core::Layer *) override;
  void signalActiveLayersChanged() override;
  void signalObjectFilteringChanged() override;
  void signalInvalidateCachedObjects() override;
  void signalAnimationTimeChanged(float) override;

 private:
  tsd::core::Scene *m_scene{nullptr};
  tsd::network::NetworkChannel *m_channel{nullptr};
  bool m_enabled{true};
};

} // namespace tsd::network
