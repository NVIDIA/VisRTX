// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_core
#include "tsd/scene/Scene.hpp"
#include "tsd/scene/UpdateDelegate.hpp"
// tsd_network
#include "tsd/network/NetworkChannel.hpp"

namespace tsd::network {

struct NetworkUpdateDelegate : public tsd::scene::BaseUpdateDelegate
{
  NetworkUpdateDelegate(
      tsd::scene::Scene *scene, tsd::network::NetworkChannel *channel);
  ~NetworkUpdateDelegate() override = default;

  void setEnabled(bool enabled);
  void setNetworkChannel(tsd::network::NetworkChannel *channel);

  // Update signals //

  void signalObjectAdded(const tsd::scene::Object *) override;
  void signalParameterUpdated(
      const tsd::scene::Object *, const tsd::scene::Parameter *) override;
  void signalParameterRemoved(
      const tsd::scene::Object *, const tsd::scene::Parameter *) override;
  void signalParameterBatchUpdated(const tsd::scene::Object *,
      const std::vector<const tsd::scene::Parameter *> &) override;
  void signalArrayMapped(const tsd::scene::Array *) override;
  void signalArrayUnmapped(const tsd::scene::Array *) override;
  void signalObjectParameterUseCountZero(const tsd::scene::Object *obj) override;
  void signalObjectLayerUseCountZero(const tsd::scene::Object *obj) override;
  void signalObjectRemoved(const tsd::scene::Object *) override;
  void signalRemoveAllObjects() override;
  void signalLayerAdded(const tsd::scene::Layer *) override;
  void signalLayerStructureUpdated(const tsd::scene::Layer *) override;
  void signalLayerTransformUpdated(const tsd::scene::Layer *) override;
  void signalLayerRemoved(const tsd::scene::Layer *) override;
  void signalActiveLayersChanged() override;
  void signalObjectFilteringChanged() override;
  void signalInvalidateCachedObjects() override;
  void signalAnimationTimeChanged(float) override;

 private:
  bool isReady(const char *fcn) const;

  tsd::scene::Scene *m_scene{nullptr};
  tsd::network::NetworkChannel *m_channel{nullptr};
  bool m_enabled{true};
};

} // namespace tsd::network
