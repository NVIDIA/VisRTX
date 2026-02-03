// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "NetworkUpdateDelegate.hpp"
// tsd_core
#include "tsd/core/Logging.hpp"
// tsd_network
#include "tsd/network/messages/ParameterChange.hpp"

#include "../RenderSession.hpp"

namespace tsd::network {

NetworkUpdateDelegate::NetworkUpdateDelegate(
    tsd::network::NetworkChannel *channel)
    : m_channel(channel)
{
  // no-op
}

void NetworkUpdateDelegate::setNetworkChannel(
    tsd::network::NetworkChannel *channel)
{
  m_channel = channel;
}

void NetworkUpdateDelegate::signalObjectAdded(const tsd::core::Object *)
{
  tsd::core::logWarning(
      "NetworkUpdateDelegate::signalObjectAdded not implemented");
}

void NetworkUpdateDelegate::signalParameterUpdated(
    const tsd::core::Object *o, const tsd::core::Parameter *p)
{
  if (!m_channel) {
    tsd::core::logError(
        "NetworkUpdateDelegate::signalParameterUpdated: no network channel");
    return;
  }
  auto msg = tsd::network::messages::ParameterChange(o, p);
  m_channel->send(MessageType::SERVER_SET_OBJECT_PARAMETER, std::move(msg));
}

void NetworkUpdateDelegate::signalParameterRemoved(
    const tsd::core::Object *, const tsd::core::Parameter *)
{
  tsd::core::logWarning(
      "NetworkUpdateDelegate::signalParameterRemoved not implemented");
}

void NetworkUpdateDelegate::signalArrayMapped(const tsd::core::Array *)
{
  tsd::core::logWarning(
      "NetworkUpdateDelegate::signalArrayMapped not implemented");
}

void NetworkUpdateDelegate::signalArrayUnmapped(const tsd::core::Array *)
{
  tsd::core::logWarning(
      "NetworkUpdateDelegate::signalArrayUnmapped not implemented");
}

void NetworkUpdateDelegate::signalObjectParameterUseCountZero(
    const tsd::core::Object *obj)
{
  tsd::core::logWarning(
      "NetworkUpdateDelegate::signalObjectParameterUseCountZero not implemented");
}

void NetworkUpdateDelegate::signalObjectLayerUseCountZero(
    const tsd::core::Object *obj)
{
  tsd::core::logWarning(
      "NetworkUpdateDelegate::signalObjectLayerUseCountZero not implemented");
}

void NetworkUpdateDelegate::signalObjectRemoved(const tsd::core::Object *)
{
  tsd::core::logWarning(
      "NetworkUpdateDelegate::signalObjectRemoved not implemented");
}

void NetworkUpdateDelegate::signalRemoveAllObjects()
{
  tsd::core::logWarning(
      "NetworkUpdateDelegate::signalRemoveAllObjects not implemented");
}

void NetworkUpdateDelegate::signalLayerAdded(const tsd::core::Layer *)
{
  tsd::core::logWarning(
      "NetworkUpdateDelegate::signalLayerAdded not implemented");
}

void NetworkUpdateDelegate::signalLayerUpdated(const tsd::core::Layer *)
{
  tsd::core::logWarning(
      "NetworkUpdateDelegate::signalLayerUpdated not implemented");
}

void NetworkUpdateDelegate::signalLayerRemoved(const tsd::core::Layer *)
{
  tsd::core::logWarning(
      "NetworkUpdateDelegate::signalLayerRemoved not implemented");
}

void NetworkUpdateDelegate::signalActiveLayersChanged()
{
  tsd::core::logWarning(
      "NetworkUpdateDelegate::signalActiveLayersChanged not implemented");
}

void NetworkUpdateDelegate::signalObjectFilteringChanged()
{
  tsd::core::logWarning(
      "NetworkUpdateDelegate::signalObjectFilteringChanged not implemented");
}

void NetworkUpdateDelegate::signalInvalidateCachedObjects()
{
  tsd::core::logWarning(
      "NetworkUpdateDelegate::signalInvalidateCachedObjects not implemented");
}

void NetworkUpdateDelegate::signalAnimationTimeChanged(float)
{
  tsd::core::logWarning(
      "NetworkUpdateDelegate::signalAnimationTimeChanged not implemented");
}

} // namespace tsd::network
