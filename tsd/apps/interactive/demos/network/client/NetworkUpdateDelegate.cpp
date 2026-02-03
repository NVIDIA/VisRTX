// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "NetworkUpdateDelegate.hpp"
// tsd_core
#include "tsd/core/Logging.hpp"
// tsd_network
#include "tsd/network/messages/NewObject.hpp"
#include "tsd/network/messages/ParameterChange.hpp"
#include "tsd/network/messages/TransferArrayData.hpp"
#include "tsd/network/messages/TransferLayer.hpp"

#include "../RenderSession.hpp"

namespace tsd::network {

NetworkUpdateDelegate::NetworkUpdateDelegate(
    tsd::core::Scene *scene, tsd::network::NetworkChannel *channel)
    : m_scene(scene)
{
  setNetworkChannel(channel);
}

void NetworkUpdateDelegate::setEnabled(bool enabled)
{
  m_enabled = enabled;
}

void NetworkUpdateDelegate::setNetworkChannel(
    tsd::network::NetworkChannel *channel)
{
  m_channel = channel;
}

void NetworkUpdateDelegate::signalObjectAdded(const tsd::core::Object *o)
{
  if (!m_enabled)
    return;
  else if (!m_channel) {
    tsd::core::logError(
        "NetworkUpdateDelegate::signalObjectAdded: no network channel");
    return;
  }
  auto msg = tsd::network::messages::NewObject(o);
  m_channel->send(MessageType::SERVER_ADD_OBJECT, std::move(msg));
}

void NetworkUpdateDelegate::signalParameterUpdated(
    const tsd::core::Object *o, const tsd::core::Parameter *p)
{
  if (!m_enabled)
    return;
  else if (!m_channel) {
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
  if (!m_enabled)
    return;
  tsd::core::logWarning(
      "NetworkUpdateDelegate::signalParameterRemoved not implemented");
}

void NetworkUpdateDelegate::signalArrayMapped(const tsd::core::Array *)
{
  if (!m_enabled)
    return;
  // no-op
}

void NetworkUpdateDelegate::signalArrayUnmapped(const tsd::core::Array *a)
{
  if (!m_enabled)
    return;
  else if (!m_channel) {
    tsd::core::logError(
        "NetworkUpdateDelegate::signalArrayUnmapped: no network channel");
    return;
  }
  auto msg = tsd::network::messages::TransferArrayData(a);
  m_channel->send(MessageType::SERVER_SET_ARRAY_DATA, std::move(msg));
}

void NetworkUpdateDelegate::signalObjectParameterUseCountZero(
    const tsd::core::Object *obj)
{
  if (!m_enabled)
    return;
  // no-op
}

void NetworkUpdateDelegate::signalObjectLayerUseCountZero(
    const tsd::core::Object *obj)
{
  if (!m_enabled)
    return;
  // no-op
}

void NetworkUpdateDelegate::signalObjectRemoved(const tsd::core::Object *)
{
  if (!m_enabled)
    return;
  tsd::core::logWarning(
      "NetworkUpdateDelegate::signalObjectRemoved not implemented");
}

void NetworkUpdateDelegate::signalRemoveAllObjects()
{
  if (!m_enabled)
    return;
  tsd::core::logWarning(
      "NetworkUpdateDelegate::signalRemoveAllObjects not implemented");
}

void NetworkUpdateDelegate::signalLayerAdded(const tsd::core::Layer *l)
{
  if (!m_enabled)
    return;
  else if (!m_channel) {
    tsd::core::logError(
        "NetworkUpdateDelegate::signalLayerAdded: no network channel");
    return;
  }
  auto msg = tsd::network::messages::TransferLayer(
      m_scene, const_cast<tsd::core::Layer *>(l));
  m_channel->send(MessageType::SERVER_UPDATE_LAYER, std::move(msg));
}

void NetworkUpdateDelegate::signalLayerUpdated(const tsd::core::Layer *l)
{
  if (!m_enabled)
    return;
  else if (!m_channel) {
    tsd::core::logError(
        "NetworkUpdateDelegate::signalLayerUpdated: no network channel");
    return;
  }
  auto msg = tsd::network::messages::TransferLayer(
      m_scene, const_cast<tsd::core::Layer *>(l));
  m_channel->send(MessageType::SERVER_UPDATE_LAYER, std::move(msg));
}

void NetworkUpdateDelegate::signalLayerRemoved(const tsd::core::Layer *)
{
  if (!m_enabled)
    return;
  tsd::core::logWarning(
      "NetworkUpdateDelegate::signalLayerRemoved not implemented");
}

void NetworkUpdateDelegate::signalActiveLayersChanged()
{
  if (!m_enabled)
    return;
  tsd::core::logWarning(
      "NetworkUpdateDelegate::signalActiveLayersChanged not implemented");
}

void NetworkUpdateDelegate::signalObjectFilteringChanged()
{
  if (!m_enabled)
    return;
  tsd::core::logWarning(
      "NetworkUpdateDelegate::signalObjectFilteringChanged not implemented");
}

void NetworkUpdateDelegate::signalInvalidateCachedObjects()
{
  if (!m_enabled)
    return;
  tsd::core::logWarning(
      "NetworkUpdateDelegate::signalInvalidateCachedObjects not implemented");
}

void NetworkUpdateDelegate::signalAnimationTimeChanged(float)
{
  if (!m_enabled)
    return;
  tsd::core::logWarning(
      "NetworkUpdateDelegate::signalAnimationTimeChanged not implemented");
}

} // namespace tsd::network
