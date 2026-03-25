// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "NetworkUpdateDelegate.hpp"
// tsd_core
#include "tsd/core/Logging.hpp"
// tsd_network
#include "tsd/network/messages/NewObject.hpp"
#include "tsd/network/messages/ParameterChange.hpp"
#include "tsd/network/messages/ParameterRemove.hpp"
#include "tsd/network/messages/RemoveObject.hpp"
#include "tsd/network/messages/TransferArrayData.hpp"
#include "tsd/network/messages/TransferLayer.hpp"

#include "../RenderSession.hpp"

#define CHECK_READY_OR_RETURN()                                                \
  if (!isReady(__func__))                                                      \
    return;

namespace tsd::network {

NetworkUpdateDelegate::NetworkUpdateDelegate(
    tsd::scene::Scene *scene, tsd::network::NetworkChannel *channel)
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

void NetworkUpdateDelegate::signalObjectAdded(const tsd::scene::Object *o)
{
  CHECK_READY_OR_RETURN();
  auto msg = tsd::network::messages::NewObject(o);
  m_channel->send(MessageType::SERVER_ADD_OBJECT, std::move(msg));
}

void NetworkUpdateDelegate::signalParameterUpdated(
    const tsd::scene::Object *o, const tsd::scene::Parameter *p)
{
  CHECK_READY_OR_RETURN();
  auto msg = tsd::network::messages::ParameterChange(o, p);
  m_channel->send(MessageType::SERVER_SET_OBJECT_PARAMETER, std::move(msg));
}

void NetworkUpdateDelegate::signalParameterRemoved(
    const tsd::scene::Object *o, const tsd::scene::Parameter *p)
{
  CHECK_READY_OR_RETURN();
  auto msg = tsd::network::messages::ParameterRemove(o, p);
  m_channel->send(MessageType::SERVER_REMOVE_OBJECT_PARAMETER, std::move(msg));
}

void NetworkUpdateDelegate::signalParameterBatchUpdated(
    const tsd::scene::Object *o,
    const std::vector<const tsd::scene::Parameter *> &ps)
{
  CHECK_READY_OR_RETURN();
  auto msg = tsd::network::messages::ParameterChange(o, ps.data(), ps.size());
  m_channel->send(MessageType::SERVER_SET_OBJECT_PARAMETER, std::move(msg));
}

void NetworkUpdateDelegate::signalArrayMapped(const tsd::scene::Array *)
{
  CHECK_READY_OR_RETURN();
  // no-op
}

void NetworkUpdateDelegate::signalArrayUnmapped(const tsd::scene::Array *a)
{
  CHECK_READY_OR_RETURN();
  auto msg = tsd::network::messages::TransferArrayData(a);
  m_channel->send(MessageType::SERVER_SET_ARRAY_DATA, std::move(msg));
}

void NetworkUpdateDelegate::signalObjectParameterUseCountZero(
    const tsd::scene::Object *obj)
{
  CHECK_READY_OR_RETURN();
  // no-op
}

void NetworkUpdateDelegate::signalObjectLayerUseCountZero(
    const tsd::scene::Object *obj)
{
  CHECK_READY_OR_RETURN();
  // no-op
}

void NetworkUpdateDelegate::signalObjectRemoved(const tsd::scene::Object *o)
{
  CHECK_READY_OR_RETURN();
  auto msg = tsd::network::messages::RemoveObject(o);
  m_channel->send(MessageType::SERVER_REMOVE_OBJECT, std::move(msg));
}

void NetworkUpdateDelegate::signalRemoveAllObjects()
{
  CHECK_READY_OR_RETURN();
  m_channel->send(MessageType::SERVER_REMOVE_ALL_OBJECTS);
}

void NetworkUpdateDelegate::signalLayerAdded(const tsd::scene::Layer *l)
{
  CHECK_READY_OR_RETURN();
  auto msg = tsd::network::messages::TransferLayer(m_scene, l);
  m_channel->send(MessageType::SERVER_UPDATE_LAYER, std::move(msg));
}

void NetworkUpdateDelegate::signalLayerStructureUpdated(
    const tsd::scene::Layer *l)
{
  CHECK_READY_OR_RETURN();
  auto msg = tsd::network::messages::TransferLayer(m_scene, l);
  m_channel->send(MessageType::SERVER_UPDATE_LAYER, std::move(msg));
}

void NetworkUpdateDelegate::signalLayerTransformUpdated(
    const tsd::scene::Layer *l)
{
  CHECK_READY_OR_RETURN();
  auto msg = tsd::network::messages::TransferLayer(m_scene, l);
  m_channel->send(MessageType::SERVER_UPDATE_LAYER, std::move(msg));
}

void NetworkUpdateDelegate::signalLayerRemoved(const tsd::scene::Layer *)
{
  CHECK_READY_OR_RETURN();
  tsd::core::logWarning(
      "NetworkUpdateDelegate::signalLayerRemoved not implemented");
}

void NetworkUpdateDelegate::signalActiveLayersChanged()
{
  CHECK_READY_OR_RETURN();
  tsd::core::logWarning(
      "NetworkUpdateDelegate::signalActiveLayersChanged not implemented");
}

void NetworkUpdateDelegate::signalObjectFilteringChanged()
{
  CHECK_READY_OR_RETURN();
  tsd::core::logWarning(
      "NetworkUpdateDelegate::signalObjectFilteringChanged not implemented");
}

void NetworkUpdateDelegate::signalInvalidateCachedObjects()
{
  CHECK_READY_OR_RETURN();
  tsd::core::logWarning(
      "NetworkUpdateDelegate::signalInvalidateCachedObjects not implemented");
}

void NetworkUpdateDelegate::signalAnimationTimeChanged(float t)
{
  CHECK_READY_OR_RETURN();
  m_channel->send(MessageType::SERVER_UPDATE_TIME, &t);
}

bool NetworkUpdateDelegate::isReady(const char *fcn) const
{
  if (!m_enabled) {
    return false;
  } else if (!m_channel) {
    tsd::core::logError("%s: no network channel", fcn);
    return false;
  }
  return true;
}

} // namespace tsd::network
