// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "TransferLayer.hpp"
// tsd_core
#include "tsd/core/Logging.hpp"
// tsd_io
#include "tsd/io/serialization.hpp"

namespace tsd::network::messages {

TransferLayer::TransferLayer(tsd::core::Scene *scene, tsd::core::Layer *layer)
{
  if (!(scene && layer)) {
    tsd::core::logError(
        "[message::TransferLayer] Both scene and layer required");
    return;
  }

  auto root = m_tree.root();
  root["n"] = scene->getLayerName(layer).str();
  tsd::io::layerToNode(*layer, root["l"]);
}

TransferLayer::TransferLayer(const Message &msg, tsd::core::Scene *scene)
    : StructuredMessage(msg), m_scene(scene)
{
  tsd::core::logDebug(
      "[message::TransferLayer] Received scene from server"
      " (%zu bytes)",
      msg.header.payload_length);
}

void TransferLayer::execute()
{
  if (!m_scene) {
    tsd::core::logError(
        "[message::TransferLayer] No scene set to transfer data into");
    return;
  }

  auto &root = m_tree.root();
  auto layerName = root["n"].getValueAs<std::string>();
  auto *layer = m_scene->layer(layerName);
  if (!layer) {
    layer = m_scene->addLayer(layerName.c_str());
    tsd::core::logDebug(
        "[message::TransferLayer] Creating new layer '%s'", layerName.c_str());
  } else {
    tsd::core::logDebug(
        "[message::TransferLayer] Updating existing layer '%s'",
        layerName.c_str());
  }
  tsd::io::nodeToLayer(root["l"], *layer, *m_scene);
  m_scene->signalLayerChange(layer);
}

} // namespace tsd::network::messages
