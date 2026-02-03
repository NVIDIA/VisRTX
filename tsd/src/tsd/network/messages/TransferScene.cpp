// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "TransferScene.hpp"
// tsd_core
#include "tsd/core/Logging.hpp"
// tsd_io
#include "tsd/io/serialization.hpp"

namespace tsd::network::messages {

TransferScene::TransferScene(tsd::core::Scene *scene, bool includeArrayData)
{
  if (!scene) {
    tsd::core::logError(
        "[message::TransferScene] No scene set to transfer data from");
    return;
  }

  tsd::io::save_Scene(*scene, m_tree.root(), !includeArrayData);
}

TransferScene::TransferScene(const Message &msg, tsd::core::Scene *scene)
    : StructuredMessage(msg), m_scene(scene)
{
  tsd::core::logStatus(
      "[message::TransferScene] Received scene from server"
      " (%zu bytes)",
      msg.header.payload_length);
}

void TransferScene::execute()
{
  if (!m_scene) {
    tsd::core::logError(
        "[message::TransferScene] No scene set to transfer data into");
    return;
  }

  tsd::io::load_Scene(*m_scene, m_tree.root());
}

} // namespace tsd::network::messages
