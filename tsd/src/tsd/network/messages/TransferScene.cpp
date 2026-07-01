// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "TransferScene.hpp"
// tsd_core
#include "tsd/core/Logging.hpp"
// tsd_io
#include "tsd/io/archives/SceneArchive.hpp"

namespace tsd::network::messages {

TransferScene::TransferScene(tsd::scene::Scene *scene, bool includeArrayData)
{
  if (!scene) {
    tsd::core::logError(
        "[message::TransferScene] No scene set to transfer data from");
    return;
  }

  if (!tsd::io::serialize_SceneArchive(*scene,
          m_tree.root(),
          includeArrayData ? tsd::io::ArrayDataPolicy::IncludeData
                           : tsd::io::ArrayDataPolicy::ProxyOnly)) {
    tsd::core::logError(
        "[message::TransferScene] Failed to serialize Scene Archive");
  }
}

TransferScene::TransferScene(const Message &msg, tsd::scene::Scene *scene)
    : StructuredMessage(msg), m_scene(scene)
{
  tsd::core::logStatus("[message::TransferScene] Received message (%zu bytes)",
      msg.header.payload_length);
}

void TransferScene::execute()
{
  if (!m_scene) {
    tsd::core::logError(
        "[message::TransferScene] No scene set to transfer data into");
    return;
  }

  if (!tsd::io::deserialize_SceneArchive(*m_scene, m_tree.root())) {
    tsd::core::logError(
        "[message::TransferScene] Failed to deserialize Scene Archive");
  }
}

} // namespace tsd::network::messages
