// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "NewObject.hpp"
// tsd_core
#include "tsd/core/Logging.hpp"
// tsd_io
#include "tsd/io/serialization.hpp"

namespace tsd::network::messages {

NewObject::NewObject(const tsd::core::Object *o)
{
  if (!o) {
    tsd::core::logError("[message::NewObject] No client object provided");
    return;
  }

  tsd::io::objectToNode(*o, m_tree.root());
}

NewObject::NewObject(const Message &msg, tsd::core::Scene *scene)
    : StructuredMessage(msg), m_scene(scene)
{
  tsd::core::logDebug(
      "[message::NewObject] Received new object from server"
      " (%zu bytes)",
      msg.header.payload_length);
}

void NewObject::execute()
{
  if (!m_scene) {
    tsd::core::logError("[message::NewObject] No scene provided for exec");
    return;
  }

  tsd::io::nodeToNewObject(*m_scene, m_tree.root());
}

} // namespace tsd::network::messages
