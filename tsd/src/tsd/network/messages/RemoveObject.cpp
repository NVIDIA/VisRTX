// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "RemoveObject.hpp"
// tsd_core
#include "tsd/core/Logging.hpp"
// tsd_io
#include "tsd/io/serialization.hpp"

namespace tsd::network::messages {

RemoveObject::RemoveObject(const tsd::core::Object *o)
{
  if (!o) {
    tsd::core::logError("[message::RemoveObject] No client object provided");
    return;
  }

  m_tree.root() = tsd::core::Any(o->type(), o->index());
}

RemoveObject::RemoveObject(const Message &msg, tsd::core::Scene *scene)
    : StructuredMessage(msg), m_scene(scene)
{
  tsd::core::logDebug("[message::RemoveObject] Received message (%zu bytes)",
      msg.header.payload_length);
}

void RemoveObject::execute()
{
  if (!m_scene) {
    tsd::core::logError("[message::RemoveObject] No scene provided for exec");
    return;
  }

  m_scene->removeObject(m_tree.root().getValue());
}

} // namespace tsd::network::messages
