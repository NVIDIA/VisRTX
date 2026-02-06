// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/network/Message.hpp"
// tsd_core
#include "tsd/core/scene/Scene.hpp"

namespace tsd::network::messages {

struct NewObject : public StructuredMessage
{
  // Sender -- will serialize the data on construction
  NewObject(const tsd::core::Object *obj);

  // Receiver -- will setup deserialization on execute()
  NewObject(const Message &msg, tsd::core::Scene *scene);

  // Receiver behavior
  void execute() override;

 private:
  tsd::core::Scene *m_scene{nullptr};
};

} // namespace tsd::network::messages
