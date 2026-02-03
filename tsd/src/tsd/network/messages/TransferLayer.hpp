// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/network/Message.hpp"
// tsd_core
#include "tsd/core/scene/Scene.hpp"

namespace tsd::network::messages {

struct TransferLayer : public StructuredMessage
{
  // Sender -- will serialize the scene into the message on construction
  TransferLayer(tsd::core::Scene *scene, tsd::core::Layer *layer);

  // Receiver -- will setup deserialization on execute()
  TransferLayer(const Message &msg, tsd::core::Scene *scene);

  // Receiver behavior
  void execute() override;

 private:
  tsd::core::Scene *m_scene{nullptr};
};

} // namespace tsd::network::messages
