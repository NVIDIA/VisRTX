// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/network/Message.hpp"
// tsd_core
#include "tsd/scene/Scene.hpp"

namespace tsd::network::messages {

struct TransferLayer : public StructuredMessage
{
  // Sender -- will serialize the scene into the message on construction
  TransferLayer(tsd::scene::Scene *scene, const tsd::scene::Layer *layer);

  // Receiver -- will setup deserialization on execute()
  TransferLayer(const Message &msg, tsd::scene::Scene *scene);

  // Receiver behavior
  void execute() override;

 private:
  tsd::scene::Scene *m_scene{nullptr};
};

} // namespace tsd::network::messages
