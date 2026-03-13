// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/network/Message.hpp"
// tsd_core
#include "tsd/scene/Scene.hpp"

namespace tsd::network::messages {

struct TransferScene : public StructuredMessage
{
  // Sender -- will serialize the scene into the message on construction
  TransferScene(tsd::scene::Scene *scene, bool includeArrayData = false);

  // Receiver -- will setup deserialization on execute()
  TransferScene(const Message &msg, tsd::scene::Scene *scene);

  // Receiver behavior
  void execute() override;

 private:
  tsd::scene::Scene *m_scene{nullptr};
};

} // namespace tsd::network::messages
