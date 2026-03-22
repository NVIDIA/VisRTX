// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ImagePass.h"
// std
#include <functional>

namespace tsd::rendering {

/*
 * ImagePass that invokes a caller-supplied callback with the current
 * ImageBuffers to perform picking or hit-testing using the ID AOV channels.
 *
 * Example:
 *   auto *pass = pipeline.emplace_back<PickPass>();
 *   pass->setPickOperation([&](ImageBuffers &b) {
 *     uint32_t id = b.objectId[clickY * width + clickX];
 *   });
 */
struct PickPass : public ImagePass
{
  using PickOpFunc = std::function<void(ImageBuffers &b)>;

  PickPass();
  ~PickPass() override;
  const char *name() const override { return "Pick"; }

  void setPickOperation(PickOpFunc &&f);

 private:
  void render(ImageBuffers &b, int stageId) override;

  PickOpFunc m_op;
};

} // namespace tsd::rendering
