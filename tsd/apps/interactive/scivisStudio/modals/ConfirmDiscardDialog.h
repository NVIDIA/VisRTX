// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/ui/imgui/modals/Modal.h"

#include <functional>

namespace tsd::scivis_studio {

struct ConfirmDiscardDialog : public tsd::ui::imgui::Modal
{
  explicit ConfirmDiscardDialog(tsd::ui::imgui::Application *app);
  ~ConfirmDiscardDialog() override;

  void configure(std::function<void()> onDiscard, std::function<void()> onCancel);

 private:
  void buildUI() override;

  std::function<void()> m_onDiscard;
  std::function<void()> m_onCancel;
};

} // namespace tsd::scivis_studio
