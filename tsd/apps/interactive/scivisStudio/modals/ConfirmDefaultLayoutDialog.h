// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/ui/imgui/modals/Modal.h"

#include <functional>

namespace tsd::scivis_studio {

struct ConfirmDefaultLayoutDialog : public tsd::ui::imgui::Modal
{
  explicit ConfirmDefaultLayoutDialog(tsd::ui::imgui::Application *app);
  ~ConfirmDefaultLayoutDialog() override;

  void configure(std::function<void()> onConfirm);

 private:
  void buildUI() override;

  std::function<void()> m_onConfirm;
};

} // namespace tsd::scivis_studio
