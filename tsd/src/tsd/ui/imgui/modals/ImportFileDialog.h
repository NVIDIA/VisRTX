// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Modal.h"
// tsd_app
#include "tsd/app/Core.h"

namespace tsd::ui::imgui {

struct ImportFileDialog : public Modal
{
  ImportFileDialog(tsd::app::Core *ctx);
  ~ImportFileDialog() override;

  void buildUI() override;

 private:
  tsd::app::Core *m_core{nullptr};
  std::string m_filename;
  int m_selectedFileType{0};
};

} // namespace tsd::ui::imgui
