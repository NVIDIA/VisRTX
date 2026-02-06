// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Modal.h"

namespace tsd::ui::imgui {

struct ImportFileDialog : public Modal
{
  ImportFileDialog(Application *app);
  ~ImportFileDialog() override;

  void buildUI() override;

 private:
  std::string m_filename;
  int m_selectedFileType{0};
};

} // namespace tsd::ui::imgui
