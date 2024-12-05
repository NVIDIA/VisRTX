// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../AppCore.h"
#include "Modal.h"
// tsd
#include "tsd/TSD.hpp"

namespace tsd_viewer {

struct ImportFileDialog : public Modal
{
  ImportFileDialog(AppCore *ctx);
  ~ImportFileDialog() override;

  void buildUI() override;

 private:
  AppCore *m_core{nullptr};
  std::string m_filename;
  int m_selectedFileType{0};
};

} // namespace tsd_viewer
