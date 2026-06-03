// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ObjectFileDialog.h"
// tsd_core
#include "tsd/core/Logging.hpp"
// tsd_io
#include "tsd/io/serialization.hpp"
// tsd_ui_imgui
#include "tsd/ui/imgui/Application.h"
// imgui
#include <misc/cpp/imgui_stdlib.h>

namespace tsd::ui::imgui {

ObjectFileDialog::ObjectFileDialog(Application *app)
    : Modal(app, "TSD Object File")
{}

ObjectFileDialog::~ObjectFileDialog() = default;

void ObjectFileDialog::showImport(
    TSDObjectFileType fileType, tsd::scene::LayerNodeRef importRoot)
{
  m_mode = Mode::Import;
  m_fileType = fileType;
  m_importRoot = importRoot;
  m_exportObjectType = ANARI_UNKNOWN;
  m_exportObjectIndex = tsd::core::INVALID_INDEX;
  m_filename.clear();
  m_dialogFilename.clear();
  show();
}

void ObjectFileDialog::showExport(TSDObjectFileType fileType,
    anari::DataType objectType,
    size_t objectIndex)
{
  m_mode = Mode::Export;
  m_fileType = fileType;
  m_importRoot = {};
  m_exportObjectType = objectType;
  m_exportObjectIndex = objectIndex;
  m_filename.clear();
  m_dialogFilename.clear();
  show();
}

void ObjectFileDialog::buildUI()
{
  ImGui::Text("%s %s", actionLabel(), fileTypeLabel());

  if (ImGui::Button("...")) {
    m_dialogFilename.clear();
    m_app->getFilenameFromDialog(m_dialogFilename, m_mode == Mode::Export);
  }

  if (!m_dialogFilename.empty()) {
    m_filename = m_dialogFilename;
    m_dialogFilename.clear();
  }

  ImGui::SameLine();
  ImGui::SetNextItemWidth(800.f);
  ImGui::InputText("##filename", &m_filename);

  ImGui::NewLine();

  if (ImGui::Button("cancel") || ImGui::IsKeyDown(ImGuiKey_Escape)) {
    hide();
    return;
  }

  ImGui::SameLine();

  ImGui::BeginDisabled(m_filename.empty());
  if (ImGui::Button(m_mode == Mode::Import ? "import" : "export")) {
    hide();
    if (m_mode == Mode::Import)
      importFile();
    else
      exportFile();
  }
  ImGui::EndDisabled();
}

const char *ObjectFileDialog::fileTypeLabel() const
{
  switch (m_fileType) {
  case TSDObjectFileType::Surface:
    return "TSD Surface";
  case TSDObjectFileType::Volume:
    return "TSD Volume";
  }

  return "TSD Object";
}

const char *ObjectFileDialog::actionLabel() const
{
  return m_mode == Mode::Import ? "Import" : "Export";
}

const char *ObjectFileDialog::taskLabel() const
{
  return m_mode == Mode::Import ? "Please Wait: Importing TSD Object..."
                                : "Please Wait: Exporting TSD Object...";
}

anari::DataType ObjectFileDialog::anariObjectType() const
{
  switch (m_fileType) {
  case TSDObjectFileType::Surface:
    return ANARI_SURFACE;
  case TSDObjectFileType::Volume:
    return ANARI_VOLUME;
  }

  return ANARI_UNKNOWN;
}

void ObjectFileDialog::importFile()
{
  auto filename = m_filename;
  auto fileType = m_fileType;
  auto importRoot = m_importRoot;
  auto *app = m_app;

  auto doImport = [filename, fileType, importRoot, app]() mutable {
    auto *ctx = app->appContext();
    auto &scene = ctx->tsd.scene;

    if (!importRoot.valid())
      importRoot = scene.defaultLayer()->root();

    tsd::scene::Object *importedObject = nullptr;
    if (fileType == TSDObjectFileType::Surface) {
      auto importedSurface = tsd::io::import_Surface(scene, filename.c_str());
      importedObject = importedSurface.data();
    } else {
      auto importedVolume = tsd::io::import_Volume(scene, filename.c_str());
      importedObject = importedVolume.data();
    }

    if (!importedObject)
      return;

    const auto nodeName = importedObject->name().empty()
        ? std::string(fileType == TSDObjectFileType::Surface ? "surface"
                                                             : "volume")
        : importedObject->name();
    scene.insertChildObjectNode(importRoot,
        importedObject->type(),
        importedObject->index(),
        nodeName.c_str());
  };

  m_app->showTaskModal(doImport, taskLabel());
}

void ObjectFileDialog::exportFile()
{
  auto filename = m_filename;
  auto expectedType = anariObjectType();
  auto objectType = m_exportObjectType;
  auto objectIndex = m_exportObjectIndex;
  auto *app = m_app;

  auto doExport = [filename, expectedType, objectType, objectIndex, app]() {
    auto *ctx = app->appContext();
    auto &scene = ctx->tsd.scene;
    auto *object = scene.getObject(objectType, objectIndex);

    if (!object) {
      tsd::core::logError("[ObjectFileDialog] No object selected for export.");
      return;
    }

    if (object->type() != expectedType) {
      tsd::core::logError("[ObjectFileDialog] Selected object is not a %s.",
          anari::toString(expectedType));
      return;
    }

    tsd::io::export_Object(filename.c_str(), *object);
  };

  m_app->showTaskModal(doExport, taskLabel());
}

} // namespace tsd::ui::imgui
