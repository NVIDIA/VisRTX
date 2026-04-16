// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Timeline.h"
// tsd_core
#include "tsd/animation/Animation.hpp"
#include "tsd/animation/AnimationManager.hpp"
#include "tsd/core/ObjectPool.hpp"
#include "tsd/scene/objects/Camera.hpp"
// tsd_ui_imgui
#include "tsd/ui/imgui/Application.h"
#include "tsd/ui/imgui/tsd_ui_imgui.h"
// std
#include <algorithm>
#include <cmath>
#include <cstring>

namespace tsd::ui::imgui {

// Helpers for keyframe-style editing on vector-based bindings ////////////////

static void captureCurrentCameraKeyframe(
    tsd::animation::Animation &anim, float t)
{
  // Find a camera-targeting binding to identify the camera
  const tsd::scene::Object *cam = nullptr;
  for (auto &b : anim.objectParameterBindings()) {
    auto *t = b.target();
    if (t && t->type() == ANARI_CAMERA) {
      cam = t;
      break;
    }
  }
  if (!cam)
    return;

  auto captureParam = [&](const char *_paramName, anari::DataType type) {
    auto paramName = tsd::core::Token(_paramName);
    for (size_t i = 0; i < anim.objectParameterBindings().size(); i++) {
      auto &b = *anim.editableObjectParameterBinding(i);
      if (b.paramName() == paramName) {
        auto val = cam->parameterValueAs<math::float3>(paramName);
        if (val)
          b.insertKeyframe(t, *val);
        return;
      }
    }
  };

  captureParam("position", ANARI_FLOAT32_VEC3);
  captureParam("direction", ANARI_FLOAT32_VEC3);
  captureParam("up", ANARI_FLOAT32_VEC3);
}

// Timeline definitions ///////////////////////////////////////////////////////

Timeline::Timeline(Application *app, const char *name) : Window(app, name) {}

void Timeline::buildUI()
{
  auto *ctx = appContext();
  auto &scene = ctx->tsd.scene;
  auto &animMgr = ctx->tsd.animationMgr;

  // K shortcut: set keyframe on selected tracks
  if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)
      && ImGui::IsKeyPressed(ImGuiKey_K)) {
    float t = animMgr.getAnimationTime();
    auto &anims = animMgr.animations();
    for (size_t i = 0; i < anims.size(); i++) {
      if (!m_selectedTracks.empty() && m_selectedTracks.count(i) == 0)
        continue;
      auto &anim = anims[i];
      auto *transform = anim.editableTransformBinding(0);
      if (transform && transform->target()) {
        math::mat4 mat = (*transform->target())->getTransform();
        transform->insertKeyframe(t, mat);
      } else {
        captureCurrentCameraKeyframe(anim, t);
      }
    }
  }

  buildUI_transport();
  ImGui::Separator();
  buildUI_canvas();
}

void Timeline::buildUI_transport()
{
  auto *ctx = appContext();
  auto &scene = ctx->tsd.scene;
  auto &animMgr = ctx->tsd.animationMgr;
  const float uiScale = ImGui::GetIO().FontGlobalScale;

  ImGui::PushStyleVar(
      ImGuiStyleVar_CellPadding, ImVec2(16.f * uiScale, 2.f * uiScale));

  if (ImGui::BeginTable("##controls",
          5,
          ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_BordersInnerV)) {
    ImGui::TableNextRow();

    ImGui::PushStyleVar(
        ImGuiStyleVar_FramePadding, ImVec2(8.f * uiScale, 4.f * uiScale));

    // Play/Pause
    ImGui::TableNextColumn();
    if (animMgr.isPlaying()) {
      if (ImGui::Button("||"))
        animMgr.stop();
      if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Pause (Space)");
    } else {
      if (ImGui::Button(" > "))
        animMgr.play();
      if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Play (Space)");
    }

    ImGui::SameLine();

    // Stop
    if (ImGui::Button(" [] ")) {
      animMgr.stop();
      animMgr.setAnimationFrame(0);
    }

    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("Stop");

    ImGui::SameLine();

    // Loop toggle
    bool loop = animMgr.isLoop();
    if (ImGui::Checkbox("Loop", &loop))
      animMgr.setLoop(loop);

    // Frame counter
    ImGui::TableNextColumn();
    int frame = animMgr.getAnimationFrame();
    int totalFrames = animMgr.getAnimationTotalFrames();
    if (ImGui::InputInt("##frame", &frame, 1, 10))
      animMgr.setAnimationFrame(frame);

    ImGui::TableNextColumn();
    ImGui::Text("%d / %d", frame, totalFrames - 1);

    // Total frames
    ImGui::TableNextColumn();
    if (ImGui::InputInt("Frames", &totalFrames, 1, 10))
      animMgr.setAnimationTotalFrames(totalFrames);

    ImGui::PopStyleVar();

    ImGui::EndTable();
  }

  ImGui::PopStyleVar();
}

void Timeline::buildUI_canvas()
{
  auto *ctx = appContext();
  auto &scene = ctx->tsd.scene;
  auto &animMgr = ctx->tsd.animationMgr;
  const float uiScale = ImGui::GetIO().FontGlobalScale;
  auto scaled = [uiScale](float value) { return value * uiScale; };

  const float rowHeight = scaled(28.f);
  const float rulerHeight = scaled(24.f);
  const float pixelsPerFrame = m_pixelsPerFrame * uiScale;
  const float cellPaddingX = scaled(4.f);
  const float cellPaddingY = scaled(8.f);

  int totalFrames = animMgr.getAnimationTotalFrames();
  int currentFrame = animMgr.getAnimationFrame();
  auto &anims = animMgr.animations();

  const auto writeTrackLabel =
      [](const tsd::animation::Animation &anim, char *label, size_t size) {
    if (anim.name().size() > 20)
      std::snprintf(label, size, "%.17s...", anim.name().c_str());
    else
      std::snprintf(label, size, "%s", anim.name().c_str());
  };

  const ImGuiStyle &style = ImGui::GetStyle();
  float maxTrackLabelWidth = ImGui::CalcTextSize("+ Add Track").x;
  for (const auto &anim : anims) {
    char label[64];
    writeTrackLabel(anim, label, sizeof(label));
    maxTrackLabelWidth =
        std::max(maxTrackLabelWidth, ImGui::CalcTextSize(label).x);
  }

  const float removeButtonWidth =
      ImGui::CalcTextSize("x").x + style.FramePadding.x * 2.f;
  const float nameButtonWidth =
      maxTrackLabelWidth + style.FramePadding.x * 2.f;
  const float nameColWidth = nameButtonWidth + style.ItemSpacing.x
      + removeButtonWidth + cellPaddingX * 2.f;
  float canvasWidth = ImGui::GetContentRegionAvail().x - nameColWidth;

  // Zoom with scroll wheel when hovering canvas
  if (ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows)) {
    float wheel = ImGui::GetIO().MouseWheel;
    if (wheel != 0.f && ImGui::GetIO().KeyCtrl) {
      m_pixelsPerFrame =
          std::clamp(m_pixelsPerFrame * (1.f + wheel * 0.1f), 1.f, 50.f);
    }
  }

  float totalCanvasWidth = pixelsPerFrame * (totalFrames - 1);
  size_t numAnims = anims.size();
  float keysColWidth = std::max(canvasWidth, totalCanvasWidth + scaled(20.f));

  ImGui::PushStyleVar(
      ImGuiStyleVar_CellPadding, ImVec2(cellPaddingX, cellPaddingY));

  const ImGuiTableFlags tableFlags = ImGuiTableFlags_ScrollX
      | ImGuiTableFlags_ScrollY | ImGuiTableFlags_BordersInnerV
      | ImGuiTableFlags_Resizable;

  if (ImGui::BeginTable(
          "##timeline_table", 2, tableFlags, ImGui::GetContentRegionAvail())) {
    ImGui::TableSetupScrollFreeze(1, 0);
    ImGui::TableSetupColumn(
        "##names", ImGuiTableColumnFlags_WidthFixed, nameColWidth);
    ImGui::TableSetupColumn("##canvas", ImGuiTableColumnFlags_WidthStretch);

    // --- Ruler row ---
    ImGui::TableNextRow(0, rulerHeight);
    ImGui::TableSetColumnIndex(0);

    ImGui::TableSetColumnIndex(1);
    {
      ImVec2 rulerPos = ImGui::GetCursorScreenPos();
      ImDrawList *draw = ImGui::GetWindowDrawList();

      draw->AddRectFilled(rulerPos,
          ImVec2(rulerPos.x + keysColWidth, rulerPos.y + rulerHeight),
          IM_COL32(50, 50, 50, 255));

      const int tickInterval =
          m_pixelsPerFrame < 4.f ? 20 : (m_pixelsPerFrame < 10.f ? 10 : 5);
      for (int f = 0; f < totalFrames; f += tickInterval) {
        float x = rulerPos.x + f * pixelsPerFrame;
        draw->AddLine(ImVec2(x, rulerPos.y + rulerHeight - scaled(8.f)),
            ImVec2(x, rulerPos.y + rulerHeight),
            IM_COL32(200, 200, 200, 255));
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%d", f);
        draw->AddText(ImVec2(x + scaled(2.f), rulerPos.y + scaled(2.f)),
            IM_COL32(200, 200, 200, 255),
            buf);
      }

      float scrubX = rulerPos.x + currentFrame * pixelsPerFrame;
      float scrubLineHeight = rulerHeight + numAnims * rowHeight;
      draw->AddLine(ImVec2(scrubX, rulerPos.y),
          ImVec2(scrubX, rulerPos.y + scrubLineHeight),
          IM_COL32(255, 80, 80, 255),
          scaled(2.f));

      ImGui::SetCursorScreenPos(ImVec2(scrubX - scaled(6.f), rulerPos.y));
      ImGui::InvisibleButton("##scrubber", ImVec2(scaled(12.f), rulerHeight));
      if (ImGui::IsItemActive() && ImGui::IsMouseDragging(0)) {
        float dx = ImGui::GetIO().MouseDelta.x;
        int newFrame =
            std::clamp(currentFrame + static_cast<int>(dx / pixelsPerFrame),
                0,
                totalFrames - 1);
        animMgr.setAnimationFrame(newFrame);
      }

      ImGui::SetCursorScreenPos(rulerPos);
      ImGui::InvisibleButton("##ruler_seek", ImVec2(keysColWidth, rulerHeight));
      if (ImGui::IsItemClicked(0)) {
        float mx = ImGui::GetMousePos().x - rulerPos.x;
        int newFrame = std::clamp(
            static_cast<int>(mx / pixelsPerFrame), 0, totalFrames - 1);
        animMgr.setAnimationFrame(newFrame);
      }
    }

    // --- Animation track rows ---
    for (size_t i = 0; i < numAnims; i++) {
      auto &anim = anims[i];

      ImGui::TableNextRow(0, rowHeight);

      // Name column
      ImGui::TableSetColumnIndex(0);
      ImGui::PushID(static_cast<int>(i));

      bool selected = m_selectedTracks.count(i) > 0;
      if (selected)
        ImGui::PushStyleColor(
            ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));

      char label[64];
      writeTrackLabel(anim, label, sizeof(label));
      float colW = ImGui::GetContentRegionAvail().x;
      float trackButtonWidth = std::max(
          0.f, colW - removeButtonWidth - style.ItemSpacing.x);
      if (ImGui::Button(label, ImVec2(trackButtonWidth, rowHeight))) {
        if (ImGui::GetIO().KeyCtrl) {
          if (selected)
            m_selectedTracks.erase(i);
          else
            m_selectedTracks.insert(i);
        } else {
          m_selectedTracks.clear();
          m_selectedTracks.insert(i);
        }
      }

      if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s", anim.name().c_str());

      if (selected)
        ImGui::PopStyleColor();

      ImGui::SameLine();
      bool removed = false;
      if (ImGui::Button("x")) {
        animMgr.removeAnimation(i);
        m_selectedTracks.erase(i);
        removed = true;
      }
      ImGui::PopID();

      if (removed)
        break;

      // Keyframe column — draw diamonds for each binding's keyframes
      ImGui::TableSetColumnIndex(1);
      {
        ImVec2 rowPos = ImGui::GetCursorScreenPos();
        ImDrawList *draw = ImGui::GetWindowDrawList();

        uint32_t rowBg = (i % 2 == 0) ? IM_COL32(40, 40, 40, 200)
                                      : IM_COL32(35, 35, 35, 200);
        draw->AddRectFilled(rowPos,
            ImVec2(rowPos.x + keysColWidth, rowPos.y + rowHeight),
            rowBg);

        // Draw binding keyframe diamonds
        for (size_t bi = 0; bi < anim.objectParameterBindings().size(); bi++) {
          const auto &b = anim.objectParameterBindings()[bi];
          if (b.timeBase().empty())
            continue;
          for (size_t ki = 0; ki < b.timeBase().size(); ki++) {
            float kx = rowPos.x
                + b.timeBase()[ki] * (totalFrames - 1) * pixelsPerFrame;
            float ky = rowPos.y + rowHeight * 0.5f;
            float r = scaled(4.f);
            uint32_t fillCol = (bi == 0) ? IM_COL32(80, 200, 255, 255)
                                         : IM_COL32(80, 255, 120, 255);
            draw->AddQuadFilled(ImVec2(kx, ky - r),
                ImVec2(kx + r, ky),
                ImVec2(kx, ky + r),
                ImVec2(kx - r, ky),
                fillCol);
            draw->AddQuad(ImVec2(kx, ky - r),
                ImVec2(kx + r, ky),
                ImVec2(kx, ky + r),
                ImVec2(kx - r, ky),
                IM_COL32(200, 240, 255, 180));

            ImGui::SetCursorScreenPos(ImVec2(kx - r, ky - r));
            ImGui::PushID(static_cast<int>(i * 10000 + bi * 100 + ki));
            ImGui::InvisibleButton("##bkf", ImVec2(r * 2.f, r * 2.f));
            if (ImGui::IsItemHovered()) {
              int kframe = static_cast<int>(
                  std::round(b.timeBase()[ki] * (totalFrames - 1)));
              ImGui::SetTooltip("%s @ frame %d", b.paramName().c_str(), kframe);
            }
            if (ImGui::BeginPopupContextItem("##bkf_ctx")) {
              if (ImGui::MenuItem("Delete keyframe"))
                anim.editableObjectParameterBinding(bi)->removeKeyframe(ki);
              ImGui::EndPopup();
            }
            ImGui::PopID();
          }
        }

        // Draw transform binding keyframe diamonds
        for (size_t ti = 0; ti < anim.transformBindings().size(); ti++) {
          const auto &tfb = *anim.editableTransformBinding(ti);
          if (tfb.timeBase().empty())
            continue;
          for (size_t ki = 0; ki < tfb.timeBase().size(); ki++) {
            float kx = rowPos.x
                + tfb.timeBase()[ki] * (totalFrames - 1) * pixelsPerFrame;
            float ky = rowPos.y + rowHeight * 0.5f;
            float r = scaled(5.f);
            uint32_t fillCol = selected ? IM_COL32(255, 200, 50, 255)
                                        : IM_COL32(200, 160, 30, 255);
            draw->AddQuadFilled(ImVec2(kx, ky - r),
                ImVec2(kx + r, ky),
                ImVec2(kx, ky + r),
                ImVec2(kx - r, ky),
                fillCol);
            draw->AddQuad(ImVec2(kx, ky - r),
                ImVec2(kx + r, ky),
                ImVec2(kx, ky + r),
                ImVec2(kx - r, ky),
                IM_COL32(255, 255, 255, 180));

            ImGui::SetCursorScreenPos(ImVec2(kx - r, ky - r));
            ImGui::PushID(static_cast<int>(i * 10000 + 8000 + ki));
            ImGui::InvisibleButton("##tkf", ImVec2(r * 2.f, r * 2.f));
            if (ImGui::IsItemHovered()) {
              int kframe = static_cast<int>(
                  std::round(tfb.timeBase()[ki] * (totalFrames - 1)));
              ImGui::SetTooltip("Transform @ frame %d", kframe);
            }
            ImGui::PopID();
          }
        }

        ImGui::Dummy(ImVec2(keysColWidth, rowHeight));
      }
    }

    // --- Footer row: Add Track | Set Keyframe ---
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    if (ImGui::Button(
            "+ Add Track", ImVec2(ImGui::GetContentRegionAvail().x, 0)))
      ImGui::OpenPopup("##add_track_popup");

    if (ImGui::BeginPopup("##add_track_popup")) {
      if (ImGui::BeginMenu("Transform Nodes")) {
        for (size_t li = 0; li < scene.numberOfLayers(); li++) {
          auto *layer = scene.layer(li);
          if (!layer)
            continue;

          int nodeIdx = 0;
          layer->traverse(
              layer->root(), [&](tsd::scene::LayerNode &node, int level) {
                if (!node->isTransform() && node->name().empty())
                  return true;

                char menuLabel[128];
                if (!node->name().empty()) {
                  std::snprintf(menuLabel,
                      sizeof(menuLabel),
                      "%*s%s",
                      level * 2,
                      "",
                      node->name().c_str());
                } else {
                  std::snprintf(menuLabel,
                      sizeof(menuLabel),
                      "%*s[xform %d]",
                      level * 2,
                      "",
                      nodeIdx);
                }
                nodeIdx++;

                ImGui::PushID(&node);
                if (ImGui::MenuItem(menuLabel)) {
                  auto nodeRef = layer->at(node.index());
                  const char *animName =
                      !node->name().empty() ? node->name().c_str() : menuLabel;
                  auto &newAnim = animMgr.addAnimation(animName);
                  newAnim.addTransformBinding(nodeRef);
                  ImGui::CloseCurrentPopup();
                }
                ImGui::PopID();
                return true;
              });
        }

        ImGui::EndMenu();
      }

      ImGui::Separator();

      if (scene.numberOfObjects(ANARI_CAMERA) > 0
          && ImGui::BeginMenu("Cameras")) {
        auto t = ANARI_CAMERA;
        if (auto i = tsd::ui::buildUI_objects_menulist(scene, t);
            i != TSD_INVALID_INDEX) {
          auto cam = scene.getObject<tsd::scene::Camera>(i);
          if (cam) {
            auto &newAnim = animMgr.addAnimation(cam->name().c_str());
            newAnim.addObjectParameterBinding(cam.data(),
                "position",
                ANARI_FLOAT32_VEC3,
                (const void *)nullptr,
                nullptr,
                0);
            newAnim.addObjectParameterBinding(cam.data(),
                "direction",
                ANARI_FLOAT32_VEC3,
                (const void *)nullptr,
                nullptr,
                0);
            newAnim.addObjectParameterBinding(cam.data(),
                "up",
                ANARI_FLOAT32_VEC3,
                (const void *)nullptr,
                nullptr,
                0);
          }
        }
        ImGui::EndMenu();
      }

      ImGui::EndPopup();
    }

    ImGui::TableSetColumnIndex(1);
    if (ImGui::Button("Set Keyframe (K)")) {
      float t = animMgr.getAnimationTime();
      for (size_t i = 0; i < anims.size(); i++) {
        if (!m_selectedTracks.empty() && m_selectedTracks.count(i) == 0)
          continue;
        auto &anim = anims[i];
        auto *transform = anim.editableTransformBinding(0);
        if (transform && transform->target()) {
          math::mat4 mat = (*transform->target())->getTransform();
          transform->insertKeyframe(t, mat);
        } else {
          captureCurrentCameraKeyframe(anim, t);
        }
      }
    }

    ImGui::EndTable();
  }

  ImGui::PopStyleVar();
}

} // namespace tsd::ui::imgui
