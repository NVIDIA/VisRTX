// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Timeline.h"
// tsd_core
#include "tsd/core/ObjectPool.hpp"
#include "tsd/scene/Animation.hpp"
#include "tsd/scene/objects/Camera.hpp"
// tsd_ui_imgui
#include "tsd/ui/imgui/Application.h"
// std
#include <algorithm>
#include <cmath>
#include <cstring>

namespace tsd::ui::imgui {

static void captureCurrentCameraKeyframe(tsd::scene::Animation *anim, float t)
{
  const auto *obj = anim->keyframeTargetObject();
  if (!obj || obj->type() != ANARI_CAMERA)
    return;
  const auto *cam = static_cast<const tsd::scene::Camera *>(obj);

  if (auto v = cam->parameterValueAs<math::float3>("position"))
    anim->addValueKeyframe("position", ANARI_FLOAT32_VEC3, t, &*v);
  if (auto v = cam->parameterValueAs<math::float3>("direction"))
    anim->addValueKeyframe("direction", ANARI_FLOAT32_VEC3, t, &*v);
  if (auto v = cam->parameterValueAs<math::float3>("up"))
    anim->addValueKeyframe("up", ANARI_FLOAT32_VEC3, t, &*v);
  if (cam->subtype() == tsd::scene::tokens::camera::perspective) {
    if (auto v = cam->parameterValueAs<float>("fovy"))
      anim->addValueKeyframe("fovy", ANARI_FLOAT32, t, &*v);
  }
}

Timeline::Timeline(Application *app, const char *name) : Window(app, name) {}

void Timeline::buildUI()
{
  auto *ctx = appContext();
  auto &scene = ctx->tsd.scene;

  // Space toggles play
  if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)
      && ImGui::IsKeyPressed(ImGuiKey_Space)) {
    m_playing = !m_playing;
  }

  if (m_playing) {
    scene.incrementAnimationFrame();
    if (!m_loop && scene.getAnimationFrame() == 0)
      m_playing = false;
  }

  // K shortcut: set keyframe on selected tracks
  if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)
      && ImGui::IsKeyPressed(ImGuiKey_K)) {
    float t = scene.getAnimationTime();
    for (size_t i = 0; i < scene.numberOfAnimations(); i++) {
      if (!m_selectedTracks.empty() && m_selectedTracks.count(i) == 0)
        continue;
      auto *anim = scene.animation(i);
      if (!anim->hasKeyframes() && anim->timeStepCount() > 0)
        continue; // skip baked timestep animations
      auto nodeRef = anim->keyframeTargetNode();
      if (nodeRef) {
        math::mat4 mat = (*nodeRef)->getTransform();
        anim->addTransformKeyframe(t, mat);
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

  ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(16.f, 2.f));

  if (ImGui::BeginTable("##controls",
          5,
          ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_BordersInnerV)) {
    ImGui::TableNextRow();

    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8.f, 4.f));

    // Play/Pause
    ImGui::TableNextColumn();
    if (m_playing) {
      if (ImGui::Button("||"))
        m_playing = false;
      if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Pause (Space)");
    } else {
      if (ImGui::Button(" > "))
        m_playing = true;
      if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Play (Space)");
    }

    ImGui::SameLine();

    // Stop
    if (ImGui::Button(" [] ")) {
      m_playing = false;
      scene.setAnimationFrame(0);
    }
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("Stop");

    ImGui::SameLine();

    // Loop toggle
    ImGui::Checkbox("Loop", &m_loop);

    // Frame counter
    ImGui::TableNextColumn();
    int frame = scene.getAnimationFrame();
    int totalFrames = scene.getAnimationTotalFrames();
    if (ImGui::InputInt("##frame", &frame, 1, 10))
      scene.setAnimationFrame(frame);

    ImGui::TableNextColumn();
    ImGui::Text("%d / %d", frame, totalFrames - 1);

    // Total frames
    ImGui::TableNextColumn();
    if (ImGui::InputInt("Frames", &totalFrames, 1, 10))
      scene.setAnimationTotalFrames(totalFrames);

    // FPS
    float fps = scene.getAnimationFPS();
    ImGui::TableNextColumn();
    if (ImGui::DragFloat("FPS", &fps, 0.5f, 1.f, 240.f, "%.1f"))
      scene.setAnimationFPS(fps);

    ImGui::PopStyleVar();

    ImGui::EndTable();
  }

  ImGui::PopStyleVar();
}

void Timeline::buildUI_canvas()
{
  auto *ctx = appContext();
  auto &scene = ctx->tsd.scene;

  const float nameColWidth = 150.f;
  const float rowHeight = 20.f;
  const float rulerHeight = 24.f;

  int totalFrames = scene.getAnimationTotalFrames();
  int currentFrame = scene.getAnimationFrame();
  float canvasWidth = ImGui::GetContentRegionAvail().x - nameColWidth;

  // Zoom with scroll wheel when hovering canvas
  if (ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows)) {
    float wheel = ImGui::GetIO().MouseWheel;
    if (wheel != 0.f && ImGui::GetIO().KeyCtrl) {
      m_pixelsPerFrame =
          std::clamp(m_pixelsPerFrame * (1.f + wheel * 0.1f), 1.f, 50.f);
    }
  }

  float totalCanvasWidth = m_pixelsPerFrame * (totalFrames - 1);
  size_t numAnims = scene.numberOfAnimations();
  float keysColWidth = std::max(canvasWidth, totalCanvasWidth + 20.f);

  ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(4.f, 8.f));

  const ImGuiTableFlags tableFlags = ImGuiTableFlags_ScrollX
      | ImGuiTableFlags_ScrollY | ImGuiTableFlags_BordersInnerV
      | ImGuiTableFlags_Resizable;

  if (ImGui::BeginTable(
          "##timeline_table", 2, tableFlags, ImGui::GetContentRegionAvail())) {
    ImGui::TableSetupScrollFreeze(1, 0); // freeze name column
    ImGui::TableSetupColumn(
        "##names", ImGuiTableColumnFlags_WidthFixed, nameColWidth);
    ImGui::TableSetupColumn("##canvas", ImGuiTableColumnFlags_WidthStretch);

    // --- Ruler row ---
    ImGui::TableNextRow(0, rulerHeight);
    ImGui::TableSetColumnIndex(0);
    // (empty name cell — spacer for ruler)

    ImGui::TableSetColumnIndex(1);
    {
      ImVec2 rulerPos = ImGui::GetCursorScreenPos();
      ImDrawList *draw = ImGui::GetWindowDrawList();

      // Ruler background
      draw->AddRectFilled(rulerPos,
          ImVec2(rulerPos.x + keysColWidth, rulerPos.y + rulerHeight),
          IM_COL32(50, 50, 50, 255));

      // Ruler ticks
      const int tickInterval =
          m_pixelsPerFrame < 4.f ? 20 : (m_pixelsPerFrame < 10.f ? 10 : 5);
      for (int f = 0; f < totalFrames; f += tickInterval) {
        float x = rulerPos.x + f * m_pixelsPerFrame;
        draw->AddLine(ImVec2(x, rulerPos.y + rulerHeight - 8.f),
            ImVec2(x, rulerPos.y + rulerHeight),
            IM_COL32(200, 200, 200, 255));
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%d", f);
        draw->AddText(ImVec2(x + 2.f, rulerPos.y + 2.f),
            IM_COL32(200, 200, 200, 255),
            buf);
      }

      // Scrubber line spanning ruler + all track rows
      float scrubX = rulerPos.x + currentFrame * m_pixelsPerFrame;
      float scrubLineHeight = rulerHeight + numAnims * rowHeight;
      draw->AddLine(ImVec2(scrubX, rulerPos.y),
          ImVec2(scrubX, rulerPos.y + scrubLineHeight),
          IM_COL32(255, 80, 80, 255),
          2.f);

      // Scrubber drag handle
      ImGui::SetCursorScreenPos(ImVec2(scrubX - 6.f, rulerPos.y));
      ImGui::InvisibleButton("##scrubber", ImVec2(12.f, rulerHeight));
      if (ImGui::IsItemActive() && ImGui::IsMouseDragging(0)) {
        float dx = ImGui::GetIO().MouseDelta.x;
        int newFrame =
            std::clamp(currentFrame + static_cast<int>(dx / m_pixelsPerFrame),
                0,
                totalFrames - 1);
        scene.setAnimationFrame(newFrame);
      }

      // Click on ruler to seek
      ImGui::SetCursorScreenPos(rulerPos);
      ImGui::InvisibleButton("##ruler_seek", ImVec2(keysColWidth, rulerHeight));
      if (ImGui::IsItemClicked(0)) {
        float mx = ImGui::GetMousePos().x - rulerPos.x;
        int newFrame = std::clamp(
            static_cast<int>(mx / m_pixelsPerFrame), 0, totalFrames - 1);
        scene.setAnimationFrame(newFrame);
      }
    }

    // --- Animation track rows ---
    for (size_t i = 0; i < numAnims; i++) {
      auto *anim = scene.animation(i);

      ImGui::TableNextRow(0, rowHeight);

      // Name column
      ImGui::TableSetColumnIndex(0);
      ImGui::PushID(static_cast<int>(i));

      bool selected = m_selectedTracks.count(i) > 0;
      if (selected)
        ImGui::PushStyleColor(
            ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));

      char label[64];
      const auto &animName = anim->name();
      if (animName.size() > 20)
        std::snprintf(label, sizeof(label), "%.17s...", animName.c_str());
      else
        std::snprintf(label, sizeof(label), "%s", animName.c_str());

      float colW = ImGui::GetContentRegionAvail().x;
      if (ImGui::Button(label, ImVec2(colW - 30.f, rowHeight))) {
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

      if (selected)
        ImGui::PopStyleColor();

      ImGui::SameLine();
      bool removed = false;
      if (ImGui::SmallButton("x")) {
        scene.removeAnimation(anim);
        m_selectedTracks.erase(i);
        removed = true;
      }
      ImGui::PopID();

      if (removed)
        break;

      // Keyframe column
      ImGui::TableSetColumnIndex(1);
      {
        ImVec2 rowPos = ImGui::GetCursorScreenPos();
        ImDrawList *draw = ImGui::GetWindowDrawList();

        // Row background (alternating)
        uint32_t rowBg = (i % 2 == 0) ? IM_COL32(40, 40, 40, 200)
                                      : IM_COL32(35, 35, 35, 200);
        draw->AddRectFilled(rowPos,
            ImVec2(rowPos.x + keysColWidth, rowPos.y + rowHeight),
            rowBg);

        // Transform keyframe diamonds
        const auto &tkfs = anim->transformKeyframes();
        for (size_t ki = 0; ki < tkfs.size(); ki++) {
          float kx =
              rowPos.x + tkfs[ki].time * (totalFrames - 1) * m_pixelsPerFrame;
          float ky = rowPos.y + rowHeight * 0.5f;
          float r = 5.f;

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
          ImGui::PushID(static_cast<int>(i * 10000 + ki));
          ImGui::InvisibleButton("##kf", ImVec2(r * 2.f, r * 2.f));

          if (ImGui::IsItemHovered()) {
            int kframe =
                static_cast<int>(std::round(tkfs[ki].time * (totalFrames - 1)));
            ImGui::SetTooltip("Frame %d", kframe);
          }
          if (ImGui::IsItemClicked(0))
            scene.setAnimationFrame(static_cast<int>(
                std::round(tkfs[ki].time * (totalFrames - 1))));

          if (ImGui::BeginPopupContextItem("##kf_ctx")) {
            if (ImGui::MenuItem("Delete keyframe"))
              anim->removeTransformKeyframe(ki);
            if (ImGui::MenuItem("Move to current frame"))
              anim->addTransformKeyframe(
                  scene.getAnimationTime(), tkfs[ki].matrix);
            ImGui::EndPopup();
          }

          if (ImGui::IsItemActive() && ImGui::IsMouseDragging(0)) {
            float dx = ImGui::GetIO().MouseDelta.x;
            float newTime =
                tkfs[ki].time + dx / ((totalFrames - 1) * m_pixelsPerFrame);
            newTime = std::clamp(newTime, 0.f, 1.f);
            math::mat4 mat = tkfs[ki].matrix;
            anim->removeTransformKeyframe(ki);
            anim->addTransformKeyframe(newTime, mat);
          }
          ImGui::PopID();
        }

        // Value channel keyframe diamonds
        const auto &channels = anim->keyframeChannels();
        for (size_t ci = 0; ci < channels.size(); ci++) {
          const auto &chan = channels[ci];
          for (size_t ki = 0; ki < chan.keyframes.size(); ki++) {
            float kx = rowPos.x
                + chan.keyframes[ki].time * (totalFrames - 1)
                    * m_pixelsPerFrame;
            float ky = rowPos.y + rowHeight * 0.5f;
            float r = 4.f;
            draw->AddQuadFilled(ImVec2(kx, ky - r),
                ImVec2(kx + r, ky),
                ImVec2(kx, ky + r),
                ImVec2(kx - r, ky),
                IM_COL32(80, 200, 255, 255));
            draw->AddQuad(ImVec2(kx, ky - r),
                ImVec2(kx + r, ky),
                ImVec2(kx, ky + r),
                ImVec2(kx - r, ky),
                IM_COL32(200, 240, 255, 180));

            ImGui::SetCursorScreenPos(ImVec2(kx - r, ky - r));
            ImGui::PushID(static_cast<int>(i * 10000 + 5000 + ci * 100 + ki));
            ImGui::InvisibleButton("##vkf", ImVec2(r * 2.f, r * 2.f));
            if (ImGui::IsItemHovered()) {
              int kframe = static_cast<int>(
                  std::round(chan.keyframes[ki].time * (totalFrames - 1)));
              ImGui::SetTooltip(
                  "%s @ frame %d", chan.parameterName.c_str(), kframe);
            }
            if (ImGui::BeginPopupContextItem("##vkf_ctx")) {
              if (ImGui::MenuItem("Delete keyframe"))
                anim->removeValueKeyframe(chan.parameterName, ki);
              ImGui::EndPopup();
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
    if (ImGui::Button("+ Add Track", ImVec2(ImGui::GetContentRegionAvail().x, 0)))
      ImGui::OpenPopup("##add_track_popup");

    if (ImGui::BeginPopup("##add_track_popup")) {
      ImGui::Text("Select a transform node to animate:");
      ImGui::Separator();

      for (size_t li = 0; li < scene.numberOfLayers(); li++) {
        auto *layer = scene.layer(li);
        if (!layer)
          continue;

        int nodeIdx = 0;
        layer->traverse(
            layer->root(), [&](tsd::scene::LayerNode &node, int level) {
              if (!node->isTransform() && node->name().empty())
                return true;

              char label[128];
              if (!node->name().empty()) {
                std::snprintf(label,
                    sizeof(label),
                    "%*s%s",
                    level * 2,
                    "",
                    node->name().c_str());
              } else {
                std::snprintf(label,
                    sizeof(label),
                    "%*s[xform %d]",
                    level * 2,
                    "",
                    nodeIdx);
              }
              nodeIdx++;

              ImGui::PushID(static_cast<int>(li * 100000 + node.index()));
              if (ImGui::MenuItem(label)) {
                auto nodeRef = layer->at(node.index());
                const char *animName =
                    !node->name().empty() ? node->name().c_str() : label;
                scene.addKeyframeAnimation(animName, nodeRef);
                ImGui::CloseCurrentPopup();
              }
              ImGui::PopID();
              return true;
            });
      }

      ImGui::Separator();
      ImGui::Text("Cameras:");

      const auto &cameraDB = scene.objectDB().camera;
      tsd::core::foreach_item_const(cameraDB, [&](const auto *cam) {
        if (!cam)
          return;
        char label[128];
        const auto &camName = cam->name();
        if (!camName.empty())
          std::snprintf(label, sizeof(label), "%s", camName.c_str());
        else
          std::snprintf(label, sizeof(label), "Camera [%zu]", cam->index());
        ImGui::PushID(static_cast<int>(cam->index() + 900000));
        if (ImGui::MenuItem(label)) {
          scene.addKeyframeAnimationForCamera(label, cam->self());
          ImGui::CloseCurrentPopup();
        }
        ImGui::PopID();
      });

      ImGui::EndPopup();
    }

    ImGui::TableSetColumnIndex(1);
    if (ImGui::Button("Set Keyframe (K)")) {
      float t = scene.getAnimationTime();
      for (size_t i = 0; i < scene.numberOfAnimations(); i++) {
        if (!m_selectedTracks.empty() && m_selectedTracks.count(i) == 0)
          continue;
        auto *anim = scene.animation(i);
        auto nodeRef = anim->keyframeTargetNode();
        if (nodeRef) {
          math::mat4 mat = (*nodeRef)->getTransform();
          anim->addTransformKeyframe(t, mat);
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
