// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Timeline.h"
// tsd_core
#include "tsd/animation/Animation.hpp"
#include "tsd/core/ObjectPool.hpp"
#include "tsd/scene/objects/Camera.hpp"
// tsd_ui_imgui
#include "tsd/ui/imgui/Application.h"
// std
#include <algorithm>
#include <cmath>
#include <cstring>

namespace tsd::ui::imgui {

// Helpers for keyframe-style editing on vector-based bindings ////////////////

static void insertKeyframe(tsd::animation::ObjectParameterBinding &b,
    float time,
    const void *value)
{
  size_t elemSize = anari::sizeOf(b.dataType);
  if (elemSize == 0)
    return;

  size_t count = b.timeBase.size();

  // Find insertion point (maintain time sort)
  size_t insertIdx = count;
  for (size_t i = 0; i < count; i++) {
    if (std::abs(b.timeBase[i] - time) < 1e-4f) {
      // Replace existing keyframe at this time
      auto *dst = static_cast<uint8_t *>(b.data.map());
      std::memcpy(dst + i * elemSize, value, elemSize);
      b.data.unmap();
      return;
    }
    if (b.timeBase[i] > time) {
      insertIdx = i;
      break;
    }
  }

  // Insert into timeBase vector
  b.timeBase.insert(b.timeBase.begin() + insertIdx, time);

  // Rebuild data TimeSamples with new element
  size_t newCount = count + 1;
  tsd::animation::TimeSamples newData(b.dataType, newCount);
  auto *newVals = static_cast<uint8_t *>(newData.map());

  const auto *oldData =
      static_cast<const uint8_t *>(b.data.data());

  if (oldData && insertIdx > 0)
    std::memcpy(newVals, oldData, insertIdx * elemSize);

  std::memcpy(newVals + insertIdx * elemSize, value, elemSize);

  if (oldData && insertIdx < count)
    std::memcpy(newVals + (insertIdx + 1) * elemSize,
        oldData + insertIdx * elemSize,
        (count - insertIdx) * elemSize);

  newData.unmap();
  b.data = std::move(newData);
}

static void removeKeyframe(
    tsd::animation::ObjectParameterBinding &b, size_t index)
{
  size_t count = b.timeBase.size();
  if (index >= count)
    return;

  size_t elemSize = anari::sizeOf(b.dataType);
  if (count <= 1) {
    b.timeBase.clear();
    b.data = tsd::animation::TimeSamples();
    return;
  }

  b.timeBase.erase(b.timeBase.begin() + index);

  size_t newCount = count - 1;
  tsd::animation::TimeSamples newData(b.dataType, newCount);
  auto *newVals = static_cast<uint8_t *>(newData.map());
  const auto *oldData = static_cast<const uint8_t *>(b.data.data());

  if (index > 0)
    std::memcpy(newVals, oldData, index * elemSize);
  if (index < newCount)
    std::memcpy(newVals + index * elemSize,
        oldData + (index + 1) * elemSize,
        (newCount - index) * elemSize);

  newData.unmap();
  b.data = std::move(newData);
}

static void insertTransformKeyframe(tsd::animation::TransformBinding &tb,
    float time,
    const math::mat4 &m)
{
  // Decompose mat4 -> rotation quaternion, translation, scale
  math::float3 c0 = {m[0][0], m[0][1], m[0][2]};
  math::float3 c1 = {m[1][0], m[1][1], m[1][2]};
  math::float3 c2 = {m[2][0], m[2][1], m[2][2]};

  auto vecLen = [](math::float3 v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
  };
  math::float3 scl = {vecLen(c0), vecLen(c1), vecLen(c2)};
  if (scl.x > 0.f)
    c0 = {c0.x / scl.x, c0.y / scl.x, c0.z / scl.x};
  if (scl.y > 0.f)
    c1 = {c1.x / scl.y, c1.y / scl.y, c1.z / scl.y};
  if (scl.z > 0.f)
    c2 = {c2.x / scl.z, c2.y / scl.z, c2.z / scl.z};

  // Shepperd's method (mat3 -> unit quaternion)
  float trace = c0.x + c1.y + c2.z;
  math::float4 rot;
  if (trace > 0.f) {
    float s = 0.5f / std::sqrt(trace + 1.f);
    rot = {(c1.z - c2.y) * s, (c2.x - c0.z) * s, (c0.y - c1.x) * s, 0.25f / s};
  } else if (c0.x > c1.y && c0.x > c2.z) {
    float s = 0.5f / std::sqrt(1.f + c0.x - c1.y - c2.z);
    rot = {0.25f / s, (c0.y + c1.x) * s, (c2.x + c0.z) * s, (c1.z - c2.y) * s};
  } else if (c1.y > c2.z) {
    float s = 0.5f / std::sqrt(1.f + c1.y - c0.x - c2.z);
    rot = {(c0.y + c1.x) * s, 0.25f / s, (c1.z + c2.y) * s, (c2.x - c0.z) * s};
  } else {
    float s = 0.5f / std::sqrt(1.f + c2.z - c0.x - c1.y);
    rot = {(c2.x + c0.z) * s, (c1.z + c2.y) * s, 0.25f / s, (c0.y - c1.x) * s};
  }
  float qlen =
      std::sqrt(rot.x * rot.x + rot.y * rot.y + rot.z * rot.z + rot.w * rot.w);
  rot = {rot.x / qlen, rot.y / qlen, rot.z / qlen, rot.w / qlen};

  math::float3 trans = {m[3][0], m[3][1], m[3][2]};

  // Find insertion point (maintain time sort)
  size_t count = tb.timeBase.size();
  size_t insertIdx = count;
  for (size_t i = 0; i < count; i++) {
    if (std::abs(tb.timeBase[i] - time) < 1e-4f) {
      tb.rotation[i] = rot;
      tb.translation[i] = trans;
      tb.scale[i] = scl;
      return;
    }
    if (tb.timeBase[i] > time) {
      insertIdx = i;
      break;
    }
  }

  tb.timeBase.insert(tb.timeBase.begin() + insertIdx, time);
  tb.rotation.insert(tb.rotation.begin() + insertIdx, rot);
  tb.translation.insert(tb.translation.begin() + insertIdx, trans);
  tb.scale.insert(tb.scale.begin() + insertIdx, scl);
}

static void captureCurrentCameraKeyframe(
    tsd::animation::Animation &anim, float t)
{
  // Find a camera-targeting binding to identify the camera
  tsd::scene::Object *camObj = nullptr;
  for (auto &b : anim.bindings) {
    if (b.target && b.target->type() == ANARI_CAMERA) {
      camObj = b.target;
      break;
    }
  }
  if (!camObj)
    return;
  const auto *cam = static_cast<const tsd::scene::Camera *>(camObj);

  auto captureParam = [&](const char *paramName, ANARIDataType type) {
    for (auto &b : anim.bindings) {
      if (b.paramName == tsd::core::Token(paramName)) {
        auto val = cam->parameterValueAs<math::float3>(paramName);
        if (val)
          insertKeyframe(b, t, &*val);
        return;
      }
    }
  };

  captureParam("position", ANARI_FLOAT32_VEC3);
  captureParam("direction", ANARI_FLOAT32_VEC3);
  captureParam("up", ANARI_FLOAT32_VEC3);
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
    scene.sceneAnimation().incrementAnimationFrame();
    if (!m_loop && scene.sceneAnimation().getAnimationFrame() == 0)
      m_playing = false;
  }

  // K shortcut: set keyframe on selected tracks
  if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)
      && ImGui::IsKeyPressed(ImGuiKey_K)) {
    float t = scene.sceneAnimation().getAnimationTime();
    auto &anims = scene.sceneAnimation().animations();
    for (size_t i = 0; i < anims.size(); i++) {
      if (!m_selectedTracks.empty() && m_selectedTracks.count(i) == 0)
        continue;
      auto &anim = anims[i];
      if (!anim.transforms.empty() && anim.transforms[0].target) {
        auto nodeRef = anim.transforms[0].target;
        math::mat4 mat = (*nodeRef)->getTransform();
        insertTransformKeyframe(anim.transforms[0], t, mat);
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
      scene.sceneAnimation().setAnimationFrame(0);
    }
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("Stop");

    ImGui::SameLine();

    // Loop toggle
    ImGui::Checkbox("Loop", &m_loop);

    // Frame counter
    ImGui::TableNextColumn();
    int frame = scene.sceneAnimation().getAnimationFrame();
    int totalFrames = scene.sceneAnimation().getAnimationTotalFrames();
    if (ImGui::InputInt("##frame", &frame, 1, 10))
      scene.sceneAnimation().setAnimationFrame(frame);

    ImGui::TableNextColumn();
    ImGui::Text("%d / %d", frame, totalFrames - 1);

    // Total frames
    ImGui::TableNextColumn();
    if (ImGui::InputInt("Frames", &totalFrames, 1, 10))
      scene.sceneAnimation().setAnimationTotalFrames(totalFrames);

    // FPS
    float fps = scene.sceneAnimation().getAnimationFPS();
    ImGui::TableNextColumn();
    if (ImGui::DragFloat("FPS", &fps, 0.5f, 1.f, 240.f, "%.1f"))
      scene.sceneAnimation().setAnimationFPS(fps);

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

  int totalFrames = scene.sceneAnimation().getAnimationTotalFrames();
  int currentFrame = scene.sceneAnimation().getAnimationFrame();
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
  auto &anims = scene.sceneAnimation().animations();
  size_t numAnims = anims.size();
  float keysColWidth = std::max(canvasWidth, totalCanvasWidth + 20.f);

  ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(4.f, 8.f));

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

      float scrubX = rulerPos.x + currentFrame * m_pixelsPerFrame;
      float scrubLineHeight = rulerHeight + numAnims * rowHeight;
      draw->AddLine(ImVec2(scrubX, rulerPos.y),
          ImVec2(scrubX, rulerPos.y + scrubLineHeight),
          IM_COL32(255, 80, 80, 255),
          2.f);

      ImGui::SetCursorScreenPos(ImVec2(scrubX - 6.f, rulerPos.y));
      ImGui::InvisibleButton("##scrubber", ImVec2(12.f, rulerHeight));
      if (ImGui::IsItemActive() && ImGui::IsMouseDragging(0)) {
        float dx = ImGui::GetIO().MouseDelta.x;
        int newFrame =
            std::clamp(currentFrame + static_cast<int>(dx / m_pixelsPerFrame),
                0,
                totalFrames - 1);
        scene.sceneAnimation().setAnimationFrame(newFrame);
      }

      ImGui::SetCursorScreenPos(rulerPos);
      ImGui::InvisibleButton("##ruler_seek", ImVec2(keysColWidth, rulerHeight));
      if (ImGui::IsItemClicked(0)) {
        float mx = ImGui::GetMousePos().x - rulerPos.x;
        int newFrame = std::clamp(
            static_cast<int>(mx / m_pixelsPerFrame), 0, totalFrames - 1);
        scene.sceneAnimation().setAnimationFrame(newFrame);
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
      if (anim.name.size() > 20)
        std::snprintf(label, sizeof(label), "%.17s...", anim.name.c_str());
      else
        std::snprintf(label, sizeof(label), "%s", anim.name.c_str());

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
        scene.sceneAnimation().removeAnimation(i);
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
        for (size_t bi = 0; bi < anim.bindings.size(); bi++) {
          const auto &b = anim.bindings[bi];
          if (b.timeBase.empty())
            continue;
          for (size_t ki = 0; ki < b.timeBase.size(); ki++) {
            float kx =
                rowPos.x + b.timeBase[ki] * (totalFrames - 1) * m_pixelsPerFrame;
            float ky = rowPos.y + rowHeight * 0.5f;
            float r = 4.f;
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
                  std::round(b.timeBase[ki] * (totalFrames - 1)));
              ImGui::SetTooltip("%s @ frame %d", b.paramName.c_str(), kframe);
            }
            if (ImGui::BeginPopupContextItem("##bkf_ctx")) {
              if (ImGui::MenuItem("Delete keyframe"))
                removeKeyframe(anim.bindings[bi], ki);
              ImGui::EndPopup();
            }
            ImGui::PopID();
          }
        }

        // Draw transform binding keyframe diamonds
        for (size_t ti = 0; ti < anim.transforms.size(); ti++) {
          const auto &tfb = anim.transforms[ti];
          if (tfb.timeBase.empty())
            continue;
          for (size_t ki = 0; ki < tfb.timeBase.size(); ki++) {
            float kx = rowPos.x
                + tfb.timeBase[ki] * (totalFrames - 1) * m_pixelsPerFrame;
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
            ImGui::PushID(static_cast<int>(i * 10000 + 8000 + ki));
            ImGui::InvisibleButton("##tkf", ImVec2(r * 2.f, r * 2.f));
            if (ImGui::IsItemHovered()) {
              int kframe = static_cast<int>(
                  std::round(tfb.timeBase[ki] * (totalFrames - 1)));
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

              ImGui::PushID(static_cast<int>(li * 100000 + node.index()));
              if (ImGui::MenuItem(menuLabel)) {
                auto nodeRef = layer->at(node.index());
                const char *animName =
                    !node->name().empty() ? node->name().c_str() : menuLabel;
                auto &newAnim = scene.sceneAnimation().addAnimation(animName);
                newAnim.transforms.push_back({nodeRef, {}, {}, {}, {}});
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
        char menuLabel[128];
        const auto &camName = cam->name();
        if (!camName.empty())
          std::snprintf(menuLabel, sizeof(menuLabel), "%s", camName.c_str());
        else
          std::snprintf(
              menuLabel, sizeof(menuLabel), "Camera [%zu]", cam->index());
        ImGui::PushID(static_cast<int>(cam->index() + 900000));
        if (ImGui::MenuItem(menuLabel)) {
          auto &newAnim = scene.sceneAnimation().addAnimation(menuLabel);
          auto *camPtr = const_cast<tsd::scene::Camera *>(cam);
          using tsd::animation::ObjectParameterBinding;
          using tsd::animation::InterpolationRule;
          auto makeEmptyBinding =
              [&](const char *param) -> ObjectParameterBinding {
            ObjectParameterBinding b;
            b.target = camPtr;
            b.paramName = param;
            b.dataType = ANARI_FLOAT32_VEC3;
            b.interp = InterpolationRule::LINEAR;
            return b;
          };
          newAnim.bindings.push_back(makeEmptyBinding("position"));
          newAnim.bindings.push_back(makeEmptyBinding("direction"));
          newAnim.bindings.push_back(makeEmptyBinding("up"));
          ImGui::CloseCurrentPopup();
        }
        ImGui::PopID();
      });

      ImGui::EndPopup();
    }

    ImGui::TableSetColumnIndex(1);
    if (ImGui::Button("Set Keyframe (K)")) {
      float t = scene.sceneAnimation().getAnimationTime();
      for (size_t i = 0; i < anims.size(); i++) {
        if (!m_selectedTracks.empty() && m_selectedTracks.count(i) == 0)
          continue;
        auto &anim = anims[i];
        if (!anim.transforms.empty() && anim.transforms[0].target) {
          auto nodeRef = anim.transforms[0].target;
          math::mat4 mat = (*nodeRef)->getTransform();
          insertTransformKeyframe(anim.transforms[0], t, mat);
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
