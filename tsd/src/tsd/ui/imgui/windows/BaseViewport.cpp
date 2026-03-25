// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/ui/imgui/windows/BaseViewport.h"
// tsd_rendering
#include "tsd/rendering/view/ManipulatorToTSD.hpp"
// tsd_ui_imgui
#include "tsd/ui/imgui/Application.h"
#include "tsd/ui/imgui/tsd_ui_imgui.h"

namespace tsd::ui::imgui {

BaseViewport::BaseViewport(Application *app, const char *name)
    : Window(app, name)
{
  setManipulator(nullptr);
}

BaseViewport::~BaseViewport()
{
  imagePipeline_teardown();
}

void BaseViewport::buildUI()
{
  ImVec2 _viewportSize = ImGui::GetContentRegionAvail();
  tsd::math::int2 viewportSize(_viewportSize.x, _viewportSize.y);

  if (m_viewport.size != viewportSize)
    viewport_reshape(viewportSize);
}

void BaseViewport::setManipulator(tsd::rendering::Manipulator *m)
{
  m_camera.arcball = m ? m : &m_camera.localArcball;
}

void BaseViewport::saveSettings(tsd::core::DataNode &root)
{
  // Viewport settings //

  root["viewport.scale"] = m_viewport.resolutionScale;

  // Gizmo settings //

  root["gizmo.active"] = m_gizmo.active;
  root["gizmo.operation"] = static_cast<int>(m_gizmo.operation);
  root["gizmo.mode"] = static_cast<int>(m_gizmo.mode);

  // Orientation widget settings //

  root["orientationWidget.show"] = m_showOrientationWidget;
}

void BaseViewport::loadSettings(tsd::core::DataNode &root)
{
  // Viewport settings //

  root["viewport.scale"].getValue(ANARI_FLOAT32, &m_viewport.resolutionScale);

  // Gizmo settings //

  root["gizmo.active"].getValue(ANARI_BOOL, &m_gizmo.active);
  int gizmoOp = static_cast<int>(m_gizmo.operation);
  root["gizmo.operation"].getValue(ANARI_INT32, &gizmoOp);
  m_gizmo.operation = static_cast<ImGuizmo::OPERATION>(gizmoOp);
  int gizmoMode = static_cast<int>(m_gizmo.mode);
  root["gizmo.mode"].getValue(ANARI_INT32, &gizmoMode);
  m_gizmo.mode = static_cast<ImGuizmo::MODE>(gizmoMode);

  // Orientation widget settings //

  root["orientationWidget.show"].getValue(ANARI_BOOL, &m_showOrientationWidget);
}

void BaseViewport::viewport_setActive(bool active)
{
  m_viewport.active = active;
}

bool BaseViewport::viewport_isActive() const
{
  return m_viewport.active;
}

void BaseViewport::viewport_reshape(tsd::math::int2 newWindowSize)
{
  if (newWindowSize.x <= 0 || newWindowSize.y <= 0)
    return;

  m_viewport.size = newWindowSize;
  m_viewport.renderSize = tsd::math::int2(
      tsd::math::float2(m_viewport.size) * m_viewport.resolutionScale);

  BaseViewport::imagePipeline_setDimensions(
      m_viewport.renderSize.x, m_viewport.renderSize.y);
}

void BaseViewport::imagePipeline_setup()
{
  imagePipeline_teardown();
  imagePipeline_populate(m_pipeline);
  viewport_reshape(m_viewport.size);
}

bool BaseViewport::imagePipeline_isSetup() const
{
  return !m_pipeline.empty();
}

void BaseViewport::imagePipeline_setDimensions(uint32_t width, uint32_t height)
{
  m_pipeline.setDimensions(width, height);
}

void BaseViewport::imagePipeline_render()
{
  m_pipeline.render();
}

const tsd::rendering::ImagePipeline &BaseViewport::imagePipeline() const
{
  return m_pipeline;
}

void BaseViewport::imagePipeline_teardown()
{
  m_pipeline.clear();
}

void BaseViewport::camera_update(bool force)
{
  if (!m_viewport.active)
    return;

  if (!m_camera.current)
    camera_setCurrent(appContext()->tsd.scene.defaultCamera());

  if (!force && !m_camera.arcball->hasChanged(m_camera.arcballToken))
    return;

  tsd::rendering::updateCameraObject(*m_camera.current, *m_camera.arcball);
}

void BaseViewport::camera_setCurrent(tsd::scene::CameraAppRef c)
{
  m_camera.current = c;
}

bool BaseViewport::gizmo_canShow() const
{
  if (!m_gizmo.active || !m_viewport.active)
    return false;

  // Check if we have a selected node with a transform
  auto selectedNode = appContext()->getFirstSelected();
  if (selectedNode.valid()) {
    return (*selectedNode)->isTransform();
  }

  return false;
}

void BaseViewport::ui_handleInput()
{
  // Handle gizmo keyboard shortcuts. Handle those before checking for
  // window focus so they can act globally.
  // When a new manipulator mode is selected, we default to world mode.
  // Otherwise, toggle between local and global modes.
  if (ImGui::IsKeyPressed(ImGuiKey_Q, false)
      || ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
    m_gizmo.active = false;
  } else if (ImGui::IsKeyPressed(ImGuiKey_W, false)) {
    if (m_gizmo.active && m_gizmo.operation == ImGuizmo::TRANSLATE) {
      m_gizmo.mode =
          (m_gizmo.mode == ImGuizmo::WORLD) ? ImGuizmo::LOCAL : ImGuizmo::WORLD;
    } else {
      m_gizmo.active = true;
      m_gizmo.operation = ImGuizmo::TRANSLATE;
      m_gizmo.mode = ImGuizmo::WORLD;
    }
  } else if (ImGui::IsKeyPressed(ImGuiKey_E, false)) {
    if (m_gizmo.active && m_gizmo.operation == ImGuizmo::SCALE) {
      m_gizmo.mode =
          (m_gizmo.mode == ImGuizmo::WORLD) ? ImGuizmo::LOCAL : ImGuizmo::WORLD;
    } else {
      m_gizmo.active = true;
      m_gizmo.operation = ImGuizmo::SCALE;
      m_gizmo.mode = ImGuizmo::WORLD;
    }
  } else if (ImGui::IsKeyPressed(ImGuiKey_R, false)) {
    if (m_gizmo.active && m_gizmo.operation == ImGuizmo::ROTATE) {
      m_gizmo.mode =
          (m_gizmo.mode == ImGuizmo::WORLD) ? ImGuizmo::LOCAL : ImGuizmo::WORLD;
    } else {
      m_gizmo.active = true;
      m_gizmo.operation = ImGuizmo::ROTATE;
      m_gizmo.mode = ImGuizmo::WORLD;
    }
  }

  // Enforce global Gizmo state so that it actually stops tracking
  // interactions when disabled.
  ImGuizmo::Enable(m_gizmo.active);

  // Block arcball input and picking when ImGuizmo is being used
  if (ImGuizmo::IsUsing())
    return;

  // Do not bother with events if the window is not hovered
  // or no interaction is ongoing.
  // We'll use that hovering status to check for starting an
  // event below.
  if (!ImGui::IsWindowHovered() && !m_input.manipulating)
    return;

  ImGuiIO &io = ImGui::GetIO();

  const bool dolly = ImGui::IsMouseDown(ImGuiMouseButton_Right)
      || (ImGui::IsMouseDown(ImGuiMouseButton_Left)
          && ImGui::IsKeyDown(ImGuiKey_LeftShift));
  const bool pan = ImGui::IsMouseDown(ImGuiMouseButton_Middle)
      || (ImGui::IsMouseDown(ImGuiMouseButton_Left)
          && ImGui::IsKeyDown(ImGuiKey_LeftAlt));
  const bool orbit = ImGui::IsMouseDown(ImGuiMouseButton_Left);

  const bool anyMovement = dolly || pan || orbit;
  if (!anyMovement) {
    m_input.manipulating = false;
    m_input.previousMouse = tsd::math::float2(-1);
  } else if (ImGui::IsItemHovered() && !m_input.manipulating) {
    m_input.manipulating = true;
    ImGui::SetWindowFocus(); // ensure we keep focus while manipulating
  }

  if (m_input.mouseRotating && !orbit)
    m_input.mouseRotating = false;

  if (m_input.manipulating) {
    tsd::math::float2 position;
    std::memcpy(&position, &io.MousePos, sizeof(position));

    const tsd::math::float2 mouse(position.x, position.y);

    if (anyMovement && m_input.previousMouse != tsd::math::float2(-1)) {
      const tsd::math::float2 prev = m_input.previousMouse;

      const tsd::math::float2 mouseFrom =
          prev * 2.f / tsd::math::float2(m_viewport.size);
      const tsd::math::float2 mouseTo =
          mouse * 2.f / tsd::math::float2(m_viewport.size);

      const tsd::math::float2 mouseDelta = mouseTo - mouseFrom;

      if (mouseDelta != tsd::math::float2(0.f)) {
        if (orbit && !(pan || dolly)) {
          if (!m_input.mouseRotating) {
            m_camera.arcball->startNewRotation();
            m_input.mouseRotating = true;
          }

          m_camera.arcball->rotate(mouseDelta);
        } else if (dolly)
          m_camera.arcball->zoom(mouseDelta.y);
        else if (pan)
          m_camera.arcball->pan(mouseDelta);
      }
    }

    m_input.previousMouse = mouse;
  }
}

void BaseViewport::ui_gizmo()
{
  if (!gizmo_canShow())
    return;

  auto computeWorldTransform = [](tsd::scene::LayerNodeRef node) -> math::mat4 {
    auto world = math::IDENTITY_MAT4;
    for (; node; node = node->parent())
      world = mul((*node)->getTransform(), world);

    return world;
  };

  auto selectedNodeRef = appContext()->getFirstSelected();
  auto parentNodeRef = selectedNodeRef->parent();

  auto localTransform = (*selectedNodeRef)->getTransform();
  auto parentWorldTransform = computeWorldTransform(parentNodeRef);
  auto worldTransform = mul(parentWorldTransform, localTransform);

  ImGuizmo::SetOrthographic(
      m_camera.current->subtype() == scene::tokens::camera::orthographic);
  ImGuizmo::BeginFrame();

  // Setup ImGuizmo with window and relative viewport information
  ImVec2 viewportPos = ImGui::GetCursorScreenPos();
  ImVec2 viewportSize = ImGui::GetContentRegionAvail();
  ImVec2 imageMin = ImGui::GetItemRectMin();
  ImVec2 imageMax = ImGui::GetItemRectMax();
  ImVec2 imageSize(imageMax.x - imageMin.x, imageMax.y - imageMin.y);
  ImGuizmo::SetRect(imageMin.x, imageMin.y, imageSize.x, imageSize.y);

  // Build view matrix and projection matrices from manipulator
  // Not sure if we can get those more directly...
  const auto eye = m_camera.arcball->eye();
  const auto at = m_camera.arcball->at();
  const auto up = m_camera.arcball->up();
  const auto view = linalg::lookat_matrix(eye, at, up);

  const float aspect = m_viewport.size.x / float(m_viewport.size.y);
  math::mat4 proj;

  // Try and get some legroom for ImGuizmo get precision on depth.
  // We don't know the extent of scene, so try and estimate a good enough near
  // plane position based on the distance to the select object
  const auto selectedObjectPos =
      math::float3(
          worldTransform[3][0], worldTransform[3][1], worldTransform[3][2])
      - eye;
  const float distanceToSelectedObject =
      dot(selectedObjectPos, normalize(at - eye));

  float near = std::max(1e-8f, distanceToSelectedObject * 1e-2f);
  float far = std::max(1e-6f, distanceToSelectedObject * 1e2f);

  if (m_camera.current->subtype() == scene::tokens::camera::perspective) {
    const float fovRadians =
        m_camera.current->parameterValueAs<float>("fovy").value_or(
            math::radians(40.f));
    float oneOverTanFov = 1.0f / tan(fovRadians / 2.0f);
    proj = math::mat4{
        {oneOverTanFov / aspect, 0.0f, 0.0f, 0.0f},
        {0.0f, oneOverTanFov, 0.0f, 0.0f},
        {0.0f, 0.0f, -(far + near) / (far - near), -1.0f},
        {0.0f, 0.0f, -2.0f * far * near / (far - near), 0.0f},
    };
  } else if (m_camera.current->subtype()
      == scene::tokens::camera::orthographic) {
    // The 0.75 factor is to match updateCameraParametersOrthographic
    const float height = m_camera.arcball->distance() * 0.75f;
    const float halfHeight = height * 0.5f;
    const float halfWidth = halfHeight * aspect;
    const float left = -halfWidth;
    const float right = halfWidth;
    const float bottom = -halfHeight;
    const float top = halfHeight;

    proj = math::mat4{{2.0f / (right - left), 0.0f, 0.0f, 0.0f},
        {0.0f, 2.0f / (top - bottom), 0.0f, 0.0f},
        {0.0f, 0.0f, -2.0f / (far - near), 0.0f},
        {-(right + left) / (right - left),
            -(top + bottom) / (top - bottom),
            -(far + near) / (far - near),
            1.0f}};
  } else {
    // No support for omnidirectional camera, bail out.
    return;
  }

  // Draw and manipulate the gizmo
  ImGuizmo::SetDrawlist();
  if (ImGuizmo::Manipulate(&view[0].x,
          &proj[0].x,
          m_gizmo.operation,
          m_gizmo.mode,
          &worldTransform[0].x)) {
    auto invParent = linalg::inverse(parentWorldTransform);
    localTransform = mul(invParent, worldTransform);
    (*selectedNodeRef)->setAsTransform(localTransform);
    appContext()->tsd.scene.signalLayerTransformChanged(
        (*selectedNodeRef)->layer());
  }
}

bool BaseViewport::ui_orientationWidget()
{
  if (!m_showOrientationWidget || !m_camera.arcball)
    return false;

  auto *uiConfig = m_app->uiConfig();

  // Position widget in the bottom-right corner of the window
  const float size = 150.f * uiConfig->fontScale;
  const ImVec2 winPos = ImGui::GetWindowPos();
  const ImVec2 winSize = ImGui::GetWindowSize();
  const float x = winPos.x + winSize.x - size - 5.f;
  const float y = size * 0.6f;
  ImOGuizmo::SetRect(x, y, size);

  // Build view matrix from arcball (same approach as ui_gizmo())
  const auto eye = m_camera.arcball->eye();
  const auto at = m_camera.arcball->at();
  const auto up = m_camera.arcball->up();
  const auto view = linalg::lookat_matrix(eye, at, up);
  float viewMat[16];
  std::memcpy(viewMat, &view[0].x, sizeof(viewMat));

  // Simple fixed perspective projection for the widget (fov=90deg, aspect=1)
  const float near = 0.1f, far = 100.f;
  const float f = 1.f; // 1/tan(45 deg)
  // clang-format off
  const float projMat[16] = {
      f, 0, 0, 0,
      0, f, 0, 0,
      0, 0, -(far + near) / (far - near), -1.f,
      0, 0, -2.f * far * near / (far - near), 0};
  // clang-format on

  // Block arcball input when the mouse is inside the widget circle
  const ImVec2 center{x + size * 0.5f, y + size * 0.5f};
  const ImVec2 mousePos = ImGui::GetIO().MousePos;
  const float dx = mousePos.x - center.x;
  const float dy = mousePos.y - center.y;
  const bool mouseInWidget = (dx * dx + dy * dy) <= (size * 0.5f * size * 0.5f);

  ImOGuizmo::SetDrawList(ImGui::GetWindowDrawList());
  const float pivotDist = m_camera.arcball->distance();
  const bool snapped = ImOGuizmo::DrawGizmo(viewMat, projMat, pivotDist);
  if (snapped)
    applyViewMatrixToArcball(viewMat);

  return snapped || mouseInWidget;
}

void BaseViewport::ui_animationSlider()
{
  if (!m_showAnimationSlider)
    return;

  auto &animMgr = appContext()->tsd.animationMgr;

  const ImVec2 contentStart = ImGui::GetCursorStartPos();
  const float vpW = static_cast<float>(m_viewport.size.x);
  const float vpH = static_cast<float>(m_viewport.size.y);
  const float sliderW = vpW * 0.8f;
  const float itemH = ImGui::GetFrameHeight();
  const float padBottom = vpH * 0.05f;

  const float x = contentStart.x + (vpW - sliderW) * 0.5f;
  const float y = contentStart.y + vpH - padBottom - itemH;
  ImGui::SetCursorPos(ImVec2(x, y));

  ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.f, 0.f, 0.f, 0.5f));
  const ImGuiWindowFlags flags =
      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
  if (ImGui::BeginChild("##animSlider",
          ImVec2(sliderW, itemH),
          ImGuiChildFlags_None,
          flags)) {
    ImGui::SetNextItemWidth(-1.f);
    float time = animMgr.getAnimationTime();
    if (ImGui::SliderFloat("##animTime", &time, 0.f, 1.f))
      animMgr.setAnimationTime(time);
  }
  ImGui::EndChild();
  ImGui::PopStyleColor();
}

void BaseViewport::ui_menubar_Renderer()
{
  if (ImGui::BeginMenu("Renderer")) {
    if (m_renderers.objects.size() > 1) {
      ImGui::Text("Subtype:");
      ImGui::Indent(INDENT_AMOUNT);
      for (int i = 0; i < m_renderers.objects.size(); i++) {
        auto ro = m_renderers.objects[i];
        const char *rName = ro->subtype().c_str();
        if (ImGui::RadioButton(rName, m_renderers.current == ro))
          m_renderers.current = ro;
      }
      ImGui::Unindent(INDENT_AMOUNT);
    }

    ImGui::Separator();

    if (!m_renderers.objects.empty()) {
      ImGui::Text("Parameters:");
      ImGui::Indent(INDENT_AMOUNT);

      tsd::ui::buildUI_object(
          *m_renderers.current, appContext()->tsd.scene, true);

      ImGui::Unindent(INDENT_AMOUNT);
      ImGui::Separator();
      ImGui::Separator();
      ImGui::Indent(INDENT_AMOUNT);

      if (ImGui::BeginMenu("Reset to Defaults?")) {
        if (ImGui::MenuItem("Yes"))
          renderer_resetParameterDefaults();
        ImGui::EndMenu();
      }

      ImGui::Unindent(INDENT_AMOUNT);
    }
    ImGui::EndMenu();
  }
}

void BaseViewport::ui_menubar_Camera()
{
  if (ImGui::BeginMenu("Camera")) {
    auto &scene = appContext()->tsd.scene;

    ImGui::Text("Manipulator:");
    {
      ImGui::Indent(INDENT_AMOUNT);

      auto axis = static_cast<int>(m_camera.arcball->axis());
      if (ImGui::Combo("Up", &axis, "+x\0+y\0+z\0-x\0-y\0-z\0\0"))
        m_camera.arcball->setAxis(static_cast<tsd::rendering::UpAxis>(axis));

      auto at = m_camera.arcball->at();
      auto azel = m_camera.arcball->azel();
      auto dist = m_camera.arcball->distance();
      auto fixedDist = m_camera.arcball->fixedDistance();

      bool update = ImGui::SliderFloat("Azimuth", &azel.x, 0.f, 360.f);
      update |= ImGui::SliderFloat("Elevation", &azel.y, 0.f, 360.f);
      update |= ImGui::DragFloat("Distance", &dist);
      update |= ImGui::DragFloat3("At", &at.x);
      ImGui::BeginDisabled(
          m_camera.current->subtype() != scene::tokens::camera::orthographic);
      update |= ImGui::DragFloat("Near", &fixedDist);
      if (ImGui::IsItemHovered())
        ImGui::SetTooltip("near plane distance for orthographic camera");
      ImGui::EndDisabled();

      if (update) {
        m_camera.arcball->setConfig(at, dist, azel);
        m_camera.arcball->setFixedDistance(fixedDist);
      }

      ImGui::Separator();

      if (ImGui::BeginMenu("Reset View")) {
        if (ImGui::MenuItem("Center"))
          camera_centerView();
        if (ImGui::MenuItem("Distance"))
          camera_resetView(false);
        if (ImGui::MenuItem("Angle + Distance + Center"))
          camera_resetView(true);
        ImGui::EndMenu();
      }

      ImGui::Unindent(INDENT_AMOUNT);
    }

    ImGui::Separator();

    ImGui::Text("Current Camera:");
    {
      ImGui::Indent(INDENT_AMOUNT);

      if (ImGui::BeginMenu("Select Camera")) {
        if (ImGui::BeginMenu("New")) {
          tsd::scene::CameraRef newCam;
          if (ImGui::MenuItem("Perspective")) {
            newCam = scene.createObject<tsd::scene::Camera>(
                tsd::scene::tokens::camera::perspective);
          }
          if (ImGui::MenuItem("Orthographic")) {
            newCam = scene.createObject<tsd::scene::Camera>(
                tsd::scene::tokens::camera::orthographic);
          }
          if (ImGui::MenuItem("Omnidirectional")) {
            newCam = scene.createObject<tsd::scene::Camera>(
                tsd::scene::tokens::camera::omnidirectional);
          }

          if (newCam) {
            camera_setCurrent(newCam);
            newCam->setName("camera" + std::to_string(newCam->index()));
            BaseViewport::camera_update(true);
          }

          ImGui::EndMenu();
        }

        ImGui::Separator();

        auto t = ANARI_CAMERA;
        if (auto i = tsd::ui::buildUI_objects_menulist(scene, t);
            i != TSD_INVALID_INDEX) {
          camera_setCurrent(scene.getObject<tsd::scene::Camera>(i));
          tsd::rendering::updateManipulatorFromCamera(
              *m_camera.arcball, *m_camera.current);
        }

        ImGui::EndMenu();
      }

      ImGui::Separator();
      tsd::ui::buildUI_object(*m_camera.current, scene, true);
      ImGui::Unindent(INDENT_AMOUNT);
    }

    ImGui::EndMenu();
  }
}

void BaseViewport::ui_menubar_TransformManipulator()
{
  if (ImGui::BeginMenu("Transform Manipulator")) {
    ImGui::Checkbox("Enabled", &m_gizmo.active);

    ImGui::Separator();
    ImGui::Text("Operation:");
    ImGui::Indent(INDENT_AMOUNT);
    const auto &gOp = m_gizmo.operation;
    if (ImGui::RadioButton("(w) Translate", gOp == ImGuizmo::TRANSLATE))
      m_gizmo.operation = ImGuizmo::TRANSLATE;
    if (ImGui::RadioButton("(e) Scale", gOp == ImGuizmo::SCALE))
      m_gizmo.operation = ImGuizmo::SCALE;
    if (ImGui::RadioButton("(r) Rotate", gOp == ImGuizmo::ROTATE))
      m_gizmo.operation = ImGuizmo::ROTATE;
    ImGui::Unindent(INDENT_AMOUNT);

    ImGui::Separator();
    ImGui::Text("Mode:");
    ImGui::Indent(INDENT_AMOUNT);
    if (ImGui::RadioButton("Local", m_gizmo.mode == ImGuizmo::LOCAL))
      m_gizmo.mode = ImGuizmo::LOCAL;
    if (ImGui::RadioButton("World", m_gizmo.mode == ImGuizmo::WORLD))
      m_gizmo.mode = ImGuizmo::WORLD;
    ImGui::Unindent(INDENT_AMOUNT);

    ImGui::EndMenu();
  }
}

int BaseViewport::windowFlags() const
{
  return ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoScrollbar;
}

void BaseViewport::applyViewMatrixToArcball(const float *viewMat)
{
  // Extract forward direction from column-major view matrix produced by
  // imoguizmo's lookAt: col2 = {r[2], u[2], -f[2], 0}, so f = {-[2],-[6],-[10]}
  const tsd::math::float3 forward{-viewMat[2], -viewMat[6], -viewMat[10]};

  // d = direction from 'at' toward new 'eye' = -forward
  const tsd::math::float3 d = -forward;

  // Invert azelToDirection for the current up axis to recover azel.
  // Manipulator::update() uses: az = radians(-m_azel.x), el =
  // radians(-m_azel.y) so we store: m_azel.x = -degrees(az), m_azel.y =
  // -degrees(el)
  float az_rad = 0.f, el_rad = 0.f;
  switch (m_camera.arcball->axis()) {
  case tsd::rendering::UpAxis::POS_Y: {
    // azelToDirection = -normalize({sin(az)*cos(el), sin(el), cos(az)*cos(el)})
    const tsd::math::float3 D = -d;
    el_rad = std::asin(D.y);
    az_rad = std::atan2(D.x, D.z);
    break;
  }
  case tsd::rendering::UpAxis::NEG_Y: {
    // azelToDirection = normalize({sin(az)*cos(el), sin(el), cos(az)*cos(el)})
    const tsd::math::float3 D = d;
    el_rad = std::asin(D.y);
    az_rad = std::atan2(D.x, D.z);
    break;
  }
  case tsd::rendering::UpAxis::POS_Z: {
    // azelToDirection = -normalize({sin(az)*cos(el), cos(az)*cos(el), sin(el)})
    const tsd::math::float3 D = -d;
    el_rad = std::asin(D.z);
    az_rad = std::atan2(D.x, D.y);
    break;
  }
  case tsd::rendering::UpAxis::NEG_Z: {
    // azelToDirection = normalize({sin(az)*cos(el), cos(az)*cos(el), sin(el)})
    const tsd::math::float3 D = d;
    el_rad = std::asin(D.z);
    az_rad = std::atan2(D.x, D.y);
    break;
  }
  case tsd::rendering::UpAxis::POS_X: {
    // azelToDirection = -normalize({sin(el), cos(az)*cos(el), sin(az)*cos(el)})
    const tsd::math::float3 D = -d;
    el_rad = std::asin(D.x);
    az_rad = std::atan2(D.z, D.y);
    break;
  }
  case tsd::rendering::UpAxis::NEG_X: {
    // azelToDirection = normalize({sin(el), cos(az)*cos(el), sin(az)*cos(el)})
    const tsd::math::float3 D = d;
    el_rad = std::asin(D.x);
    az_rad = std::atan2(D.z, D.y);
    break;
  }
  }

  const tsd::math::float2 newAzel{
      -tsd::math::degrees(az_rad), -tsd::math::degrees(el_rad)};
  m_camera.arcball->setConfig(
      m_camera.arcball->at(), m_camera.arcball->distance(), newAzel);
}

} // namespace tsd::ui::imgui
