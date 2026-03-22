// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ImagePass.h"
// anari
#include <anari/anari_cpp.hpp>

namespace tsd::rendering {

/*
 * ImagePass that drives a single ANARI Frame with a configurable camera,
 * renderer, and world; optionally captures auxiliary AOV buffers
 * (depth, normals, albedo, object/primitive/instance IDs).
 *
 * Example:
 *   auto *pass = pipeline.emplace_back<AnariSceneRenderPass>(device);
 *   pass->setCamera(cam); pass->setRenderer(rend); pass->setWorld(world);
 */
struct AnariSceneRenderPass : public ImagePass
{
  AnariSceneRenderPass(anari::Device d);
  ~AnariSceneRenderPass() override;
  const char *name() const override { return "ANARI Scene"; }

  void setCamera(anari::Camera c);
  void setRenderer(anari::Renderer r);
  void setWorld(anari::World w);
  void setColorFormat(anari::DataType t);
  void setEnableIDs(bool on);
  void setEnablePrimitiveId(bool on);
  void setEnableInstanceId(bool on);
  void setEnableAlbedo(bool on);
  void setEnableNormals(bool on);

  // default' true', if 'false', then anari::wait() on each pass
  void setRunAsync(bool on);

  anari::Frame getFrame() const;

 private:
  void updateSize() override;
  void render(ImageBuffers &b, int stageId) override;
  void copyFrameData();
  void composite(ImageBuffers &b, int stageId);
  void cleanup();

  ImageBuffers m_buffers;

  bool m_firstFrame{true};
  bool m_deviceSupportsCUDAFrames{false};
  bool m_enableIDs{false};
  bool m_enablePrimitiveId{false};
  bool m_enableInstanceId{false};
  bool m_enableAlbedo{false};
  bool m_enableNormals{false};
  bool m_runAsync{true};

  anari::DataType m_format{ANARI_UFIXED8_RGBA_SRGB};

  anari::Device m_device{nullptr};
  anari::Camera m_camera{nullptr};
  anari::Renderer m_renderer{nullptr};
  anari::World m_world{nullptr};
  anari::Frame m_frame{nullptr};
};

} // namespace tsd::rendering
