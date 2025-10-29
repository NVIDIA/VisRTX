/*
 * Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "Denoiser.h"
#include "camera/Camera.h"
#include "gpu/gpu_objects.h"
#include "renderer/Renderer.h"
#include "world/World.h"
#include "utility/DeviceObject.h"
// helium
#include "helium/BaseFrame.h"
// std
#include <memory>

namespace visrtx {

struct Frame : public helium::BaseFrame, public DeviceObject<FrameGPUData>
{
  Frame(DeviceGlobalState *d);
  ~Frame();

  bool isValid() const override;

  DeviceGlobalState *deviceState() const;

  bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint64_t size,
      uint32_t flags) override;

  void commitParameters() override;
  void finalize() override;

  void renderFrame() override;

  void *map(std::string_view channel,
      uint32_t *width,
      uint32_t *height,
      ANARIDataType *pixelType) override;
  void unmap(std::string_view channel) override;
  int frameReady(ANARIWaitMask m) override;
  void discard() override;

  void *mapColorBuffer(bool gpu);
  void *mapDepthBuffer(bool gpu);
  void *mapPrimIDBuffer(bool gpu);
  void *mapObjIDBuffer(bool gpu);
  void *mapInstIDBuffer(bool gpu);
  void *mapAlbedoBuffer(bool gpu);
  void *mapNormalBuffer(bool gpu);

 private:
  bool ready() const;
  void wait() const;
  bool checkerboarding() const;
  void checkAccumulationReset();
  void newFrame();
  size_t numPixels() const;

  //// Data ////

  float m_invFrameID{1.f};
  int m_perPixelBytes{1};
  bool m_denoise{false};
  bool m_nextFrameReset{true};
  bool m_frameMappedOnce{false}; // NOTE(jda) - for instrumented events

  anari::DataType m_colorType{ANARI_UNKNOWN};
  anari::DataType m_depthType{ANARI_UNKNOWN};
  anari::DataType m_primIDType{ANARI_UNKNOWN};
  anari::DataType m_objIDType{ANARI_UNKNOWN};
  anari::DataType m_instIDType{ANARI_UNKNOWN};
  anari::DataType m_albedoType{ANARI_UNKNOWN};
  anari::DataType m_normalType{ANARI_UNKNOWN};

  HostDeviceArray<uint8_t> m_pixelBuffer;
  HostDeviceArray<float> m_depthBuffer;
  HostDeviceArray<vec3> m_normalBuffer;
  HostDeviceArray<vec3> m_albedoBuffer;
  HostDeviceArray<uint32_t> m_primIDBuffer;
  HostDeviceArray<uint32_t> m_objIDBuffer;
  HostDeviceArray<uint32_t> m_instIDBuffer;

  DeviceBuffer m_accumColor; // vec4
  DeviceBuffer m_accumAlbedo; // vec3
  DeviceBuffer m_accumNormal; // vec3

  helium::IntrusivePtr<Renderer> m_renderer;
  helium::IntrusivePtr<Camera> m_camera;
  helium::IntrusivePtr<World> m_world;

  cudaEvent_t m_eventStart;
  cudaEvent_t m_eventEnd;

  float m_duration{0.f};

  bool m_frameChanged{false};
  helium::TimeStamp m_cameraLastChanged{0};
  helium::TimeStamp m_rendererLastChanged{0};
  helium::TimeStamp m_worldLastChanged{0};
  helium::TimeStamp m_lastCommitFlushOccured{0};
  helium::TimeStamp m_lastUploadFlushOccured{0};

  Denoiser m_denoiser;

  anari::FrameCompletionCallback m_callback{nullptr};
  const void *m_callbackUserPtr{nullptr};
};

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_SPECIALIZATION(visrtx::Frame *, ANARI_FRAME);
