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

#include "Frame.h"
#include "utility/instrument.h"
// std
#include <algorithm>
#include <random>
// thrust
#include <cuda_runtime_api.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/transform.h>

namespace visrtx {

// Frame definitions //////////////////////////////////////////////////////////

Frame::Frame(DeviceGlobalState *d) : helium::BaseFrame(d), m_denoiser(d)
{
  cudaEventCreate(&m_eventStart);
  cudaEventCreate(&m_eventEnd);

  cudaEventRecord(m_eventStart, d->stream);
  cudaEventRecord(m_eventEnd, d->stream);
}

Frame::~Frame()
{
  wait();

  cudaEventDestroy(m_eventStart);
  cudaEventDestroy(m_eventEnd);
}

bool Frame::isValid() const
{
  return m_renderer && m_renderer->isValid() && m_camera && m_camera->isValid()
      && m_world && m_world->isValid();
}

DeviceGlobalState *Frame::deviceState() const
{
  return (DeviceGlobalState *)helium::BaseObject::m_state;
}

void Frame::commitParameters()
{
  m_renderer = getParamObject<Renderer>("renderer");
  m_camera = getParamObject<Camera>("camera");
  m_world = getParamObject<World>("world");
  m_callback = getParam<ANARIFrameCompletionCallback>(
      "frameCompletionCallback", nullptr);
  m_callbackUserPtr =
      getParam<void *>("frameCompletionCallbackUserData", nullptr);
  m_colorType =
      getParam<ANARIDataType>("channel.color", ANARI_UFIXED8_RGBA_SRGB);
  auto &hd = data();
  hd.fb.size = getParam<uvec2>("size", uvec2(10));
  m_depthType = getParam<ANARIDataType>("channel.depth", ANARI_UNKNOWN);
  m_primIDType = getParam<ANARIDataType>("channel.primitiveId", ANARI_UNKNOWN);
  m_objIDType = getParam<ANARIDataType>("channel.objectId", ANARI_UNKNOWN);
  m_instIDType = getParam<ANARIDataType>("channel.instanceId", ANARI_UNKNOWN);
  m_albedoType = getParam<ANARIDataType>("channel.albedo", ANARI_UNKNOWN);
  m_normalType = getParam<ANARIDataType>("channel.normal", ANARI_UNKNOWN);
}

void Frame::finalize()
{
  if (!isValid())
    return;

  auto &hd = data();

  const bool useFloatFB = m_denoise || m_colorType == ANARI_FLOAT32_VEC4;
  if (useFloatFB)
    hd.fb.format = FrameFormat::FLOAT;
  else if (m_colorType == ANARI_UFIXED8_RGBA_SRGB)
    hd.fb.format = FrameFormat::SRGB;
  else
    hd.fb.format = FrameFormat::UINT;

  hd.fb.invSize = 1.f / vec2(hd.fb.size);

  const bool channelPrimID = m_primIDType == ANARI_UINT32;
  const bool channelObjID = m_objIDType == ANARI_UINT32;
  const bool channelInstID = m_instIDType == ANARI_UINT32;
  const bool channelAlbedo = m_albedoType == ANARI_FLOAT32;
  const bool channelNormal = m_normalType == ANARI_FLOAT32;

  const bool channelDepth = m_depthType == ANARI_FLOAT32 || channelPrimID
      || channelObjID || channelInstID;
  if (channelDepth && m_depthType != ANARI_FLOAT32)
    m_depthType = ANARI_FLOAT32;

  m_perPixelBytes = 4 * (useFloatFB ? 4 : 1);

  m_pixelBuffer.resize(numPixels() * m_perPixelBytes);
  m_depthBuffer.resize(channelDepth ? numPixels() : 0);
  m_normalBuffer.resize(channelNormal ? numPixels() : 0);
  m_albedoBuffer.resize(channelAlbedo ? numPixels() : 0);
  m_primIDBuffer.resize(channelPrimID ? numPixels() : 0);
  m_objIDBuffer.resize(channelObjID ? numPixels() : 0);
  m_instIDBuffer.resize(channelInstID ? numPixels() : 0);

  m_accumColor.reserve(numPixels() * sizeof(vec4));
  m_accumAlbedo.reserve((channelAlbedo ? numPixels() : 0) * sizeof(vec3));
  m_accumNormal.reserve((channelNormal ? numPixels() : 0) * sizeof(vec3));

  hd.fb.buffers.colorAccumulation = m_accumColor.ptrAs<vec4>();

  hd.fb.buffers.outColorVec4 = nullptr;
  hd.fb.buffers.outColorUint = nullptr;

  if (useFloatFB)
    hd.fb.buffers.outColorVec4 = (vec4 *)m_pixelBuffer.dataDevice();
  else
    hd.fb.buffers.outColorUint = (uint32_t *)m_pixelBuffer.dataDevice();

  hd.fb.buffers.depth = channelDepth ? m_depthBuffer.dataDevice() : nullptr;
  hd.fb.buffers.primID = channelPrimID ? m_primIDBuffer.dataDevice() : nullptr;
  hd.fb.buffers.objID = channelObjID ? m_objIDBuffer.dataDevice() : nullptr;
  hd.fb.buffers.instID = channelInstID ? m_instIDBuffer.dataDevice() : nullptr;
  hd.fb.buffers.albedo = channelAlbedo ? m_accumAlbedo.ptrAs<vec3>() : nullptr;
  hd.fb.buffers.normal = channelNormal ? m_accumNormal.ptrAs<vec3>() : nullptr;

  if (m_denoise)
    m_denoiser.setup(hd.fb.size, m_pixelBuffer, m_colorType);
  else
    m_denoiser.cleanup();

  m_frameChanged = true;
}

bool Frame::getProperty(const std::string_view &name,
    ANARIDataType type,
    void *ptr,
    uint64_t size,
    uint32_t flags)
{
  if (type == ANARI_FLOAT32 && name == "duration") {
    if (flags & ANARI_WAIT)
      wait();
    cudaEventElapsedTime(&m_duration, m_eventStart, m_eventEnd);
    m_duration /= 1000;
    helium::writeToVoidP(ptr, m_duration);
    return true;
  } else if (type == ANARI_INT32 && name == "numSamples") {
    if (flags & ANARI_WAIT)
      wait();
    auto &hd = data();
    helium::writeToVoidP(ptr, hd.fb.frameID);
    return true;
  } else if (type == ANARI_BOOL && name == "nextFrameReset") {
    if (flags & ANARI_WAIT)
      wait();
    if (ready())
      deviceState()->commitBuffer.flush();
    checkAccumulationReset();
    helium::writeToVoidP(ptr, m_nextFrameReset);
    return true;
  }

  return 0;
}

void Frame::renderFrame()
{
  wait();

  auto &state = *deviceState();

  instrument::rangePush("update scene");
  instrument::rangePush("flush commits");
  state.commitBuffer.flush();
  instrument::rangePop(); // flush commits

  instrument::rangePush("flush array uploads");
  state.uploadBuffer.flush();
  instrument::rangePop(); // flush array uploads

  instrument::rangePush("rebuild BVHs");
  auto worldLock = m_world->scopeLockObject();
  m_world->rebuildWorld();
  instrument::rangePop(); // rebuild BVHs
  instrument::rangePop(); // update scene

  if (!isValid()) {
    std::string problemMsg = "<unknown>";
    if (!m_renderer)
      problemMsg = "missing ANARIRenderer";
    else if (!m_renderer->isValid())
      problemMsg = "invalid ANARIRenderer";
    else if (!m_camera)
      problemMsg = "missing ANARICamera";
    else if (!m_camera->isValid())
      problemMsg = "invalid ANARICamera";
    else if (!m_world)
      problemMsg = "missing ANARIWorld";
    else if (!m_world->isValid())
      problemMsg = "invalid ANARIWorld";
    reportMessage(ANARI_SEVERITY_ERROR,
        "skipping render of incomplete or invalid frame object -- issue: %s",
        problemMsg.c_str());
    return;
  }

  bool wasDenoising = m_denoise;
  m_denoise = m_renderer->denoise();
  if (m_denoise != wasDenoising)
    this->finalize();

  m_frameMappedOnce = false;

  instrument::rangePush("frame + map");
  instrument::rangePush("Frame::renderFrame()");
  instrument::rangePush("frame setup");

  checkAccumulationReset();

  auto &hd = data();

  const int sampleLimit = m_renderer->sampleLimit();
  if (!m_nextFrameReset && sampleLimit > 0 && hd.fb.frameID >= sampleLimit)
    return;

  cudaEventRecord(m_eventStart, state.stream);

  m_renderer->populateFrameData(hd);

  hd.camera = (CameraGPUData *)m_camera->deviceData();
  hd.world = m_world->gpuData();

  hd.registry.samplers = state.registry.samplers.devicePtr();
  hd.registry.geometries = state.registry.geometries.devicePtr();
  hd.registry.materials = state.registry.materials.devicePtr();
  hd.registry.surfaces = state.registry.surfaces.devicePtr();
  hd.registry.lights = state.registry.lights.devicePtr();
  hd.registry.fields = state.registry.fields.devicePtr();
  hd.registry.volumes = state.registry.volumes.devicePtr();

  instrument::rangePop(); // frame setup
  instrument::rangePush("render all frames");

  instrument::rangePush("Frame::newFrame()");
  newFrame();
  instrument::rangePop(); // Frame::newFrame()

  instrument::rangePush("Frame::upload()");
  upload();
  instrument::rangePop(); // Frame::upload()

  instrument::rangePush("optixLaunch()");
  OPTIX_CHECK(optixLaunch(m_renderer->pipeline(),
      state.stream,
      (CUdeviceptr)deviceData(),
      payloadBytes(),
      m_renderer->sbt(),
      checkerboarding() ? (hd.fb.size.x + 1) / 2 : hd.fb.size.x,
      checkerboarding() ? (hd.fb.size.y + 1) / 2 : hd.fb.size.y,
      1));
  instrument::rangePop(); // optixLaunch()

  if (m_denoise)
    m_denoiser.launch();

  if (m_callback) {
    cudaLaunchHostFunc(
        state.stream,
        [](void *_this) {
          auto &self = *(Frame *)_this;
          auto *d = self.deviceState()->anariDevice;
          self.m_callback(self.m_callbackUserPtr, d, (ANARIFrame)_this);
        },
        this);
  }

  instrument::rangePop(); // render all frames
  cudaEventRecord(m_eventEnd, state.stream);
  instrument::rangePop(); // Frame::renderFrame()
  instrument::rangePush("time until FB map");
}

void *Frame::map(std::string_view channel,
    uint32_t *width,
    uint32_t *height,
    ANARIDataType *pixelType)
{
  wait();

  ANARIDataType type = ANARI_UNKNOWN;
  void *retval = nullptr;

  const bool channelDepth = m_depthType == ANARI_FLOAT32;
  const bool channelPrimID = m_primIDType == ANARI_UINT32;
  const bool channelObjID = m_objIDType == ANARI_UINT32;
  const bool channelInstID = m_instIDType == ANARI_UINT32;
  const bool channelAlbedo = m_albedoType == ANARI_FLOAT32;
  const bool channelNormal = m_normalType == ANARI_FLOAT32;

  if (channel == "channel.colorCUDA") {
    type = m_colorType;
    retval = mapColorBuffer(true);
  } else if (channelDepth && channel == "channel.depthCUDA") {
    type = ANARI_FLOAT32;
    retval = mapDepthBuffer(true);
  } else if (channelPrimID && channel == "channel.primitiveIdCUDA") {
    type = ANARI_UINT32;
    retval = mapPrimIDBuffer(true);
  } else if (channelObjID && channel == "channel.objectIdCUDA") {
    type = ANARI_UINT32;
    retval = mapObjIDBuffer(true);
  } else if (channelInstID && channel == "channel.instanceIdCUDA") {
    type = ANARI_UINT32;
    retval = mapInstIDBuffer(true);
  } else if (channelNormal && channel == "channel.normalCUDA") {
    type = ANARI_FLOAT32_VEC3;
    retval = mapNormalBuffer(true);
  } else if (channelAlbedo && channel == "channel.albedoCUDA") {
    type = ANARI_FLOAT32_VEC3;
    retval = mapAlbedoBuffer(true);
  } else if (channel == "channel.color") {
    type = m_colorType;
    retval = mapColorBuffer(false);
  } else if (channelDepth && channel == "channel.depth") {
    type = ANARI_FLOAT32;
    retval = mapDepthBuffer(false);
  } else if (channelPrimID && channel == "channel.primitiveId") {
    type = ANARI_UINT32;
    retval = mapPrimIDBuffer(false);
  } else if (channelObjID && channel == "channel.objectId") {
    type = ANARI_UINT32;
    retval = mapObjIDBuffer(false);
  } else if (channelInstID && channel == "channel.instanceId") {
    type = ANARI_UINT32;
    retval = mapInstIDBuffer(false);
  } else if (channelNormal && channel == "channel.normal") {
    type = ANARI_FLOAT32_VEC3;
    retval = mapNormalBuffer(false);
  } else if (channelAlbedo && channel == "channel.albedo") {
    type = ANARI_FLOAT32_VEC3;
    retval = mapAlbedoBuffer(false);
  } else if (channel == "channel.colorGPU") {
    reportMessage(ANARI_SEVERITY_WARNING,
        "channel.colorGPU is deprecated, please use channel.colorCUDA instead");
    type = m_colorType;
    retval = mapColorBuffer(true);
  } else if (channelDepth && channel == "channel.depthGPU") {
    reportMessage(ANARI_SEVERITY_WARNING,
        "channel.depthGPU is deprecated, please use channel.depthCUDA instead");
    type = ANARI_FLOAT32;
    retval = mapDepthBuffer(true);
  } else if (channelPrimID && channel == "channel.primitiveIdGPU") {
    reportMessage(ANARI_SEVERITY_WARNING,
        "channel.primitiveIdGPU is deprecated, please use "
        "channel.primitiveIdCUDA instead");
    type = ANARI_UINT32;
    retval = mapPrimIDBuffer(true);
  } else if (channelObjID && channel == "channel.objectIdGPU") {
    reportMessage(ANARI_SEVERITY_WARNING,
        "channel.objectIdGPU is deprecated, please use "
        "channel.objectIdCUDA instead");
    type = ANARI_UINT32;
    retval = mapObjIDBuffer(true);
  } else if (channelInstID && channel == "channel.instanceIdGPU") {
    reportMessage(ANARI_SEVERITY_WARNING,
        "channel.instanceIdGPU is deprecated, please use "
        "channel.instanceIdCUDA instead");
    type = ANARI_UINT32;
    retval = mapInstIDBuffer(true);
  } else if (channelNormal && channel == "channel.normalGPU") {
    reportMessage(ANARI_SEVERITY_WARNING,
        "channel.normalGPU is deprecated, please use "
        "channel.normalCUDA instead");
    type = ANARI_FLOAT32_VEC3;
    retval = mapNormalBuffer(true);
  } else if (channelAlbedo && channel == "channel.albedoGPU") {
    reportMessage(ANARI_SEVERITY_WARNING,
        "channel.albedoGPU is deprecated, please use "
        "channel.albedoCUDA instead");
    type = ANARI_FLOAT32_VEC3;
    retval = mapAlbedoBuffer(true);
  }

  if (type != ANARI_UNKNOWN) {
    const auto &hd = data();
    *width = hd.fb.size.x;
    *height = hd.fb.size.y;
    m_frameMappedOnce = true;
  }

  *pixelType = type;

  return retval;
}

void Frame::unmap(std::string_view channel)
{
  // no-op
}

int Frame::frameReady(ANARIWaitMask m)
{
  if (m == ANARI_NO_WAIT)
    return ready();
  else {
    wait();
    return 1;
  }
}

void Frame::discard()
{
  // no-op
}

void *Frame::mapColorBuffer(bool gpu)
{
  void *retval = nullptr;

  if (gpu) {
    if (!m_frameMappedOnce) {
      instrument::rangePop(); // time until FB map
      instrument::rangePop(); // frame + map
    }

    m_frameMappedOnce = true;

    retval =
        m_denoise ? m_denoiser.mapGPUColorBuffer() : m_pixelBuffer.dataDevice();
  } else {
    if (!m_frameMappedOnce)
      instrument::rangePop(); // time until FB map

    instrument::rangePush("copy to host");

    if (m_denoise)
      retval = m_denoiser.mapColorBuffer();
    else {
      m_pixelBuffer.download();
      retval = m_pixelBuffer.dataHost();
    }

    instrument::rangePop(); // copy to host

    if (!m_frameMappedOnce)
      instrument::rangePop(); // frame + map
  }

  return retval;
}

void *Frame::mapDepthBuffer(bool gpu)
{
  if (gpu)
    return m_depthBuffer.dataDevice();
  else {
    m_depthBuffer.download();
    return m_depthBuffer.dataHost();
  }
}

void *Frame::mapPrimIDBuffer(bool gpu)
{
  if (gpu)
    return m_primIDBuffer.dataDevice();
  else {
    m_primIDBuffer.download();
    return m_primIDBuffer.dataHost();
  }
}

void *Frame::mapObjIDBuffer(bool gpu)
{
  if (gpu)
    return m_objIDBuffer.dataDevice();
  else {
    m_objIDBuffer.download();
    return m_objIDBuffer.dataHost();
  }
}

void *Frame::mapInstIDBuffer(bool gpu)
{
  if (gpu)
    return m_instIDBuffer.dataDevice();
  else {
    m_instIDBuffer.download();
    return m_instIDBuffer.dataHost();
  }
}

void *Frame::mapAlbedoBuffer(bool gpu)
{
  auto &state = *deviceState();
  const float invFrameID = m_invFrameID;
  auto begin = thrust::device_pointer_cast<vec3>((vec3 *)m_accumAlbedo.ptr());
  auto end = begin + numPixels();
  thrust::transform(thrust::cuda::par.on(state.stream),
      begin,
      end,
      thrust::device_pointer_cast<vec3>(m_albedoBuffer.dataDevice()),
      [=] __device__(const vec3 &in) { return in * invFrameID; });
  if (gpu)
    return m_albedoBuffer.dataDevice();
  else {
    m_albedoBuffer.download();
    return m_albedoBuffer.dataHost();
  }
}

void *Frame::mapNormalBuffer(bool gpu)
{
  auto &state = *deviceState();
  const float invFrameID = m_invFrameID;
  auto begin = thrust::device_pointer_cast<vec3>((vec3 *)m_accumNormal.ptr());
  auto end = begin + numPixels();
  thrust::transform(thrust::cuda::par.on(state.stream),
      begin,
      end,
      thrust::device_pointer_cast<vec3>(m_normalBuffer.dataDevice()),
      [=] __device__(const vec3 &in) { return in * invFrameID; });
  if (gpu)
    return m_normalBuffer.dataDevice();
  else {
    m_normalBuffer.download();
    return m_normalBuffer.dataHost();
  }
}

bool Frame::ready() const
{
  return cudaEventQuery(m_eventEnd) == cudaSuccess;
}

void Frame::wait() const
{
  cudaEventSynchronize(m_eventEnd);
}

bool Frame::checkerboarding() const
{
  return m_renderer ? m_renderer->checkerboarding() : false;
}

void Frame::checkAccumulationReset()
{
  if (m_nextFrameReset)
    return;

  auto &state = *deviceState();
  if (m_lastCommitFlushOccured < state.commitBuffer.lastObjectFinalization()) {
    m_lastCommitFlushOccured = state.commitBuffer.lastObjectFinalization();
    m_nextFrameReset = true;
  }
  if (m_lastUploadFlushOccured < state.uploadBuffer.lastUpload()) {
    m_lastUploadFlushOccured = state.uploadBuffer.lastUpload();
    m_nextFrameReset = true;
  }
}

void Frame::newFrame()
{
  auto &hd = data();
  if (m_nextFrameReset) {
    hd.fb.frameID = 0;
    hd.fb.checkerboardID = checkerboarding() ? 0 : -1;
    m_nextFrameReset = false;

    // Reset buffers if needed
    const bool channelPrimID = m_primIDType == ANARI_UINT32;
    const bool channelObjID = m_objIDType == ANARI_UINT32;
    const bool channelInstID = m_instIDType == ANARI_UINT32;
    const bool channelAlbedo = m_albedoType == ANARI_FLOAT32;
    const bool channelNormal = m_normalType == ANARI_FLOAT32;

    const bool channelDepth = m_depthType == ANARI_FLOAT32 || channelPrimID
        || channelObjID || channelInstID;

    // Always clear the color accumulation buffer
    thrust::fill_n(thrust::device_pointer_cast(m_accumColor.ptrAs<vec4>()),
        numPixels(),
        vec4(0.0f));

    // Conditionally initialize other buffers
    if (channelDepth) {
      thrust::fill_n(thrust::device_pointer_cast(m_depthBuffer.dataDevice()),
          numPixels(),
          std::numeric_limits<float>::max());
    }

    if (channelPrimID) {
      thrust::fill_n(thrust::device_pointer_cast(m_primIDBuffer.dataDevice()),
          numPixels(),
          uint32_t(0));
    }

    if (channelObjID) {
      thrust::fill_n(thrust::device_pointer_cast(m_objIDBuffer.dataDevice()),
          numPixels(),
          uint32_t(0));
    }

    if (channelInstID) {
      thrust::fill_n(thrust::device_pointer_cast(m_instIDBuffer.dataDevice()),
          numPixels(),
          uint32_t(0));
    }

    if (channelAlbedo) {
      thrust::fill_n(thrust::device_pointer_cast(m_accumAlbedo.ptrAs<vec3>()),
          numPixels(),
          vec3(0.0f));
    }

    if (channelNormal) {
      thrust::fill_n(thrust::device_pointer_cast(m_accumNormal.ptrAs<vec3>()),
          numPixels(),
          vec3(0.0f));
    }
  } else {
    if (checkerboarding())
      hd.fb.frameID += int(hd.fb.checkerboardID == 3);
    else
      hd.fb.frameID += m_renderer->spp();
    hd.fb.checkerboardID =
        checkerboarding() ? ((hd.fb.checkerboardID + 1) & 0x3) : -1;
  }

  hd.fb.invFrameID = m_invFrameID = 1.f / (hd.fb.frameID + 1);
  m_frameChanged = false;
}

size_t Frame::numPixels() const
{
  auto &hd = data();
  return size_t(hd.fb.size.x) * size_t(hd.fb.size.y);
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_DEFINITION(visrtx::Frame *);
