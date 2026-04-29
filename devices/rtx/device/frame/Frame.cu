/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "gpu/gpu_tonemap.h"
#include "gpu/gpu_util.h"
#include "utility/instrument.h"
// std
#include <algorithm>
#include <glm/ext/vector_float4.hpp>
#include <random>
// thrust
#include <cuda_runtime_api.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/transform.h>

namespace visrtx {

namespace {

// Resolve per-pixel (sourceIdx, divisor) for the current sub-frame. Mirrors
// compositeBackground so both kernels agree on which accumulator sample count
// and source pixel to read under checkerboarding.
__device__ bool resolveSample(uint32_t idx,
    uvec2 size,
    int frameID,
    int checkerboardID,
    uint32_t &sourceIdx,
    int &divisor)
{
  sourceIdx = idx;
  divisor = frameID;
  if (checkerboardID >= 0 && checkerboardID < 3) {
    const uint32_t px = idx % size.x;
    const uint32_t py = idx / size.x;
    const int pixTile = (px & 1) | ((py & 1) << 1);
    if (pixTile <= checkerboardID) {
      divisor = frameID + 1;
    } else if (frameID == 0) {
      sourceIdx = (px & ~1u) + (py & ~1u) * size.x;
      divisor = 1;
    }
  }
  return divisor > 0;
}

__global__ void prepareDenoiseInput(const vec4 *__restrict__ accumColor,
    vec4 *__restrict__ denoiseInput,
    uvec2 size,
    int frameID,
    int checkerboardID,
    bool fireflyFilter)
{
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size.x * size.y)
    return;

  uint32_t srcIdx;
  int divisor;
  if (!resolveSample(idx, size, frameID, checkerboardID, srcIdx, divisor)) {
    denoiseInput[idx] = vec4(0.f);
    return;
  }

  vec4 c = accumColor[srcIdx] / float(divisor);
  if (fireflyFilter)
    c = detail::inverseTonemap(c);
  denoiseInput[idx] = c;
}

void launchPrepareDenoiseInput(const vec4 *accumColor,
    vec4 *denoiseInput,
    uvec2 size,
    int frameID,
    int checkerboardID,
    bool fireflyFilter,
    cudaStream_t stream)
{
  const uint32_t nPixels = size.x * size.y;
  const uint32_t blockSize = 256;
  const uint32_t gridSize = (nPixels + blockSize - 1) / blockSize;
  prepareDenoiseInput<<<gridSize, blockSize, 0, stream>>>(
      accumColor, denoiseInput, size, frameID, checkerboardID, fireflyFilter);
}

__global__ void prepareDenoiseGuides(const vec3 *__restrict__ accumAlbedo,
    const vec3 *__restrict__ accumNormal,
    vec3 *__restrict__ denoiseAlbedo,
    vec3 *__restrict__ denoiseNormal,
    uvec2 size,
    int frameID,
    int checkerboardID)
{
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size.x * size.y)
    return;

  uint32_t srcIdx;
  int divisor;
  if (!resolveSample(idx, size, frameID, checkerboardID, srcIdx, divisor)) {
    if (denoiseAlbedo)
      denoiseAlbedo[idx] = vec3(0.f);
    if (denoiseNormal)
      denoiseNormal[idx] = vec3(0.f);
    return;
  }

  const float invDivisor = 1.0f / float(divisor);
  if (denoiseAlbedo)
    denoiseAlbedo[idx] = accumAlbedo[srcIdx] * invDivisor;

  if (denoiseNormal) {
    const vec3 n = accumNormal[srcIdx];
    const float len = glm::length(n);
    constexpr float NORMAL_EPSILON = 1e-6f;
    denoiseNormal[idx] = len > NORMAL_EPSILON ? n * (1.0f / len) : vec3(0.f);
  }
}

void launchPrepareDenoiseGuides(const vec3 *accumAlbedo,
    const vec3 *accumNormal,
    vec3 *denoiseAlbedo,
    vec3 *denoiseNormal,
    uvec2 size,
    int frameID,
    int checkerboardID,
    cudaStream_t stream)
{
  const uint32_t nPixels = size.x * size.y;
  const uint32_t blockSize = 256;
  const uint32_t gridSize = (nPixels + blockSize - 1) / blockSize;
  prepareDenoiseGuides<<<gridSize, blockSize, 0, stream>>>(accumAlbedo,
      accumNormal,
      denoiseAlbedo,
      denoiseNormal,
      size,
      frameID,
      checkerboardID);
}

__global__ void compositeBackground(vec4 *__restrict__ accumColor,
    vec4 *__restrict__ pixelBuf,
    uint32_t *__restrict__ uintBuf,
    RendererGPUData renderer,
    uvec2 size,
    vec2 invSize,
    FrameFormat format,
    int frameID,
    int checkerboardID,
    bool isDenoised)
{
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size.x * size.y)
    return;

  uint32_t sourceIdx;
  int divisor;
  if (!resolveSample(idx, size, frameID, checkerboardID, sourceIdx, divisor))
    return;

  const uint32_t px = idx % size.x;
  const uint32_t py = idx / size.x;

  vec4 rendered;
  if (isDenoised) {
    // The denoiser fills pixelBuf at every pixel, so reading from sourceIdx
    // would race against another thread compositing into that same slot.
    // Read RGB from this thread's own pixel; only the alpha needs the
    // checkerboard source redirect because accumColor is sparse.
    rendered = pixelBuf[idx];
    rendered.a = accumColor[sourceIdx].a / float(divisor);
  } else {
    rendered = accumColor[sourceIdx] / float(divisor);
    if (renderer.fireflyFilter)
      rendered = detail::inverseTonemap(rendered);
  }

  const vec2 uv = (vec2(px, py) + 0.5f) * invSize;

  vec4 bg;
  if (renderer.backgroundMode == BackgroundMode::COLOR) {
    bg = renderer.background.color;
  } else {
    const auto s = tex2D<float4>(renderer.background.texobj, uv.x, uv.y);
    bg = vec4(s.x, s.y, s.z, s.w);
  }

  vec3 rgb = vec3(rendered);
  float alpha = rendered.a;
  accumulateValue(rgb, vec3(bg) * bg.a, alpha);
  accumulateValue(alpha, bg.a, alpha);

  if (!renderer.premultipliedAlpha && alpha > 0.0f)
    rgb *= 1.0f / alpha;

  vec4 rgba = vec4(rgb, alpha);
  if (format == FrameFormat::SRGB) {
    uintBuf[idx] = glm::packUnorm4x8(glm::convertLinearToSRGB(rgba));
  } else if (format == FrameFormat::UINT) {
    uintBuf[idx] = glm::packUnorm4x8(rgba);
  } else {
    pixelBuf[idx] = rgba;
  }
}

void launchCompositeBackground(vec4 *accumColor,
    vec4 *pixelBuf,
    uint32_t *uintBuf,
    const RendererGPUData &renderer,
    uvec2 size,
    vec2 invSize,
    FrameFormat format,
    int frameID,
    int checkerboardID,
    bool isDenoised,
    cudaStream_t stream)
{
  const uint32_t nPixels = size.x * size.y;
  const uint32_t blockSize = 256;
  const uint32_t gridSize = (nPixels + blockSize - 1) / blockSize;
  compositeBackground<<<gridSize, blockSize, 0, stream>>>(accumColor,
      pixelBuf,
      uintBuf,
      renderer,
      size,
      invSize,
      format,
      frameID,
      checkerboardID,
      isDenoised);
}

} // anonymous namespace

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
  m_manualAccumulationRestart = getParam(
      "accumulationVersion", ANARI_UINT64, &m_applicationAccumulationVersion);
}

void Frame::finalize()
{
  if (!isValid())
    return;

  if (!m_manualAccumulationRestart || m_applicationAccumulationVersion == 0) {
    m_applicationAccumulationVersion = 0;
    m_lastRenderedAccumulationVersion = 0;
    m_manualAccumulationRestart = false;
  } else {
    reportMessage(ANARI_SEVERITY_DEBUG,
        "Frame using manual accumulation restart with version %zu",
        m_applicationAccumulationVersion);
  }

  auto &hd = data();

  const bool useFloatFB = m_denoise || m_colorType == ANARI_FLOAT32_VEC4;

  hd.fb.invSize = 1.f / vec2(hd.fb.size);

  const bool channelPrimID = m_primIDType == ANARI_UINT32;
  const bool channelObjID = m_objIDType == ANARI_UINT32;
  const bool channelInstID = m_instIDType == ANARI_UINT32;
  const bool channelAlbedo =
      m_denoiseUsingAlbedo || (m_albedoType == ANARI_FLOAT32_VEC3);
  const bool channelNormal =
      m_denoiseUsingNormal || (m_normalType == ANARI_FLOAT32_VEC3);

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
  if (channelAlbedo)
    m_accumAlbedo.reserve(numPixels() * sizeof(vec3));
  else
    m_accumAlbedo.reset();
  if (channelNormal)
    m_accumNormal.reserve(numPixels() * sizeof(vec3));
  else
    m_accumNormal.reset();

  if (m_denoise) {
    m_denoiseInput.reserve(numPixels() * sizeof(vec4));
    if (m_denoiseUsingAlbedo)
      m_denoiseAlbedo.reserve(numPixels() * sizeof(vec3));
    else
      m_denoiseAlbedo.reset();
    if (m_denoiseUsingNormal)
      m_denoiseNormal.reserve(numPixels() * sizeof(vec3));
    else
      m_denoiseNormal.reset();
  } else {
    m_denoiseInput.reset();
    m_denoiseAlbedo.reset();
    m_denoiseNormal.reset();
  }

  hd.fb.buffers.colorAccumulation = m_accumColor.ptrAs<vec4>();

  hd.fb.buffers.depth = channelDepth ? m_depthBuffer.dataDevice() : nullptr;
  hd.fb.buffers.primID = channelPrimID ? m_primIDBuffer.dataDevice() : nullptr;
  hd.fb.buffers.objID = channelObjID ? m_objIDBuffer.dataDevice() : nullptr;
  hd.fb.buffers.instID = channelInstID ? m_instIDBuffer.dataDevice() : nullptr;
  hd.fb.buffers.albedo = channelAlbedo ? m_accumAlbedo.ptrAs<vec3>() : nullptr;
  hd.fb.buffers.normal = channelNormal ? m_accumNormal.ptrAs<vec3>() : nullptr;

  if (m_denoise)
    m_denoiser.setup(hd.fb.size,
        m_pixelBuffer,
        m_colorType,
        m_denoiseInput,
        m_denoiseAlbedo,
        m_denoiseNormal);
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
  } else if (type == ANARI_FLOAT32 && name == "refinementProgress") {
    if (!m_renderer)
      return false;
    if (flags & ANARI_WAIT)
      wait();
    const auto sampleLimit = m_renderer->sampleLimit();
    if (sampleLimit <= 0)
      return false;
    else {
      auto &hd = data();
      const auto progress = float(hd.fb.frameID) / float(sampleLimit);
      helium::writeToVoidP(ptr, progress);
      return true;
    }
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
  bool wasDenoisingUsingAlbedo = m_denoiseUsingAlbedo;
  bool wasDenoisingUsingNormal = m_denoiseUsingNormal;
  m_denoise = m_renderer->denoise();
  m_denoiseUsingAlbedo = m_renderer->denoiseUsingAlbedo();
  m_denoiseUsingNormal = m_renderer->denoiseUsingNormal();
  if (m_denoise != wasDenoising
      || m_denoiseUsingAlbedo != wasDenoisingUsingAlbedo
      || m_denoiseUsingNormal != wasDenoisingUsingNormal)
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
  m_camera->populateFrameData(hd.camera, hd.fb.size);
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

  // Increment frameID after rendering completes
  if (checkerboarding())
    hd.fb.frameID += int(hd.fb.checkerboardID == 3);
  else
    hd.fb.frameID += m_renderer->spp();

  const bool useFloatOutput = m_denoise || m_colorType == ANARI_FLOAT32_VEC4;

  if (m_denoise) {
    launchPrepareDenoiseInput(m_accumColor.ptrAs<vec4>(),
        m_denoiseInput.ptrAs<vec4>(),
        hd.fb.size,
        hd.fb.frameID,
        hd.fb.checkerboardID,
        hd.renderer.fireflyFilter,
        state.stream);

    if (m_denoiseUsingAlbedo || m_denoiseUsingNormal) {
      launchPrepareDenoiseGuides(m_accumAlbedo.ptrAs<vec3>(),
          m_accumNormal.ptrAs<vec3>(),
          m_denoiseAlbedo.ptrAs<vec3>(),
          m_denoiseNormal.ptrAs<vec3>(),
          hd.fb.size,
          hd.fb.frameID,
          hd.fb.checkerboardID,
          state.stream);
    }

    m_denoiser.launch();

    launchCompositeBackground(m_accumColor.ptrAs<vec4>(),
        (vec4 *)m_pixelBuffer.dataDevice(),
        nullptr,
        hd.renderer,
        hd.fb.size,
        hd.fb.invSize,
        FrameFormat::FLOAT,
        hd.fb.frameID,
        hd.fb.checkerboardID,
        /*isDenoised=*/true,
        state.stream);

    m_denoiser.convertOutput();
  } else {
    const FrameFormat outFormat = useFloatOutput ? FrameFormat::FLOAT
        : m_colorType == ANARI_UFIXED8_RGBA_SRGB ? FrameFormat::SRGB
                                                 : FrameFormat::UINT;
    launchCompositeBackground(m_accumColor.ptrAs<vec4>(),
        useFloatOutput ? (vec4 *)m_pixelBuffer.dataDevice() : nullptr,
        useFloatOutput ? nullptr : (uint32_t *)m_pixelBuffer.dataDevice(),
        hd.renderer,
        hd.fb.size,
        hd.fb.invSize,
        outFormat,
        hd.fb.frameID,
        hd.fb.checkerboardID,
        /*isDenoised=*/false,
        state.stream);
  }

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
  const bool channelAlbedo = m_albedoType == ANARI_FLOAT32_VEC3;
  const bool channelNormal = m_normalType == ANARI_FLOAT32_VEC3;

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
  auto begin = thrust::device_pointer_cast<vec3>((vec3 *)m_accumNormal.ptr());
  auto end = begin + numPixels();
  thrust::transform(thrust::cuda::par.on(state.stream),
      begin,
      end,
      thrust::device_pointer_cast<vec3>(m_normalBuffer.dataDevice()),
      [=] __device__(const vec3 &in) { return normalize(in); });
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
  if (m_manualAccumulationRestart
      && m_lastRenderedAccumulationVersion < m_applicationAccumulationVersion) {
    m_lastRenderedAccumulationVersion = m_applicationAccumulationVersion;
    m_nextFrameReset = true;
  } else if (!m_manualAccumulationRestart) { // automatic accumulation restart
    if (m_lastCommitFlushOccured
        < state.commitBuffer.lastObjectFinalization()) {
      m_lastCommitFlushOccured = state.commitBuffer.lastObjectFinalization();
      m_nextFrameReset = true;
    }
    if (m_lastUploadFlushOccured < state.uploadBuffer.lastUpload()) {
      m_lastUploadFlushOccured = state.uploadBuffer.lastUpload();
      m_nextFrameReset = true;
    }
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
    const bool channelAlbedo =
        m_denoiseUsingAlbedo || (m_albedoType == ANARI_FLOAT32_VEC3);
    const bool channelNormal =
        m_denoiseUsingNormal || (m_normalType == ANARI_FLOAT32_VEC3);

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
          ~0u);
    }

    if (channelObjID) {
      thrust::fill_n(thrust::device_pointer_cast(m_objIDBuffer.dataDevice()),
          numPixels(),
          ~0u);
    }

    if (channelInstID) {
      thrust::fill_n(thrust::device_pointer_cast(m_instIDBuffer.dataDevice()),
          numPixels(),
          ~0u);
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
