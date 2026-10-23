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

#include "optix_visrtx.h"
#include "Object.h"

#ifdef USE_MDL
#include <algorithm>
#include <cstdlib>
#include <thread>
#endif

namespace visrtx {

void buildOptixBVH(std::vector<OptixBuildInput> buildInput,
    DeviceBuffer &bvh,
    OptixTraversableHandle &traversable,
    box3 &bounds,
    Object *obj)
{
  traversable = {};
  bounds = {};

  if (buildInput.empty()) {
    obj->reportMessage(ANARI_SEVERITY_DEBUG, "skipping BVH build");
    return;
  }

  auto &state = *obj->deviceState();

  // TLAS setup //

  OptixAccelBuildOptions accelOptions{};
  accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes tlasBufferSizes;
  OPTIX_CHECK_OBJECT(optixAccelComputeMemoryUsage(state.optixContext,
                         &accelOptions,
                         buildInput.data(),
                         buildInput.size(),
                         &tlasBufferSizes),
      obj);

  // Prepare compaction //

  DeviceBuffer compactedSizeBuffer;
  compactedSizeBuffer.reserve(sizeof(uint64_t));

  OptixAccelEmitDesc emitDesc[2];
  emitDesc[0].type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emitDesc[0].result = (CUdeviceptr)compactedSizeBuffer.ptr();

  DeviceBuffer aabbBuffer;
  aabbBuffer.reserve(sizeof(box3));

  emitDesc[1].type = OPTIX_PROPERTY_TYPE_AABBS;
  emitDesc[1].result = (CUdeviceptr)aabbBuffer.ptr();

  // Execute BVH build //

  DeviceBuffer tempBuffer;
  tempBuffer.reserve(tlasBufferSizes.tempSizeInBytes);

  DeviceBuffer outputBuffer;
  outputBuffer.reserve(tlasBufferSizes.outputSizeInBytes);

  OPTIX_CHECK_OBJECT(optixAccelBuild(state.optixContext,
                         state.stream,
                         &accelOptions,
                         buildInput.data(),
                         buildInput.size(),
                         (CUdeviceptr)tempBuffer.ptr(),
                         tempBuffer.bytes(),
                         (CUdeviceptr)outputBuffer.ptr(),
                         outputBuffer.bytes(),
                         &traversable,
                         emitDesc,
                         2),
      obj);
  CUDA_SYNC_CHECK_OBJECT(obj);

  aabbBuffer.download(&bounds);

  // Perform compaction //

  uint64_t compactedSize;
  compactedSizeBuffer.download(&compactedSize, 1);

  bvh.reserve(compactedSize);
  OPTIX_CHECK_OBJECT(optixAccelCompact(state.optixContext,
                         state.stream,
                         traversable,
                         (CUdeviceptr)bvh.ptr(),
                         bvh.bytes(),
                         &traversable),
      obj);
  CUDA_SYNC_CHECK_OBJECT(obj);
}

///////////////////////////////////////////////////////////////////////////////
// DeviceGlobalState definitions //////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

DeviceGlobalState::DeviceGlobalState(ANARIDevice d)
    : helium::BaseGlobalDeviceState(d), anariDevice(d)
{}

DeviceGlobalState::~DeviceGlobalState() = default;

#ifdef USE_MDL
namespace {

// MDL policy, not executor policy. Eight is a conservative default resource
// limit, not a measured scaling ceiling. The runtime environment overrides it;
// a hard ceiling protects against accidentally requesting thousands of threads.
std::size_t mdlCompileWorkerCount()
{
  constexpr std::size_t maxDefaultWorkers = 8;
  constexpr std::size_t maxWorkers = 256;
  if (const char *env = std::getenv("VISRTX_MDL_COMPILE_THREADS")) {
    char *end = nullptr;
    const long value = std::strtol(env, &end, 10);
    if (end != env && value >= 1)
      return std::clamp<std::size_t>(
          static_cast<std::size_t>(value), 1, maxWorkers);
  }
  return std::clamp<std::size_t>(
      std::thread::hardware_concurrency(), 1, maxDefaultWorkers);
}

} // namespace

DeviceGlobalState::MDL::MDL(DeviceGlobalState *deviceState)
    : core(new mdl::Logger(deviceState)),
      materialRegistry(&core),
      samplerRegistry(&core, deviceState),
      coordinator(mdlCompileWorkerCount())
{}

DeviceGlobalState::MDL::MDL(
    DeviceGlobalState *deviceState, mi::neuraylib::INeuray *neuray)
    : core(neuray),
      materialRegistry(&core),
      samplerRegistry(&core, deviceState),
      coordinator(mdlCompileWorkerCount())
{}
#endif

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_DEFINITION(visrtx::box1);
