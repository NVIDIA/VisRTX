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
#include <nonstd/scope.hpp>
#include "Object.h"
#include "geometry/Geometry.h"

#ifdef USE_MDL
#include <algorithm>
#include <cstdlib>
#include <thread>
#endif

namespace visrtx {

bool buildOptixBVH(std::vector<OptixBuildInput> buildInput,
    DeviceBuffer &bvh,
    OptixTraversableHandle &traversable,
    box3 &bounds,
    Object *obj)
{
  traversable = {};
  bounds = {};
  if (buildInput.empty()) {
    obj->reportMessage(ANARI_SEVERITY_DEBUG, "skipping BVH build");
    return true;
  }

  auto &state = *obj->deviceState();
  auto &preparation = state.preparation;
  const auto stream = preparation.stream();
  auto check = [&](cudaError_t error, const char *operation) {
    if (error == cudaSuccess)
      return true;
    preparation.finish(); // Drain queued consumers before scratch is freed.
    cudaGetLastError();
    obj->reportMessage(ANARI_SEVERITY_ERROR,
        "BVH preparation failed during %s: %s",
        operation,
        cudaGetErrorString(error));
    return false;
  };
  auto checkOptix = [&](OptixResult result, const char *operation) {
    if (result == OPTIX_SUCCESS)
      return true;
    preparation.finish();
    obj->reportMessage(ANARI_SEVERITY_ERROR,
        "BVH preparation failed during %s: %s",
        operation,
        optixGetErrorName(result));
    return false;
  };
  auto reserve = [&](DeviceBuffer &buffer, size_t bytes) {
    if (!buffer)
      buffer.reset();
    buffer.reserve(bytes);
    const auto error = cudaGetLastError();
    return check(error != cudaSuccess ? error
            : bytes && (!buffer || buffer.bytes() < bytes)
            ? cudaErrorMemoryAllocation
            : cudaSuccess,
        "allocating device storage");
  };

  // Legacy producers include field/isosurface work on the render stream and
  // authored arrays/instance uploads on the default stream. Previous rendering
  // must also finish before the existing BVH storage can be reused.
  if (!check(preparation.waitFor(state.stream), "previous BVH consumers")
      || !check(preparation.waitFor(nullptr), "input upload dependency"))
    return false;

  OptixAccelBuildOptions options{};
  options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  options.operation = OPTIX_BUILD_OPERATION_BUILD;
  OptixAccelBufferSizes sizes{};
  if (!checkOptix(optixAccelComputeMemoryUsage(state.optixContext,
                      &options,
                      buildInput.data(),
                      buildInput.size(),
                      &sizes),
          "memory requirements"))
    return false;

  DeviceBuffer compactedSizeBuffer, aabbBuffer, tempBuffer, outputBuffer;
  if (!reserve(compactedSizeBuffer, sizeof(uint64_t))
      || !reserve(aabbBuffer, sizeof(box3))
      || !reserve(tempBuffer, sizes.tempSizeInBytes)
      || !reserve(outputBuffer, sizes.outputSizeInBytes))
    return false;

  OptixAccelEmitDesc emit[2]{};
  emit[0].type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emit[0].result = (CUdeviceptr)compactedSizeBuffer.ptr();
  emit[1].type = OPTIX_PROPERTY_TYPE_AABBS;
  emit[1].result = (CUdeviceptr)aabbBuffer.ptr();
  OptixTraversableHandle built{};
  if (!checkOptix(optixAccelBuild(state.optixContext,
                      stream,
                      &options,
                      buildInput.data(),
                      buildInput.size(),
                      (CUdeviceptr)tempBuffer.ptr(),
                      tempBuffer.bytes(),
                      (CUdeviceptr)outputBuffer.ptr(),
                      outputBuffer.bytes(),
                      &built,
                      emit,
                      2),
          "build"))
    return false;

  box3 builtBounds;
  uint64_t compactedSize = 0;
  if (!check(cudaMemcpyAsync(&builtBounds,
                 aabbBuffer.ptr(),
                 sizeof(box3),
                 cudaMemcpyDeviceToHost,
                 stream),
          "bounds readback")
      || !check(cudaMemcpyAsync(&compactedSize,
                    compactedSizeBuffer.ptr(),
                    sizeof(compactedSize),
                    cudaMemcpyDeviceToHost,
                    stream),
          "compaction size readback")
      || !check(preparation.finish(), "build completion"))
    return false;

  if (!reserve(bvh, compactedSize)
      || !checkOptix(optixAccelCompact(state.optixContext,
                         stream,
                         built,
                         (CUdeviceptr)bvh.ptr(),
                         bvh.bytes(),
                         &built),
          "compaction")
      || !check(preparation.finish(), "compaction completion"))
    return false;

  // Publication and scratch destruction happen only after this stream's work
  // completes. Subsequent tracing needs no device-wide barrier.
  traversable = built;
  bounds = builtBounds;
  return true;
}

///////////////////////////////////////////////////////////////////////////////
// DeviceGlobalState definitions //////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

DeviceGlobalState::DeviceGlobalState(ANARIDevice d)
    : helium::BaseGlobalDeviceState(d), anariDevice(d)
{}

DeviceGlobalState::~DeviceGlobalState() = default;

void DeviceGlobalState::enqueueGeometryInputUploads(Geometry *geometry)
{
  m_geometryInputUploads.emplace_back(geometry);
}

cudaError_t DeviceGlobalState::stageGeometryArray(
    Array *array, helium::TimeStamp modified)
{
  const auto found = m_geometryInputVersions.find(array);
  if (found != m_geometryInputVersions.end() && found->second == modified)
    return cudaSuccess;
  const auto error = array->enqueueUpload(preparation.stream());
  if (error == cudaSuccess)
    m_geometryInputVersions.insert_or_assign(array, modified);
  return error;
}

bool DeviceGlobalState::collectingGeometryInputUploads() const
{
  return m_collectingGeometryInputUploads;
}

bool DeviceGlobalState::collectGeometryInputUploads()
{
  if (m_collectingGeometryInputUploads)
    return false;
  if (m_geometryInputUploads.empty())
    return true;
  m_collectingGeometryInputUploads = true;
  bool completed = false;
  auto collecting = nonstd::make_scope_exit([&] {
    if (!completed) {
      // A host allocation or callback can unwind after copies were submitted.
      // Keep sources alive and drain before permitting public release again.
      for (auto &geometry : m_geometryInputUploads)
        geometry->failInputUploads();
      finishGeometryPreparation();
      m_geometryInputVersions.clear();
    }
    m_collectingGeometryInputUploads = false;
  });
  // Commits capture sources; the first geometry finalizer submits every input
  // batch before waiting. No legacy finalizer sees partially uploaded arrays.
  // Retire obsolete cached copies before allocating a whole replacement cohort.
  // A single geometry retains the existing allocation/retirement order.
  if (m_geometryInputUploads.size() > 1)
    for (auto &geometry : m_geometryInputUploads)
      geometry->retirePreparationSnapshot();
  auto error = preparation.waitFor(stream);
  if (error == cudaSuccess) {
    for (auto &geometry : m_geometryInputUploads)
      geometry->submitInputUploads();
    error = finishGeometryPreparation();
  }
  if (error != cudaSuccess) {
    for (auto &geometry : m_geometryInputUploads)
      geometry->failInputUploads();
    checkPreparation(m_geometryInputUploads.front().ptr,
        error,
        "geometry input batch completion");
  }
  completed = true;
  m_geometryInputVersions.clear();
  auto pending = std::move(m_geometryInputUploads);
  for (auto &geometry : pending)
    geometry->completeInputUploads();
  pending.clear();
  m_geometryInputUploads.swap(pending);
  return true;
}

void DeviceGlobalState::enqueueGeometryPreparation(Geometry *geometry)
{
  m_geometryPreparations.emplace_back(geometry);
}

bool DeviceGlobalState::collectingGeometryPreparations() const
{
  return m_collectingGeometryPreparations || m_collectingGeometryInputUploads;
}

bool DeviceGlobalState::hasPendingGeometryPreparations() const
{
  return m_collectingGeometryPreparations || !m_geometryPreparations.empty()
      || m_collectingGeometryInputUploads;
}

cudaError_t DeviceGlobalState::finishGeometryPreparation()
{
  const auto error = preparation.finish();
  if (error != cudaSuccess) {
    // An intervening upload/legacy consumer may wait before the collection
    // point. Its stream-completion error affects all already queued results.
    for (auto &geometry : m_geometryPreparations)
      geometry->setPreparationFailed(true);
    for (auto &geometry : m_geometryInputUploads)
      geometry->failInputUploads();
  }
  return error;
}

bool DeviceGlobalState::collectGeometryPreparations()
{
  if (!collectGeometryInputUploads())
    return false;
  if (m_collectingGeometryPreparations)
    return false;
  if (m_geometryPreparations.empty())
    return true;
  m_collectingGeometryPreparations = true;
  auto collecting = nonstd::make_scope_exit(
      [&] { m_collectingGeometryPreparations = false; });
  auto pending = std::move(m_geometryPreparations);
  const auto error = preparation.finish();
  if (error != cudaSuccess) {
    // A stream error invalidates every generation in this batch. Mark them
    // before reporting, so callbacks cannot see a partially valid generation.
    for (auto &geometry : pending)
      geometry->setPreparationFailed(true);
    checkPreparation(pending.front().ptr, error, "batch completion");
  }
  for (auto &geometry : pending)
    geometry->completePreparation();
  pending.clear();
  m_geometryPreparations.swap(pending); // Reuse queue capacity next batch.
  // Failure is local to the affected geometries; unrelated worlds remain
  // usable.
  return true;
}

cudaError_t DeviceGlobalState::DeviceObjectRegistry::upload(
    PreparationStream &preparation, cudaStream_t renderStream)
{
  if (!(samplers.dirty() || geometries.dirty() || materials.dirty()
          || surfaces.dirty() || lights.dirty() || fields.dirty()
          || volumes.dirty()))
    return cudaSuccess;
  const auto error = preparation.upload(renderStream,
      samplers,
      geometries,
      materials,
      surfaces,
      lights,
      fields,
      volumes);
  if (error != cudaSuccess)
    return error;
  samplers.markUploaded();
  geometries.markUploaded();
  materials.markUploaded();
  surfaces.markUploaded();
  lights.markUploaded();
  fields.markUploaded();
  volumes.markUploaded();
  return cudaSuccess;
}

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
