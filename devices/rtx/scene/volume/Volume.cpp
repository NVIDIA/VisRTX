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

#include "Volume.h"
// specific types
#include "TransferFunction1D.h"
#include "UnknownVolume.h"

namespace visrtx {

Volume::Volume(DeviceGlobalState *d)
    : RegisteredObject<VolumeGPUData>(ANARI_VOLUME, d)
{
  setRegistry(d->registry.volumes);
}

void Volume::commitParameters()
{
  m_id = getParam<uint32_t>("id", ~0u);
}

void Volume::markFinalized()
{
  Object::markFinalized();
  deviceState()->objectUpdates.lastBLASChange = helium::newTimeStamp();
}

OptixBuildInput Volume::buildInput() const
{
  OptixBuildInput buildInput{};

  auto gd = gpuData();
  m_aabbsBuffer.upload(&gd.bounds);
  m_aabbsBufferPtr = (CUdeviceptr)m_aabbsBuffer.ptr();

  buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;

  buildInput.customPrimitiveArray.aabbBuffers = &m_aabbsBufferPtr;
  buildInput.customPrimitiveArray.numPrimitives = 1;

  static uint32_t buildInputFlags[1] = {OPTIX_GEOMETRY_FLAG_NONE};

  buildInput.customPrimitiveArray.flags = buildInputFlags;
  buildInput.customPrimitiveArray.numSbtRecords = 1;

  return buildInput;
}

Volume *Volume::createInstance(std::string_view subtype, DeviceGlobalState *d)
{
  if (subtype == "transferFunction1D" || subtype == "scivis")
    return new TransferFunction1D(d);
  else
    return new UnknownVolume(subtype, d);
}

VolumeGPUData Volume::gpuData() const
{
  VolumeGPUData retval{};
  retval.id = m_id;
  return retval;
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_DEFINITION(visrtx::Volume *);
