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

#pragma once

#include <libmdl/ArgumentBlockDescriptor.h>
#include <libmdl/Core.h>

#include <anari/anari_cpp.hpp>

#include <mi/neuraylib/itransaction.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace visrtx {
class DeviceGlobalState;
class Sampler;
} // namespace visrtx

namespace visrtx::mdl {

class MdlCompileCoordinator;

class SamplerRegistry
{
 public:
  SamplerRegistry(libmdl::Core *core, DeviceGlobalState *deviceState);
  ~SamplerRegistry();

  Sampler *acquireSampler(
      const std::string &filePath, libmdl::ColorSpace colorSpace);
  Sampler *acquireSampler(const libmdl::TextureDescriptor &textureDesc);

  // Decode the material's texture files across the compile pool, then wait, so a
  // material referencing many texture files decodes them in parallel. Called on
  // the commit thread; it skips descriptors already in the sampler cache and
  // de-duplicates repeated keys, so each unique uncached texture is decoded
  // exactly once. acquireSampler then consumes the staged pixels (falling back
  // to an inline decode for anything not staged -- BSDF-data, .dds, or a decode
  // that failed). Pure stb decode runs on the workers; the sampler cache is only
  // read here on the commit thread, which is blocked in the wait meanwhile.
  void stageDecodeBatch(MdlCompileCoordinator &coordinator,
      const std::vector<libmdl::TextureDescriptor> &descriptors);

  bool releaseSampler(const Sampler *);

 private:
  libmdl::Core *m_core = {};
  DeviceGlobalState *m_deviceState = {};

  struct CacheEntry
  {
    Sampler *sampler{nullptr};
    // The registry's own outstanding acquires. The entry is erased when THIS
    // hits zero — never inferred from the object's refcount: other holders
    // (e.g. the deferred commit buffer) can outlive the final release, and a
    // count-snapshot heuristic leaves a dangling cache pointer once they drop
    // theirs — a use-after-free on the next same-key acquire (heap corruption
    // that surfaced as an optixPipelineDestroy crash under material churn).
    int acquires{0};
  };
  // Guards m_dbToSampler. acquireSampler runs on the (flush-serialized) commit
  // thread, but releaseSampler runs whenever a material's refcount hits zero --
  // which, under the device's khr_device_synchronization, is any app thread --
  // so cache mutation must be locked. Held only around the map operations, never
  // the decode/upload.
  std::mutex m_cacheMutex;
  std::unordered_map<std::string, CacheEntry> m_dbToSampler;

  // Frees an stb_image allocation; defined where stb is included.
  struct StbDeleter
  {
    void operator()(void *p) const;
  };
  // Host pixels decoded off the commit thread, awaiting sampler creation.
  struct StagedImage
  {
    std::unique_ptr<void, StbDeleter> data;
    int width{};
    int height{};
    int channels{};
    bool isHdr{};
  };
  // Decoded-but-not-yet-created textures, keyed like m_dbToSampler. Filled by
  // decodeToStaging() on pool workers, drained by acquireSampler() on the commit
  // thread; its own lock keeps it independent of the (commit-thread-only)
  // sampler cache.
  std::mutex m_stagingMutex;
  std::unordered_map<std::string, StagedImage> m_staging;

  Sampler *loadFromFile(
      const std::string_view &filePath, libmdl::ColorSpace colorSpace);

  Sampler *loadFromDDS(
      const std::string_view &filePath, libmdl::ColorSpace colorSpace);
  Sampler *loadFromImage(
      const std::string_view &filePath, libmdl::ColorSpace colorSpace);
  Sampler *loadFromTextureDesc(const libmdl::TextureDescriptor &textureDesc);

  // stb decode split out of loadFromImage so it can run off the commit thread;
  // createFromStb builds the sampler from already-decoded pixels (commit thread,
  // device work). loadFromImage is decode + create.
  StagedImage decodeStb(const std::string_view &filePath);
  Sampler *createFromStb(StagedImage image, libmdl::ColorSpace colorSpace);
  StagedImage takeStaged(const std::string &key);
  // One texture's worker-side decode into the staging map (pure stb).
  void decodeToStaging(const libmdl::TextureDescriptor &textureDesc);
};

} // namespace visrtx::mdl
