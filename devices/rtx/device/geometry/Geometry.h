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

#include "Preprocessing.h"
#include "RegisteredObject.h"
#include "array/Array1D.h"
#include "gpu/gpu_objects.h"
// std
#include <initializer_list>
#include <optional>
#include <string>
#include <vector>

namespace visrtx {

struct UniformAttributes
{
  helium::IntrusivePtr<Array1D> attribute0Array;
  helium::IntrusivePtr<Array1D> attribute1Array;
  helium::IntrusivePtr<Array1D> attribute2Array;
  helium::IntrusivePtr<Array1D> attribute3Array;
  helium::IntrusivePtr<Array1D> colorArray;

  std::optional<vec4> attribute0;
  std::optional<vec4> attribute1;
  std::optional<vec4> attribute2;
  std::optional<vec4> attribute3;
  std::optional<vec4> color;
};

struct GeometryAttributes
{
  helium::IntrusivePtr<Array1D> attribute0;
  helium::IntrusivePtr<Array1D> attribute1;
  helium::IntrusivePtr<Array1D> attribute2;
  helium::IntrusivePtr<Array1D> attribute3;
  helium::IntrusivePtr<Array1D> color;
};

struct Geometry : public RegisteredObject<GeometryGPUData>
{
  Geometry(DeviceGlobalState *d);

  static Geometry *createInstance(
      std::string_view subtype, DeviceGlobalState *d);

  void commitParameters() override;
  void markFinalized() override;
  // Defers registry publication while preprocessing results are pending.
  void upload() override;
  // Queues the geometry for collection once; returns its queue position.
  size_t deferPreparation();
  void completePreparation();
  void markPreparationCollected() noexcept;
  void retirePreparationSnapshot();
  void submitInputUploads();
  void completeInputUploads();
  void failInputUploads();

  virtual void populateBuildInput(OptixBuildInput &) const = 0;
  virtual int optixGeometryType() const = 0;

  // Geometry Light area sampling. ensureAreaData() submits a per-primitive
  // CDF if needed. Collect device geometry preparations before reading
  // totalArea(), the object-space emitting area, or publishing a registry.
  virtual bool isAreaSamplingSupported() const;
  virtual void ensureAreaData();
  virtual float totalArea() const;

  // True in any of the failed preparation states below.
  bool preparationFailed() const;
  // Fails the current generation; keeps a more specific failed state.
  void markPreparationFailed();

 protected:
  // Declare once in the subtype constructor, before the first commit. List
  // every consumed parameter except geometry-wide attributes; the base adds
  // primitive and vertex arrays. These inputs determine preparation reuse.
  void declarePreparationParameters(
      std::initializer_list<const char *> parameters);

  enum class PreparationStart
  {
    Prepare, // The subtype prepares and publishes the new generation.
    Reused, // Inputs are unchanged; the registry is already republished.
    Blocked, // Failed, an earlier failure awaits retry, or an input
             // collection is in progress. The subtype returns.
  };
  PreparationStart beginPreparation();
  GeometryGPUData gpuData() const override = 0;

  void commitAttributes(const char *prefix, GeometryAttributes &attrs);
  void populateAttributeDataSet(
      const GeometryAttributes &hostAttrs, AttributeDataSet &gpuAttrs) const;

  GeometryAttributes m_primitiveAttributes;
  UniformAttributes m_uniformAttributes;
  helium::IntrusivePtr<Array1D> m_primitiveId;
  Preprocessing m_preprocessing;

 private:
  struct PreparationParameter
  {
    size_t parameterIndex{0};
    helium::AnariAny value;
    helium::TimeStamp updated{0};
    helium::TimeStamp finalized{0};
    helium::TimeStamp modified{0};
    const void *devicePointer{nullptr};

    bool operator==(const PreparationParameter &other) const;
  };
  bool canRetainPreparationInputs() const;
  void stageInputUploads();

  struct InputUpload
  {
    helium::IntrusivePtr<Array> array;
    helium::TimeStamp modified{0};
    bool submitted{false};
  };
  std::vector<InputUpload> m_inputUploads;
  std::vector<PreparationParameter> m_preparedParameters;
  std::vector<std::string> m_preparationParameterNames;

  // Preparation state of the current generation.
  //
  //   Unprepared         Nothing published for the current inputs: initial,
  //                      preparing, or published while !isValid().
  //   Prepared           Published; a commit may retain its inputs.
  //   Reused             Prepared, and the latest finalize reused it, so
  //                      markFinalized() leaves the BLAS change stamp alone.
  //   Failed             The next finalize retries.
  //   InputUploadFailed  An input upload batch failed. A parameter commit, or
  //                      an observer update newer than m_failedInputUpdate,
  //                      retries.
  //   RetirementFailed   Retiring the previous inputs failed. Only a
  //                      parameter commit retries.
  //
  // Transitions. An event leaves every state it does not list unchanged.
  //
  //   Event                             From               To
  //   commitParameters()                InputUploadFailed  Failed
  //                                     RetirementFailed   Failed
  //     then retiring inputs fails      any                RetirementFailed
  //   beginPreparation()                Reused             Prepared
  //     observer update newer than      InputUploadFailed  Failed
  //     m_failedInputUpdate
  //     inputs unchanged, isValid()     Prepared           Reused
  //     full preparation                Unprepared         Unprepared
  //                                     Prepared           Unprepared
  //                                     Failed             Unprepared
  //   upload(), isValid()               Unprepared         Prepared
  //                                     Prepared           Prepared
  //                                     Reused             Reused
  //   upload(), !isValid()              any not failed     Unprepared
  //   markPreparationFailed() (every    any not failed     Failed
  //   CUDA or numeric failure, and
  //   retirePreparationSnapshot())
  //   failInputUploads()                any but            InputUploadFailed
  //                                     RetirementFailed
  //
  // beginPreparation() returns Blocked in InputUploadFailed and
  // RetirementFailed (retryBlocked()). RetirementFailed outranks
  // InputUploadFailed: both need a commit, and the commit clears both. Whether
  // the queued input batch failed is kept in m_inputBatch, so a batch that
  // fails in RetirementFailed still leaves its arrays dirty.
  enum class PreparationState
  {
    Unprepared,
    Prepared,
    Reused,
    Failed,
    InputUploadFailed,
    RetirementFailed,
  };
  PreparationState m_preparationState{PreparationState::Unprepared};
  // lastUpdated() when the input upload batch failed (InputUploadFailed).
  helium::TimeStamp m_failedInputUpdate{0};

  bool retryBlocked() const;

  // Queue membership, independent of the state above.
  enum class InputBatch
  {
    None, // Not queued for input uploads.
    Queued,
    Failed, // Queued; the batch failed, so submit nothing and mark no array
            // uploaded. A commit retries it.
  };
  InputBatch m_inputBatch{InputBatch::None};
  // Position in the device's preparation queue while queued.
  std::optional<size_t> m_preparationPosition;
};

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_SPECIALIZATION(visrtx::Geometry *, ANARI_GEOMETRY);
