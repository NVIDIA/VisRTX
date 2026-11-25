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
#include <utility>
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
  // Final so that no subtype can skip the preparation protocol; subtypes
  // implement prepare() instead.
  void finalize() final;
  void markFinalized() override;
  // Defers registry publication while preprocessing results are pending.
  void upload() override;

  virtual void populateBuildInput(OptixBuildInput &) const = 0;
  virtual int optixGeometryType() const = 0;

  // Geometry Light area sampling, for the subtypes that implement areaInput().
  // ensureAreaData() submits the per-primitive area CDF if it is not current,
  // and republishes the geometry. Lazy: only Emissive Surfaces call it, so
  // ordinary geometry never pays for it. The request persists, so every later
  // finalize rebuilds the CDF whatever the order of surface and geometry
  // commits. Collect device geometry preparations before reading totalArea(),
  // the object-space emitting area, or publishing a registry.
  bool isAreaSamplingSupported() const;
  void ensureAreaData();
  float totalArea() const;

  // True from a failure until a finalize prepares the geometry again.
  bool preparationFailed() const;

 protected:
  // Declare once in the subtype constructor, before the first commit. List
  // every consumed parameter except geometry-wide attributes; the base adds
  // primitive and vertex arrays. These inputs determine preparation reuse.
  void declarePreparationParameters(
      std::initializer_list<const char *> parameters);

  // Builds the subtype's device data for a new generation; finalize() calls
  // it only when one is needed, and uploads when it returns true. Return false
  // for every reason not to upload: a missing required parameter, or a step
  // that failed the geometry. finalize() does not ask preparationFailed()
  // afterwards, so a failed step's result must not be discarded. Never call
  // upload() here.
  virtual bool prepare() = 0;
  GeometryGPUData gpuData() const override = 0;

  // The input of the per-primitive area CDF, or nothing when the subtype
  // cannot be sampled as a Geometry Light. Asked in any state: return an empty
  // input while !isValid().
  virtual std::optional<PrimitiveInput> areaInput() const;
  // Fills the Geometry Light fields of the subtype's GPU record: the CDF is
  // null and the other two zero until ensureAreaData() ran and was collected.
  // `count` is the subtype's primitive count, the length of the CDF.
  template <typename RECORD_T>
  void populateAreaData(RECORD_T &record, size_t count) const;

  void commitAttributes(const char *prefix, GeometryAttributes &attrs);
  void populateAttributeDataSet(
      const GeometryAttributes &hostAttrs, AttributeDataSet &gpuAttrs) const;

  GeometryAttributes m_primitiveAttributes;
  UniformAttributes m_uniformAttributes;
  helium::IntrusivePtr<Array1D> m_primitiveId;
  Preprocessing m_preprocessing;

 private:
  // The device's GeometryPreparation drives the steps below; see its header
  // for the order.
  friend struct GeometryPreparation;
  void completePreparation();
  void markPreparationCollected() noexcept;
  void retirePreparationSnapshot();
  void submitInputUploads();
  void completeInputUploads() noexcept;
  void failInputUploads();
  // What retries a failed preparation, weakest first. Until it arrives,
  // beginPreparation() returns Blocked for the two stronger ones.
  enum class Retry
  {
    Finalize, // The next finalize: every CUDA or numeric failure, a retired
              // snapshot, and an input batch that a finalize queued itself.
    Update, // A parameter commit, or an observer update newer than
            // m_failedInputUpdate: an input batch queued by a commit failed.
    Commit, // Only a parameter commit: retiring the previous inputs failed.
  };
  // Fails the current generation. Never weakens the retry already recorded.
  void markPreparationFailed(Retry retry = Retry::Finalize);

  // After prepare(): resubmits the area CDF if a surface ever asked for it,
  // else drops it. False means the geometry failed.
  bool refreshAreaData();

  enum class PreparationStart
  {
    Prepare, // finalize() runs prepare() and uploads the new generation.
    Reused, // Inputs are unchanged; the registry is already republished.
    Blocked, // Failed, an earlier failure awaits retry, or an input
             // collection is in progress. Nothing is prepared or uploaded.
  };
  PreparationStart beginPreparation();

  struct PreparationParameter
  {
    size_t parameterIndex{0};
    helium::AnariAny value;
    helium::TimeStamp updated{0};
    helium::TimeStamp finalized{0};
    helium::TimeStamp modified{0};
    const void *devicePointer{nullptr};

    // Equal in everything a commit can read. The device pointer is not: it is
    // known only once the uploads finished.
    bool sameInput(const PreparationParameter &other) const;
    bool operator==(const PreparationParameter &other) const;
  };
  // The one walk over the declared parameters. Calls fcn(PreparationParameter
  // &&, Array *) for each one that is set, with the committed value, its
  // versions and no device pointer; the array is null unless the value is one.
  // Never calls dataGPU(), which can upload or replace storage. Stops, and
  // returns false, when fcn does.
  template <typename FCN_T>
  bool forEachPreparationParameter(FCN_T &&fcn) const;
  bool canRetainPreparationInputs() const;
  // Queues every declared array with host data newer than its device copy as
  // this geometry's input batch.
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
  //   Unprepared  Nothing published for the current inputs: initial,
  //               preparing, or published while !isValid().
  //   Prepared    Published; a commit may retain its inputs.
  //   Failed      Nothing is prepared or published until m_retry arrives.
  //
  //   Event                                    To
  //   beginPreparation(), full preparation     Unprepared
  //   upload(), unless Failed                  Prepared if isValid(), else
  //                                            Unprepared
  //   markPreparationFailed(retry)             Failed
  //
  // When its retry arrives a failure only drops to Retry::Finalize; the full
  // preparation of that or the next finalize is what leaves Failed.
  enum class PreparationState
  {
    Unprepared,
    Prepared,
    Failed,
  };
  PreparationState m_preparationState{PreparationState::Unprepared};
  // Meaningful while Failed.
  Retry m_retry{Retry::Finalize};
  // lastUpdated() when the latest input upload batch failed (Retry::Update).
  helium::TimeStamp m_failedInputUpdate{0};
  // The latest finalize reused the preparation, so markFinalized() leaves the
  // BLAS change stamp alone. Only ever set while Prepared.
  bool m_preparationReused{false};

  bool retryBlocked() const;

  // Queue membership, independent of the state above.
  enum class InputBatch
  {
    None, // Not queued for input uploads.
    Queued,
    Failed, // Queued; the batch failed, so submit nothing and mark no array
            // uploaded. A commit retries it. Kept apart from Retry::Update
            // because a batch can fail under Retry::Commit too.
  };
  InputBatch m_inputBatch{InputBatch::None};
  // Position in the device's preparation queue while queued.
  std::optional<size_t> m_preparationPosition;

  // Geometry Light area data. Valid: the submitted CDF matches the current
  // generation. Wanted: set by the first ensureAreaData() and kept across
  // recommits.
  bool m_areaDataValid{false};
  bool m_areaDataWanted{false};
};

// Inlined definitions ////////////////////////////////////////////////////////

template <typename FCN_T>
inline bool Geometry::forEachPreparationParameter(FCN_T &&fcn) const
{
  for (size_t i = 0; i < m_preparationParameterNames.size(); ++i) {
    // The committed value, not staging.
    PreparationParameter parameter{
        i, getParamDirect(m_preparationParameterNames[i])};
    if (!parameter.value.valid())
      continue; // Keep only present values, retaining their parameter identity.
    Array *array = nullptr;
    if (anari::isObject(parameter.value.type())) {
      if (auto *object = parameter.value.getObject<helium::BaseObject>()) {
        parameter.updated = object->lastUpdated();
        parameter.finalized = object->lastFinalized();
        array = dynamic_cast<Array *>(object);
        if (array)
          parameter.modified = array->lastDataModified();
      }
    }
    if (!fcn(std::move(parameter), array))
      return false;
  }
  return true;
}

template <typename RECORD_T>
inline void Geometry::populateAreaData(RECORD_T &record, size_t count) const
{
  record.primAreaCdf = m_preprocessing.areaCdf();
  record.numPrimitives = record.primAreaCdf ? uint32_t(count) : 0;
  record.totalArea = m_preprocessing.totalArea();
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_SPECIALIZATION(visrtx::Geometry *, ANARI_GEOMETRY);
