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

#include "Geometry.h"
// specific types
#include "Cone.h"
#include "Curve.h"
#include "Cylinder.h"
#include "Isosurface.h"
#include "Quad.h"
#include "Sphere.h"
#include "Triangle.h"

#ifdef VISRTX_USE_NEURAL
#include "Neural.h"
#endif
#include "SDF.h"
#include "UnknownGeometry.h"
// std
#include <cstring>
#include <string_view>
#include <utility>
// helium
#include <helium/helium_math.h>

#include "utility/AnariTypeHelpers.h"

namespace visrtx {

// Helper functions ///////////////////////////////////////////////////////////

static void populateAttributeData(
    helium::IntrusivePtr<Array1D> array, AttributeData &attr)
{
  attr.type = ANARI_UNKNOWN;
  attr.numChannels = 0;
  attr.data = nullptr;

  if (!array)
    return;

  auto type = array->elementType();

  if (!isColor(type))
    return;

  attr.type = type;
  attr.numChannels = numANARIChannels(attr.type);
  attr.data = array->dataGPU();
}

// Geometry definitions ///////////////////////////////////////////////////////

Geometry::Geometry(DeviceGlobalState *s)
    : RegisteredObject<GeometryGPUData>(ANARI_GEOMETRY, s)
{
  setRegistry(s->registry.geometries);
}

Geometry *Geometry::createInstance(
    std::string_view subtype, DeviceGlobalState *d)
{
  if (subtype == "triangle")
    return new Triangle(d);
  else if (subtype == "quad")
    return new Quad(d);
  else if (subtype == "sphere")
    return new Sphere(d);
  else if (subtype == "cylinder")
    return new Cylinder(d);
  else if (subtype == "cone")
    return new Cone(d);
  else if (subtype == "curve")
    return new Curve(d);
  else if (subtype == "sdf")
    return new SDF(d);
  else if (subtype == "isosurface")
    return new Isosurface(d);
#ifdef VISRTX_USE_NEURAL
  else if (subtype == "neural")
    return new Neural(d);
#endif
  else
    return new UnknownGeometry(subtype, d);
}

void Geometry::commitParameters()
{
  if (m_inputBatch != InputBatch::None)
    deviceState()->preparation.uploadInputs();
  // A parameter commit retries every failure.
  if (m_inputBatch == InputBatch::Failed)
    m_inputBatch = InputBatch::Queued;
  if (preparationFailed())
    m_retry = Retry::Finalize;
  if (m_preparationPosition)
    deviceState()->preparation.collect();
  // Array observers can re-finalize without a parameter commit, but parameter
  // replacement can release the last reference to an aliased device array.
  // Complete earlier consumers before subclasses replace those references.
  const bool retain = canRetainPreparationInputs();
  const bool retired = retain
      || deviceState()->preparation.check(this,
          deviceState()->preparation.retireRender(),
          "retiring previous geometry inputs");
  if (!retired)
    markPreparationFailed(Retry::Commit);
  commitAttributes("primitive.", m_primitiveAttributes);

  auto getUniformAttribute =
      [&](const std::string &pName) -> std::optional<vec4> {
    vec4 v(0.f, 0.f, 0.f, 1.f);
    if (getParam(pName, ANARI_FLOAT32_VEC4, &v))
      return v;
    else
      return {};
  };

  m_uniformAttributes.attribute0 = getUniformAttribute("attribute0");
  m_uniformAttributes.attribute1 = getUniformAttribute("attribute1");
  m_uniformAttributes.attribute2 = getUniformAttribute("attribute2");
  m_uniformAttributes.attribute3 = getUniformAttribute("attribute3");
  m_uniformAttributes.color = getUniformAttribute("color");
  m_primitiveId = getParamObject<Array1D>("primitive.id");
  if (!retain && retired)
    stageInputUploads();
}

void Geometry::markFinalized()
{
  Object::markFinalized();
  if (!m_preparationReused)
    deviceState()->objectUpdates.lastSurfaceBLASChange = helium::newTimeStamp();
}

bool Geometry::PreparationParameter::sameInput(
    const PreparationParameter &other) const
{
  return parameterIndex == other.parameterIndex && value == other.value
      && updated == other.updated && finalized == other.finalized
      && modified == other.modified;
}

bool Geometry::PreparationParameter::operator==(
    const PreparationParameter &other) const
{
  return sameInput(other) && devicePointer == other.devicePointer;
}

bool Geometry::canRetainPreparationInputs() const
{
  if (m_preparationState != PreparationState::Prepared
      || m_preparationParameterNames.empty())
    return false;

  // Inspect committed values only, before subclasses replace their references.
  // Do not call dataGPU(): it can upload/replace storage. Pointer validation
  // remains in beginPreparation(), after references have been retained safely.
  size_t present = 0;
  const bool same = forEachPreparationParameter(
      [&](PreparationParameter &&parameter, Array *) {
        return present < m_preparedParameters.size()
            && m_preparedParameters[present++].sameInput(parameter);
      });
  return same && present == m_preparedParameters.size();
}

void Geometry::completePreparation()
{
  // complete() fails the geometry whenever it returns false.
  if (m_preprocessing.complete(this))
    upload();
}

void Geometry::markPreparationCollected() noexcept
{
  m_preparationPosition.reset();
}

void Geometry::upload()
{
  if (m_preprocessing.pending()) {
    deviceState()->preparation.defer(this);
    return;
  }
  RegisteredObject<GeometryGPUData>::upload();
  if (preparationFailed())
    return;
  if (isValid())
    m_preparationState = PreparationState::Prepared;
  else {
    m_preparationState = PreparationState::Unprepared;
    m_preparationReused = false;
  }
}

void Geometry::declarePreparationParameters(
    std::initializer_list<const char *> parameters)
{
  for (const char *name : {"primitive.attribute0",
           "primitive.attribute1",
           "primitive.attribute2",
           "primitive.attribute3",
           "primitive.color",
           "primitive.id",
           "vertex.attribute0",
           "vertex.attribute1",
           "vertex.attribute2",
           "vertex.attribute3",
           "vertex.color"})
    m_preparationParameterNames.emplace_back(name);
  for (const char *name : parameters)
    m_preparationParameterNames.emplace_back(name);
}

static bool needsInputUpload(const Array *array)
{
  return array && array->needToUploadData() && array->dataHost();
}

void Geometry::stageInputUploads()
{
  m_inputUploads.clear();
  forEachPreparationParameter(
      [&](PreparationParameter &&parameter, Array *array) {
        if (needsInputUpload(array))
          m_inputUploads.push_back({array, parameter.modified, false});
        return true;
      });
  if (m_inputUploads.empty())
    return;
  deviceState()->preparation.stageInputs(this);
  m_inputBatch = InputBatch::Queued;
}

void Geometry::retirePreparationSnapshot()
{
  // Only queued by a commit whose render-consumer retirement check succeeded.
  // Discard obsolete references before allocating a multi-geometry cohort.
  // Until beginPreparation establishes the new generation, old GPU records
  // must not be usable if a host exception aborts finalization.
  markPreparationFailed();
  m_preparedParameters.clear();
}

void Geometry::submitInputUploads()
{
  if (m_inputBatch == InputBatch::Failed)
    return;
  auto &state = *deviceState();
  auto fail = [&](cudaError_t error, const char *operation) {
    if (error == cudaSuccess)
      return false;
    failInputUploads();
    state.preparation.fail(this, error, operation);
    return true;
  };
  for (auto &upload : m_inputUploads) {
    upload.modified = upload.array->lastDataModified();
    const auto error =
        state.preparation.stageArray(upload.array.ptr, upload.modified);
    upload.submitted = error == cudaSuccess;
    if (fail(error, "geometry array upload"))
      return;
  }
}

void Geometry::failInputUploads()
{
  m_inputBatch = InputBatch::Failed;
  markPreparationFailed(Retry::Update);
}

void Geometry::completeInputUploads() noexcept
{
  for (const auto &upload : m_inputUploads)
    if (!upload.submitted)
      failInputUploads();
  const bool failed = m_inputBatch == InputBatch::Failed;
  m_inputBatch = InputBatch::None;
  if (failed)
    m_failedInputUpdate = lastUpdated();
  else {
    for (const auto &upload : m_inputUploads) {
      // Array finalization or a coalesced unmap may have changed the version
      // since submission. Leave that array dirty for the conservative path.
      if (upload.array->lastDataModified() == upload.modified)
        upload.array->markDataUploaded();
    }
  }
  m_inputUploads.clear();
}

void Geometry::finalize()
{
  if (beginPreparation() != PreparationStart::Prepare)
    return;
  if (prepare() && refreshAreaData())
    upload();
}

Geometry::PreparationStart Geometry::beginPreparation()
{
  if (!deviceState()->preparation.uploadInputs())
    return PreparationStart::Blocked;
  // Observer-only finalization can arrive before a prior submission is
  // collected.
  if (m_preparationPosition)
    deviceState()->preparation.collect();
  m_preparationReused = false;
  // A later observer update is a new generation too; it may retry a failed
  // input batch without requiring a parameter commit on the geometry.
  if (preparationFailed() && m_retry == Retry::Update
      && lastUpdated() > m_failedInputUpdate)
    m_retry = Retry::Finalize;
  if (retryBlocked())
    return PreparationStart::Blocked;

  std::vector<PreparationParameter> current;
  if (!m_preparationParameterNames.empty()) {
    current.reserve(m_preparedParameters.capacity());
    // Indices remain valid if current grows; its AnariAny values retain every
    // source array until the batch completes (including on failure).
    std::vector<std::pair<size_t, Array *>> pendingUploads;
    forEachPreparationParameter(
        [&](PreparationParameter &&parameter, Array *array) {
          // Privatization can change backing pointers without changing values
          // or the array timestamp. Never reuse cached aliases in that case.
          if (needsInputUpload(array))
            pendingUploads.emplace_back(current.size(), array);
          else if (array)
            parameter.devicePointer = array->dataGPU();
          current.push_back(std::move(parameter));
          return true;
        });

    if (!pendingUploads.empty()) {
      // No commit captured these: an observer-only update, or an array
      // modified after its batch was submitted. Upload them the same way, as
      // a batch of this geometry alone. Its failure is this finalize's, so
      // the next one retries.
      stageInputUploads();
      deviceState()->preparation.uploadInputs();
      if (retryBlocked()) {
        m_retry = Retry::Finalize;
        return PreparationStart::Blocked;
      }
      for (const auto &[index, array] : pendingUploads)
        current[index].devicePointer = array->dataGPU();
    }

    // Observer-only updates always take the preparation path. A geometry-wide
    // attribute edit coalesced with an array change cannot hide that change;
    // the versions/pointers above still invalidate preparation reuse.
    if (m_preparationState == PreparationState::Prepared && isValid()
        && lastCommitted() > lastFinalized()
        && current == m_preparedParameters) {
      m_preparationReused = true;
      upload();
      return PreparationStart::Reused;
    }
  }
  m_preparedParameters = std::move(current);
  m_preparationState = PreparationState::Unprepared;
  // Full preparation can overwrite derived buffers read by earlier frames.
  // Keep that dependency on the GPU; parameter-reference retirement above
  // still requires a host wait before releasing old application storage.
  return deviceState()->preparation.check(this,
             deviceState()->preparationStream.orderAfterRender(),
             "previous geometry consumers")
      ? PreparationStart::Prepare
      : PreparationStart::Blocked;
}

bool Geometry::preparationFailed() const
{
  return m_preparationState == PreparationState::Failed;
}

bool Geometry::retryBlocked() const
{
  return preparationFailed() && m_retry != Retry::Finalize;
}

void Geometry::markPreparationFailed(Retry retry)
{
  if (!preparationFailed() || retry > m_retry)
    m_retry = retry;
  m_preparationState = PreparationState::Failed;
  m_preparationReused = false;
}

std::optional<PrimitiveInput> Geometry::areaInput() const
{
  return {};
}

bool Geometry::isAreaSamplingSupported() const
{
  return areaInput().has_value();
}

void Geometry::ensureAreaData()
{
  if (m_areaDataValid)
    return;
  const auto input = areaInput();
  if (!input)
    return;
  m_areaDataWanted = true;
  m_areaDataValid = m_preprocessing.areas(this, *input);
  if (m_areaDataValid)
    upload(); // republish only successfully prepared sampling data
}

bool Geometry::refreshAreaData()
{
  // prepare() may have changed what the CDF was computed from.
  m_areaDataValid = false;
  if (!m_areaDataWanted) {
    m_preprocessing.releaseAreas();
    return true;
  }
  // Wanted implies that the subtype has an area input.
  m_areaDataValid = m_preprocessing.areas(this, *areaInput());
  return m_areaDataValid;
}

float Geometry::totalArea() const
{
  return m_preprocessing.totalArea();
}

GeometryGPUData Geometry::gpuData() const
{
  GeometryGPUData retval{};

  const vec4 defaultAttr(0.f, 0.f, 0.f, 1.f);
  retval.attrUniform[0] = m_uniformAttributes.attribute0.value_or(defaultAttr);
  retval.attrUniform[1] = m_uniformAttributes.attribute1.value_or(defaultAttr);
  retval.attrUniform[2] = m_uniformAttributes.attribute2.value_or(defaultAttr);
  retval.attrUniform[3] = m_uniformAttributes.attribute3.value_or(defaultAttr);
  retval.attrUniform[4] = m_uniformAttributes.color.value_or(defaultAttr);
  retval.attrUniformPresent[0] = m_uniformAttributes.attribute0.has_value();
  retval.attrUniformPresent[1] = m_uniformAttributes.attribute1.has_value();
  retval.attrUniformPresent[2] = m_uniformAttributes.attribute2.has_value();
  retval.attrUniformPresent[3] = m_uniformAttributes.attribute3.has_value();
  retval.attrUniformPresent[4] = m_uniformAttributes.color.has_value();
  populateAttributeDataSet(m_primitiveAttributes, retval.attr);
  retval.primitiveId =
      (const uint32_t *)(m_primitiveId ? m_primitiveId->dataGPU() : nullptr);
  retval.epsilonScale = m_preprocessing.epsilonScale();

  return retval;
}

void Geometry::commitAttributes(const char *_prefix, GeometryAttributes &attrs)
{
  std::string prefix = _prefix;
  attrs.attribute0 = getParamObject<Array1D>(prefix + "attribute0");
  attrs.attribute1 = getParamObject<Array1D>(prefix + "attribute1");
  attrs.attribute2 = getParamObject<Array1D>(prefix + "attribute2");
  attrs.attribute3 = getParamObject<Array1D>(prefix + "attribute3");
  attrs.color = getParamObject<Array1D>(prefix + "color");
}

void Geometry::populateAttributeDataSet(
    const GeometryAttributes &hostAttrs, AttributeDataSet &gpuAttrs) const
{
  populateAttributeData(hostAttrs.attribute0, gpuAttrs[0]);
  populateAttributeData(hostAttrs.attribute1, gpuAttrs[1]);
  populateAttributeData(hostAttrs.attribute2, gpuAttrs[2]);
  populateAttributeData(hostAttrs.attribute3, gpuAttrs[3]);
  populateAttributeData(hostAttrs.color, gpuAttrs[4]);
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_DEFINITION(visrtx::Geometry *);
