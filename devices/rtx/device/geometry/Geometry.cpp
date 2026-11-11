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
    deviceState()->collectGeometryInputUploads();
  // A parameter commit retries every failure.
  if (m_inputBatch == InputBatch::Failed)
    m_inputBatch = InputBatch::Queued;
  if (retryBlocked())
    m_preparationState = PreparationState::Failed;
  if (m_preparationPosition)
    deviceState()->collectGeometryPreparations();
  // Array observers can re-finalize without a parameter commit, but parameter
  // replacement can release the last reference to an aliased device array.
  // Complete earlier consumers before subclasses replace those references.
  const bool retain = canRetainPreparationInputs();
  if (!retain
      && !failPreparationOnError(this,
          deviceState()->renderRetirement.retire(),
          "retiring previous geometry inputs"))
    m_preparationState = PreparationState::RetirementFailed;
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
  if (!retain && m_preparationState != PreparationState::RetirementFailed)
    stageInputUploads();
}

void Geometry::markFinalized()
{
  Object::markFinalized();
  if (m_preparationState != PreparationState::Reused)
    deviceState()->objectUpdates.lastSurfaceBLASChange = helium::newTimeStamp();
}

bool Geometry::isAreaSamplingSupported() const
{
  return false;
}

bool Geometry::PreparationParameter::operator==(
    const PreparationParameter &other) const
{
  return parameterIndex == other.parameterIndex && value == other.value
      && updated == other.updated && finalized == other.finalized
      && modified == other.modified && devicePointer == other.devicePointer;
}

bool Geometry::canRetainPreparationInputs() const
{
  if ((m_preparationState != PreparationState::Prepared
          && m_preparationState != PreparationState::Reused)
      || m_preparationParameterNames.empty())
    return false;

  // Inspect committed values only, before subclasses replace their references.
  // Do not call dataGPU(): it can upload/replace storage. Pointer validation
  // remains in beginPreparation(), after references have been retained safely.
  size_t present = 0;
  for (size_t i = 0; i < m_preparationParameterNames.size(); ++i) {
    const auto value = getParamDirect(m_preparationParameterNames[i]);
    if (!value.valid())
      continue;
    if (present == m_preparedParameters.size())
      return false;
    const auto &previous = m_preparedParameters[present++];
    if (previous.parameterIndex != i || !(previous.value == value))
      return false;
    if (anari::isObject(value.type())) {
      auto *object = value.getObject<helium::BaseObject>();
      if (object) {
        if (previous.updated != object->lastUpdated()
            || previous.finalized != object->lastFinalized())
          return false;
        if (auto *array = dynamic_cast<Array *>(object)) {
          if (previous.modified != array->lastDataModified())
            return false;
        }
      }
    }
  }
  return present == m_preparedParameters.size();
}

size_t Geometry::deferPreparation()
{
  if (!m_preparationPosition)
    m_preparationPosition = deviceState()->enqueueGeometryPreparation(this);
  return *m_preparationPosition;
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
  if (m_preprocessing.status() == Preprocessing::Status::Pending) {
    deferPreparation();
    return;
  }
  RegisteredObject<GeometryGPUData>::upload();
  if (preparationFailed())
    return;
  if (!isValid())
    m_preparationState = PreparationState::Unprepared;
  else if (m_preparationState != PreparationState::Reused)
    m_preparationState = PreparationState::Prepared;
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

void Geometry::stageInputUploads()
{
  m_inputUploads.clear();
  for (const auto &name : m_preparationParameterNames) {
    const auto value = getParamDirect(name);
    if (!anari::isObject(value.type()))
      continue;
    auto *array = dynamic_cast<Array *>(value.getObject<helium::BaseObject>());
    if (array && array->needToUploadData() && array->dataHost())
      m_inputUploads.push_back({array, array->lastDataModified(), false});
  }
  if (m_inputUploads.empty())
    return;
  deviceState()->enqueueGeometryInputUploads(this);
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
    failPreparation(this, error, operation);
    return true;
  };
  for (auto &upload : m_inputUploads) {
    upload.modified = upload.array->lastDataModified();
    const auto error =
        state.stageGeometryArray(upload.array.ptr, upload.modified);
    upload.submitted = error == cudaSuccess;
    if (fail(error, "geometry array upload"))
      return;
  }
}

void Geometry::failInputUploads()
{
  m_inputBatch = InputBatch::Failed;
  if (m_preparationState != PreparationState::RetirementFailed)
    m_preparationState = PreparationState::InputUploadFailed;
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

Geometry::PreparationStart Geometry::beginPreparation()
{
  if (!deviceState()->collectGeometryInputUploads())
    return PreparationStart::Blocked;
  // Observer-only finalization can arrive before a prior submission is
  // collected.
  if (m_preparationPosition)
    deviceState()->collectGeometryPreparations();
  if (m_preparationState == PreparationState::Reused)
    m_preparationState = PreparationState::Prepared;
  // A later observer update is a new generation too; it may retry a failed
  // input batch without requiring a parameter commit on the geometry.
  if (m_preparationState == PreparationState::InputUploadFailed
      && lastUpdated() > m_failedInputUpdate)
    m_preparationState = PreparationState::Failed;
  if (retryBlocked())
    return PreparationStart::Blocked;

  std::vector<PreparationParameter> current;
  if (!m_preparationParameterNames.empty()) {
    current.reserve(m_preparedParameters.capacity());
    // Indices remain valid if current grows; its AnariAny values retain every
    // source array until the batch completes (including on failure).
    std::vector<std::pair<size_t, Array *>> pendingUploads;
    size_t parameterIndex = 0;
    auto append = [&](const std::string &name) {
      PreparationParameter parameter;
      parameter.parameterIndex = parameterIndex++;
      parameter.value = getParamDirect(name); // committed snapshot, not staging
      if (!parameter.value.valid())
        return; // Keep only present values, retaining their parameter identity.
      if (anari::isObject(parameter.value.type())) {
        auto *object = parameter.value.getObject<helium::BaseObject>();
        if (object) {
          parameter.updated = object->lastUpdated();
          parameter.finalized = object->lastFinalized();
          if (auto *array = dynamic_cast<Array *>(object)) {
            parameter.modified = array->lastDataModified();
            // Privatization can change backing pointers without changing values
            // or the array timestamp. Never reuse cached aliases in that case.
            if (array->needToUploadData() && array->dataHost())
              pendingUploads.emplace_back(current.size(), array);
            else
              parameter.devicePointer = array->dataGPU();
          }
        }
      }
      current.push_back(std::move(parameter));
    };
    for (const auto &name : m_preparationParameterNames)
      append(name);

    if (!pendingUploads.empty()) {
      auto &state = *deviceState();
      if (!failPreparationOnError(this,
              state.preparation.waitFor(state.stream),
              "previous array consumers"))
        return PreparationStart::Blocked;
      for (const auto &[index, array] : pendingUploads) {
        if (!failPreparationOnError(this,
                array->enqueueUpload(state.preparation.stream()),
                "geometry array upload"))
          return PreparationStart::Blocked;
      }
      if (!failPreparationOnError(this,
              state.finishGeometryPreparation(),
              "geometry array upload completion"))
        return PreparationStart::Blocked;
      // Never mark a partial/failed batch uploaded: a later commit must retry.
      for (const auto &[index, array] : pendingUploads) {
        array->markDataUploaded();
        current[index].devicePointer = array->dataGPU();
      }
    }

    // Observer-only updates always take the preparation path. A geometry-wide
    // attribute edit coalesced with an array change cannot hide that change;
    // the versions/pointers above still invalidate preparation reuse.
    if (m_preparationState == PreparationState::Prepared && isValid()
        && lastCommitted() > lastFinalized()
        && current == m_preparedParameters) {
      m_preparationState = PreparationState::Reused;
      upload();
      return PreparationStart::Reused;
    }
  }
  m_preparedParameters = std::move(current);
  m_preparationState = PreparationState::Unprepared;
  // Full preparation can overwrite derived buffers read by earlier frames.
  // Keep that dependency on the GPU; parameter-reference retirement above
  // still requires a host wait before releasing old application storage.
  return failPreparationOnError(this,
             deviceState()->preparation.waitFor(deviceState()->stream),
             "previous geometry consumers")
      ? PreparationStart::Prepare
      : PreparationStart::Blocked;
}

bool Geometry::preparationFailed() const
{
  return m_preparationState == PreparationState::Failed
      || m_preparationState == PreparationState::InputUploadFailed
      || m_preparationState == PreparationState::RetirementFailed;
}

bool Geometry::retryBlocked() const
{
  return m_preparationState == PreparationState::InputUploadFailed
      || m_preparationState == PreparationState::RetirementFailed;
}

void Geometry::markPreparationFailed()
{
  if (!preparationFailed())
    m_preparationState = PreparationState::Failed;
}

void Geometry::ensureAreaData() {}

float Geometry::totalArea() const
{
  return 0.f;
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
