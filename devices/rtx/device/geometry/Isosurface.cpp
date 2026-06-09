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

#include "Isosurface.h"
#include "IsosurfaceBricks.h"

namespace visrtx {

// hitKind is a 7-bit OptiX field; the matched isovalue index must fit.
static constexpr uint32_t MAX_ISOVALUES = 128;
static_assert(MAX_ISOVALUES <= 128,
    "isovalue index is packed into OptiX's 7-bit hitKind; must fit in [0,127]");

Isosurface::Isosurface(DeviceGlobalState *d)
    : Geometry(d), m_field(this), m_isovalueArray(this)
{}

Isosurface::~Isosurface() = default;

void Isosurface::commitParameters()
{
  Geometry::commitParameters();
  m_field = getParamObject<SpatialField>("field");
  m_isovalueArray = getParamObject<Array1D>("isovalue");
  m_hasScalarIsovalue =
      getParam("isovalue", ANARI_FLOAT32, &m_isovalueScalar);
}

void Isosurface::finalize()
{
  m_numBricks = 0;
  m_isovaluesDev = nullptr;
  m_numIsovalues = 0;

  if (!m_field) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'field' on isosurface geometry");
    return;
  }
  if (m_field->m_uniformGrid.m_valueRanges == nullptr) {
    // The field's space-skipping grid isn't built yet; a field committed before
    // this geometry re-triggers finalize via the field ChangeObserverPtr once
    // its grid is ready. (Custom fields build the grid in their finalize too.)
    reportMessage(ANARI_SEVERITY_WARNING,
        "isosurface 'field' space-skipping grid not ready; nothing will render "
        "yet");
    return;
  }
  if (!m_isovalueArray && !m_hasScalarIsovalue) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'isovalue' on isosurface geometry");
    return;
  }
  if (m_isovalueArray && m_isovalueArray->size() == 0) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "empty 'isovalue' array on isosurface geometry; one or more values are "
        "required, nothing will render");
    return;
  }
  if (m_isovalueArray && m_isovalueArray->elementType() != ANARI_FLOAT32) {
    // beginAs<float>() only asserts the element type, so under NDEBUG a non-
    // FLOAT32 array would be silently reinterpreted as float garbage. Reject it.
    reportMessage(ANARI_SEVERITY_WARNING,
        "isosurface 'isovalue' array must be ANARI_FLOAT32; nothing will render");
    return;
  }

  // Resolve the device isovalue array (array form preferred).
  if (m_isovalueArray) {
    m_numIsovalues = uint32_t(m_isovalueArray->size());
    if (m_numIsovalues > MAX_ISOVALUES) {
      reportMessage(ANARI_SEVERITY_WARNING,
          "isosurface 'isovalue' array has %u entries; clamping to %u",
          m_numIsovalues,
          MAX_ISOVALUES);
      m_numIsovalues = MAX_ISOVALUES;
    }
    m_isovaluesDev = m_isovalueArray->beginAs<float>(AddressSpace::GPU);
  } else {
    m_numIsovalues = 1;
    m_scalarIsovalueBuffer.reserve(sizeof(float));
    m_scalarIsovalueBuffer.upload(&m_isovalueScalar, 1);
    m_isovaluesDev = (const float *)m_scalarIsovalueBuffer.ptr();
  }

  dropUndersizedPrimitiveArrays();

  // Coarse active-region BLAS: a small set of bricks tightly bounding the
  // active macrocells, so the BVH culls rays missing the surface while the
  // in-program DDA still does fine per-cell skipping inside each brick.
  auto &state = *deviceState();
  m_numBricks = buildIsosurfaceBricks(state.stream,
      m_field->m_uniformGrid.gpuData(),
      m_isovaluesDev,
      m_numIsovalues,
      m_aabbs);
  m_aabbsBufferPtr = (CUdeviceptr)m_aabbs.ptr();

  reportMessage(ANARI_SEVERITY_DEBUG,
      "finalizing isosurface geometry: %u isovalue(s), %zu active bricks",
      m_numIsovalues,
      m_numBricks);

  upload();
}

void Isosurface::dropUndersizedPrimitiveArrays()
{
  auto dropIfShort = [&](helium::IntrusivePtr<Array1D> &array,
                         const char *name) {
    if (array && array->size() < m_numIsovalues) {
      reportMessage(ANARI_SEVERITY_WARNING,
          "isosurface '%s' array has %zu element(s) but the geometry has %u "
          "isovalue(s); ignoring it",
          name,
          array->size(),
          m_numIsovalues);
      array = {};
    }
  };

  dropIfShort(m_primitiveAttributes.attribute0, "primitive.attribute0");
  dropIfShort(m_primitiveAttributes.attribute1, "primitive.attribute1");
  dropIfShort(m_primitiveAttributes.attribute2, "primitive.attribute2");
  dropIfShort(m_primitiveAttributes.attribute3, "primitive.attribute3");
  dropIfShort(m_primitiveAttributes.color, "primitive.color");
  dropIfShort(m_primitiveId, "primitive.id");
}

bool Isosurface::isValid() const
{
  return m_field && m_field->isValid()
      && m_field->m_uniformGrid.m_valueRanges != nullptr
      && ((m_isovalueArray && m_isovalueArray->size() > 0) || m_hasScalarIsovalue);
}

void Isosurface::populateBuildInput(OptixBuildInput &buildInput) const
{
  buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
  buildInput.customPrimitiveArray.numPrimitives = uint32_t(m_numBricks);
  // OptiX requires a null aabb buffer when there are no primitives. An empty
  // isosurface (isovalue outside the field's value range) has zero active
  // bricks; passing the (non-null) buffer with numPrimitives == 0 is a fatal
  // ACCEL build error.
  buildInput.customPrimitiveArray.aabbBuffers =
      m_numBricks ? &m_aabbsBufferPtr : nullptr;

  static uint32_t buildInputFlags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
  buildInput.customPrimitiveArray.flags = buildInputFlags;
  buildInput.customPrimitiveArray.numSbtRecords = 1;
}

GeometryGPUData Isosurface::gpuData() const
{
  auto retval = Geometry::gpuData();
  retval.type = GeometryType::ISOSURFACE;

  auto &iso = retval.isosurface;
  iso.field = m_field->index();
  iso.isovalues = m_isovaluesDev;
  iso.numIsovalues = m_numIsovalues;
  iso.brickBounds = (const box3 *)m_aabbs.ptr();
  iso.stepSize = m_field->stepSize();

  return retval;
}

int Isosurface::optixGeometryType() const
{
  return OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
}

} // namespace visrtx
