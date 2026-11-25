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

#include "GeometryPreparation.h"
#include "Geometry.h"
#include "array/Array.h"

namespace visrtx {

GeometryPreparation::GeometryPreparation(PreparationStream &stream)
    : m_stream(stream)
{}

GeometryPreparation::~GeometryPreparation() = default;

void GeometryPreparation::shutdown()
{
  // Disable cached returns and free idle/pending blocks before the device
  // destroys its streams. Any allocation outliving the device falls back to
  // ordinary free.
  m_inputAllocations->close();
}

bool GeometryPreparation::busy() const
{
  return m_preparations.collecting() || m_inputUploads.collecting();
}

bool GeometryPreparation::collect()
{
  if (!uploadInputs())
    return false;
  // Owners remain queued if descriptor allocation or registry publication
  // throws. CUDA failure still invalidates the pending batch before reporting,
  // so callbacks cannot see a partially valid generation. Numeric failures
  // remain local to their geometries during publication. Every publication
  // must succeed before any geometry leaves the queue.
  return m_preparations.collect(
      [&](Geometry *first) {
        const auto error = flush();
        if (error != cudaSuccess)
          fail(first, error, "batch completion");
        m_preparations.forEach(
            [](Geometry *geometry) { geometry->completePreparation(); });
      },
      [](Geometry *geometry) noexcept { geometry->markPreparationCollected(); },
      [] {});
}

cudaError_t GeometryPreparation::waitForHostReaders()
{
  // Only an input-upload collection has copies in flight. Draining does not
  // release the queue's retained array references.
  return uploadingInputs() ? flush() : cudaSuccess;
}

cudaError_t GeometryPreparation::waitForDeviceReaders()
{
  auto completion = retireRender();
  if (completion == cudaSuccess && pending())
    completion = flush();
  return completion;
}

std::shared_ptr<DeviceAllocationCache> GeometryPreparation::inputAllocations()
    const
{
  return m_inputAllocations;
}

void GeometryPreparation::stageInputs(Geometry *geometry)
{
  m_inputUploads.enqueue(helium::IntrusivePtr<Geometry>(geometry));
}

bool GeometryPreparation::uploadInputs()
{
  return m_inputUploads.collect(
      [&](Geometry *first) {
        // The only route by which a geometry's host inputs reach the device.
        // Commits capture sources; the first geometry finalizer submits every
        // input batch before waiting. No legacy finalizer sees partially
        // uploaded arrays. A finalizer then queues what no commit captured
        // as a batch of its own. Retire obsolete cached copies before
        // allocating a whole replacement cohort. A single geometry retains
        // the existing allocation/retirement order.
        if (m_inputUploads.size() > 1)
          m_inputUploads.forEach([](Geometry *geometry) {
            geometry->retirePreparationSnapshot();
          });
        auto error = m_inputAllocations->reclaim([&] { return flush(); });
        if (error == cudaSuccess)
          error = m_stream.orderAfterRender();
        if (error == cudaSuccess) {
          m_inputUploads.forEach(
              [](Geometry *geometry) { geometry->submitInputUploads(); });
          error = flush();
        }
        if (error != cudaSuccess) {
          failInputUploads();
          fail(first, error, "geometry input batch completion");
        }
        m_inputVersions.clear();
      },
      [](Geometry *geometry) noexcept { geometry->completeInputUploads(); },
      [&] {
        // A host allocation or callback can unwind after copies were
        // submitted. Keep sources alive and drain before permitting public
        // release again.
        failInputUploads();
        flush();
        m_inputVersions.clear();
      });
}

cudaError_t GeometryPreparation::stageArray(
    Array *array, helium::TimeStamp modified)
{
  const auto found = m_inputVersions.find(array);
  if (found != m_inputVersions.end() && found->second == modified)
    return cudaSuccess;
  const auto error = array->enqueueUpload(m_stream.stream());
  if (error == cudaSuccess)
    m_inputVersions.insert_or_assign(array, modified);
  return error;
}

void GeometryPreparation::defer(Geometry *geometry)
{
  if (!geometry->m_preparationPosition)
    geometry->m_preparationPosition =
        m_preparations.enqueue(helium::IntrusivePtr<Geometry>(geometry));
}

void GeometryPreparation::deferSmall(Geometry *geometry, Preprocessing *job)
{
  m_smallBatch.enqueue(job, position(geometry));
}

GeometryResults *GeometryPreparation::deviceResults(Geometry *geometry)
{
  const size_t at = position(geometry);
  if (!check(geometry,
          m_results.reserve(at + 1, m_stream.stream()),
          "allocating preparation results"))
    return nullptr;
  return m_results.device(at);
}

const GeometryResults &GeometryPreparation::hostResults(
    const Geometry *geometry) const
{
  return m_results.host(position(geometry));
}

size_t GeometryPreparation::position(const Geometry *geometry) const
{
  return *geometry->m_preparationPosition;
}

cudaError_t GeometryPreparation::flush()
{
  // Flush deferred small jobs even at intervening upload/retirement waits:
  // their input pointers must be consumed before a backing allocation changes.
  // Then read back every queued result in one transfer.
  const auto stream = m_stream.stream();
  auto error = m_smallBatch.launch(m_results, stream);
  if (error == cudaSuccess)
    error = m_results.readback(m_preparations.size(), stream);
  const auto completion = cudaStreamSynchronize(stream);
  if (error == cudaSuccess)
    error = completion;
  if (error != cudaSuccess) {
    cudaGetLastError();
    // An intervening upload/legacy consumer may wait before the collection
    // point. Its stream-completion error affects all already queued results.
    m_preparations.forEach(
        [](Geometry *geometry) { geometry->markPreparationFailed(); });
    failInputUploads();
  }
  return error;
}

void GeometryPreparation::fail(
    Geometry *geometry, cudaError_t error, const char *operation)
{
  geometry->markPreparationFailed();
  // A failed dispatch/readback can follow successful queued kernels. Drain
  // preparation before retrying or freeing scratch. Keep the render-stream
  // drain on this error path too: this also handles input retirement and
  // dependency failures, which may precede a successful producer edge.
  flush();
  cudaStreamSynchronize(m_stream.renderStream());
  geometry->reportMessage(ANARI_SEVERITY_ERROR,
      "geometry preprocessing failed during %s: %s",
      operation,
      cudaGetErrorString(error));
}

bool GeometryPreparation::check(
    Geometry *geometry, cudaError_t error, const char *operation)
{
  if (error == cudaSuccess)
    return true;
  fail(geometry, error, operation);
  return false;
}

void GeometryPreparation::reject(Geometry *geometry, const char *reason)
{
  geometry->markPreparationFailed();
  geometry->reportMessage(ANARI_SEVERITY_ERROR,
      "geometry preprocessing rejected input: %s",
      reason);
}

bool GeometryPreparation::finish(Geometry *geometry)
{
  // Also establishes an explicit boundary for legacy default-stream consumers.
  return check(geometry, cudaGetLastError(), "launch")
      && check(geometry, flush(), "execution completion");
}

cudaError_t GeometryPreparation::retireRender()
{
  // Anything but success, a failed query included, takes the wait: only a
  // completed stream permits releasing what its work reads.
  const auto renderStream = m_stream.renderStream();
  return cudaStreamQuery(renderStream) == cudaSuccess
      ? cudaSuccess
      : cudaStreamSynchronize(renderStream);
}

bool GeometryPreparation::uploadingInputs() const
{
  return m_inputUploads.collecting();
}

bool GeometryPreparation::pending() const
{
  return m_preparations.collecting() || !m_preparations.empty()
      || m_inputUploads.collecting();
}

void GeometryPreparation::failInputUploads()
{
  m_inputUploads.forEach(
      [](Geometry *geometry) { geometry->failInputUploads(); });
}

} // namespace visrtx
