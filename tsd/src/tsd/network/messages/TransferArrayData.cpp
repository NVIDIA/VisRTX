// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "TransferArrayData.hpp"
// tsd_core
#include "tsd/core/Logging.hpp"
#if TSD_USE_CUDA
// cuda
#include <cuda_runtime.h>
#endif

namespace tsd::network::messages {

TransferArrayData::TransferArrayData(const tsd::core::Array *array)
{
  if (!array) {
    tsd::core::logError(
        "[message::TransferArrayData] No array provided for transfer");
    return;
  } else if (array->isProxy()) {
    tsd::core::logError(
        "[message::TransferArrayData] Cannot transfer data for proxy array "
        "(%s, %zu)",
        anari::toString(array->type()),
        array->index());
    return;
  } else if (array->isEmpty()) {
    tsd::core::logStatus(
        "[message::TransferArrayData] Array is empty, no data to transfer "
        "(%s, %zu)",
        anari::toString(array->type()),
        array->index());
    return;
  }

  auto root = m_tree.root();
  root["a"] = tsd::core::Any(array->type(), array->index()); // array

  auto &d = root["d"];
  d.setValueAsExternalArray(array->elementType(), array->data(), array->size());

#if TSD_USE_CUDA
  if (array->kind() == tsd::core::Array::MemoryKind::CUDA) {
    const size_t numBytes = array->size() * array->elementSize();
    std::vector<std::byte> hostBuf(numBytes);
    cudaMemcpy(hostBuf.data(), array->data(), numBytes, cudaMemcpyDeviceToHost);
    d.setValueAsArray(array->elementType(), hostBuf.data(), array->size());
  } else
#endif
  {
    d.setValueAsExternalArray(
        array->elementType(), array->data(), array->size());
  }
}

TransferArrayData::TransferArrayData(
    const Message &msg, tsd::core::Scene *scene)
    : StructuredMessage(msg), m_scene(scene)
{
  tsd::core::logDebug(
      "[message::TransferArrayData] Received message (%zu bytes)",
      msg.header.payload_length);
}

void TransferArrayData::execute()
{
  if (!m_scene) {
    tsd::core::logError(
        "[message::TransferArrayData] No scene provided for exec");
    return;
  }

  auto a = m_tree.root()["a"].getValue();
  auto array = m_scene->getObject<tsd::core::Array>(a.getAsObjectIndex());
  if (!array) {
    tsd::core::logError(
        "[message::TransferArrayData] Unable to find array (%s, %zu)",
        anari::toString(a.type()),
        a.getAsObjectIndex());
    return;
  }

  auto &d = m_tree.root()["d"];
  anari::DataType type = ANARI_UNKNOWN;
  const void *ptr = nullptr;
  size_t size = 0;
  d.getValueAsArray(&type, &ptr, &size);

  if (array->elementType() != type) {
    tsd::core::logError(
        "[message::TransferArrayData] Array type mismatch (%s != %s) for "
        "array (%s, %zu)",
        anari::toString(array->elementType()),
        anari::toString(type),
        anari::toString(a.type()),
        a.getAsObjectIndex());
    return;
  } else if (array->size() != size) {
    tsd::core::logError(
        "[message::TransferArrayData] Array size mismatch (%zu != %zu) for "
        "array (%s, %zu)",
        array->size(),
        size,
        anari::toString(a.type()),
        a.getAsObjectIndex());
    return;
  }

  if (array->isProxy())
    array->convertProxyToHost();
  array->setData(ptr);
}

} // namespace tsd::network::messages
