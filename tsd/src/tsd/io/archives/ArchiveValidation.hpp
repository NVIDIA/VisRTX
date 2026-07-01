// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// std
#include <string>

namespace tsd::io {

enum class ArchiveValidationStatus
{
  Valid,
  MissingMetadataAccepted,
  UnknownSchema,
  IncompatibleSchema,
  UnsupportedEnvelopeVersion,
  UnsupportedSchemaVersion,
  MalformedMetadata,
  MissingRequiredNode
};

struct ArchiveValidationResult
{
  ArchiveValidationStatus status{ArchiveValidationStatus::Valid};
  std::string fileType;
  std::string schema;
  int envelopeVersion{0};
  int schemaVersion{0};
  std::string message;

  bool accepted() const;
};

inline bool ArchiveValidationResult::accepted() const
{
  return status == ArchiveValidationStatus::Valid
      || status == ArchiveValidationStatus::MissingMetadataAccepted;
}

// Temporary source compatibility for callers migrated in Handoff 04.
using PayloadValidationResult = ArchiveValidationResult;

} // namespace tsd::io
