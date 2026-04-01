// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pxr/pxr.h>
#include <pxr/usd/sdf/fileFormat.h>
#include <pxr/base/tf/staticTokens.h>

PXR_NAMESPACE_OPEN_SCOPE

#define CASE_FILE_FORMAT_TOKENS \
    ((Id, "caseFileFormat"))((Version, "1.0"))((Target, "usd"))

TF_DECLARE_PUBLIC_TOKENS(CaseFileFormatTokens, CASE_FILE_FORMAT_TOKENS);

TF_DECLARE_WEAK_AND_REF_PTRS(CaseFileFormat);

class CaseFileFormat : public SdfFileFormat
{
public:
  SDF_FILE_FORMAT_FACTORY_ACCESS;

  CaseFileFormat();
  ~CaseFileFormat() override;

  bool CanRead(const std::string &file) const override;

  bool Read(SdfLayer *layer,
      const std::string &resolvedPath,
      bool metadataOnly) const override;

  bool WriteToString(const SdfLayer &layer,
      std::string *str,
      const std::string &comment = std::string()) const override;

  bool WriteToStream(const SdfSpecHandle &spec,
      std::ostream &out,
      size_t indent) const override;
};

PXR_NAMESPACE_CLOSE_SCOPE
