// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "CaseFileFormat.h"

#include <tsd/io/importers/detail/ensight_io.hpp>

#include <pxr/base/tf/registryManager.h>
#include <pxr/base/vt/dictionary.h>
#include <pxr/base/vt/value.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/primSpec.h>
#include <pxr/usd/sdf/types.h>

#include <algorithm>
#include <cctype>
#include <fstream>

PXR_NAMESPACE_OPEN_SCOPE

TF_DEFINE_PUBLIC_TOKENS(CaseFileFormatTokens, CASE_FILE_FORMAT_TOKENS);

TF_REGISTRY_FUNCTION(TfType)
{
  SDF_DEFINE_FILE_FORMAT(CaseFileFormat, SdfFileFormat);
}

CaseFileFormat::CaseFileFormat()
    : SdfFileFormat(CaseFileFormatTokens->Id,
        CaseFileFormatTokens->Version,
        CaseFileFormatTokens->Target,
        "case")
{
}

CaseFileFormat::~CaseFileFormat() = default;

bool CaseFileFormat::CanRead(const std::string &file) const
{
  static const std::string ext = ".case";
  if (file.size() < ext.size())
    return false;
  if (file.compare(file.size() - ext.size(), ext.size(), ext) != 0)
    return false;

  std::ifstream in(file);
  if (!in.is_open())
    return false;

  std::string line;
  while (std::getline(in, line)) {
    // Skip blank lines
    if (line.find_first_not_of(" \t\r\n") == std::string::npos)
      continue;
    return line.find("FORMAT") != std::string::npos;
  }

  return false;
}

static std::string sanitizePrimName(const std::string &desc, int id)
{
  if (desc.empty())
    return "part_" + std::to_string(id);

  std::string name = desc;
  for (char &c : name) {
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_')
      c = '_';
  }

  // USD prim names cannot start with a digit
  if (std::isdigit(static_cast<unsigned char>(name[0])))
    name = "_" + name;

  return name;
}

bool CaseFileFormat::Read(SdfLayer *layer,
    const std::string &resolvedPath,
    bool /*metadataOnly*/) const
{
  tsd::io::ensight::CaseData caseData;
  if (!tsd::io::ensight::parseCase(resolvedPath, caseData))
    return false;

  // Resolve geo file path relative to case file directory
  auto lastSlash = resolvedPath.rfind('/');
  std::string caseDir =
      (lastSlash != std::string::npos) ? resolvedPath.substr(0, lastSlash + 1) : "";
  std::string geoPath = caseDir + caseData.geoPattern;

  // If geoPattern has wildcards, expand to get the first frame
  if (geoPath.find('*') != std::string::npos) {
    auto expanded = tsd::io::ensight::expandPattern(
        caseData.geoPattern, caseData.startNumber, caseData.increment, 1);
    if (!expanded.empty())
      geoPath = caseDir + expanded[0];
  }

  std::vector<tsd::io::ensight::PartHeader> partHeaders;
  tsd::io::ensight::readGeoFileHeader(geoPath, partHeaders);

  // Collect per-vertex scalar and vector field names
  VtStringArray scalarFields, vectorFields;
  for (const auto &[name, info] : caseData.variables) {
    if (info.association != "vertex")
      continue;
    if (info.type == "scalar")
      scalarFields.push_back(name);
    else if (info.type == "vector")
      vectorFields.push_back(name);
  }

  // Populate the layer

  SdfLayerHandle layerHandle(layer);

  VtDictionary ensightLayerData;
  ensightLayerData["caseFile"] = VtValue(resolvedPath);
  VtDictionary layerData;
  layerData["ensight"] = VtValue(ensightLayerData);
  layer->SetCustomLayerData(layerData);

  double startTC = static_cast<double>(caseData.startNumber);
  double endTC = startTC + (caseData.numSteps - 1) * caseData.increment;
  layer->SetStartTimeCode(startTC);
  layer->SetEndTimeCode(endTC);

  // Root scope prim
  SdfPath defaultPath =
      SdfPath::AbsoluteRootPath().AppendChild(TfToken("default"));
  SdfPrimSpecHandle defaultPrim = SdfCreatePrimInLayer(layerHandle, defaultPath);
  defaultPrim->SetSpecifier(SdfSpecifierDef);
  defaultPrim->SetTypeName("Scope");
  layer->SetDefaultPrim(TfToken("default"));

  // One Mesh prim per EnSight part
  for (const auto &ph : partHeaders) {
    std::string primName = sanitizePrimName(ph.description, ph.id);
    SdfPath partPath = defaultPath.AppendChild(TfToken(primName));
    SdfPrimSpecHandle partPrim = SdfCreatePrimInLayer(layerHandle, partPath);
    partPrim->SetSpecifier(SdfSpecifierDef);
    partPrim->SetTypeName("Mesh");

    VtDictionary ensightData;
    ensightData["partId"] = VtValue(ph.id);
    ensightData["scalarFields"] = VtValue(scalarFields);
    ensightData["vectorFields"] = VtValue(vectorFields);
    partPrim->SetCustomData("ensight", VtValue(ensightData));
  }

  return true;
}

bool CaseFileFormat::WriteToString(
    const SdfLayer &, std::string *, const std::string &) const
{
  return false;
}

bool CaseFileFormat::WriteToStream(
    const SdfSpecHandle &, std::ostream &, size_t) const
{
  return false;
}

PXR_NAMESPACE_CLOSE_SCOPE
