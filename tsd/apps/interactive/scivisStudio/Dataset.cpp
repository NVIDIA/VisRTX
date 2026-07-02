// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Dataset.h"

namespace tsd::scivis_studio::dataset {

const char *toString(DatasetSourceKind kind)
{
  switch (kind) {
  case DatasetSourceKind::Static:
    return "Static";
  case DatasetSourceKind::FileAnimation:
    return "FileAnimation";
  case DatasetSourceKind::Live:
    return "Live";
  }
  return "Static";
}

const char *toString(DatasetStatus status)
{
  switch (status) {
  case DatasetStatus::Available:
    return "Available";
  case DatasetStatus::Unavailable:
    return "Unavailable";
  case DatasetStatus::Importing:
    return "Importing";
  case DatasetStatus::ImportFailed:
    return "ImportFailed";
  }
  return "Missing";
}

const char *toString(DatasetResidency residency)
{
  switch (residency) {
  case DatasetResidency::Loaded:
    return "Loaded";
  case DatasetResidency::Unloaded:
    return "Unloaded";
  }
  return "Loaded";
}

DatasetSourceKind sourceKindFromString(const std::string &s)
{
  if (s == "FileAnimation" || s == "TimeSeries")
    return DatasetSourceKind::FileAnimation;
  if (s == "Live")
    return DatasetSourceKind::Live;
  return DatasetSourceKind::Static;
}

DatasetStatus statusFromString(const std::string &s)
{
  if (s == "Available")
    return DatasetStatus::Available;
  if (s == "Importing")
    return DatasetStatus::Importing;
  if (s == "ImportFailed")
    return DatasetStatus::ImportFailed;
  return DatasetStatus::Unavailable;
}

DatasetResidency residencyFromString(const std::string &s)
{
  return s == "Unloaded" ? DatasetResidency::Unloaded
                         : DatasetResidency::Loaded;
}

const char *displayStatus(const Dataset &dataset)
{
  switch (dataset.status) {
  case DatasetStatus::Importing:
    return "Importing";
  case DatasetStatus::ImportFailed:
    return "Import Failed";
  case DatasetStatus::Unavailable:
    return "Unavailable";
  case DatasetStatus::Available:
    break;
  }
  return dataset.residency == DatasetResidency::Unloaded ? "Unloaded"
                                                         : "Loaded";
}

} // namespace tsd::scivis_studio::dataset
