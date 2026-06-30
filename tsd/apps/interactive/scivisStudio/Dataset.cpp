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

} // namespace tsd::scivis_studio::dataset
