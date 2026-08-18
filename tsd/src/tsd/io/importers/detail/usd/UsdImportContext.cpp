// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/importers/detail/usd/UsdImportContext.h"
#include "tsd/animation/AnimationManager.hpp"
// usd
#include <pxr/usd/usd/attribute.h>
// std
#include <algorithm>
#include <vector>

namespace tsd::io::usd {

tsd::animation::Animation &ImportContext::animation()
{
  if (!importAnimation)
    importAnimation = &animMgr.addAnimation(filePath);
  return *importAnimation;
}

void ImportContext::reportSampleCount(size_t count)
{
  report.animatedPrims++;
  report.sampleCount = std::max(report.sampleCount, count);
  if (session)
    report.timeCodesPerSecond = float(session->timeCodesPerSecond());
}

bool attributeValueVaries(const pxr::UsdAttribute &attribute)
{
  if (!attribute)
    return false;

  std::vector<double> times;
  attribute.GetTimeSamples(&times);
  if (times.size() < 2)
    return false;

  // Reading every sample of an array attribute means reading the whole file.
  if (attribute.GetTypeName().IsArray())
    return true;

  pxr::VtValue first;
  if (!attribute.Get(&first, pxr::UsdTimeCode(times.front())))
    return false;

  for (size_t i = 1; i < times.size(); ++i) {
    pxr::VtValue value;
    if (!attribute.Get(&value, pxr::UsdTimeCode(times[i])))
      return true;
    if (value != first)
      return true;
  }
  return false;
}

} // namespace tsd::io::usd
