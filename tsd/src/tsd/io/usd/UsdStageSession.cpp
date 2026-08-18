// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/usd/UsdStageSession.h"
#include "tsd/core/Logging.hpp"
// usd
#include <pxr/imaging/hd/retainedDataSource.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hdsi/implicitSurfaceSceneIndex.h>
#include <pxr/imaging/hdsi/nurbsApproximatingSceneIndex.h>
#include <pxr/imaging/hdsi/pinnedCurveExpandingSceneIndex.h>
#include <pxr/imaging/hdsi/tetMeshConversionSceneIndex.h>
#include <pxr/usdImaging/usdImaging/sceneIndices.h>
// std
#include <algorithm>
#include <filesystem>
#include <map>
#include <mutex>

namespace tsd::io::usd {

namespace {

// Everything OpenUSD can resolve for us, resolved before TSD sees it: sphere,
// cone and cylinder stay analytic for TSD's native quadrics while capsule,
// cube and plane become meshes; NURBS are approximated; pinned curves are
// expanded; tetrahedral meshes are converted.
pxr::HdSceneIndexBaseRefPtr buildFilterChain(pxr::HdSceneIndexBaseRefPtr input)
{
  using pxr::HdsiImplicitSurfaceSceneIndexTokens;

  auto implicitArgs =
      pxr::HdRetainedContainerDataSource::New(pxr::HdPrimTypeTokens->capsule,
          pxr::HdRetainedTypedSampledDataSource<pxr::TfToken>::New(
              HdsiImplicitSurfaceSceneIndexTokens->toMesh),
          pxr::HdPrimTypeTokens->cube,
          pxr::HdRetainedTypedSampledDataSource<pxr::TfToken>::New(
              HdsiImplicitSurfaceSceneIndexTokens->toMesh),
          pxr::HdPrimTypeTokens->plane,
          pxr::HdRetainedTypedSampledDataSource<pxr::TfToken>::New(
              HdsiImplicitSurfaceSceneIndexTokens->toMesh));
  // Cone and cylinder are deliberately left alone: axisToTransform would move
  // the shape's spine into a transform this importer does not read (local
  // transforms come from the Stage, not the resolved scene), so the converter
  // folds the axis into the emitted endpoints instead.

  auto retval = pxr::HdsiImplicitSurfaceSceneIndex::New(input, implicitArgs);
  auto nurbs = pxr::HdsiNurbsApproximatingSceneIndex::New(retval);
  auto curves = pxr::HdsiPinnedCurveExpandingSceneIndex::New(nurbs);
  return pxr::HdsiTetMeshConversionSceneIndex::New(curves);
}

// The registry key. A Stage that cannot be resolved to an absolute path is
// keyed by what the caller said, which is still stable within one process.
std::string sessionKeyOf(const std::string &filePath)
{
  std::error_code ec;
  auto absolute = std::filesystem::weakly_canonical(filePath, ec);
  return ec ? filePath : absolute.string();
}

std::mutex &registryMutex()
{
  static std::mutex mutex;
  return mutex;
}

std::map<std::string, std::weak_ptr<UsdStageSession>> &registry()
{
  static std::map<std::string, std::weak_ptr<UsdStageSession>> sessions;
  return sessions;
}

} // namespace

UsdStageSession::UsdStageSession(
    std::string key, std::string filePath, pxr::UsdStageRefPtr stage)
    : m_key(std::move(key)),
      m_filePath(std::move(filePath)),
      m_stage(std::move(stage))
{
  pxr::UsdImagingCreateSceneIndicesInfo createInfo;
  createInfo.stage = m_stage;
  createInfo.addDrawModeSceneIndex = false;
  auto sceneIndices = pxr::UsdImagingCreateSceneIndices(createInfo);
  m_stageSceneIndex = sceneIndices.stageSceneIndex;
  m_sceneIndex = buildFilterChain(sceneIndices.finalSceneIndex);

  // Values authored only as time samples do not resolve at UsdTimeCode's
  // default, so a Stage with no authored range is still read at a real time.
  m_authoredTimeRange = m_stage->HasAuthoredTimeCodeRange();
  if (m_authoredTimeRange) {
    m_startTimeCode = m_stage->GetStartTimeCode();
    m_endTimeCode = m_stage->GetEndTimeCode();
    if (!(m_endTimeCode > m_startTimeCode))
      m_endTimeCode = m_startTimeCode;
  }

  setTime(pxr::UsdTimeCode(m_startTimeCode));
}

UsdStageSession::~UsdStageSession()
{
  // Take the registry entry out with the Session, so a later acquire of this
  // file opens a fresh Stage rather than finding a corpse. The entry may
  // already have been replaced by a newer Session for the same file, which
  // `expired()` distinguishes.
  std::lock_guard<std::mutex> guard(registryMutex());
  auto &sessions = registry();
  if (auto found = sessions.find(m_key);
      found != sessions.end() && found->second.expired())
    sessions.erase(found);
}

const std::string &UsdStageSession::filePath() const
{
  return m_filePath;
}

const pxr::UsdStageRefPtr &UsdStageSession::stage() const
{
  return m_stage;
}

const pxr::HdSceneIndexBaseRefPtr &UsdStageSession::sceneIndex() const
{
  return m_sceneIndex;
}

double UsdStageSession::startTimeCode() const
{
  return m_startTimeCode;
}

double UsdStageSession::endTimeCode() const
{
  return m_endTimeCode;
}

double UsdStageSession::timeCodesPerSecond() const
{
  return m_stage->GetTimeCodesPerSecond();
}

bool UsdStageSession::hasAuthoredTimeRange() const
{
  return m_authoredTimeRange;
}

void UsdStageSession::noteAuthoredSampleTimes(const std::vector<double> &times)
{
  if (m_authoredTimeRange || times.empty())
    return;

  if (!m_sawSampleTimes) {
    m_sawSampleTimes = true;
    m_startTimeCode = times.front();
    m_endTimeCode = times.back();
    return;
  }
  m_startTimeCode = std::min(m_startTimeCode, times.front());
  m_endTimeCode = std::max(m_endTimeCode, times.back());
}

pxr::UsdTimeCode UsdStageSession::timeCodeAt(float t) const
{
  const double span = m_endTimeCode - m_startTimeCode;
  return pxr::UsdTimeCode(m_startTimeCode + double(t) * span);
}

pxr::UsdTimeCode UsdStageSession::currentTime() const
{
  return m_currentTime;
}

void UsdStageSession::setTime(pxr::UsdTimeCode time)
{
  if (m_currentTime == time)
    return;
  m_currentTime = time;
  if (!m_stageSceneIndex)
    return;
  m_stageSceneIndex->SetTime(time);
  m_stageSceneIndex->ApplyPendingUpdates();
}

std::shared_ptr<UsdStageSession> acquireUsdSession(const std::string &filePath)
{
  const auto key = sessionKeyOf(filePath);

  std::lock_guard<std::mutex> guard(registryMutex());

  auto &sessions = registry();
  if (auto found = sessions.find(key); found != sessions.end()) {
    if (auto existing = found->second.lock())
      return existing;
  }

  auto stage = pxr::UsdStage::Open(filePath, pxr::UsdStage::LoadAll);
  if (!stage)
    return {};

  auto session = std::make_shared<UsdStageSession>(key, filePath, stage);
  sessions[key] = session;
  return session;
}

} // namespace tsd::io::usd
