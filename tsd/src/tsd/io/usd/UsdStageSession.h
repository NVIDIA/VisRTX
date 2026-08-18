// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_core
#include "tsd/core/TypeMacros.hpp"
// usd
#include <pxr/imaging/hd/sceneIndex.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usdImaging/usdImaging/stageSceneIndex.h>
// std
#include <memory>
#include <string>
#include <vector>

namespace tsd::io::usd {

/*
 * A Stage held open together with the resolution chain that turns it into the
 * scene TSD reads, and the Time Code both are currently evaluated at. One
 * Session is shared by the Import that created it and by every animation
 * binding that Import produced, so a scrub resolves through exactly the chain
 * the Import converted from.
 *
 * setTime() is the one place SetTime/ApplyPendingUpdates happen, and it does
 * them once per distinct Time Code no matter how many bindings ask for it.
 * Nothing here depends on Import Options: two Imports of one file with
 * different options still share one Session.
 *
 * Example:
 *   auto session = acquireUsdSession("/data/sim.usd");
 *   session->setTime(session->timeCodeAt(0.5f));
 *   auto prim = session->sceneIndex()->GetPrim(primPath);
 */
struct UsdStageSession
{
  TSD_NOT_COPYABLE(UsdStageSession)
  TSD_NOT_MOVEABLE(UsdStageSession)

  // Use acquireUsdSession(); this is public only so the registry can build a
  // Session with make_shared. `key` is what the registry filed it under, which
  // the Session needs in order to take itself back out again.
  UsdStageSession(
      std::string key, std::string filePath, pxr::UsdStageRefPtr stage);
  ~UsdStageSession();

  const std::string &filePath() const;
  const pxr::UsdStageRefPtr &stage() const;
  const pxr::HdSceneIndexBaseRefPtr &sceneIndex() const;

  // The Stage's own clock.
  double startTimeCode() const;
  double endTimeCode() const;
  double timeCodesPerSecond() const;

  // Whether the Stage authored a time-code range of its own. When it did not,
  // the range is whatever noteAuthoredSampleTimes() has been told about.
  bool hasAuthoredTimeRange() const;

  // Widen the fallback range with times authored on one attribute. Does
  // nothing when the Stage authored a range, which is the authority. Without
  // this a Stage that has time samples but no `startTimeCode`/`endTimeCode`
  // would map every animation time onto one Time Code and never move.
  void noteAuthoredSampleTimes(const std::vector<double> &times);

  // Normalized animation time onto the Stage's clock. USD evaluates
  // continuously at the result, so no snapping to an authored sample happens.
  pxr::UsdTimeCode timeCodeAt(float t) const;

  pxr::UsdTimeCode currentTime() const;
  void setTime(pxr::UsdTimeCode time);

 private:
  std::string m_key;
  std::string m_filePath;
  pxr::UsdStageRefPtr m_stage;
  pxr::UsdImagingStageSceneIndexRefPtr m_stageSceneIndex;
  pxr::HdSceneIndexBaseRefPtr m_sceneIndex;
  pxr::UsdTimeCode m_currentTime{pxr::UsdTimeCode::EarliestTime()};
  double m_startTimeCode{0.0};
  double m_endTimeCode{0.0};
  bool m_authoredTimeRange{false};
  bool m_sawSampleTimes{false};
};

// Open `filePath`, or join the Session already open on it. Sessions are keyed
// by absolute path in a process-wide registry that holds them weakly: the last
// holder to let go closes the Stage. Returns null if the Stage cannot open.
std::shared_ptr<UsdStageSession> acquireUsdSession(const std::string &filePath);

} // namespace tsd::io::usd
