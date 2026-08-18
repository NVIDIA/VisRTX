// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// usd
#include <pxr/imaging/hd/sceneIndex.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usdImaging/usdImaging/stageSceneIndex.h>
// std
#include <memory>
#include <string>

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
  // Use acquireUsdSession(); the constructor is public only so the registry
  // can build one with make_shared.
  UsdStageSession(std::string filePath, pxr::UsdStageRefPtr stage);
  ~UsdStageSession();

  const std::string &filePath() const;
  const pxr::UsdStageRefPtr &stage() const;
  const pxr::HdSceneIndexBaseRefPtr &sceneIndex() const;

  // The Stage's own clock.
  double startTimeCode() const;
  double endTimeCode() const;
  double timeCodesPerSecond() const;

  // Normalized animation time onto the Stage's clock. USD evaluates
  // continuously at the result, so no snapping to an authored sample happens.
  pxr::UsdTimeCode timeCodeAt(float t) const;

  pxr::UsdTimeCode currentTime() const;
  void setTime(pxr::UsdTimeCode time);

 private:
  std::string m_filePath;
  pxr::UsdStageRefPtr m_stage;
  pxr::UsdImagingStageSceneIndexRefPtr m_stageSceneIndex;
  pxr::HdSceneIndexBaseRefPtr m_sceneIndex;
  pxr::UsdTimeCode m_currentTime{pxr::UsdTimeCode::EarliestTime()};
  double m_startTimeCode{0.0};
  double m_endTimeCode{0.0};
};

// Open `filePath`, or join the Session already open on it. Sessions are keyed
// by absolute path in a process-wide registry that holds them weakly: the last
// holder to let go closes the Stage. Returns null if the Stage cannot open.
std::shared_ptr<UsdStageSession> acquireUsdSession(const std::string &filePath);

} // namespace tsd::io::usd
