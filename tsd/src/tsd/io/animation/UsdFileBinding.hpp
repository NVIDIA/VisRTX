// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/animation/FileBinding.hpp"
// std
#include <memory>
#include <string>
#include <vector>

namespace tsd::io {

namespace usd {
struct UsdStageSession;
} // namespace usd

/*
 * What every USD animation binding needs regardless of what it drives: the
 * Stage Session it resolves through, and the file and prim paths that identify
 * what it resolves. Bindings created by an Import are handed the Session the
 * Import used; bindings reconstructed from an Archive carry only the paths and
 * join the Session for that file on the first update.
 *
 * Example:
 *   struct MyBinding : UsdFileBinding {
 *     void update(float t) override {
 *       if (!ensureSession()) return;
 *       ...
 *     }
 *   };
 */
struct UsdFileBinding : public tsd::animation::FileBinding
{
  UsdFileBinding(scene::Scene *scene,
      std::shared_ptr<usd::UsdStageSession> session,
      std::string stageFile,
      std::string primPath);
  ~UsdFileBinding() override;

  const std::string &stageFile() const;
  const std::string &primPath() const;

 protected:
  // Join the Session for this binding's file, or report why not. Retried at
  // most once: a file that failed to open is not reopened on every scrub.
  bool ensureSession();

  usd::UsdStageSession *session() const;

  // Write the file and prim paths every USD binding serializes.
  void writePathsToDataNode(tsd::core::DataNode &node) const;

  // Tell the Session about times authored on this binding's own prim, so a
  // Stage that carries samples without authoring a time-code range still maps
  // animation time onto a range that moves.
  void noteAuthoredSampleTimes(const std::vector<double> &times);

  // Named in the warning ensureSession() emits, so a failure says which kind
  // of binding could not resolve.
  virtual const char *logTag() const = 0;

 private:
  std::shared_ptr<usd::UsdStageSession> m_session;
  std::string m_stageFile;
  std::string m_primPath;
  bool m_sessionFailed{false};
};

} // namespace tsd::io
