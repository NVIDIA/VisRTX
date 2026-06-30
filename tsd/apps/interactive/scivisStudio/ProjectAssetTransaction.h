// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace tsd::scivis_studio {

using ProjectAssetWriter =
    std::function<bool(const std::filesystem::path &, std::string *)>;
using ProjectAssetValidator =
    std::function<bool(const std::filesystem::path &, std::string *)>;

/*
 * One managed file in a ProjectSavePlan. target and ownedTarget are relative
 * to the plan directory. An existing target is replaceable only when it is the
 * case-insensitive match for ownedTarget. The writer must create the supplied
 * staging file, and the validator must accept that file before commit begins.
 */
struct ProjectAssetWrite
{
  std::string description;
  std::filesystem::path target;
  std::optional<std::filesystem::path> ownedTarget;
  ProjectAssetWriter writer;
  ProjectAssetValidator validator;
};

/*
 * The complete, explicit filesystem effect of one project save. Only listed
 * assets and removals participate; the manifest is always installed last.
 */
struct ProjectSavePlan
{
  std::filesystem::path directory;
  std::vector<std::filesystem::path> directories;
  std::vector<ProjectAssetWrite> assets;
  ProjectAssetWrite manifest;
  std::vector<std::filesystem::path> removals;
};

enum class AssetTransactionPhase
{
  Validation,
  Staging,
  Backup,
  AssetInstall,
  ManifestInstall,
  Rollback
};

/* Test adapter for forcing a failure immediately before a transaction phase. */
struct AssetTransactionFailureInjector
{
  virtual ~AssetTransactionFailureInjector();

  virtual bool fail(AssetTransactionPhase phase,
      const std::filesystem::path &path,
      std::string &message) = 0;
};

/*
 * Validate, stage, and transactionally install one ProjectSavePlan with
 * rollback.
 */
struct AssetTransaction
{
  explicit AssetTransaction(
      AssetTransactionFailureInjector *failureInjector = nullptr);

  bool commit(const ProjectSavePlan &plan, std::string *error = nullptr);

 private:
  AssetTransactionFailureInjector *m_failureInjector{nullptr};
};

} // namespace tsd::scivis_studio
