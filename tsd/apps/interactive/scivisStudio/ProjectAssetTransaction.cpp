// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ProjectAssetTransaction.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <system_error>

namespace tsd::scivis_studio {

namespace {

struct TransactionEntry
{
  const ProjectAssetWrite *write{nullptr};
  std::filesystem::path target;
  std::filesystem::path existing;
  std::filesystem::path stage;
  std::filesystem::path backup;
  bool isManifest{false};
  bool backedUp{false};
  bool installed{false};
};

static bool fail(const std::string &message, std::string *error)
{
  if (error)
    *error = message;
  return false;
}

static bool namesEqual(const std::string &a, const std::string &b)
{
  if (a.size() != b.size())
    return false;
  return std::equal(a.begin(), a.end(), b.begin(), [](char x, char y) {
    return std::tolower(static_cast<unsigned char>(x))
        == std::tolower(static_cast<unsigned char>(y));
  });
}

static bool pathsEqual(
    const std::filesystem::path &a, const std::filesystem::path &b)
{
  return namesEqual(a.lexically_normal().generic_string(),
      b.lexically_normal().generic_string());
}

static bool validRelativePath(const std::filesystem::path &path)
{
  if (path.empty() || path.is_absolute())
    return false;
  const auto normalized = path.lexically_normal();
  if (normalized.empty() || normalized == ".")
    return false;
  return std::none_of(normalized.begin(),
      normalized.end(),
      [](const auto &part) { return part == ".."; });
}

static bool findExistingTarget(const std::filesystem::path &directory,
    const std::filesystem::path &relativeTarget,
    std::filesystem::path &existing,
    std::string *error)
{
  existing.clear();
  const auto parent = directory / relativeTarget.parent_path();
  std::error_code ec;
  if (!std::filesystem::exists(parent, ec)) {
    if (ec)
      return fail("failed to inspect target directory: " + ec.message(), error);
    return true;
  }

  for (const auto &entry : std::filesystem::directory_iterator(parent, ec)) {
    if (ec)
      break;
    if (!namesEqual(entry.path().filename().string(),
            relativeTarget.filename().string()))
      continue;
    if (!existing.empty()) {
      return fail("target is ambiguous under case-insensitive comparison: "
              + relativeTarget.generic_string(),
          error);
    }
    existing = entry.path();
  }
  if (ec)
    return fail("failed to inspect target directory: " + ec.message(), error);
  return true;
}

static std::filesystem::path relativeExistingPath(
    const std::filesystem::path &relativeTarget,
    const std::filesystem::path &existing)
{
  return relativeTarget.parent_path() / existing.filename();
}

static bool validateWrite(const ProjectSavePlan &plan,
    const ProjectAssetWrite &write,
    bool isManifest,
    TransactionEntry &entry,
    std::string *error)
{
  if (write.description.empty())
    return fail("asset description is empty", error);
  if (!validRelativePath(write.target)) {
    return fail("invalid target for " + write.description + ": "
            + write.target.generic_string(),
        error);
  }
  if (!write.writer || !write.validator)
    return fail(
        "missing writer or validator for " + write.description, error);
  if (write.ownedTarget && !validRelativePath(*write.ownedTarget)) {
    return fail("invalid owned target for " + write.description + ": "
            + write.ownedTarget->generic_string(),
        error);
  }

  std::filesystem::path existing;
  if (!findExistingTarget(plan.directory, write.target, existing, error))
    return false;
  if (!existing.empty()) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(existing, ec) || ec) {
      return fail("target for " + write.description
              + " is not a regular file: " + existing.string(),
          error);
    }
    const auto relativeExisting = relativeExistingPath(write.target, existing);
    if (!write.ownedTarget
        || !pathsEqual(relativeExisting, *write.ownedTarget)) {
      return fail("unowned target blocks " + write.description + ": "
              + relativeExisting.generic_string(),
          error);
    }
  }

  entry.write = &write;
  entry.target = plan.directory / write.target;
  entry.existing = std::move(existing);
  entry.isManifest = isManifest;
  return true;
}

static bool containsPath(const std::vector<std::filesystem::path> &paths,
    const std::filesystem::path &candidate)
{
  return std::any_of(paths.begin(), paths.end(), [&](const auto &path) {
    return pathsEqual(path, candidate);
  });
}

static void removeIfPresent(const std::filesystem::path &path)
{
  if (path.empty())
    return;
  std::error_code ec;
  std::filesystem::remove(path, ec);
}

} // namespace

AssetTransactionFailureInjector::~AssetTransactionFailureInjector() = default;

AssetTransaction::AssetTransaction(
    AssetTransactionFailureInjector *failureInjector)
    : m_failureInjector(failureInjector)
{}

bool AssetTransaction::commit(const ProjectSavePlan &plan, std::string *error)
{
  auto injectedFailure = [&](AssetTransactionPhase phase,
                             const std::filesystem::path &path,
                             std::string &message) {
    return m_failureInjector && m_failureInjector->fail(phase, path, message);
  };

  std::string failure;
  if (injectedFailure(
          AssetTransactionPhase::Validation, plan.directory, failure))
    return fail(failure, error);
  if (plan.directory.empty())
    return fail("project directory is empty", error);

  std::vector<std::filesystem::path> claimedPaths;
  std::vector<TransactionEntry> entries;
  entries.reserve(plan.assets.size() + plan.removals.size() + 1);
  for (const auto &asset : plan.assets) {
    if (containsPath(claimedPaths, asset.target))
      return fail(
          "duplicate transaction target: " + asset.target.string(), error);
    claimedPaths.push_back(asset.target);
    entries.emplace_back();
    if (!validateWrite(plan, asset, false, entries.back(), error))
      return false;
  }

  if (containsPath(claimedPaths, plan.manifest.target)) {
    return fail(
        "duplicate transaction target: " + plan.manifest.target.string(),
        error);
  }
  claimedPaths.push_back(plan.manifest.target);

  std::vector<TransactionEntry> removals;
  removals.reserve(plan.removals.size());
  for (const auto &removal : plan.removals) {
    if (!validRelativePath(removal))
      return fail("invalid removal target: " + removal.generic_string(), error);
    if (containsPath(claimedPaths, removal))
      return fail("duplicate transaction target: " + removal.string(), error);
    claimedPaths.push_back(removal);

    std::filesystem::path existing;
    if (!findExistingTarget(plan.directory, removal, existing, error))
      return false;
    if (existing.empty())
      continue;
    std::error_code ec;
    if (!std::filesystem::is_regular_file(existing, ec) || ec)
      return fail(
          "removal target is not a regular file: " + existing.string(), error);
    TransactionEntry entry;
    entry.target = plan.directory / removal;
    entry.existing = std::move(existing);
    removals.push_back(std::move(entry));
  }

  TransactionEntry manifest;
  if (!validateWrite(plan, plan.manifest, true, manifest, error))
    return false;

  std::error_code ec;
  std::filesystem::create_directories(plan.directory, ec);
  if (ec)
    return fail("failed to create project directory: " + ec.message(), error);
  for (const auto &relativeDirectory : plan.directories) {
    if (!validRelativePath(relativeDirectory)) {
      return fail(
          "invalid project subdirectory: " + relativeDirectory.string(), error);
    }
    std::filesystem::create_directories(plan.directory / relativeDirectory, ec);
    if (ec)
      return fail(
          "failed to create project subdirectory: " + ec.message(), error);
  }
  for (const auto &asset : plan.assets) {
    std::filesystem::create_directories(
        (plan.directory / asset.target).parent_path(), ec);
    if (ec)
      return fail("failed to create asset directory: " + ec.message(), error);
  }
  std::filesystem::create_directories(
      (plan.directory / plan.manifest.target).parent_path(), ec);
  if (ec)
    return fail("failed to create manifest directory: " + ec.message(), error);

  entries.insert(entries.end(), removals.begin(), removals.end());
  entries.push_back(std::move(manifest));

  const auto transactionTag = std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count());
  size_t index = 0;
  for (auto &entry : entries) {
    const auto &filename = entry.target.filename().string();
    const auto parent = entry.target.parent_path();
    entry.backup = parent
        / ("." + filename + ".backup-" + transactionTag + "-"
            + std::to_string(index));
    if (entry.write) {
      entry.stage = parent
          / ("." + filename + ".stage-" + transactionTag + "-"
              + std::to_string(index));
    }
    ++index;
  }

  auto cleanupStages = [&]() {
    for (const auto &entry : entries)
      removeIfPresent(entry.stage);
  };

  for (auto &entry : entries) {
    if (!entry.write)
      continue;
    failure.clear();
    if (injectedFailure(
            AssetTransactionPhase::Staging, entry.target, failure)) {
      cleanupStages();
      return fail(failure, error);
    }
    std::string stageError;
    if (!entry.write->writer(entry.stage, &stageError)) {
      cleanupStages();
      return fail(
          "failed to stage " + entry.write->description + ": " + stageError,
          error);
    }
    if (!std::filesystem::is_regular_file(entry.stage, ec) || ec) {
      cleanupStages();
      return fail("writer did not stage a regular file for "
              + entry.write->description,
          error);
    }
    if (!entry.write->validator(entry.stage, &stageError)) {
      cleanupStages();
      return fail("failed to validate staged " + entry.write->description + ": "
              + stageError,
          error);
    }
  }

  auto rollback = [&](std::string primaryFailure) {
    std::vector<std::string> rollbackFailures;
    for (auto itr = entries.rbegin(); itr != entries.rend(); ++itr) {
      if (!itr->installed)
        continue;
      failure.clear();
      if (injectedFailure(
              AssetTransactionPhase::Rollback, itr->target, failure)) {
        rollbackFailures.push_back(failure);
        continue;
      }
      std::filesystem::remove(itr->target, ec);
      if (ec)
        rollbackFailures.push_back("failed to remove installed '"
            + itr->target.string() + "': " + ec.message());
    }
    for (auto itr = entries.rbegin(); itr != entries.rend(); ++itr) {
      if (!itr->backedUp)
        continue;
      failure.clear();
      if (injectedFailure(
              AssetTransactionPhase::Rollback, itr->existing, failure)) {
        rollbackFailures.push_back(failure);
        continue;
      }
      std::filesystem::rename(itr->backup, itr->existing, ec);
      if (ec)
        rollbackFailures.push_back("failed to restore '"
            + itr->existing.string() + "': " + ec.message());
    }
    cleanupStages();

    std::string message = primaryFailure;
    if (!rollbackFailures.empty()) {
      message += "\nrollback failures:";
      for (const auto &rollbackFailure : rollbackFailures)
        message += "\n- " + rollbackFailure;
    }
    return fail(message, error);
  };

  for (auto &entry : entries) {
    if (entry.existing.empty())
      continue;
    failure.clear();
    if (injectedFailure(AssetTransactionPhase::Backup, entry.existing, failure))
      return rollback(failure);
    std::filesystem::rename(entry.existing, entry.backup, ec);
    if (ec)
      return rollback("failed to back up '" + entry.existing.string()
          + "': " + ec.message());
    entry.backedUp = true;
  }

  for (auto &entry : entries) {
    if (!entry.write || entry.isManifest)
      continue;
    failure.clear();
    if (injectedFailure(
            AssetTransactionPhase::AssetInstall, entry.target, failure))
      return rollback(failure);
    std::filesystem::rename(entry.stage, entry.target, ec);
    if (ec)
      return rollback("failed to install " + entry.write->description + ": "
          + ec.message());
    entry.installed = true;
  }

  auto &manifestEntry = entries.back();
  failure.clear();
  if (injectedFailure(AssetTransactionPhase::ManifestInstall,
          manifestEntry.target,
          failure))
    return rollback(failure);
  std::filesystem::rename(manifestEntry.stage, manifestEntry.target, ec);
  if (ec) {
    return rollback("failed to install " + manifestEntry.write->description
        + ": " + ec.message());
  }
  manifestEntry.installed = true;

  for (const auto &entry : entries) {
    if (entry.backedUp)
      removeIfPresent(entry.backup);
  }
  cleanupStages();
  return true;
}

} // namespace tsd::scivis_studio
