// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "catch.hpp"

#include "ProjectAssetTransaction.h"

#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace tsd::scivis_studio::test {

static void writeText(
    const std::filesystem::path &file, const std::string &text)
{
  std::ofstream output(file, std::ios::binary);
  output << text;
}

static std::string readText(const std::filesystem::path &file)
{
  std::ifstream input(file, std::ios::binary);
  return {
      std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

static ProjectAssetWrite textAsset(const std::string &description,
    const std::filesystem::path &target,
    const std::string &contents,
    const std::optional<std::filesystem::path> &ownedTarget = std::nullopt)
{
  ProjectAssetWrite asset;
  asset.description = description;
  asset.target = target;
  asset.ownedTarget = ownedTarget;
  asset.writer = [contents](const std::filesystem::path &stage, std::string *) {
    writeText(stage, contents);
    return true;
  };
  asset.validator = [contents](const std::filesystem::path &stage,
                        std::string *error) {
    if (readText(stage) == contents)
      return true;
    if (error)
      *error = "unexpected staged contents";
    return false;
  };
  return asset;
}

static ProjectSavePlan twoAssetPlan(const std::filesystem::path &root)
{
  ProjectSavePlan plan;
  plan.directory = root;
  plan.directories = {"assets"};
  plan.assets.push_back(textAsset(
      "first asset", "assets/first.tsd", "new first", "assets/first.tsd"));
  plan.assets.push_back(textAsset(
      "second asset", "assets/second.tsd", "new second", "assets/second.tsd"));
  plan.manifest =
      textAsset("manifest", "project.tsd", "new manifest", "project.tsd");
  return plan;
}

static void createOldProject(const std::filesystem::path &root)
{
  std::filesystem::create_directories(root / "assets");
  writeText(root / "assets/first.tsd", "old first");
  writeText(root / "assets/second.tsd", "old second");
  writeText(root / "project.tsd", "old manifest");
}

static void requireOldProject(const std::filesystem::path &root)
{
  REQUIRE(readText(root / "assets/first.tsd") == "old first");
  REQUIRE(readText(root / "assets/second.tsd") == "old second");
  REQUIRE(readText(root / "project.tsd") == "old manifest");
}

static void requireNoTransactionFiles(const std::filesystem::path &root)
{
  for (const auto &entry :
      std::filesystem::recursive_directory_iterator(root)) {
    const auto filename = entry.path().filename().string();
    REQUIRE(filename.find(".stage-") == std::string::npos);
    REQUIRE(filename.find(".backup-") == std::string::npos);
  }
}

struct FailureRule
{
  AssetTransactionPhase phase{AssetTransactionPhase::Validation};
  int occurrence{1};
  std::string message{"injected failure"};
};

struct FailureInjector : AssetTransactionFailureInjector
{
  explicit FailureInjector(std::vector<FailureRule> rules);

  bool fail(AssetTransactionPhase phase,
      const std::filesystem::path &,
      std::string &message) override;

 private:
  std::vector<FailureRule> m_rules;
  std::map<AssetTransactionPhase, int> m_occurrences;
};

FailureInjector::FailureInjector(std::vector<FailureRule> rules)
    : m_rules(std::move(rules))
{}

bool FailureInjector::fail(AssetTransactionPhase phase,
    const std::filesystem::path &,
    std::string &message)
{
  const int occurrence = ++m_occurrences[phase];
  for (const auto &rule : m_rules) {
    if (rule.phase == phase && rule.occurrence == occurrence) {
      message = rule.message;
      return true;
    }
  }
  return false;
}

SCENARIO("project assets commit only after every file is staged",
    "[ProjectAssetTransaction]")
{
  const auto root = std::filesystem::temp_directory_path()
      / "tsd_project_asset_transaction_success";
  std::filesystem::remove_all(root);
  createOldProject(root);

  auto plan = twoAssetPlan(root);
  writeText(root / "assets/obsolete.tsd", "obsolete");
  writeText(root / "assets/unlisted.tsd", "unlisted");
  plan.removals.push_back("assets/obsolete.tsd");
  bool firstWriterSawOldProject = false;
  plan.assets.front().writer = [&](const std::filesystem::path &stage,
                                   std::string *) {
    firstWriterSawOldProject =
        readText(root / "assets/first.tsd") == "old first"
        && readText(root / "assets/second.tsd") == "old second"
        && readText(root / "project.tsd") == "old manifest";
    writeText(stage, "new first");
    return true;
  };

  AssetTransaction transaction;
  REQUIRE(transaction.commit(plan));
  REQUIRE(firstWriterSawOldProject);
  REQUIRE(readText(root / "assets/first.tsd") == "new first");
  REQUIRE(readText(root / "assets/second.tsd") == "new second");
  REQUIRE(readText(root / "project.tsd") == "new manifest");
  REQUIRE_FALSE(std::filesystem::exists(root / "assets/obsolete.tsd"));
  REQUIRE(readText(root / "assets/unlisted.tsd") == "unlisted");
  requireNoTransactionFiles(root);

  std::filesystem::remove_all(root);
}

SCENARIO("project asset validation protects unowned targets",
    "[ProjectAssetTransaction]")
{
  const auto root = std::filesystem::temp_directory_path()
      / "tsd_project_asset_transaction_collision";
  std::filesystem::remove_all(root);
  createOldProject(root);

  auto plan = twoAssetPlan(root);
  plan.assets.front().target = "assets/FIRST.tsd";
  plan.assets.front().ownedTarget.reset();
  bool writerCalled = false;
  plan.assets.front().writer = [&](const std::filesystem::path &,
                                   std::string *) {
    writerCalled = true;
    return true;
  };

  std::string error;
  AssetTransaction transaction;
  REQUIRE_FALSE(transaction.commit(plan, &error));
  REQUIRE(error.find("unowned target") != std::string::npos);
  REQUIRE_FALSE(writerCalled);
  requireOldProject(root);

  std::filesystem::remove_all(root);
}

SCENARIO("project asset transaction rolls back every commit phase",
    "[ProjectAssetTransaction]")
{
  const std::vector<FailureRule> failures = {
      {AssetTransactionPhase::Validation, 1, "validation failure"},
      {AssetTransactionPhase::Staging, 2, "staging failure"},
      {AssetTransactionPhase::Backup, 2, "backup failure"},
      {AssetTransactionPhase::AssetInstall, 2, "asset install failure"},
      {AssetTransactionPhase::ManifestInstall, 1, "manifest install failure"}};

  for (size_t i = 0; i < failures.size(); ++i) {
    CAPTURE(i);
    const auto root = std::filesystem::temp_directory_path()
        / ("tsd_project_asset_transaction_phase_" + std::to_string(i));
    std::filesystem::remove_all(root);
    createOldProject(root);

    auto plan = twoAssetPlan(root);
    FailureInjector injector({failures[i]});
    AssetTransaction transaction(&injector);
    std::string error;
    REQUIRE_FALSE(transaction.commit(plan, &error));
    INFO(error);
    REQUIRE(error.find(failures[i].message) != std::string::npos);
    requireOldProject(root);
    requireNoTransactionFiles(root);

    std::filesystem::remove_all(root);
  }
}

SCENARIO("project asset transaction reports rollback failures",
    "[ProjectAssetTransaction]")
{
  const auto root = std::filesystem::temp_directory_path()
      / "tsd_project_asset_transaction_rollback_failure";
  std::filesystem::remove_all(root);
  createOldProject(root);

  auto plan = twoAssetPlan(root);
  FailureInjector injector({
      {AssetTransactionPhase::AssetInstall, 2, "install stopped"},
      {AssetTransactionPhase::Rollback, 1, "restore blocked"},
  });
  AssetTransaction transaction(&injector);
  std::string error;
  REQUIRE_FALSE(transaction.commit(plan, &error));
  INFO(error);
  REQUIRE(error.find("install stopped") != std::string::npos);
  REQUIRE(error.find("rollback failures") != std::string::npos);
  REQUIRE(error.find("restore blocked") != std::string::npos);

  std::filesystem::remove_all(root);
}

} // namespace tsd::scivis_studio::test
