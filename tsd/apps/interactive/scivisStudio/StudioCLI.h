// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Project.h"

#include <filesystem>
#include <iosfwd>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tsd::scivis_studio {

// The scivisStudioCLI noun-verb surface: observe project contents, create
// datasets (including Declared Datasets authored without reading any source
// file), revise File Animation Source Lists, and flip Dataset Residency —
// all headless.
enum class StudioCommand
{
  None,
  ProjectInit,
  ProjectShow,
  DatasetList,
  DatasetShow,
  DatasetCreateStatic,
  DatasetCreateFileAnimation,
  DatasetSourcesSet,
  DatasetSourcesAppend,
  DatasetSourcesRemap,
  DatasetRename,
  DatasetRemove,
  DatasetLoad,
  DatasetUnload,
};

struct StudioCommandLine
{
  StudioCommand command{StudioCommand::None};
  std::filesystem::path projectDirectory;
  // Dataset addressing: exact Dataset ID first, then unique case-insensitive
  // name.
  std::string dataset;
  std::string name; // --name, or the rename target
  std::string importerType; // --importer
  bool declare{false}; // --declare: create a Declared Dataset (ADR 0014)
  bool keepAsset{false}; // --keep-asset on remove
  // Mirrors FileAnimationDatasetOptions::setActiveShotFrameCount; opt out
  // with --no-shot-frame-count.
  bool setShotFrameCount{true};
  // Source-list entries from positional arguments; entries also come from
  // --paths-from FILE or stdin (see gatherSourceListEntries).
  std::vector<std::string> paths;
  std::optional<std::filesystem::path> pathsFrom;
  std::optional<std::string> remapFrom; // --from
  std::optional<std::string> remapTo; // --to
  bool showHelp{false};
};

bool parseStudioCommandLine(const std::vector<std::string> &args,
    StudioCommandLine &commandLine,
    std::string &error);

std::string studioCLIUsage(const std::string &programName);

// Exact Dataset ID match first, then case-insensitive unique name; returns
// nullptr with an error listing the candidates otherwise.
Dataset *resolveDatasetSelector(
    Project &project, const std::string &selector, std::string &error);

// The command's source-list entries: positional paths win, then --paths-from
// FILE, then lines read from input (one entry per line, trimmed, blank lines
// skipped — the Source List File rules of ADR 0013).
bool gatherSourceListEntries(const StudioCommandLine &commandLine,
    std::istream &input,
    std::vector<std::string> &entries,
    std::string &error);

// Literal prefix substitution on raw entries; entries without the prefix pass
// through unchanged and nothing is validated — entries stay opaque
// (ADR 0007). Returns the number of entries changed.
size_t remapSourceListEntries(std::vector<std::string> &entries,
    const std::string &fromPrefix,
    const std::string &toPrefix);

// Command results are plain structs with plain-text formatters so a
// structured output format can be added later without rewriting the command
// logic. stdout carries only these results; diagnostics go to the log.

struct DatasetSummary
{
  DatasetID id;
  std::string name;
  std::string sourceKind;
  std::string importerType;
  std::string residency;
  std::string status;
};

struct ShotSummary
{
  ShotID id;
  std::string name;
  int frameCount{0};
  bool active{false};
};

struct ProjectShowResult
{
  std::string name;
  std::string directory;
  std::vector<ShotSummary> shots;
  std::vector<DatasetSummary> datasets;
  size_t cameraRigCount{0};
  size_t lightRigCount{0};
};

struct DatasetListResult
{
  std::vector<DatasetSummary> datasets;
};

struct DatasetShowResult
{
  DatasetSummary summary;
  std::string provenanceSourcePath;
  std::vector<std::pair<std::string, std::string>> importerSettings;
  std::string assetFile;
  std::string sourcesFile;
  bool legacyEmbeddedSourceList{false};
  bool sourceListReadable{false};
  std::string sourceListError;
  std::vector<std::string> sourceList;
};

struct ProjectInitResult
{
  std::string name;
  std::string directory;
};

struct DatasetCreateResult
{
  DatasetSummary dataset;
  size_t frameCount{0};
  bool declared{false};
};

struct DatasetRenameResult
{
  DatasetID id;
  std::string oldName;
  std::string newName;
};

struct DatasetRemoveResult
{
  DatasetID id;
  std::string name;
  bool assetKept{false};
};

struct DatasetResidencyResult
{
  DatasetID id;
  std::string name;
  std::string residency;
  bool changed{false};
  bool materialized{false};
};

struct DatasetSourcesResult
{
  DatasetID id;
  std::string name;
  std::string sourcesFile;
  size_t entryCount{0};
  size_t changedEntries{0};
  bool migratedLegacy{false};
};

DatasetSummary makeDatasetSummary(const Dataset &dataset);
ProjectShowResult makeProjectShowResult(const Project &project);
DatasetListResult makeDatasetListResult(const Project &project);

std::string formatProjectShow(const ProjectShowResult &result);
std::string formatDatasetList(const DatasetListResult &result);
std::string formatDatasetShow(const DatasetShowResult &result);
std::string formatProjectInit(const ProjectInitResult &result);
std::string formatDatasetCreate(const DatasetCreateResult &result);
std::string formatDatasetRename(const DatasetRenameResult &result);
std::string formatDatasetRemove(const DatasetRemoveResult &result);
std::string formatDatasetResidency(const DatasetResidencyResult &result);
std::string formatDatasetSources(const DatasetSourcesResult &result);

// Execute a parsed command end to end (open -> mutate -> save -> exit for
// mutations; observe commands never write) and return the process exit code:
// 0 means any mutation was persisted, nonzero means the project on disk is
// untouched. Results are written to output; input feeds stdin-supplied
// source-list entries.
int runStudioCommand(const StudioCommandLine &commandLine,
    std::istream &input,
    std::ostream &output);

} // namespace tsd::scivis_studio
