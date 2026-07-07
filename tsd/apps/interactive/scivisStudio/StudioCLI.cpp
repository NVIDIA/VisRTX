// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "StudioCLI.h"

#include "DatasetIO.h"
#include "ProjectContext.h"
#include "ProjectPersistence.h"

#include "tsd/app/Context.h"
#include "tsd/core/Logging.hpp"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <sstream>

namespace tsd::scivis_studio {

namespace {

bool namesEqualCaseInsensitive(const std::string &a, const std::string &b)
{
  if (a.size() != b.size())
    return false;
  return std::equal(a.begin(), a.end(), b.begin(), [](char x, char y) {
    return std::tolower(static_cast<unsigned char>(x))
        == std::tolower(static_cast<unsigned char>(y));
  });
}

std::string trimmed(const std::string &line)
{
  const auto first = line.find_first_not_of(" \t\r\n");
  if (first == std::string::npos)
    return {};
  const auto last = line.find_last_not_of(" \t\r\n");
  return line.substr(first, last - first + 1);
}

std::filesystem::path datasetAssetFile(
    const Project &project, const Dataset &dataset)
{
  return project.projectDirectory / "datasets"
      / (dataset.persistedName + ".tsd");
}

int commandFailed(const std::string &message)
{
  tsd::core::logError("[scivisStudioCLI] %s", message.c_str());
  return 1;
}

// One headless TSD context per command invocation, opened in bookkeeping
// mode: no dataset runtime is built and recorded residency round-trips
// unchanged.
struct BookkeepingSession
{
  tsd::app::Context appContext;
  ProjectContext context{&appContext};

  bool open(const std::filesystem::path &directory, std::string &error)
  {
    ProjectOpenOptions options;
    options.bookkeeping = true;
    return context.openProject(
        directory, nullptr, nullptr, nullptr, &error, options);
  }
};

bool persistProject(ProjectContext &context, std::string &error)
{
  return context.saveProject(
      context.project().projectDirectory, nullptr, "", nullptr, &error);
}

bool parseImporterType(const std::string &importerString,
    tsd::io::ImporterType &importerType,
    std::string &error)
{
  if (importerString.empty()) {
    error = "--importer is required";
    return false;
  }
  importerType = importerTypeFromString(importerString);
  if (importerType == tsd::io::ImporterType::NONE) {
    error = "unknown importer type: " + importerString;
    return false;
  }
  return true;
}

std::string defaultDatasetName(
    const std::string &explicitName, const std::string &firstEntry)
{
  if (!explicitName.empty())
    return explicitName;
  // Purely textual: the stem of the first entry, never a filesystem probe.
  return std::filesystem::path(firstEntry).stem().string();
}

std::string padded(const std::string &text, size_t width)
{
  if (text.size() >= width)
    return text;
  return text + std::string(width - text.size(), ' ');
}

std::string formatDatasetTable(const std::vector<DatasetSummary> &datasets)
{
  std::ostringstream out;
  const std::vector<std::string> header = {
      "ID", "NAME", "KIND", "IMPORTER", "RESIDENCY", "STATUS"};
  std::vector<size_t> widths(header.size());
  const auto columns = [](const DatasetSummary &d) {
    return std::vector<std::string>{
        d.id, d.name, d.sourceKind, d.importerType, d.residency, d.status};
  };
  for (size_t i = 0; i < header.size(); ++i)
    widths[i] = header[i].size();
  for (const auto &dataset : datasets) {
    const auto row = columns(dataset);
    for (size_t i = 0; i < row.size(); ++i)
      widths[i] = std::max(widths[i], row[i].size());
  }
  const auto writeRow = [&](const std::vector<std::string> &row) {
    for (size_t i = 0; i < row.size(); ++i)
      out << (i ? "  " : "") << padded(row[i], i + 1 < row.size() ? widths[i] : 0);
    out << '\n';
  };
  writeRow(header);
  for (const auto &dataset : datasets)
    writeRow(columns(dataset));
  return out.str();
}

} // namespace

bool parseStudioCommandLine(const std::vector<std::string> &args,
    StudioCommandLine &commandLine,
    std::string &error)
{
  commandLine = {};
  error.clear();

  std::vector<std::string> positionals;
  for (size_t i = 1; i < args.size(); ++i) {
    const auto &arg = args[i];
    const auto valueFor = [&](const char *option, std::string &value) {
      if (i + 1 >= args.size()) {
        error = std::string(option) + " requires a value";
        return false;
      }
      value = args[++i];
      return true;
    };

    if (arg == "-h" || arg == "--help") {
      commandLine.showHelp = true;
      return true;
    } else if (arg == "--name") {
      if (!valueFor("--name", commandLine.name))
        return false;
    } else if (arg == "--importer") {
      if (!valueFor("--importer", commandLine.importerType))
        return false;
    } else if (arg == "--declare") {
      commandLine.declare = true;
    } else if (arg == "--keep-asset") {
      commandLine.keepAsset = true;
    } else if (arg == "--no-shot-frame-count") {
      commandLine.setShotFrameCount = false;
    } else if (arg == "--paths-from") {
      std::string value;
      if (!valueFor("--paths-from", value))
        return false;
      commandLine.pathsFrom = value;
    } else if (arg == "--from") {
      std::string value;
      if (!valueFor("--from", value))
        return false;
      commandLine.remapFrom = value;
    } else if (arg == "--to") {
      std::string value;
      if (!valueFor("--to", value))
        return false;
      commandLine.remapTo = value;
    } else if (!arg.empty() && arg.front() == '-') {
      error = "unknown option: " + arg;
      return false;
    } else {
      positionals.push_back(arg);
    }
  }

  size_t next = 0;
  const auto positional = [&](const char *what, std::string &value) {
    if (next >= positionals.size()) {
      error = std::string("missing ") + what;
      return false;
    }
    value = positionals[next++];
    return true;
  };

  std::string noun;
  if (!positional("command (project|dataset)", noun))
    return false;

  std::string verb;
  if (noun == "project") {
    if (!positional("project verb (init|show)", verb))
      return false;
    if (verb == "init")
      commandLine.command = StudioCommand::ProjectInit;
    else if (verb == "show")
      commandLine.command = StudioCommand::ProjectShow;
    else {
      error = "unknown project command: " + verb;
      return false;
    }
  } else if (noun == "dataset") {
    if (!positional(
            "dataset verb (list|show|create|sources|rename|remove|load|unload)",
            verb))
      return false;
    if (verb == "list") {
      commandLine.command = StudioCommand::DatasetList;
    } else if (verb == "show") {
      commandLine.command = StudioCommand::DatasetShow;
    } else if (verb == "create") {
      std::string kind;
      if (!positional("dataset kind (static|file-animation)", kind))
        return false;
      if (kind == "static")
        commandLine.command = StudioCommand::DatasetCreateStatic;
      else if (kind == "file-animation")
        commandLine.command = StudioCommand::DatasetCreateFileAnimation;
      else {
        error = "unknown dataset kind: " + kind;
        return false;
      }
    } else if (verb == "sources") {
      std::string operation;
      if (!positional("sources operation (set|append|remap)", operation))
        return false;
      if (operation == "set")
        commandLine.command = StudioCommand::DatasetSourcesSet;
      else if (operation == "append")
        commandLine.command = StudioCommand::DatasetSourcesAppend;
      else if (operation == "remap")
        commandLine.command = StudioCommand::DatasetSourcesRemap;
      else {
        error = "unknown sources operation: " + operation;
        return false;
      }
    } else if (verb == "rename") {
      commandLine.command = StudioCommand::DatasetRename;
    } else if (verb == "remove") {
      commandLine.command = StudioCommand::DatasetRemove;
    } else if (verb == "load") {
      commandLine.command = StudioCommand::DatasetLoad;
    } else if (verb == "unload") {
      commandLine.command = StudioCommand::DatasetUnload;
    } else {
      error = "unknown dataset command: " + verb;
      return false;
    }
  } else {
    error = "unknown command: " + noun;
    return false;
  }

  std::string directory;
  if (!positional("project directory", directory))
    return false;
  commandLine.projectDirectory = directory;

  const auto needsDatasetSelector = [&]() {
    switch (commandLine.command) {
    case StudioCommand::DatasetShow:
    case StudioCommand::DatasetSourcesSet:
    case StudioCommand::DatasetSourcesAppend:
    case StudioCommand::DatasetSourcesRemap:
    case StudioCommand::DatasetRename:
    case StudioCommand::DatasetRemove:
    case StudioCommand::DatasetLoad:
    case StudioCommand::DatasetUnload:
      return true;
    default:
      return false;
    }
  };
  if (needsDatasetSelector() && !positional("dataset", commandLine.dataset))
    return false;

  if (commandLine.command == StudioCommand::DatasetRename
      && !positional("new dataset name", commandLine.name))
    return false;

  // Remaining positionals are source paths / Source List entries.
  commandLine.paths.assign(
      positionals.begin() + next, positionals.end());
  next = positionals.size();

  switch (commandLine.command) {
  case StudioCommand::DatasetCreateStatic:
    if (commandLine.paths.size() != 1) {
      error = "dataset create static requires exactly one source path";
      return false;
    }
    break;
  case StudioCommand::DatasetCreateFileAnimation:
  case StudioCommand::DatasetSourcesSet:
  case StudioCommand::DatasetSourcesAppend:
    break;
  case StudioCommand::DatasetSourcesRemap:
    if (!commandLine.remapFrom || !commandLine.remapTo) {
      error = "dataset sources remap requires --from and --to";
      return false;
    }
    if (!commandLine.paths.empty()) {
      error = "dataset sources remap takes no path arguments";
      return false;
    }
    break;
  default:
    if (!commandLine.paths.empty()) {
      error = "unexpected argument: " + commandLine.paths.front();
      return false;
    }
    break;
  }

  // Options only make sense on the commands they belong to; anything else is
  // a grammar error rather than a silently ignored flag.
  const bool isCreate =
      commandLine.command == StudioCommand::DatasetCreateStatic
      || commandLine.command == StudioCommand::DatasetCreateFileAnimation;
  const bool takesEntries =
      commandLine.command == StudioCommand::DatasetCreateFileAnimation
      || commandLine.command == StudioCommand::DatasetSourcesSet
      || commandLine.command == StudioCommand::DatasetSourcesAppend;
  if (commandLine.declare
      && commandLine.command != StudioCommand::DatasetCreateFileAnimation) {
    error = "--declare only applies to dataset create file-animation";
    return false;
  }
  if (commandLine.keepAsset
      && commandLine.command != StudioCommand::DatasetRemove) {
    error = "--keep-asset only applies to dataset remove";
    return false;
  }
  if (!commandLine.importerType.empty() && !isCreate) {
    error = "--importer only applies to dataset create";
    return false;
  }
  if (commandLine.pathsFrom && !takesEntries) {
    error = "--paths-from only applies to commands that take source-list "
            "entries";
    return false;
  }
  if (!commandLine.setShotFrameCount
      && commandLine.command != StudioCommand::DatasetCreateFileAnimation) {
    error = "--no-shot-frame-count only applies to dataset create "
            "file-animation";
    return false;
  }
  if ((commandLine.remapFrom || commandLine.remapTo)
      && commandLine.command != StudioCommand::DatasetSourcesRemap) {
    error = "--from/--to only apply to dataset sources remap";
    return false;
  }
  if (!commandLine.name.empty() && !isCreate
      && commandLine.command != StudioCommand::ProjectInit
      && commandLine.command != StudioCommand::DatasetRename) {
    error = "--name does not apply to this command";
    return false;
  }

  return true;
}

std::string studioCLIUsage(const std::string &programName)
{
  std::ostringstream out;
  out << "usage: " << programName << " <command> [options]\n"
      << "\n"
      << "observe:\n"
      << "  project show <project-dir>\n"
      << "  dataset list <project-dir>\n"
      << "  dataset show <project-dir> <dataset>\n"
      << "\n"
      << "mutate (open -> mutate -> save -> exit; exit 0 means persisted):\n"
      << "  project init <project-dir> [--name NAME]\n"
      << "  dataset create static <project-dir> --importer TYPE [--name NAME] <path>\n"
      << "  dataset create file-animation <project-dir> --importer TYPE [--name NAME]\n"
      << "      [--declare] [--no-shot-frame-count] [--paths-from FILE] [entry...]\n"
      << "  dataset sources set <project-dir> <dataset> [--paths-from FILE] [entry...]\n"
      << "  dataset sources append <project-dir> <dataset> [--paths-from FILE] [entry...]\n"
      << "  dataset sources remap <project-dir> <dataset> --from PREFIX --to PREFIX\n"
      << "  dataset rename <project-dir> <dataset> <new-name>\n"
      << "  dataset remove <project-dir> <dataset> [--keep-asset]\n"
      << "  dataset load <project-dir> <dataset>\n"
      << "  dataset unload <project-dir> <dataset>\n"
      << "\n"
      << "<dataset> is a Dataset ID, or a dataset name when it matches one\n"
      << "dataset case-insensitively. Source-list entries come from positional\n"
      << "arguments, --paths-from FILE, or stdin (one per line); entries are\n"
      << "opaque and never validated. --declare creates a Declared Dataset\n"
      << "without reading any source file; the first successful 'dataset load'\n"
      << "materializes it.\n";
  return out.str();
}

Dataset *resolveDatasetSelector(
    Project &project, const std::string &selector, std::string &error)
{
  error.clear();
  if (auto *byId = project::findDataset(project, selector))
    return byId;

  std::vector<Dataset *> matches;
  for (auto &dataset : project.datasets) {
    if (namesEqualCaseInsensitive(dataset.name, selector))
      matches.push_back(&dataset);
  }
  if (matches.size() == 1)
    return matches.front();

  std::ostringstream out;
  if (matches.empty()) {
    out << "no dataset matches '" << selector << "'";
    if (!project.datasets.empty()) {
      out << "; datasets:";
      for (const auto &dataset : project.datasets)
        out << "\n  " << dataset.id << "  " << dataset.name;
    }
  } else {
    out << "'" << selector << "' matches multiple datasets:";
    for (const auto *dataset : matches)
      out << "\n  " << dataset->id << "  " << dataset->name;
  }
  error = out.str();
  return nullptr;
}

bool gatherSourceListEntries(const StudioCommandLine &commandLine,
    std::istream &input,
    std::vector<std::string> &entries,
    std::string &error)
{
  entries.clear();
  error.clear();

  if (!commandLine.paths.empty() && commandLine.pathsFrom) {
    error = "pass source-list entries either as arguments or via "
            "--paths-from, not both";
    return false;
  }

  if (!commandLine.paths.empty()) {
    entries = commandLine.paths;
    return true;
  }

  if (commandLine.pathsFrom) {
    std::vector<DatasetSourceFile> sourceList;
    if (!readSourceListFile(*commandLine.pathsFrom, sourceList, &error))
      return false;
    entries.reserve(sourceList.size());
    for (const auto &source : sourceList)
      entries.push_back(source.path);
    return true;
  }

  std::string line;
  while (std::getline(input, line)) {
    auto entry = trimmed(line);
    if (!entry.empty())
      entries.push_back(std::move(entry));
  }
  if (entries.empty()) {
    error = "no source-list entries given (arguments, --paths-from, or stdin)";
    return false;
  }
  return true;
}

size_t remapSourceListEntries(std::vector<std::string> &entries,
    const std::string &fromPrefix,
    const std::string &toPrefix)
{
  size_t changed = 0;
  for (auto &entry : entries) {
    if (entry.compare(0, fromPrefix.size(), fromPrefix) != 0)
      continue;
    entry = toPrefix + entry.substr(fromPrefix.size());
    ++changed;
  }
  return changed;
}

DatasetSummary makeDatasetSummary(const Dataset &dataset)
{
  DatasetSummary summary;
  summary.id = dataset.id;
  summary.name = dataset.name;
  summary.sourceKind = dataset::toString(dataset.sourceKind);
  summary.importerType = dataset.importerType;
  summary.residency = dataset::toString(dataset.residency);
  summary.status = dataset::displayStatus(dataset);
  return summary;
}

namespace {

// The manifest records only id, name, and residency; kind and importer live
// in the dataset asset, which a bookkeeping open never reads. Displaying an
// unhydrated record reads the asset's metadata on demand.
void enrichSummaryFromAsset(
    const Project &project, const Dataset &dataset, DatasetSummary &summary)
{
  if (dataset.importerType != "NONE" || dataset.persistedName.empty())
    return;
  const auto validation =
      validateDatasetArchiveFile(datasetAssetFile(project, dataset));
  if (!validation.ok)
    return;
  summary.sourceKind = dataset::toString(validation.dataset.sourceKind);
  summary.importerType = validation.dataset.importerType;
}

} // namespace

ProjectShowResult makeProjectShowResult(const Project &project)
{
  ProjectShowResult result;
  result.name = project.name;
  result.directory = project.projectDirectory.string();
  result.shots.reserve(project.shots.size());
  for (const auto &shot : project.shots) {
    result.shots.push_back({shot.id,
        shot.name,
        shot.frameCount,
        shot.id == project.activeShotId});
  }
  result.datasets.reserve(project.datasets.size());
  for (const auto &dataset : project.datasets) {
    auto summary = makeDatasetSummary(dataset);
    enrichSummaryFromAsset(project, dataset, summary);
    result.datasets.push_back(std::move(summary));
  }
  result.cameraRigCount = project.cameraRigs.size();
  result.lightRigCount = project.lightRigs.size();
  return result;
}

DatasetListResult makeDatasetListResult(const Project &project)
{
  DatasetListResult result;
  result.datasets.reserve(project.datasets.size());
  for (const auto &dataset : project.datasets) {
    auto summary = makeDatasetSummary(dataset);
    enrichSummaryFromAsset(project, dataset, summary);
    result.datasets.push_back(std::move(summary));
  }
  return result;
}

std::string formatProjectShow(const ProjectShowResult &result)
{
  std::ostringstream out;
  out << "Project '" << result.name << "' at " << result.directory << '\n';
  out << "Shots (" << result.shots.size() << "):\n";
  for (const auto &shot : result.shots) {
    out << "  " << shot.id << "  " << shot.name << "  frames: "
        << shot.frameCount << (shot.active ? "  [active]" : "") << '\n';
  }
  out << "Datasets (" << result.datasets.size() << "):\n";
  if (!result.datasets.empty()) {
    std::istringstream table(formatDatasetTable(result.datasets));
    std::string line;
    while (std::getline(table, line))
      out << "  " << line << '\n';
  }
  out << "Camera Rigs: " << result.cameraRigCount << '\n';
  out << "Light Rigs: " << result.lightRigCount << '\n';
  return out.str();
}

std::string formatDatasetList(const DatasetListResult &result)
{
  if (result.datasets.empty())
    return "no datasets\n";
  return formatDatasetTable(result.datasets);
}

std::string formatDatasetShow(const DatasetShowResult &result)
{
  std::ostringstream out;
  out << "Dataset " << result.summary.id << " '" << result.summary.name
      << "'\n";
  out << "  Kind: " << result.summary.sourceKind << '\n';
  out << "  Importer: " << result.summary.importerType << '\n';
  for (const auto &setting : result.importerSettings)
    out << "  Importer setting: " << setting.first << " = " << setting.second
        << '\n';
  out << "  Residency: " << result.summary.residency << '\n';
  out << "  Status: " << result.summary.status << '\n';
  if (!result.assetFile.empty())
    out << "  Asset: " << result.assetFile << '\n';
  if (!result.provenanceSourcePath.empty())
    out << "  Provenance: " << result.provenanceSourcePath << '\n';
  if (result.summary.sourceKind
      == std::string(dataset::toString(DatasetSourceKind::FileAnimation))) {
    if (result.legacyEmbeddedSourceList)
      out << "  Source List: embedded (legacy; migrates on next edit or save)\n";
    else if (!result.sourcesFile.empty())
      out << "  Source List File: " << result.sourcesFile << '\n';
    if (result.sourceListReadable) {
      out << "  Source List (" << result.sourceList.size() << " entries):\n";
      for (const auto &entry : result.sourceList)
        out << "    " << entry << '\n';
    } else if (!result.sourceListError.empty()) {
      out << "  Source List: unreadable (" << result.sourceListError << ")\n";
    }
  }
  return out.str();
}

std::string formatProjectInit(const ProjectInitResult &result)
{
  std::ostringstream out;
  out << "Initialized project '" << result.name << "' at " << result.directory
      << '\n';
  return out.str();
}

std::string formatDatasetCreate(const DatasetCreateResult &result)
{
  std::ostringstream out;
  out << "Created ";
  if (result.declared)
    out << "declared ";
  out << (result.dataset.sourceKind
                 == std::string(dataset::toString(DatasetSourceKind::Static))
             ? "static"
             : "file-animation")
      << " dataset " << result.dataset.id << " '" << result.dataset.name
      << "'";
  if (result.frameCount)
    out << " (" << result.frameCount << " frames)";
  out << '\n';
  return out.str();
}

std::string formatDatasetRename(const DatasetRenameResult &result)
{
  std::ostringstream out;
  out << "Renamed dataset " << result.id << " from '" << result.oldName
      << "' to '" << result.newName << "'\n";
  return out.str();
}

std::string formatDatasetRemove(const DatasetRemoveResult &result)
{
  std::ostringstream out;
  out << "Removed dataset " << result.id << " '" << result.name << "'"
      << (result.assetKept ? " (asset kept)" : "") << '\n';
  return out.str();
}

std::string formatDatasetResidency(const DatasetResidencyResult &result)
{
  std::ostringstream out;
  out << "Dataset " << result.id << " '" << result.name << "' is "
      << (result.changed ? "now " : "already ") << result.residency;
  if (result.materialized)
    out << " (materialized)";
  out << '\n';
  return out.str();
}

std::string formatDatasetSources(const DatasetSourcesResult &result)
{
  std::ostringstream out;
  out << "Wrote " << result.entryCount << " entries to " << result.sourcesFile
      << " for dataset " << result.id << " '" << result.name << "'";
  if (result.changedEntries)
    out << " (" << result.changedEntries << " remapped)";
  if (result.migratedLegacy)
    out << " (migrated legacy embedded source list)";
  out << '\n';
  return out.str();
}

namespace {

int runProjectInit(
    const StudioCommandLine &commandLine, std::ostream &output)
{
  tsd::app::Context appContext;
  ProjectContext context(&appContext);
  context.createUnsavedProject();
  if (!commandLine.name.empty())
    context.project().name = commandLine.name;

  std::string error;
  if (!context.saveProject(
          commandLine.projectDirectory, nullptr, "", nullptr, &error))
    return commandFailed(error);

  output << formatProjectInit(
      {context.project().name, commandLine.projectDirectory.string()});
  return 0;
}

int runProjectShow(const StudioCommandLine &commandLine, std::ostream &output)
{
  BookkeepingSession session;
  std::string error;
  if (!session.open(commandLine.projectDirectory, error))
    return commandFailed(error);
  auto &context = session.context;
  output << formatProjectShow(makeProjectShowResult(context.project()));
  return 0;
}

int runDatasetList(const StudioCommandLine &commandLine, std::ostream &output)
{
  BookkeepingSession session;
  std::string error;
  if (!session.open(commandLine.projectDirectory, error))
    return commandFailed(error);
  auto &context = session.context;
  output << formatDatasetList(makeDatasetListResult(context.project()));
  return 0;
}

int runDatasetShow(const StudioCommandLine &commandLine, std::ostream &output)
{
  BookkeepingSession session;
  std::string error;
  if (!session.open(commandLine.projectDirectory, error))
    return commandFailed(error);
  auto &context = session.context;

  auto &project = context.project();
  auto *dataset = resolveDatasetSelector(project, commandLine.dataset, error);
  if (!dataset)
    return commandFailed(error);

  DatasetShowResult result;
  result.summary = makeDatasetSummary(*dataset);

  // An unhydrated record only knows what the manifest records; the asset's
  // metadata supplies kind, importer, settings, and provenance.
  const Dataset *details = dataset;
  DatasetAssetValidationResult validation;
  if (!dataset->persistedName.empty()) {
    const auto assetFile = datasetAssetFile(project, *dataset);
    result.assetFile = assetFile.string();
    validation = validateDatasetArchiveFile(assetFile);
    if (validation.ok && dataset->importerType == "NONE") {
      details = &validation.dataset;
      result.summary.sourceKind =
          dataset::toString(validation.dataset.sourceKind);
      result.summary.importerType = validation.dataset.importerType;
    }
  }

  result.provenanceSourcePath =
      details->sourceKind == DatasetSourceKind::Static
      ? details->source.sourcePath
      : std::string();
  for (const auto &setting : details->source.importerSettings)
    result.importerSettings.emplace_back(setting.first, setting.second);

  if (!dataset->persistedName.empty()
      && details->sourceKind == DatasetSourceKind::FileAnimation) {
    const auto assetFile = datasetAssetFile(project, *dataset);
    result.legacyEmbeddedSourceList =
        validation.ok && validation.dataset.pendingSourceListMigration;
    if (!result.legacyEmbeddedSourceList)
      result.sourcesFile = sourceListFilePath(assetFile).string();
    result.sourceListReadable = readDatasetSourceListEntries(
        assetFile, result.sourceList, &result.sourceListError);
  }

  output << formatDatasetShow(result);
  return 0;
}

int runDatasetCreateStatic(
    const StudioCommandLine &commandLine, std::ostream &output)
{
  std::string error;
  tsd::io::ImporterType importerType{tsd::io::ImporterType::NONE};
  if (!parseImporterType(commandLine.importerType, importerType, error))
    return commandFailed(error);

  BookkeepingSession session;
  if (!session.open(commandLine.projectDirectory, error))
    return commandFailed(error);
  auto &context = session.context;

  const auto &sourcePath = commandLine.paths.front();
  auto *dataset = context.addStaticDataset(
      defaultDatasetName(commandLine.name, sourcePath),
      sourcePath,
      importerType);
  if (!dataset || dataset->status != DatasetStatus::Available) {
    return commandFailed("static dataset import failed for '" + sourcePath
        + "' (see log for details); project untouched");
  }

  DatasetCreateResult result;
  result.dataset = makeDatasetSummary(*dataset);
  if (!persistProject(context, error))
    return commandFailed(error);
  output << formatDatasetCreate(result);
  return 0;
}

int runDatasetCreateFileAnimation(const StudioCommandLine &commandLine,
    std::istream &input,
    std::ostream &output)
{
  std::string error;
  tsd::io::ImporterType importerType{tsd::io::ImporterType::NONE};
  if (!parseImporterType(commandLine.importerType, importerType, error))
    return commandFailed(error);

  std::vector<std::string> entries;
  if (!gatherSourceListEntries(commandLine, input, entries, error))
    return commandFailed(error);
  if (entries.empty())
    return commandFailed("file-animation datasets require source entries");

  BookkeepingSession session;
  if (!session.open(commandLine.projectDirectory, error))
    return commandFailed(error);
  auto &context = session.context;

  FileAnimationDatasetOptions options;
  options.setActiveShotFrameCount = commandLine.setShotFrameCount;
  const auto name = defaultDatasetName(commandLine.name, entries.front());

  Dataset *dataset = nullptr;
  if (commandLine.declare) {
    dataset =
        context.addDeclaredFileAnimationDataset(name, entries, importerType, options);
    if (!dataset)
      return commandFailed("failed to declare file-animation dataset");
  } else {
    // Default creation imports eagerly and fails loudly when the data is
    // absent; declaration is an explicit mode, never a fallback (ADR 0014).
    std::vector<std::filesystem::path> sourcePaths(
        entries.begin(), entries.end());
    dataset = context.addFileAnimationDataset(
        name, sourcePaths, importerType, options);
    if (!dataset || dataset->status != DatasetStatus::Available) {
      return commandFailed(
          "file-animation dataset import failed (see log for details); "
          "project untouched");
    }
  }

  DatasetCreateResult result;
  result.dataset = makeDatasetSummary(*dataset);
  result.frameCount = dataset->sourceFiles.size();
  result.declared = commandLine.declare;
  if (!persistProject(context, error))
    return commandFailed(error);
  output << formatDatasetCreate(result);
  return 0;
}

int runDatasetSources(const StudioCommandLine &commandLine,
    std::istream &input,
    std::ostream &output)
{
  BookkeepingSession session;
  std::string error;
  if (!session.open(commandLine.projectDirectory, error))
    return commandFailed(error);
  auto &context = session.context;

  auto &project = context.project();
  auto *dataset = resolveDatasetSelector(project, commandLine.dataset, error);
  if (!dataset)
    return commandFailed(error);
  if (dataset->persistedName.empty())
    return commandFailed("dataset '" + dataset->name + "' has no saved asset");

  // The asset's own metadata is authoritative for its kind: an unhydrated
  // record does not know it (the manifest records only inventory).
  const auto assetFile = datasetAssetFile(project, *dataset);
  const auto validation = validateDatasetArchiveFile(assetFile);
  if (!validation.ok)
    return commandFailed(validation.error);
  if (validation.dataset.sourceKind != DatasetSourceKind::FileAnimation) {
    return commandFailed(
        "dataset '" + dataset->name + "' is not a file-animation dataset");
  }
  const bool legacy = validation.dataset.pendingSourceListMigration;

  std::vector<std::string> entries;
  size_t changed = 0;
  switch (commandLine.command) {
  case StudioCommand::DatasetSourcesSet:
    if (!gatherSourceListEntries(commandLine, input, entries, error))
      return commandFailed(error);
    break;
  case StudioCommand::DatasetSourcesAppend: {
    if (!readDatasetSourceListEntries(assetFile, entries, &error))
      return commandFailed(error);
    std::vector<std::string> appended;
    if (!gatherSourceListEntries(commandLine, input, appended, error))
      return commandFailed(error);
    entries.insert(entries.end(), appended.begin(), appended.end());
    break;
  }
  case StudioCommand::DatasetSourcesRemap:
    if (!readDatasetSourceListEntries(assetFile, entries, &error))
      return commandFailed(error);
    changed = remapSourceListEntries(
        entries, *commandLine.remapFrom, *commandLine.remapTo);
    break;
  default:
    return commandFailed("unsupported sources operation");
  }

  if (!writeDatasetSourceListEdit(assetFile, entries, &error))
    return commandFailed(error);

  DatasetSourcesResult result;
  result.id = dataset->id;
  result.name = dataset->name;
  result.sourcesFile = sourceListFilePath(assetFile).string();
  result.entryCount = entries.size();
  result.changedEntries = changed;
  result.migratedLegacy = legacy;
  output << formatDatasetSources(result);
  return 0;
}

int runDatasetRename(
    const StudioCommandLine &commandLine, std::ostream &output)
{
  BookkeepingSession session;
  std::string error;
  if (!session.open(commandLine.projectDirectory, error))
    return commandFailed(error);
  auto &context = session.context;

  auto *dataset =
      resolveDatasetSelector(context.project(), commandLine.dataset, error);
  if (!dataset)
    return commandFailed(error);

  DatasetRenameResult result;
  result.id = dataset->id;
  result.oldName = dataset->name;
  result.newName = commandLine.name;
  if (!context.renameDataset(dataset->id, commandLine.name, &error))
    return commandFailed(error);
  if (!persistProject(context, error))
    return commandFailed(error);
  output << formatDatasetRename(result);
  return 0;
}

int runDatasetRemove(
    const StudioCommandLine &commandLine, std::ostream &output)
{
  BookkeepingSession session;
  std::string error;
  if (!session.open(commandLine.projectDirectory, error))
    return commandFailed(error);
  auto &context = session.context;

  auto &project = context.project();
  auto *dataset = resolveDatasetSelector(project, commandLine.dataset, error);
  if (!dataset)
    return commandFailed(error);

  DatasetRemoveResult result;
  result.id = dataset->id;
  result.name = dataset->name;
  result.assetKept = commandLine.keepAsset;
  const auto assetFile = dataset->persistedName.empty()
      ? std::filesystem::path()
      : datasetAssetFile(project, *dataset);

  // Keep the asset files through the save so a failed exit leaves the project
  // fully untouched; the pair is deleted only after the removal persisted.
  if (!context.removeDataset(result.id, /* keepAssetFile = */ true, &error))
    return commandFailed(error);
  if (!persistProject(context, error))
    return commandFailed(error);

  if (!commandLine.keepAsset && !assetFile.empty()) {
    const auto removeAssetFile = [](const std::filesystem::path &file) {
      std::error_code ec;
      std::filesystem::remove(file, ec);
      if (ec) {
        tsd::core::logWarning(
            "[scivisStudioCLI] removed dataset from the project but failed "
            "to delete '%s': %s",
            file.string().c_str(),
            ec.message().c_str());
      }
    };
    removeAssetFile(assetFile);
    removeAssetFile(sourceListFilePath(assetFile));
  }

  output << formatDatasetRemove(result);
  return 0;
}

int runDatasetResidency(
    const StudioCommandLine &commandLine, std::ostream &output)
{
  const bool load = commandLine.command == StudioCommand::DatasetLoad;

  BookkeepingSession session;
  std::string error;
  if (!session.open(commandLine.projectDirectory, error))
    return commandFailed(error);
  auto &context = session.context;

  auto &project = context.project();
  auto *dataset = resolveDatasetSelector(project, commandLine.dataset, error);
  if (!dataset)
    return commandFailed(error);

  const auto id = dataset->id;
  const auto requested =
      load ? DatasetResidency::Loaded : DatasetResidency::Unloaded;

  DatasetResidencyResult result;
  result.id = id;
  result.name = dataset->name;
  result.residency = dataset::toString(requested);
  result.changed = dataset->residency != requested;

  if (!result.changed) {
    output << formatDatasetResidency(result);
    return 0;
  }

  if (load) {
    // On the machine that holds the data, 'dataset load' doubles as headless
    // materialization of a Declared Dataset (ADR 0014).
    if (!dataset->persistedName.empty()) {
      const auto validation =
          validateDatasetArchiveFile(datasetAssetFile(project, *dataset));
      result.materialized = validation.ok && validation.dataset.declared;
    }
    if (!context.loadDataset(id, &error))
      return commandFailed(error);
  } else {
    if (!context.unloadDataset(id, &error))
      return commandFailed(error);
  }

  if (!persistProject(context, error))
    return commandFailed(error);
  output << formatDatasetResidency(result);
  return 0;
}

} // namespace

int runStudioCommand(const StudioCommandLine &commandLine,
    std::istream &input,
    std::ostream &output)
{
  switch (commandLine.command) {
  case StudioCommand::ProjectInit:
    return runProjectInit(commandLine, output);
  case StudioCommand::ProjectShow:
    return runProjectShow(commandLine, output);
  case StudioCommand::DatasetList:
    return runDatasetList(commandLine, output);
  case StudioCommand::DatasetShow:
    return runDatasetShow(commandLine, output);
  case StudioCommand::DatasetCreateStatic:
    return runDatasetCreateStatic(commandLine, output);
  case StudioCommand::DatasetCreateFileAnimation:
    return runDatasetCreateFileAnimation(commandLine, input, output);
  case StudioCommand::DatasetSourcesSet:
  case StudioCommand::DatasetSourcesAppend:
  case StudioCommand::DatasetSourcesRemap:
    return runDatasetSources(commandLine, input, output);
  case StudioCommand::DatasetRename:
    return runDatasetRename(commandLine, output);
  case StudioCommand::DatasetRemove:
    return runDatasetRemove(commandLine, output);
  case StudioCommand::DatasetLoad:
  case StudioCommand::DatasetUnload:
    return runDatasetResidency(commandLine, output);
  case StudioCommand::None:
    break;
  }
  return commandFailed("no command given");
}

} // namespace tsd::scivis_studio
