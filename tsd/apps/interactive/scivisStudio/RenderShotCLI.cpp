// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "RenderShotCLI.h"

#include <algorithm>
#include <charconv>
#include <sstream>

namespace tsd::scivis_studio {

bool parseRenderShotCommandLine(const std::vector<std::string> &args,
    RenderShotCommandLine &commandLine,
    std::string &error)
{
  commandLine = {};
  error.clear();

  for (size_t i = 1; i < args.size(); ++i) {
    const auto &arg = args[i];
    if (arg == "-h" || arg == "--help") {
      commandLine.showHelp = true;
      return true;
    }

    if (arg == "--shot") {
      if (i + 1 >= args.size()) {
        error = "--shot requires a shot ID";
        return false;
      }
      commandLine.shotId = args[++i];
      if (commandLine.shotId.empty()) {
        error = "--shot requires a non-empty shot ID";
        return false;
      }
      continue;
    }

    if (!arg.empty() && arg.front() == '-') {
      error = "unknown option: " + arg;
      return false;
    }

    if (!commandLine.projectDirectory.empty()) {
      error = "multiple project directories were specified";
      return false;
    }

    commandLine.projectDirectory = arg;
  }

  if (commandLine.projectDirectory.empty()) {
    error = "missing project directory";
    return false;
  }

  return true;
}

std::string renderShotUsage(const std::string &programName)
{
  std::ostringstream out;
  out << "usage: " << programName
      << " <project-directory> [--shot <shot-id>]\n";
  return out.str();
}

std::string formatShotList(const Project &project)
{
  std::ostringstream out;
  out << "Available shots:\n";
  for (const auto &shot : project.shots)
    out << "  " << shot.id << "    " << shot.name << '\n';
  return out.str();
}

Shot *findShotById(Project &project, const std::string &shotId)
{
  auto it = std::find_if(project.shots.begin(),
      project.shots.end(),
      [&](const Shot &shot) { return shot.id == shotId; });
  return it == project.shots.end() ? nullptr : &*it;
}

const Shot *findShotById(const Project &project, const std::string &shotId)
{
  auto it = std::find_if(project.shots.begin(),
      project.shots.end(),
      [&](const Shot &shot) { return shot.id == shotId; });
  return it == project.shots.end() ? nullptr : &*it;
}

static bool parseSelection(const std::string &input, size_t &selection)
{
  size_t value = 0;
  const auto *begin = input.data();
  const auto *end = input.data() + input.size();
  auto result = std::from_chars(begin, end, value);
  if (result.ec != std::errc{} || result.ptr != end)
    return false;

  selection = value;
  return true;
}

Shot *selectShotForRender(Project &project,
    const std::string &shotId,
    bool interactive,
    std::istream &input,
    std::ostream &output,
    std::string &error)
{
  error.clear();

  if (!shotId.empty()) {
    auto *shot = findShotById(project, shotId);
    if (shot)
      return shot;

    std::ostringstream out;
    out << "unknown shot ID: " << shotId << "\n\n" << formatShotList(project);
    error = out.str();
    return nullptr;
  }

  if (project.shots.empty()) {
    error = "project has no shots";
    return nullptr;
  }

  if (project.shots.size() == 1)
    return &project.shots.front();

  if (!interactive) {
    std::ostringstream out;
    out << "multiple shots found; pass --shot <shot-id>\n\n"
        << formatShotList(project);
    error = out.str();
    return nullptr;
  }

  output << "Multiple shots found:\n";
  for (size_t i = 0; i < project.shots.size(); ++i) {
    const auto &shot = project.shots[i];
    output << "  " << i + 1 << ". " << shot.id << "    " << shot.name << '\n';
  }
  output << "\nSelect shot [1-" << project.shots.size() << "]: ";

  std::string line;
  if (!std::getline(input, line)) {
    error = "failed to read shot selection";
    return nullptr;
  }

  size_t selection = 0;
  if (!parseSelection(line, selection) || selection < 1
      || selection > project.shots.size()) {
    std::ostringstream out;
    out << "invalid shot selection: " << line;
    error = out.str();
    return nullptr;
  }

  return &project.shots[selection - 1];
}

} // namespace tsd::scivis_studio
