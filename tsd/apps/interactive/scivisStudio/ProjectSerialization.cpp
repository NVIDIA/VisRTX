// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ProjectSerialization.h"

#include "tsd/core/DataTreeMetadata.hpp"

#include <anari/anari_cpp/ext/std.h>

#include <cctype>

namespace tsd::scivis_studio {

namespace {

constexpr size_t MAX_RIG_NAME_LENGTH = 128;

bool isAllowedRigNameChar(unsigned char c)
{
  return std::isalnum(c) || c == ' ' || c == '_' || c == '-' || c == '('
      || c == ')';
}

} // namespace

bool validateRigName(const std::string &name, std::string *error)
{
  auto fail = [&](const char *msg) {
    if (error)
      *error = msg;
    return false;
  };

  if (name.empty())
    return fail("name cannot be empty");
  if (name.size() > MAX_RIG_NAME_LENGTH)
    return fail("name is too long");
  if (name == "." || name == "..")
    return fail("name is reserved");
  if (std::isspace(static_cast<unsigned char>(name.front()))
      || std::isspace(static_cast<unsigned char>(name.back())))
    return fail("name cannot start or end with whitespace");
  for (unsigned char c : name) {
    if (!isAllowedRigNameChar(c))
      return fail(
          "name may only contain letters, digits, spaces, '_', '-', '(', ')'");
  }

  if (error)
    error->clear();
  return true;
}

std::string sanitizeRigName(const std::string &name)
{
  std::string out;
  out.reserve(name.size());
  for (unsigned char c : name)
    out.push_back(isAllowedRigNameChar(c) ? static_cast<char>(c) : '_');

  // Trim leading/trailing whitespace.
  const auto first = out.find_first_not_of(' ');
  const auto last = out.find_last_not_of(' ');
  if (first == std::string::npos)
    out.clear();
  else
    out = out.substr(first, last - first + 1);

  if (out.size() > MAX_RIG_NAME_LENGTH)
    out.resize(MAX_RIG_NAME_LENGTH);

  if (out.empty() || out == "." || out == "..")
    out = "rig";

  return out;
}

static std::string cameraRigNameForShot(const Shot &shot)
{
  if (!shot.name.empty())
    return shot.name + " Camera";
  if (!shot.id.empty())
    return shot.id + " Camera";
  return "Camera Rig";
}

static void sourceMetadataToNode(
    const DatasetSourceMetadata &source, tsd::core::DataNode &node)
{
  node["absolutePath"] = source.absolutePath;
  node["projectRelativePath"] = source.projectRelativePath;
  node["fileSize"] = source.fileSize;
  node["modifiedTime"] = source.modifiedTime;
}

static void sourceFileToNode(
    const DatasetSourceFile &source, tsd::core::DataNode &node)
{
  node["absolutePath"] = source.absolutePath;
  node["projectRelativePath"] = source.projectRelativePath;
  node["fileSize"] = source.fileSize;
  node["modifiedTime"] = source.modifiedTime;
}

static void nodeToSourceMetadata(
    tsd::core::DataNode &node, DatasetSourceMetadata &source)
{
  source.absolutePath = node["absolutePath"].getValueOr<std::string>("");
  source.projectRelativePath =
      node["projectRelativePath"].getValueOr<std::string>("");
  source.fileSize = node["fileSize"].getValueOr<uint64_t>(0);
  source.modifiedTime = node["modifiedTime"].getValueOr<int64_t>(0);
}

static void nodeToSourceFile(
    tsd::core::DataNode &node, DatasetSourceFile &source)
{
  source.absolutePath = node["absolutePath"].getValueOr<std::string>("");
  source.projectRelativePath =
      node["projectRelativePath"].getValueOr<std::string>("");
  source.fileSize = node["fileSize"].getValueOr<uint64_t>(0);
  source.modifiedTime = node["modifiedTime"].getValueOr<int64_t>(0);
}

void projectToNode(const Project &project, tsd::core::DataNode &node)
{
  node.reset();
  node["name"] = project.name;
  node["projectDirectory"] = project.projectDirectory.string();
  node["activeShot"] = project.activeShotId;
  node["dirty"] = project.dirty;

  auto &datasets = node["datasets"];
  for (const auto &dataset : project.datasets) {
    auto &d = datasets.append();
    d["id"] = dataset.id;
    d["name"] = dataset.name;
    d["sourceKind"] = dataset::toString(dataset.sourceKind);
    d["importerType"] = dataset.importerType;
    d["status"] = dataset::toString(dataset.status);

    sourceMetadataToNode(dataset.source, d["source"]);

    auto &sourceFiles = d["sourceFiles"];
    for (const auto &sourceFile : dataset.sourceFiles)
      sourceFileToNode(sourceFile, sourceFiles.append());
  }

  auto &shots = node["shots"];
  for (const auto &shot : project.shots) {
    auto &s = shots.append();
    s["id"] = shot.id;
    s["name"] = shot.name;
    s["frameCount"] = shot.frameCount;
    s["fps"] = shot.fps;
    s["currentFrame"] = shot.currentFrame;
    s["playing"] = shot.playing;
    s["loop"] = shot.loop;
    s["lightRigId"] = shot.lightRigId;
    s["cameraRigId"] = shot.cameraRigId;

    auto &render = s["renderSettings"];
    render["width"] = shot.renderSettings.width;
    render["height"] = shot.renderSettings.height;
    render["samples"] = shot.renderSettings.samples;
    render["rendererLibrary"] = shot.renderSettings.rendererLibrary;
    render["rendererObjectIndex"] =
        static_cast<uint64_t>(shot.renderSettings.rendererObjectIndex);
    render["rendererSubtype"] = shot.renderSettings.rendererSubtype;
    render["outputFilePrefix"] = shot.renderSettings.outputFilePrefix;

    auto &bindings = s["datasetBindings"];
    for (const auto &binding : shot.datasetBindings) {
      auto &b = bindings.append();
      b["datasetId"] = binding.datasetId;
      b["enabled"] = binding.enabled;
    }
  }

  auto &lightRigs = node["lightRigs"];
  for (const auto &rig : project.lightRigs) {
    auto &r = lightRigs.append();
    r["id"] = rig.id;
    r["name"] = rig.name;
  }

  // Camera-rig value data is stored per-rig in cameras/<name>.tsd, so only the
  // id and name are recorded in the manifest (mirroring light rigs).
  auto &cameraRigs = node["cameraRigs"];
  for (const auto &rig : project.cameraRigs) {
    auto &r = cameraRigs.append();
    r["id"] = rig.id;
    r["name"] = rig.name;
  }

  auto &colorMaps = node["colorMaps"];
  for (const auto &colorMap : project.colorMaps) {
    auto &c = colorMaps.append();
    c["id"] = colorMap.id;
    c["name"] = colorMap.name;
  }
}

bool nodeToProject(tsd::core::DataNode &node, Project &project)
{
  Project out;
  out.name = node["name"].getValueOr<std::string>("Untitled");
  out.projectDirectory = node["projectDirectory"].getValueOr<std::string>("");
  out.activeShotId = node["activeShot"].getValueOr<std::string>("");
  out.dirty = node["dirty"].getValueOr<bool>(false);

  if (auto *datasets = node.child("datasets")) {
    datasets->foreach_child([&](tsd::core::DataNode &d) {
      Dataset dataset;
      dataset.id = d["id"].getValueOr<std::string>("");
      dataset.name = d["name"].getValueOr<std::string>(dataset.id);
      dataset.sourceKind = dataset::sourceKindFromString(
          d["sourceKind"].getValueOr<std::string>("Static"));
      dataset.importerType = d["importerType"].getValueOr<std::string>("NONE");
      dataset.status = dataset::statusFromString(
          d["status"].getValueOr<std::string>("Missing"));

      if (auto *source = d.child("source"))
        nodeToSourceMetadata(*source, dataset.source);

      if (auto *sourceFiles = d.child("sourceFiles")) {
        sourceFiles->foreach_child([&](tsd::core::DataNode &f) {
          DatasetSourceFile sourceFile;
          nodeToSourceFile(f, sourceFile);
          dataset.sourceFiles.push_back(std::move(sourceFile));
        });
      }
      out.datasets.push_back(std::move(dataset));
    });
  }

  if (auto *cameraRigs = node.child("cameraRigs")) {
    cameraRigs->foreach_child([&](tsd::core::DataNode &r) {
      CameraRig rig;
      rig.id = r["id"].getValueOr<std::string>("");
      rig.name = r["name"].getValueOr<std::string>("");
      if (auto *rigNode = r.child("rig"))
        camera_rig::nodeToCameraRig(*rigNode, rig);
      out.cameraRigs.push_back(std::move(rig));
    });
  }

  if (auto *shots = node.child("shots")) {
    shots->foreach_child([&](tsd::core::DataNode &s) {
      Shot shot;
      shot.id = s["id"].getValueOr<std::string>("");
      shot.name = s["name"].getValueOr<std::string>(shot.id);
      shot.frameCount = s["frameCount"].getValueOr<int>(120);
      shot.fps = s["fps"].getValueOr<float>(24.f);
      shot.currentFrame = s["currentFrame"].getValueOr<int>(0);
      shot.playing = s["playing"].getValueOr<bool>(false);
      shot.loop = s["loop"].getValueOr<bool>(true);
      shot.lightRigId = s["lightRigId"].getValueOr<std::string>("");
      shot.cameraRigId = s["cameraRigId"].getValueOr<std::string>("");
      if (shot.cameraRigId.empty()) {
        if (auto *cameraRigNode = s.child("cameraRig")) {
          CameraRig rig;
          rig.id = camera_rig::nextCameraRigId(out);
          rig.name = cameraRigNameForShot(shot);
          camera_rig::nodeToCameraRig(*cameraRigNode, rig);
          shot.cameraRigId = rig.id;
          out.cameraRigs.push_back(std::move(rig));
        }
      }

      if (auto *render = s.child("renderSettings")) {
        shot.renderSettings.width =
            (*render)["width"].getValueOr<uint32_t>(1024);
        shot.renderSettings.height =
            (*render)["height"].getValueOr<uint32_t>(768);
        shot.renderSettings.samples =
            (*render)["samples"].getValueOr<uint32_t>(128);
        shot.renderSettings.rendererLibrary =
            (*render)["rendererLibrary"].getValueOr<std::string>("");
        shot.renderSettings.rendererObjectIndex =
            (*render)["rendererObjectIndex"].getValueOr<uint64_t>(
                TSD_INVALID_INDEX);
        shot.renderSettings.rendererSubtype =
            (*render)["rendererSubtype"].getValueOr<std::string>("default");
        shot.renderSettings.outputFilePrefix =
            (*render)["outputFilePrefix"].getValueOr<std::string>("");
      }

      if (auto *bindings = s.child("datasetBindings")) {
        bindings->foreach_child([&](tsd::core::DataNode &b) {
          DatasetBinding binding;
          binding.datasetId = b["datasetId"].getValueOr<std::string>("");
          binding.enabled = b["enabled"].getValueOr<bool>(true);
          shot.datasetBindings.push_back(std::move(binding));
        });
      }
      out.shots.push_back(std::move(shot));
    });
  }

  if (auto *lightRigs = node.child("lightRigs")) {
    lightRigs->foreach_child([&](tsd::core::DataNode &r) {
      out.lightRigs.push_back({r["id"].getValueOr<std::string>(""),
          r["name"].getValueOr<std::string>("")});
    });
  }

  if (auto *colorMaps = node.child("colorMaps")) {
    colorMaps->foreach_child([&](tsd::core::DataNode &c) {
      out.colorMaps.push_back({c["id"].getValueOr<std::string>(""),
          c["name"].getValueOr<std::string>("")});
    });
  }

  if (out.activeShotId.empty() && !out.shots.empty())
    out.activeShotId = out.shots.front().id;

  project = std::move(out);
  return true;
}

ProjectValidationResult validateProjectRoot(
    const std::filesystem::path &directory)
{
  ProjectValidationResult result;
  result.manifestPath = directory / PROJECT_MANIFEST_FILENAME;

  if (!std::filesystem::exists(directory)) {
    result.error = "project directory does not exist";
    return result;
  }

  if (!std::filesystem::is_directory(directory)) {
    result.error = "selected path is not a directory";
    return result;
  }

  if (!std::filesystem::exists(result.manifestPath)) {
    result.error = "project.tsd does not exist";
    return result;
  }

  tsd::core::DataTree tree;
  if (!tree.load(result.manifestPath.string().c_str())) {
    result.error = "failed to load project.tsd";
    return result;
  }

  auto &root = tree.root();
  auto metadataResult = tsd::core::readDataTreeMetadata(root);
  if (metadataResult.malformed()) {
    result.error = "malformed __tsd_metadata: " + metadataResult.message;
    return result;
  }

  if (metadataResult.found()) {
    const auto &metadata = *metadataResult.metadata;
    if (metadata.envelopeVersion
        != tsd::core::DATA_TREE_METADATA_ENVELOPE_VERSION) {
      result.error = "unsupported SciVis Studio metadata envelopeVersion";
      return result;
    }

    if (metadata.fileType != PROJECT_FILE_TYPE
        || metadata.schema != PROJECT_SCHEMA) {
      result.error = "metadata schema is not SciVis Studio project";
      return result;
    }

    if (metadata.schemaVersion < 1 || metadata.schemaVersion > SCHEMA_VERSION) {
      result.error = "unsupported SciVis Studio schemaVersion";
      return result;
    }
  } else {
    const auto kind = root["projectKind"].getValueOr<std::string>("");
    if (kind != PROJECT_KIND) {
      result.error = "missing __tsd_metadata";
      return result;
    }

    const auto version = root["schemaVersion"].getValueOr<int>(0);
    if (version < 1 || version > SCHEMA_VERSION) {
      result.error = "unsupported legacy SciVis Studio schemaVersion";
      return result;
    }
  }

  result.ok = true;
  return result;
}

} // namespace tsd::scivis_studio
