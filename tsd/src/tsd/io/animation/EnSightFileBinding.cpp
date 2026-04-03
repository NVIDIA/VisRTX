// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/animation/EnSightFileBinding.hpp"
#include "tsd/io/importers/detail/ensight_io.hpp"
// tsd_core
#include "tsd/core/DataTree.hpp"
#include "tsd/core/Logging.hpp"
// std
#include <algorithm>

namespace tsd::io {

using namespace tsd::core;
using namespace tsd::scene;

EnSightFileBinding::EnSightFileBinding(scene::Scene *scene,
    std::vector<PartBinding> parts,
    std::vector<std::string> geoFiles,
    std::vector<FieldMapping> fieldMappings)
    : FileBinding(scene),
      m_parts(std::move(parts)),
      m_geoFiles(std::move(geoFiles)),
      m_fieldMappings(std::move(fieldMappings))
{}

std::optional<EnSightFileBinding::SerializedData> EnSightFileBinding::fromDataNode(
    scene::Scene &scene, core::DataNode &node)
{
  SerializedData data;

  if (auto *partsNode = node.child("parts")) {
    partsNode->foreach_child([&](core::DataNode &pn) {
      auto partId = pn["partId"].getValueAs<int>();
      auto targetIndex = pn["targetIndex"].getValueAs<size_t>();
      auto *geom = static_cast<scene::Geometry *>(
          scene.getObject(ANARI_GEOMETRY, targetIndex));
      if (!geom) {
        logWarning(
            "[EnSightFileBinding] geometry index %zu not found for part %d; "
            "skipping part",
            targetIndex,
            partId);
        return;
      }

      data.parts.push_back({partId, geom});
    });
  }

  if (auto *geoFilesNode = node.child("geoFiles")) {
    geoFilesNode->foreach_child([&](core::DataNode &gn) {
      data.geoFiles.push_back(gn.getValueAs<std::string>());
    });
  }

  if (auto *fieldMappingsNode = node.child("fieldMappings")) {
    fieldMappingsNode->foreach_child([&](core::DataNode &mn) {
      FieldMapping fm;
      fm.attributeName = mn["attributeName"].getValueAs<std::string>().c_str();
      fm.ensightVarName = mn["ensightVarName"].getValueAs<std::string>();
      fm.type = mn["type"].getValueAs<std::string>();

      if (auto *filesNode = mn.child("files")) {
        filesNode->foreach_child([&](core::DataNode &fn) {
          fm.files.push_back(fn.getValueAs<std::string>());
        });
      }

      data.fieldMappings.push_back(std::move(fm));
    });
  }

  if (data.parts.empty()) {
    logWarning(
        "[EnSightFileBinding] binding has no valid geometry targets; skipping");
    return std::nullopt;
  }

  return data;
}

std::string EnSightFileBinding::kind() const
{
  return "ensight";
}

void EnSightFileBinding::toDataNode(core::DataNode &node) const
{
  auto &partsNode = node["parts"];
  for (const auto &pb : m_parts) {
    auto &pNode = partsNode.append();
    pNode["partId"] = pb.partId;
    auto *geom = pb.geometry.get();
    pNode["targetIndex"] = geom ? geom->index() : size_t(-1);
  }

  auto &geoNode = node["geoFiles"];
  for (const auto &f : m_geoFiles)
    geoNode.append() = f;

  auto &mappingsNode = node["fieldMappings"];
  for (const auto &fm : m_fieldMappings) {
    auto &mNode = mappingsNode.append();
    mNode["attributeName"] = fm.attributeName.c_str();
    mNode["ensightVarName"] = fm.ensightVarName;
    mNode["type"] = fm.type;
    auto &filesNode = mNode["files"];
    for (const auto &f : fm.files)
      filesNode.append() = f;
  }
}

void EnSightFileBinding::onDefragment(const scene::IndexRemapper &cb)
{
  for (auto &pb : m_parts) {
    if (pb.geometry) {
      size_t newIdx = cb(pb.geometry->type(), pb.geometry->index());
      pb.geometry.updateDefragmentedIndex(newIdx);
    }
  }
}

void EnSightFileBinding::update(float t)
{
  const int N = static_cast<int>(frameCount());
  if (N == 0)
    return;

  const int idx = std::clamp(
      static_cast<int>(std::round(t * static_cast<float>(N - 1))), 0, N - 1);

  if (idx == m_currentFrame)
    return;

  loadFrame(idx);
  m_currentFrame = idx;
}

void EnSightFileBinding::loadFrame(int frameIdx)
{
  // Always read the geo file — needed as part metadata for readVarFile,
  // and for geometry update if transient.
  std::vector<ensight::Part> parts;
  if (!m_geoFiles.empty()) {
    const int geoIdx =
        std::min(frameIdx, static_cast<int>(m_geoFiles.size()) - 1);
    if (!ensight::readGeoFile(m_geoFiles[geoIdx], parts)) {
      logWarning("[EnSightFileBinding] failed to read geo file '%s'",
          m_geoFiles[geoIdx].c_str());
      return;
    }
  }

  // Update positions/indices if geometry is transient (multiple geo files)
  if (m_geoFiles.size() > 1) {
    for (auto &pb : m_parts) {
      auto *geom = pb.geometry.get();
      if (!geom)
        continue;

      const ensight::Part *ep = nullptr;
      for (const auto &p : parts) {
        if (p.id == pb.partId) {
          ep = &p;
          break;
        }
      }
      if (!ep)
        continue;

      const int numNodes = static_cast<int>(ep->x.size());
      const int numTris = static_cast<int>(ep->triIndices.size()) / 3;

      auto posArr = scene()->createArray(ANARI_FLOAT32_VEC3, numNodes);
      auto *pos = posArr->mapAs<float3>();
      for (int i = 0; i < numNodes; ++i)
        pos[i] = float3(ep->x[i], ep->y[i], ep->z[i]);
      posArr->unmap();
      geom->setParameterObject("vertex.position", *posArr);

      auto idxArr = scene()->createArray(ANARI_UINT32_VEC3, numTris);
      auto *tri = idxArr->mapAs<uint3>();
      for (int t = 0; t < numTris; ++t)
        tri[t] = uint3(
            ep->triIndices[t * 3],
            ep->triIndices[t * 3 + 1],
            ep->triIndices[t * 3 + 2]);
      idxArr->unmap();
      geom->setParameterObject("primitive.index", *idxArr);
    }
  }

  // Load per-frame variable data, distributed to each part's geometry
  for (const auto &fm : m_fieldMappings) {
    if (frameIdx >= static_cast<int>(fm.files.size()))
      continue;

    const int nc = (fm.type == "vector") ? 3 : 1;
    std::map<int, std::vector<float>> varData;
    ensight::readVarFile(fm.files[frameIdx], parts, nc, varData);

    for (auto &pb : m_parts) {
      auto *geom = pb.geometry.get();
      if (!geom)
        continue;

      auto it = varData.find(pb.partId);
      if (it == varData.end())
        continue;

      const auto &data = it->second;
      const int numNodes = static_cast<int>(data.size()) / nc;

      if (nc == 1) {
        auto arr = scene()->createArray(ANARI_FLOAT32, numNodes);
        std::copy(data.begin(), data.end(), arr->mapAs<float>());
        arr->unmap();
        geom->setParameterObject(fm.attributeName.c_str(), *arr);
      } else {
        auto arr = scene()->createArray(ANARI_FLOAT32_VEC3, numNodes);
        auto *vec = arr->mapAs<float3>();
        for (int i = 0; i < numNodes; ++i) {
          vec[i] = float3(data[i * 3], data[i * 3 + 1], data[i * 3 + 2]);
        }
        arr->unmap();
        geom->setParameterObject(fm.attributeName.c_str(), *arr);
      }
    }
  }

  scene()->removeUnusedObjects();
}

size_t EnSightFileBinding::frameCount() const
{
  if (!m_fieldMappings.empty() && !m_fieldMappings[0].files.empty())
    return m_fieldMappings[0].files.size();
  if (!m_geoFiles.empty())
    return m_geoFiles.size();
  return 0;
}

int EnSightFileBinding::currentFrame() const
{
  return m_currentFrame;
}

void EnSightFileBinding::addCallbackToAnimation(
    tsd::animation::Animation &anim)
{
  anim.addCallbackBinding([this](float t) { update(t); });
}

} // namespace tsd::io
