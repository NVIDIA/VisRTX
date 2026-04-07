// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/importers/detail/ensight_io.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
// std
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace tsd::core;

// ============================================================================
// File-local helpers
// ============================================================================

enum class EnsightElemType
{
  point,
  bar2,
  bar3,
  tria3,
  tria6,
  quad4,
  quad8,
  tetra4,
  tetra10,
  pyramid5,
  pyramid13,
  penta6,
  penta15,
  hexa8,
  hexa20,
  nsided,
  nfaced,
  unknown
};

static EnsightElemType parseElemType(const std::string &s)
{
  if (s == "point")
    return EnsightElemType::point;
  if (s == "bar2")
    return EnsightElemType::bar2;
  if (s == "bar3")
    return EnsightElemType::bar3;
  if (s == "tria3")
    return EnsightElemType::tria3;
  if (s == "tria6")
    return EnsightElemType::tria6;
  if (s == "quad4")
    return EnsightElemType::quad4;
  if (s == "quad8")
    return EnsightElemType::quad8;
  if (s == "tetra4")
    return EnsightElemType::tetra4;
  if (s == "tetra10")
    return EnsightElemType::tetra10;
  if (s == "pyramid5")
    return EnsightElemType::pyramid5;
  if (s == "pyramid13")
    return EnsightElemType::pyramid13;
  if (s == "penta6")
    return EnsightElemType::penta6;
  if (s == "penta15")
    return EnsightElemType::penta15;
  if (s == "hexa8")
    return EnsightElemType::hexa8;
  if (s == "hexa20")
    return EnsightElemType::hexa20;
  if (s == "nsided")
    return EnsightElemType::nsided;
  if (s == "nfaced")
    return EnsightElemType::nfaced;
  return EnsightElemType::unknown;
}

static int nodesPerElem(EnsightElemType t)
{
  switch (t) {
  case EnsightElemType::point:
    return 1;
  case EnsightElemType::bar2:
    return 2;
  case EnsightElemType::bar3:
    return 3;
  case EnsightElemType::tria3:
    return 3;
  case EnsightElemType::tria6:
    return 6;
  case EnsightElemType::quad4:
    return 4;
  case EnsightElemType::quad8:
    return 8;
  case EnsightElemType::tetra4:
    return 4;
  case EnsightElemType::tetra10:
    return 10;
  case EnsightElemType::pyramid5:
    return 5;
  case EnsightElemType::pyramid13:
    return 13;
  case EnsightElemType::penta6:
    return 6;
  case EnsightElemType::penta15:
    return 15;
  case EnsightElemType::hexa8:
    return 8;
  case EnsightElemType::hexa20:
    return 20;
  default:
    return 0;
  }
}

// Triangulate a single element and append 0-based triangle indices to out.
// Returns false if the element type is not a supported surface element.
static bool triangulateSurface(
    EnsightElemType t, const int32_t *c, std::vector<uint32_t> &out)
{
  auto v = [&](int i) { return static_cast<uint32_t>(c[i] - 1); };

  switch (t) {
  case EnsightElemType::tria3:
    out.push_back(v(0));
    out.push_back(v(1));
    out.push_back(v(2));
    return true;

  case EnsightElemType::tria6: // use corner nodes only
    out.push_back(v(0));
    out.push_back(v(1));
    out.push_back(v(2));
    return true;

  case EnsightElemType::quad4: // split into 2 triangles
    out.push_back(v(0));
    out.push_back(v(1));
    out.push_back(v(2));
    out.push_back(v(0));
    out.push_back(v(2));
    out.push_back(v(3));
    return true;

  case EnsightElemType::quad8: // use corner nodes (0,2,4,6)
    out.push_back(v(0));
    out.push_back(v(2));
    out.push_back(v(4));
    out.push_back(v(0));
    out.push_back(v(4));
    out.push_back(v(6));
    return true;

  default:
    return false; // 0D, 1D, and 3D elements are skipped
  }
}

static bool isVolumeElem(EnsightElemType t)
{
  switch (t) {
  case EnsightElemType::tetra4:
  case EnsightElemType::tetra10:
  case EnsightElemType::pyramid5:
  case EnsightElemType::pyramid13:
  case EnsightElemType::penta6:
  case EnsightElemType::penta15:
  case EnsightElemType::hexa8:
  case EnsightElemType::hexa20:
    return true;
  default:
    return false;
  }
}

// Face key for surface extraction: sorted 0-based global node indices.
// Triangular faces use ~0u as padding in the 4th slot.
using FaceKey = std::array<uint32_t, 4>;

struct FaceEntry
{
  FaceKey oriented{}; // nodes in original order (for triangulation)
  int count{0};
};

// Per-element face tables (local node indices; -1 = triangle, pad 4th slot).
struct FaceDef
{
  int n[4];
};

static const FaceDef kTetra4Faces[] = {
    {{0, 1, 2, -1}}, {{0, 1, 3, -1}}, {{1, 2, 3, -1}}, {{0, 2, 3, -1}}};

static const FaceDef kPyramid5Faces[] = {{{0, 1, 2, 3}},
    {{0, 1, 4, -1}},
    {{1, 2, 4, -1}},
    {{2, 3, 4, -1}},
    {{3, 0, 4, -1}}};

static const FaceDef kPenta6Faces[] = {{{0, 1, 2, -1}},
    {{3, 4, 5, -1}},
    {{0, 1, 4, 3}},
    {{1, 2, 5, 4}},
    {{0, 2, 5, 3}}};

static const FaceDef kHexa8Faces[] = {{{0, 1, 2, 3}},
    {{4, 5, 6, 7}},
    {{0, 1, 5, 4}},
    {{1, 2, 6, 5}},
    {{2, 3, 7, 6}},
    {{0, 3, 7, 4}}};

// Accumulate all faces of volume elements into faceMap.
// Boundary faces (count == 1 after all elements processed) form the surface.
static void accumulateVolumeFaces(EnsightElemType et,
    int npe,
    int numElems,
    const std::vector<int32_t> &conn,
    tsd::core::FlatMap<FaceKey, FaceEntry> &faceMap)
{
  const FaceDef *faceDefs = nullptr;
  int numFaceDefs = 0;

  switch (et) {
  case EnsightElemType::tetra4:
  case EnsightElemType::tetra10:
    faceDefs = kTetra4Faces;
    numFaceDefs = 4;
    break;
  case EnsightElemType::pyramid5:
  case EnsightElemType::pyramid13:
    faceDefs = kPyramid5Faces;
    numFaceDefs = 5;
    break;
  case EnsightElemType::penta6:
  case EnsightElemType::penta15:
    faceDefs = kPenta6Faces;
    numFaceDefs = 5;
    break;
  case EnsightElemType::hexa8:
  case EnsightElemType::hexa20:
    faceDefs = kHexa8Faces;
    numFaceDefs = 6;
    break;
  default:
    return;
  }

  for (int e = 0; e < numElems; ++e) {
    const int32_t *elem = conn.data() + e * npe;
    for (int fi = 0; fi < numFaceDefs; ++fi) {
      FaceKey oriented = {~0u, ~0u, ~0u, ~0u};
      FaceKey sorted = {~0u, ~0u, ~0u, ~0u};
      for (int k = 0; k < 4 && faceDefs[fi].n[k] >= 0; ++k) {
        uint32_t gn = static_cast<uint32_t>(elem[faceDefs[fi].n[k]] - 1);
        oriented[k] = gn;
        sorted[k] = gn;
      }
      std::sort(sorted.begin(), sorted.end());
      auto &entry = faceMap[sorted];
      if (entry.count == 0)
        entry.oriented = oriented;
      ++entry.count;
    }
  }
}

// Triangulate boundary faces (count == 1) and append to triIndices.
static void emitBoundaryFaces(
    const tsd::core::FlatMap<FaceKey, FaceEntry> &faceMap,
    std::vector<uint32_t> &out)
{
  for (const auto &[key, entry] : faceMap) {
    if (entry.count != 1)
      continue;
    const auto &o = entry.oriented;
    if (o[3] == ~0u) {
      // triangle face
      out.push_back(o[0]);
      out.push_back(o[1]);
      out.push_back(o[2]);
    } else {
      // quad face: split into 2 triangles
      out.push_back(o[0]);
      out.push_back(o[1]);
      out.push_back(o[2]);
      out.push_back(o[0]);
      out.push_back(o[2]);
      out.push_back(o[3]);
    }
  }
}

// ============================================================================
// Binary record reader
// ============================================================================

// Read an 80-byte record from a binary EnSight Gold file, trimming trailing
// null bytes and spaces.
static std::string readRecord80(std::FILE *f)
{
  char buf[81] = {};
  if (std::fread(buf, 1, 80, f) != 80)
    return "";
  int len = 80;
  while (len > 0 && (buf[len - 1] == '\0' || buf[len - 1] == ' '))
    --len;
  return std::string(buf, len);
}

static std::string lowerStr(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
  return s;
}

// ============================================================================
// Public API
// ============================================================================

namespace tsd::io::ensight {

std::vector<std::string> expandPattern(
    const std::string &pat, int start, int incr, int steps)
{
  std::vector<std::string> out;
  size_t starCount = std::count(pat.begin(), pat.end(), '*');
  if (starCount == 0) {
    out.push_back(pat);
    return out;
  }
  for (int i = 0; i < steps; ++i) {
    int num = start + i * incr;
    std::string numStr = std::to_string(num);
    while (numStr.size() < starCount)
      numStr = "0" + numStr;
    std::string s = pat;
    s.replace(s.find(std::string(starCount, '*')), starCount, numStr);
    out.push_back(s);
  }
  return out;
}

bool parseCase(const std::string &path, CaseData &out)
{
  std::FILE *f = std::fopen(path.c_str(), "r");
  if (!f) {
    logError("[import_ENSIGHT] cannot open case file '%s'", path.c_str());
    return false;
  }

  char buf[1024];
  auto nextLine = [&]() -> std::string {
    if (!std::fgets(buf, sizeof(buf), f))
      return "";
    std::string s(buf);
    while (!s.empty()
        && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) {
      s.pop_back();
    }
    size_t p = s.find_first_not_of(" \t");
    return p == std::string::npos ? "" : s.substr(p);
  };

  std::string line = nextLine();
  if (line != "FORMAT") {
    logError("[import_ENSIGHT] missing FORMAT section in '%s'", path.c_str());
    std::fclose(f);
    return false;
  }
  line = nextLine();
  if (lowerStr(line) != "type: ensight gold") {
    logError("[import_ENSIGHT] expected 'type: ensight gold', got '%s'",
        line.c_str());
    std::fclose(f);
    return false;
  }

  enum Section
  {
    NONE,
    GEOMETRY,
    VARIABLE,
    TIME
  } section = NONE;

  for (line = nextLine(); !line.empty() || !std::feof(f); line = nextLine()) {
    if (line.empty())
      continue;
    if (line == "GEOMETRY") {
      section = GEOMETRY;
      continue;
    }
    if (line == "VARIABLE") {
      section = VARIABLE;
      continue;
    }
    if (line == "TIME") {
      section = TIME;
      continue;
    }
    if (line == "FILE") {
      section = NONE;
      continue;
    }

    if (section == GEOMETRY) {
      if (line.size() >= 6 && line.substr(0, 6) == "model:") {
        auto tokens = splitString(line.substr(6), ' ');
        tokens.erase(
            std::remove(tokens.begin(), tokens.end(), ""), tokens.end());
        if (!tokens.empty())
          out.geoPattern = tokens.back();
      }
    } else if (section == TIME) {
      if (line.size() > 16 && line.substr(0, 16) == "number of steps:")
        out.numSteps = std::stoi(line.substr(16));
      else if (line.size() > 22
          && line.substr(0, 22) == "filename start number:")
        out.startNumber = std::stoi(line.substr(22));
      else if (line.size() > 19 && line.substr(0, 19) == "filename increment:")
        out.increment = std::stoi(line.substr(19));
    } else if (section == VARIABLE) {
      size_t colon = line.find(':');
      if (colon == std::string::npos)
        continue;

      std::string typeStr = lowerStr(line.substr(0, colon));
      while (!typeStr.empty() && typeStr.back() == ' ')
        typeStr.pop_back();

      std::string rest = line.substr(colon + 1);
      size_t p = rest.find_first_not_of(" \t");
      if (p != std::string::npos)
        rest = rest.substr(p);

      if (typeStr.find("complex") == 0 || typeStr.find("constant") == 0)
        continue;

      std::string varType;
      if (typeStr.find("scalar") == 0)
        varType = "scalar";
      else if (typeStr.find("vector") == 0)
        varType = "vector";
      else
        continue; // tensors not supported

      std::string assoc;
      if (typeStr.rfind("per node") != std::string::npos)
        assoc = "vertex";
      else if (typeStr.rfind("per element") != std::string::npos)
        assoc = "cell";
      else
        continue;

      // rest = "[timeset] varname filename"
      auto tokens = splitString(rest, ' ');
      tokens.erase(std::remove(tokens.begin(), tokens.end(), ""), tokens.end());
      if (tokens.size() < 2)
        continue;

      std::string varName = tokens[tokens.size() - 2];
      std::string varPat = tokens[tokens.size() - 1];
      out.variables[varName] = {varPat, assoc, varType};
    }
  }

  std::fclose(f);
  return !out.geoPattern.empty();
}

bool readGeoFile(const std::string &filename, std::vector<Part> &parts)
{
  std::FILE *f = std::fopen(filename.c_str(), "rb");
  if (!f) {
    logError(
        "[import_ENSIGHT] cannot open geometry file '%s'", filename.c_str());
    return false;
  }

  if (readRecord80(f) != "C Binary") {
    logError(
        "[import_ENSIGHT] only binary EnSight Gold geometry files are "
        "supported (got non-binary header)");
    std::fclose(f);
    return false;
  }
  readRecord80(f); // description 1
  readRecord80(f); // description 2

  auto nodeIdLine = lowerStr(readRecord80(f));
  bool hasNodeIds = nodeIdLine.find("given") != std::string::npos
      || nodeIdLine.find("ignore") != std::string::npos;

  auto elemIdLine = lowerStr(readRecord80(f));
  bool hasElemIds = elemIdLine.find("given") != std::string::npos
      || elemIdLine.find("ignore") != std::string::npos;

  std::string token = lowerStr(readRecord80(f));
  if (token == "extents") {
    float ext[6];
    std::fread(ext, sizeof(float), 6, f);
    token = lowerStr(readRecord80(f));
  }

  while (!token.empty()) {
    if (token != "part") {
      logWarning("[import_ENSIGHT] expected 'part', got '%s'", token.c_str());
      break;
    }

    Part part;
    int32_t partId;
    std::fread(&partId, sizeof(int32_t), 1, f);
    part.id = partId;
    part.description = readRecord80(f);

    auto coordTag = lowerStr(readRecord80(f));
    if (coordTag != "coordinates") {
      logWarning(
          "[import_ENSIGHT] part %d: only unstructured parts are supported "
          "(got '%s')",
          partId,
          coordTag.c_str());
      token = lowerStr(readRecord80(f));
      continue;
    }

    int32_t numNodes;
    std::fread(&numNodes, sizeof(int32_t), 1, f);

    if (hasNodeIds)
      std::fseek(f, numNodes * (int)sizeof(int32_t), SEEK_CUR);

    part.x.resize(numNodes);
    part.y.resize(numNodes);
    part.z.resize(numNodes);
    std::fread(part.x.data(), sizeof(float), numNodes, f);
    std::fread(part.y.data(), sizeof(float), numNodes, f);
    std::fread(part.z.data(), sizeof(float), numNodes, f);

    // Face map for surface extraction from volume elements.
    // Faces appearing exactly once are boundary (exterior) faces.
    tsd::core::FlatMap<FaceKey, FaceEntry> faceMap;

    token = lowerStr(readRecord80(f));
    while (!token.empty() && token != "part") {
      auto et = parseElemType(token);
      int32_t numElems;
      std::fread(&numElems, sizeof(int32_t), 1, f);

      if (hasElemIds)
        std::fseek(f, numElems * (int)sizeof(int32_t), SEEK_CUR);

      if (et == EnsightElemType::nsided) {
        // Variable polygon: read per-element node counts, then connectivity.
        // Fan-triangulate each polygon.
        std::vector<int32_t> nodeCounts(numElems);
        std::fread(nodeCounts.data(), sizeof(int32_t), numElems, f);
        int32_t totalNodes = 0;
        for (auto n : nodeCounts)
          totalNodes += n;
        std::vector<int32_t> conn(totalNodes);
        std::fread(conn.data(), sizeof(int32_t), totalNodes, f);
        int off = 0;
        for (int e = 0; e < numElems; ++e) {
          for (int i = 1; i + 1 < nodeCounts[e]; ++i) {
            part.triIndices.push_back(static_cast<uint32_t>(conn[off] - 1));
            part.triIndices.push_back(static_cast<uint32_t>(conn[off + i] - 1));
            part.triIndices.push_back(
                static_cast<uint32_t>(conn[off + i + 1] - 1));
          }
          off += nodeCounts[e];
        }
      } else if (et == EnsightElemType::nfaced) {
        // Polyhedral elements: read metadata to advance file position, skip.
        std::vector<int32_t> faceCounts(numElems);
        std::fread(faceCounts.data(), sizeof(int32_t), numElems, f);
        int32_t totalFaces = 0;
        for (auto n : faceCounts)
          totalFaces += n;
        std::vector<int32_t> faceNodeCounts(totalFaces);
        std::fread(faceNodeCounts.data(), sizeof(int32_t), totalFaces, f);
        int32_t totalNodes = 0;
        for (auto n : faceNodeCounts)
          totalNodes += n;
        std::fseek(f, totalNodes * (int)sizeof(int32_t), SEEK_CUR);
        logWarning(
            "[import_ENSIGHT] part %d: nfaced elements not supported, "
            "skipping %d elements",
            partId,
            numElems);
      } else if (isVolumeElem(et)) {
        // Volume element: accumulate faces for boundary extraction.
        int npe = nodesPerElem(et);
        std::vector<int32_t> conn(numElems * npe);
        std::fread(conn.data(), sizeof(int32_t), numElems * npe, f);
        accumulateVolumeFaces(et, npe, numElems, conn, faceMap);
      } else {
        int npe = nodesPerElem(et);
        if (npe == 0) {
          logWarning(
              "[import_ENSIGHT] part %d: unknown element type '%s', "
              "skipping",
              partId,
              token.c_str());
          token = lowerStr(readRecord80(f));
          continue;
        }
        std::vector<int32_t> conn(numElems * npe);
        std::fread(conn.data(), sizeof(int32_t), numElems * npe, f);
        for (int e = 0; e < numElems; ++e)
          triangulateSurface(et, conn.data() + e * npe, part.triIndices);
      }

      token = lowerStr(readRecord80(f));
    }

    // Emit exterior faces of all volume elements as triangles.
    emitBoundaryFaces(faceMap, part.triIndices);

    if (!part.triIndices.empty())
      parts.push_back(std::move(part));
    else
      logWarning(
          "[import_ENSIGHT] part %d has no supported surface elements, "
          "skipping",
          partId);
  }

  std::fclose(f);
  return !parts.empty();
}

bool readGeoFileHeader(
    const std::string &filename, std::vector<PartHeader> &parts)
{
  std::FILE *f = std::fopen(filename.c_str(), "rb");
  if (!f) {
    logError(
        "[import_ENSIGHT] cannot open geometry file '%s'", filename.c_str());
    return false;
  }

  if (readRecord80(f) != "C Binary") {
    logError(
        "[import_ENSIGHT] only binary EnSight Gold geometry files are "
        "supported (got non-binary header)");
    std::fclose(f);
    return false;
  }
  readRecord80(f); // description 1
  readRecord80(f); // description 2

  auto nodeIdLine = lowerStr(readRecord80(f));
  bool hasNodeIds = nodeIdLine.find("given") != std::string::npos
      || nodeIdLine.find("ignore") != std::string::npos;

  auto elemIdLine = lowerStr(readRecord80(f));
  bool hasElemIds = elemIdLine.find("given") != std::string::npos
      || elemIdLine.find("ignore") != std::string::npos;

  std::string token = lowerStr(readRecord80(f));
  if (token == "extents") {
    std::fseek(f, 6 * sizeof(float), SEEK_CUR);
    token = lowerStr(readRecord80(f));
  }

  while (!token.empty()) {
    if (token != "part") {
      logWarning("[import_ENSIGHT] expected 'part', got '%s'", token.c_str());
      break;
    }

    PartHeader ph;
    int32_t partId;
    std::fread(&partId, sizeof(int32_t), 1, f);
    ph.id = partId;
    ph.description = readRecord80(f);

    auto coordTag = lowerStr(readRecord80(f));
    if (coordTag != "coordinates") {
      token = lowerStr(readRecord80(f));
      continue;
    }

    int32_t numNodes;
    std::fread(&numNodes, sizeof(int32_t), 1, f);
    ph.numNodes = numNodes;

    // Skip node IDs
    if (hasNodeIds)
      std::fseek(f, numNodes * (long)sizeof(int32_t), SEEK_CUR);

    // Skip coordinate data (x, y, z arrays)
    std::fseek(f, 3L * numNodes * sizeof(float), SEEK_CUR);

    parts.push_back(std::move(ph));

    // Skip element blocks until next "part" or EOF
    token = lowerStr(readRecord80(f));
    while (!token.empty() && token != "part") {
      auto et = parseElemType(token);
      int32_t numElems;
      std::fread(&numElems, sizeof(int32_t), 1, f);

      if (hasElemIds)
        std::fseek(f, numElems * (long)sizeof(int32_t), SEEK_CUR);

      if (et == EnsightElemType::nsided) {
        std::vector<int32_t> nodeCounts(numElems);
        std::fread(nodeCounts.data(), sizeof(int32_t), numElems, f);
        int32_t totalNodes = 0;
        for (auto n : nodeCounts)
          totalNodes += n;
        std::fseek(f, totalNodes * (long)sizeof(int32_t), SEEK_CUR);
      } else if (et == EnsightElemType::nfaced) {
        std::vector<int32_t> faceCounts(numElems);
        std::fread(faceCounts.data(), sizeof(int32_t), numElems, f);
        int32_t totalFaces = 0;
        for (auto n : faceCounts)
          totalFaces += n;
        std::vector<int32_t> faceNodeCounts(totalFaces);
        std::fread(faceNodeCounts.data(), sizeof(int32_t), totalFaces, f);
        int32_t totalNodes = 0;
        for (auto n : faceNodeCounts)
          totalNodes += n;
        std::fseek(f, totalNodes * (long)sizeof(int32_t), SEEK_CUR);
      } else {
        int npe = nodesPerElem(et);
        if (npe > 0)
          std::fseek(f, (long)numElems * npe * sizeof(int32_t), SEEK_CUR);
      }

      token = lowerStr(readRecord80(f));
    }
  }

  std::fclose(f);
  return !parts.empty();
}

void readVarFile(const std::string &filename,
    const std::vector<Part> &parts,
    int numComponents,
    core::FlatMap<int, std::vector<float>> &out)
{
  std::FILE *f = std::fopen(filename.c_str(), "rb");
  if (!f) {
    logWarning(
        "[import_ENSIGHT] cannot open variable file '%s'", filename.c_str());
    return;
  }

  logInfo("[import_ENSIGHT] reading variable file '%s'", filename.c_str());

  readRecord80(f); // description

  core::FlatMap<int, const Part *> byId;
  for (const auto &p : parts)
    byId[p.id] = &p;

  std::string token = lowerStr(readRecord80(f));
  while (!token.empty()) {
    if (token != "part") {
      logWarning("[import_ENSIGHT] var file: expected 'part', got '%s'",
          token.c_str());
      break;
    }

    int32_t partId;
    std::fread(&partId, sizeof(int32_t), 1, f);

    auto assocTag = lowerStr(readRecord80(f));
    if (assocTag != "coordinates") {
      // Cell-centered data: we cannot skip without knowing element counts.
      logWarning(
          "[import_ENSIGHT] cell-centered variable data for part %d "
          "not supported, stopping variable read",
          partId);
      break;
    }

    if (!byId.contains(partId)) {
      logWarning("[import_ENSIGHT] var file: unknown part id %d", partId);
      token = lowerStr(readRecord80(f));
      continue;
    }

    int numNodes = (int)byId[partId]->x.size();
    std::vector<float> &data = out[partId];

    if (numComponents == 1) {
      data.resize(numNodes);
      std::fread(data.data(), sizeof(float), numNodes, f);
    } else {
      // File stores all comp-0 values, then all comp-1, ..., then all comp-N.
      // Convert to interleaved (AoS) layout: data[i*nc + c].
      std::vector<float> raw(numNodes * numComponents);
      std::fread(raw.data(), sizeof(float), numNodes * numComponents, f);
      data.resize(numNodes * numComponents);
      for (int c = 0; c < numComponents; ++c)
        for (int i = 0; i < numNodes; ++i)
          data[i * numComponents + c] = raw[c * numNodes + i];
    }

    token = lowerStr(readRecord80(f));
  }

  std::fclose(f);
}

} // namespace tsd::io::ensight
