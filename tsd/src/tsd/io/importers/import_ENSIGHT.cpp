// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/Logging.hpp"
#include "tsd/scene/algorithms/computeScalarRange.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
// std
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace tsd::io {

using namespace tsd::core;

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
    std::map<FaceKey, FaceEntry> &faceMap)
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
    const std::map<FaceKey, FaceEntry> &faceMap, std::vector<uint32_t> &out)
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

struct EnsightVarInfo
{
  std::string filenamePattern;
  std::string association; // "vertex" or "cell"
  std::string type; // "scalar" or "vector"
};

struct EnsightCaseData
{
  std::string geoPattern;
  int numSteps{1};
  int startNumber{1};
  int increment{1};
  std::map<std::string, EnsightVarInfo> variables; // varname → info
};

// Expand an EnSight wildcard pattern ("geo.***") into explicit filenames.
static std::vector<std::string> expandPattern(
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

static bool parseCase(const std::string &path, EnsightCaseData &out)
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

struct EnsightPart
{
  int id{-1};
  std::string description;
  std::vector<float> x, y, z; // per-node coordinates
  std::vector<uint32_t> triIndices; // triangulated, 0-based
};

static bool readGeoFile(
    const std::string &filename, std::vector<EnsightPart> &parts)
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

    EnsightPart part;
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
    std::map<FaceKey, FaceEntry> faceMap;

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

// Read per-node variable data from a binary variable file.
// Scalar: out[partId][i]     = value at node i.
// Vector: out[partId][i*3+c] = component c at node i (interleaved XYZ).
// Fortran-order storage in the file (all X, then all Y, then all Z) is
// converted to C-order interleaved on the way out.
static void readVarFile(const std::string &filename,
    const std::vector<EnsightPart> &parts,
    int numComponents,
    std::map<int, std::vector<float>> &out)
{
  std::FILE *f = std::fopen(filename.c_str(), "rb");
  if (!f) {
    logWarning(
        "[import_ENSIGHT] cannot open variable file '%s'", filename.c_str());
    return;
  }

  logInfo("[import_ENSIGHT] reading variable file '%s'", filename.c_str());

  readRecord80(f); // description

  std::map<int, const EnsightPart *> byId;
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

    auto it = byId.find(partId);
    if (it == byId.end()) {
      logWarning("[import_ENSIGHT] var file: unknown part id %d", partId);
      token = lowerStr(readRecord80(f));
      continue;
    }

    int numNodes = (int)it->second->x.size();
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

void import_ENSIGHT(Scene &scene,
    tsd::animation::SceneAnimation &sceneAnim,
    const char *filepath,
    LayerNodeRef location,
    const std::vector<std::string> &fields,
    int timestep)
{
  (void)sceneAnim;
  if (!location)
    location = scene.defaultLayer()->root();

  const std::string caseFile(filepath);
  const std::string caseDir = pathOf(caseFile);

  EnsightCaseData caseData;
  if (!parseCase(caseFile, caseData))
    return;

  if (timestep < 0 || timestep >= caseData.numSteps) {
    logError("[import_ENSIGHT] timestep %d out of range [0, %d)",
        timestep,
        caseData.numSteps);
    return;
  }

  logStatus("[import_ENSIGHT] loading '%s' (timestep %d/%d)",
      caseFile.c_str(),
      timestep,
      caseData.numSteps);

  const auto expandedPatterns = expandPattern(caseData.geoPattern,
      caseData.startNumber,
      caseData.increment,
      caseData.numSteps);
  const std::string geoFile = caseDir
      + (timestep < expandedPatterns.size() ? expandedPatterns[timestep]
                                            : expandedPatterns.front());

  std::vector<EnsightPart> parts;
  if (!readGeoFile(geoFile, parts)) {
    logError(
        "[import_ENSIGHT] failed to read geometry file '%s'", geoFile.c_str());
    return;
  }

  logStatus("[import_ENSIGHT] read %zu part(s) from %s",
      parts.size(),
      geoFile.c_str());

  // Read per-node variable data for all parts (first timestep only)
  struct VarData
  {
    std::map<int, std::vector<float>> perPart;
    int numComponents{1};
  };
  std::map<std::string, VarData> varData;

  // Build the ordered list of variable names to load.
  // If the caller specified a field list, use that order; otherwise use all
  // node-centered scalar/vector variables (up to the 4-slot ANARI limit).
  std::vector<std::string> varOrder;
  if (!fields.empty()) {
    for (const auto &f : fields)
      if (caseData.variables.count(f))
        varOrder.push_back(f);
      else
        logWarning(
            "[import_ENSIGHT] requested field '%s' not found in case "
            "file, skipping",
            f.c_str());
  } else {
    for (const auto &[name, info] : caseData.variables) {
      if (info.association != "vertex")
        continue;
      if (info.type != "scalar" && info.type != "vector")
        continue;
      varOrder.push_back(name);
      if (varOrder.size() == 4)
        break;
    }
  }

  for (const auto &name : varOrder) {
    const auto &info = caseData.variables.at(name);
    if (info.association != "vertex")
      continue;
    if (info.type != "scalar" && info.type != "vector")
      continue;

    int nc = (info.type == "vector") ? 3 : 1;
    const auto expandedPatterns = expandPattern(info.filenamePattern,
        caseData.startNumber,
        caseData.increment,
        caseData.numSteps);
    const std::string varFile = caseDir
        + (timestep < expandedPatterns.size() ? expandedPatterns[timestep]
                                              : expandedPatterns.front());

    VarData vd;
    vd.numComponents = nc;
    readVarFile(varFile, parts, nc, vd.perPart);
    varData[name] = std::move(vd);
  }

  auto root = scene.insertChildTransformNode(location);

  for (const auto &part : parts) {
    const int numNodes = (int)part.x.size();
    const int numTris = (int)part.triIndices.size() / 3;
    if (numTris == 0)
      continue;

    const std::string partName = part.description.empty()
        ? ("part_" + std::to_string(part.id))
        : part.description;

    auto geom = scene.createObject<Geometry>(tokens::geometry::triangle);
    geom->setName(partName.c_str());

    // Vertex positions
    auto posArr = scene.createArray(ANARI_FLOAT32_VEC3, numNodes);
    auto *pos = posArr->mapAs<float3>();
    for (int i = 0; i < numNodes; ++i)
      pos[i] = float3(part.x[i], part.y[i], part.z[i]);
    posArr->unmap();
    geom->setParameterObject("vertex.position", *posArr);

    // Triangle indices
    auto idxArr = scene.createArray(ANARI_UINT32_VEC3, numTris);
    auto *idx = idxArr->mapAs<uint3>();
    const uint32_t *src = part.triIndices.data();
    for (int t = 0; t < numTris; ++t)
      idx[t] = uint3(src[t * 3], src[t * 3 + 1], src[t * 3 + 2]);
    idxArr->unmap();
    geom->setParameterObject("primitive.index", *idxArr);

    // Attach per-node variables as vertex attributes in the requested order.
    // ANARI supports attribute0..attribute3; warn if more than 4 are requested.
    int attrSlot = 0;
    ArrayRef firstScalarArr;
    for (const auto &varName : varOrder) {
      if (attrSlot > 3) {
        logWarning(
            "[import_ENSIGHT] more than 4 fields requested; '%s' and "
            "subsequent fields exceed the ANARI attribute limit and will "
            "not be loaded",
            varName.c_str());
        break;
      }
      auto vdIt = varData.find(varName);
      if (vdIt == varData.end())
        continue;
      auto &vd = vdIt->second;
      auto it = vd.perPart.find(part.id);
      if (it == vd.perPart.end())
        continue;
      const auto &data = it->second;
      const std::string param = "vertex.attribute" + std::to_string(attrSlot);

      if (vd.numComponents == 1) {
        if ((int)data.size() != numNodes)
          continue;
        auto arr = scene.createArray(ANARI_FLOAT32, numNodes);
        std::copy(data.begin(), data.end(), arr->mapAs<float>());
        arr->unmap();
        geom->setParameterObject(param.c_str(), *arr);
        if (!firstScalarArr)
          firstScalarArr = arr;
      } else { // vector (3 components) → magnitude + x + y + z
        if ((int)data.size() != numNodes * vd.numComponents)
          continue;
        if (attrSlot + 3 > 3) {
          logWarning(
              "[import_ENSIGHT] vector field '%s' needs 4 attribute "
              "slots starting at %d, which exceeds the ANARI limit; "
              "skipping",
              varName.c_str(),
              attrSlot);
          continue;
        }
        auto magArr = scene.createArray(ANARI_FLOAT32, numNodes);
        auto xArr = scene.createArray(ANARI_FLOAT32, numNodes);
        auto yArr = scene.createArray(ANARI_FLOAT32, numNodes);
        auto zArr = scene.createArray(ANARI_FLOAT32, numNodes);
        auto *mag = magArr->mapAs<float>();
        auto *x = xArr->mapAs<float>();
        auto *y = yArr->mapAs<float>();
        auto *z = zArr->mapAs<float>();
        for (int i = 0; i < numNodes; ++i) {
          float vx = data[i * 3], vy = data[i * 3 + 1], vz = data[i * 3 + 2];
          x[i] = vx;
          y[i] = vy;
          z[i] = vz;
          mag[i] = std::sqrt(vx * vx + vy * vy + vz * vz);
        }
        magArr->unmap();
        xArr->unmap();
        yArr->unmap();
        zArr->unmap();
        ArrayRef vComponents[] = {magArr, xArr, yArr, zArr};
        for (int k = 0; k < 4; ++k) {
          auto attr = "vertex.attribute" + std::to_string(attrSlot + k);
          geom->setParameterObject(attr.c_str(), *vComponents[k]);
        }
        if (!firstScalarArr)
          firstScalarArr = magArr;
        attrSlot += 4;
      }
      if (vd.numComponents == 1)
        ++attrSlot;
    }

    // Color-mapped material when a scalar field is present, otherwise default
    MaterialRef mat;
    if (firstScalarArr) {
      mat = scene.createObject<Material>(tokens::material::physicallyBased);
      auto range = computeScalarRange(*firstScalarArr);
      mat->setParameterObject(
          "baseColor", *makeDefaultColorMapSampler(scene, range));
    } else {
      mat = scene.defaultMaterial();
    }

    auto surface = scene.createSurface(partName.c_str(), geom, mat);
    auto nodeRef = scene.insertChildObjectNode(root, surface, partName.c_str());
  }

  logStatus("[import_ENSIGHT] done, %zu part(s) loaded", parts.size());
}

} // namespace tsd::io
