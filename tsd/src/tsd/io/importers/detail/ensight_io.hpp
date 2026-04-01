// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace tsd::io::ensight {

struct VarInfo
{
  std::string filenamePattern;
  std::string association; // "vertex" or "cell"
  std::string type;        // "scalar" or "vector"
};

struct CaseData
{
  std::string geoPattern;
  int numSteps{1};
  int startNumber{1};
  int increment{1};
  std::map<std::string, VarInfo> variables; // varname -> info
};

struct Part
{
  int id{-1};
  std::string description;
  std::vector<float> x, y, z;         // per-node coordinates
  std::vector<uint32_t> triIndices;    // triangulated, 0-based
};

bool parseCase(const std::string &path, CaseData &out);

std::vector<std::string> expandPattern(
    const std::string &pat, int start, int incr, int steps);

bool readGeoFile(const std::string &filename, std::vector<Part> &parts);

// Read per-node variable data from a binary variable file.
// Scalar: out[partId][i]     = value at node i.
// Vector: out[partId][i*3+c] = component c at node i (interleaved XYZ).
void readVarFile(const std::string &filename,
    const std::vector<Part> &parts,
    int numComponents,
    std::map<int, std::vector<float>> &out);

} // namespace tsd::io::ensight
