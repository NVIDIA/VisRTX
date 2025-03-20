// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/authoring/importers.hpp"
#include "tsd/core/Logging.hpp"

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {
const std::string KEY_ATOM = "ATOM";
const std::string KEY_HETATM = "HETATM";
const std::string KEY_HEADER = "HEADER";
const std::string KEY_TITLE = "TITLE";
const std::string KEY_CONECT = "CONECT";
const std::string KEY_SEQRES = "SEQRES";
const std::string KEY_REMARK = "REMARK";

// https://en.wikipedia.org/wiki/Atomic_radius
const double DEFAULT_ATOM_RADIUS = 25.0;
const std::map<std::string, double> atomicRadii = {{{"H"}, {53.0}},
    {{"HE"}, {31.0}},
    {{"LI"}, {167.0}},
    {{"BE"}, {112.0}},
    {{"B"}, {87.0}},
    {{"C"}, {67.0}},
    {{"N"}, {56.0}},
    {{"O"}, {48.0}},
    {{"F"}, {42.0}},
    {{"NE"}, {38.0}},
    {{"NA"}, {190.0}},
    {{"MG"}, {145.0}},
    {{"AL"}, {118.0}},
    {{"SI"}, {111.0}},
    {{"P"}, {98.0}},
    {{"S"}, {88.0}},
    {{"CL"}, {79.0}},
    {{"AR"}, {71.0}},
    {{"K"}, {243.0}},
    {{"CA"}, {194.0}},
    {{"SC"}, {184.0}},
    {{"TI"}, {176.0}},
    {{"V"}, {171.0}},
    {{"CR"}, {166.0}},
    {{"MN"}, {161.0}},
    {{"FE"}, {156.0}},
    {{"CO"}, {152.0}},
    {{"NI"}, {149.0}},
    {{"CU"}, {145.0}},
    {{"ZN"}, {142.0}},
    {{"GA"}, {136.0}},
    {{"GE"}, {125.0}},
    {{"AS"}, {114.0}},
    {{"SE"}, {103.0}},
    {{"BR"}, {94.0}},
    {{"KR"}, {88.0}},
    {{"RB"}, {265.0}},
    {{"SR"}, {219.0}},
    {{"Y"}, {212.0}},
    {{"ZR"}, {206.0}},
    {{"NB"}, {198.0}},
    {{"MO"}, {190.0}},
    {{"TC"}, {183.0}},
    {{"RU"}, {178.0}},
    {{"RH"}, {173.0}},
    {{"PD"}, {169.0}},
    {{"AG"}, {165.0}},
    {{"CD"}, {161.0}},
    {{"IN"}, {156.0}},
    {{"SN"}, {145.0}},
    {{"SB"}, {133.0}},
    {{"TE"}, {123.0}},
    {{"I"}, {115.0}},
    {{"XE"}, {108.0}},
    {{"CS"}, {298.0}},
    {{"BA"}, {253.0}},
    {{"LA"}, {226.0}},
    {{"CE"}, {210.0}},
    {{"PR"}, {247.0}},
    {{"ND"}, {206.0}},
    {{"PM"}, {205.0}},
    {{"SM"}, {238.0}},
    {{"EU"}, {231.0}},
    {{"GD"}, {233.0}},
    {{"TB"}, {225.0}},
    {{"DY"}, {228.0}},
    {{"HO"}, {226.0}},
    {{"ER"}, {226.0}},
    {{"TM"}, {222.0}},
    {{"YB"}, {222.0}},
    {{"LU"}, {217.0}},
    {{"HF"}, {208.0}},
    {{"TA"}, {200.0}},
    {{"W"}, {193.0}},
    {{"RE"}, {188.0}},
    {{"OS"}, {185.0}},
    {{"IR"}, {180.0}},
    {{"PT"}, {177.0}},
    {{"AU"}, {174.0}},
    {{"HG"}, {171.0}},
    {{"TL"}, {156.0}},
    {{"PB"}, {154.0}},
    {{"BI"}, {143.0}},
    {{"PO"}, {135.0}},
    {{"AT"}, {127.0}},
    {{"RN"}, {120.0}}};

typedef struct
{
  uint8_t r, g, b;
} RGB;
using RGBMap = std::map<std::string, RGB>;
const RGBMap atomColorMap = {{"H", {0xDF, 0xDF, 0xDF}},
    {"He", {0xD9, 0xFF, 0xFF}},
    {"Li", {0xCC, 0x80, 0xFF}},
    {"Be", {0xC2, 0xFF, 0x00}},
    {"B", {0xFF, 0xB5, 0xB5}},
    {"C", {0x90, 0x90, 0x90}},
    {"N", {0x30, 0x50, 0xF8}},
    {"O", {0xFF, 0x0D, 0x0D}},
    {"F", {0x9E, 0x05, 0x1}},
    {"Ne", {0xB3, 0xE3, 0xF5}},
    {"Na", {0xAB, 0x5C, 0xF2}},
    {"Mg", {0x8A, 0xFF, 0x00}},
    {"Al", {0xBF, 0xA6, 0xA6}},
    {"Si", {0xF0, 0xC8, 0xA0}},
    {"P", {0xFF, 0x80, 0x00}},
    {"S", {0xFF, 0xFF, 0x30}},
    {"Cl", {0x1F, 0xF0, 0x1F}},
    {"Ar", {0x80, 0xD1, 0xE3}},
    {"K", {0x8F, 0x40, 0xD4}},
    {"Ca", {0x3D, 0xFF, 0x00}},
    {"Sc", {0xE6, 0xE6, 0xE6}},
    {"Ti", {0xBF, 0xC2, 0xC7}},
    {"V", {0xA6, 0xA6, 0xAB}},
    {"Cr", {0x8A, 0x99, 0xC7}},
    {"Mn", {0x9C, 0x7A, 0xC7}},
    {"Fe", {0xE0, 0x66, 0x33}},
    {"Co", {0xF0, 0x90, 0xA0}},
    {"Ni", {0x50, 0xD0, 0x50}},
    {"Cu", {0xC8, 0x80, 0x33}},
    {"Zn", {0x7D, 0x80, 0xB0}},
    {"Ga", {0xC2, 0x8F, 0x8F}},
    {"Ge", {0x66, 0x8F, 0x8F}},
    {"As", {0xBD, 0x80, 0xE3}},
    {"Se", {0xFF, 0xA1, 0x00}},
    {"Br", {0xA6, 0x29, 0x29}},
    {"Kr", {0x5C, 0xB8, 0xD1}},
    {"Rb", {0x70, 0x2E, 0xB0}},
    {"Sr", {0x00, 0xFF, 0x00}},
    {"Y", {0x94, 0xFF, 0xFF}},
    {"Zr", {0x94, 0xE0, 0xE0}},
    {"Nb", {0x73, 0xC2, 0xC9}},
    {"Mo", {0x54, 0xB5, 0xB5}},
    {"Tc", {0x3B, 0x9E, 0x9E}},
    {"Ru", {0x24, 0x8F, 0x8F}},
    {"Rh", {0x0A, 0x7D, 0x8C}},
    {"Pd", {0x69, 0x85, 0x00}},
    {"Ag", {0xC0, 0xC0, 0xC0}},
    {"Cd", {0xFF, 0xD9, 0x8F}},
    {"In", {0xA6, 0x75, 0x73}},
    {"Sn", {0x66, 0x80, 0x80}},
    {"Sb", {0x9E, 0x63, 0xB5}},
    {"Te", {0xD4, 0x7A, 0x00}},
    {"I", {0x94, 0x00, 0x94}},
    {"Xe", {0x42, 0x9E, 0xB0}},
    {"Cs", {0x57, 0x17, 0x8F}},
    {"Ba", {0x00, 0xC9, 0x00}},
    {"La", {0x70, 0xD4, 0xFF}},
    {"Ce", {0xFF, 0xFF, 0xC7}},
    {"Pr", {0xD9, 0xFF, 0xC7}},
    {"Nd", {0xC7, 0xFF, 0xC7}},
    {"Pm", {0xA3, 0xFF, 0xC7}},
    {"Sm", {0x8F, 0xFF, 0xC7}},
    {"Eu", {0x61, 0xFF, 0xC7}},
    {"Gd", {0x45, 0xFF, 0xC7}},
    {"Tb", {0x30, 0xFF, 0xC7}},
    {"Dy", {0x1F, 0xFF, 0xC7}},
    {"Ho", {0x00, 0xFF, 0x9C}},
    {"Er", {0x00, 0xE6, 0x75}},
    {"Tm", {0x00, 0xD4, 0x52}},
    {"Yb", {0x00, 0xBF, 0x38}},
    {"Lu", {0x00, 0xAB, 0x24}},
    {"Hf", {0x4D, 0xC2, 0xFF}},
    {"Ta", {0x4D, 0xA6, 0xFF}},
    {"W", {0x21, 0x94, 0xD6}},
    {"Re", {0x26, 0x7D, 0xAB}},
    {"Os", {0x26, 0x66, 0x96}},
    {"Ir", {0x17, 0x54, 0x87}},
    {"Pt", {0xD0, 0xD0, 0xE0}},
    {"Au", {0xFF, 0xD1, 0x23}},
    {"Hg", {0xB8, 0xB8, 0xD0}},
    {"Tl", {0xA6, 0x54, 0x4D}},
    {"Pb", {0x57, 0x59, 0x61}},
    {"Bi", {0x9E, 0x4F, 0xB5}},
    {"Po", {0xAB, 0x5C, 0x00}},
    {"At", {0x75, 0x4F, 0x45}},
    {"Rn", {0x42, 0x82, 0x96}},
    {"Fr", {0x42, 0x00, 0x66}},
    {"Ra", {0x00, 0x7D, 0x00}},
    {"Ac", {0x70, 0xAB, 0xFA}},
    {"Th", {0x00, 0xBA, 0xFF}},
    {"Pa", {0x00, 0xA1, 0xFF}},
    {"U", {0x00, 0x8F, 0xFF}},
    {"Np", {0x00, 0x80, 0xFF}},
    {"Pu", {0x00, 0x6B, 0xFF}},
    {"Am", {0x54, 0x5C, 0xF2}},
    {"Cm", {0x78, 0x5C, 0xE3}},
    {"Bk", {0x8A, 0x4F, 0xE3}},
    {"Cf", {0xA1, 0x36, 0xD4}},
    {"Es", {0xB3, 0x1F, 0xD4}},
    {"Fm", {0xB3, 0x1F, 0xBA}},
    {"Md", {0xB3, 0x0D, 0xA6}},
    {"No", {0xBD, 0x0D, 0x87}},
    {"Lr", {0xC7, 0x00, 0x66}},
    {"Rf", {0xCC, 0x00, 0x59}},
    {"Db", {0xD1, 0x00, 0x4F}},
    {"Sg", {0xD9, 0x00, 0x45}},
    {"Bh", {0xE0, 0x00, 0x38}},
    {"Hs", {0xE6, 0x00, 0x2E}},
    {"Mt", {0xEB, 0x00, 0x26}},
    {"O1", {0xFF, 0x0D, 0x0D}}};

} // namespace

namespace tsd {

struct Atom
{
  int serial; // Atom serial number
  char altLoc; // Alternate location indicator
  std::string residueName; // Residue name
  char chainID; // Chain identifier
  int residueSeq; // Residue sequence number
  char iCode; // Code for insertion of residues
  float3 position; // x, y, z coordinates
  double occupancy; // Occupancy
  double tempFactor; // Temperature factor
  std::string element; // Element symbol
  std::string charge; // Charge on the atom
};

using Atoms = std::vector<Atom>;
using AtomsMap = std::map<std::string, Atoms>;

/**
 * Trims a string of whitespace characters.
 *
 * @param str The string to trim.
 * @return The trimmed string.
 */
static std::string trim(const std::string &str)
{
  size_t first = str.find_first_not_of(" \t\n\r");
  if (first == std::string::npos)
    return "";
  size_t last = str.find_last_not_of(" \t\n\r");
  return str.substr(first, (last - first + 1));
}

/**
 * Parses an atom line from a PDB file.
 *
 * @param line The line to parse.
 * @param atomsMap The map of atoms to store the parsed atom in.
 */
void parseAtomLine(const std::string &line, AtomsMap &atomsMap)
{
  // PDB format specification:
  // https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM

  if (line.length() < 54) { // Minimum length for required fields
    throw std::runtime_error("Line too short for ATOM record");
  }

  Atom atom;
  atom.serial = std::stoi(line.substr(6, 5));
  atom.altLoc = line[16];
  atom.residueName = trim(line.substr(17, 3));
  atom.chainID = line[21];
  atom.residueSeq = std::stoi(line.substr(22, 4));
  atom.iCode = line[26];
  atom.position.x = std::stod(line.substr(30, 8));
  atom.position.y = std::stod(line.substr(38, 8));
  atom.position.z = std::stod(line.substr(46, 8));

  // Optional fields
  if (line.length() >= 60) {
    atom.occupancy = std::stod(line.substr(54, 6));
  }
  if (line.length() >= 66) {
    atom.tempFactor = std::stod(line.substr(60, 6));
  }
  if (line.length() >= 78) {
    atom.element = trim(line.substr(76, 2));
  }
  if (line.length() >= 80) {
    atom.charge = trim(line.substr(78, 2));
  }

  const std::string name = trim(line.substr(12, 4));
  atomsMap[name].push_back(atom);
}

/**
 * Reads a PDB file and generates a 3D representation.
 *
 * @param ctx Context in which to create the 3D representation.
 * @param filename Path to the SWC file to read.
 * @param location Node in the scene graph where the 3D representation should be
 * added.
 */
void readPDBFile(Context &ctx, const char *filename, LayerNodeRef location)
{
  std::ifstream file(filename);
  if (!file.is_open()) {
    logError("Error opening file: %s", filename);
    return;
  }

  AtomsMap atomsMap;
  std::string line;

  while (std::getline(file, line)) {
    if (line.substr(0, 4) == KEY_ATOM || line.substr(0, 6) == KEY_HETATM) {
      try {
        parseAtomLine(line, atomsMap);
      } catch (const std::exception &e) {
        logError("Error parsing line: %s", line.c_str());
        logError("Error: %s", e.what());
        return;
      }
    }
  }

  const auto pdbLocation = ctx.defaultLayer()->insert_first_child(
      location, tsd::utility::Any(ANARI_GEOMETRY, 1));

  for (const auto &[name, atoms] : atomsMap) {
    // Generate spheres for each atom
    auto spheres =
        ctx.createObject<tsd::Geometry>(tsd::tokens::geometry::sphere);
    spheres->setName(name.c_str());

    // Initialize the positions and radii of the spheres
    const size_t numAtoms = atoms.size();
    std::vector<float3> spherePositions;
    spherePositions.reserve(numAtoms);
    std::vector<float> sphereRadii;
    sphereRadii.reserve(numAtoms);

    for (const auto &atom : atoms) {
      spherePositions.push_back(atom.position);

      double radius = DEFAULT_ATOM_RADIUS;
      auto it = atomicRadii.find(atom.element);
      if (it != atomicRadii.end())
        radius = (*it).second;
      else {
        it = atomicRadii.find(name);
        if (it != atomicRadii.end())
          radius = (*it).second;
      }

      sphereRadii.push_back(radius / 1e2);
    }

    // Create arrays to store the positions and radii of the spheres
    auto spherePositionArray = ctx.createArray(ANARI_FLOAT32_VEC3, numAtoms);
    auto sphereRadiusArray = ctx.createArray(ANARI_FLOAT32, numAtoms);

    spherePositionArray->setData(spherePositions);
    sphereRadiusArray->setData(sphereRadii);

    // Set the positions and radii of the spheres
    spheres->setParameterObject("vertex.position"_t, *spherePositionArray);
    spheres->setParameterObject("vertex.radius"_t, *sphereRadiusArray);

    const auto it = atomColorMap.find(name);
    tsd::float3 baseColor{1.0, 1.0, 1.0};
    if (it != atomColorMap.end())
      baseColor = tsd::float3((*it).second.r / 255.0,
          (*it).second.g / 255.0,
          (*it).second.b / 255.0);

    const std::string basename =
        std::filesystem::path(filename).filename().string();
    const std::string spheresName = basename + " (" + name + ")";

    auto m = ctx.createObject<Material>(tokens::material::matte);
    m->setParameter("color"_t, ANARI_FLOAT32_VEC3, &baseColor);
    auto sphereSurface = ctx.createSurface(spheresName.c_str(), spheres, m);
    const auto atomsLocation = ctx.defaultLayer()->insert_first_child(
        pdbLocation, tsd::utility::Any(ANARI_GEOMETRY, 1));

    ctx.insertChildObjectNode(atomsLocation, sphereSurface);
  }
}

/**
 * Reads a PDB file and generates a 3D representation.
 *
 * @param ctx Context in which to create the 3D representation.
 * @param filename Path to the SWC file to read.
 * @param location Node in the scene graph where the 3D representation should be
 * added.
 */
void import_PDB(Context &ctx, const char *filename, LayerNodeRef location)
{
  readPDBFile(ctx, filename, location);
}
}; // namespace tsd
