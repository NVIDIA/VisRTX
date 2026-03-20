// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// std
#include <cmath>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace pbrt {

// Resolved parameter value — one of these vector types per "type name" pair.
using ParamValue = std::variant<std::vector<float>,
    std::vector<int>,
    std::vector<std::string>,
    std::vector<bool>>;

struct ParamList
{
  std::map<std::string, ParamValue> values;
  // PBRT v4 typed-parameter qualifier ("rgb", "spectrum", "blackbody",
  // "float", "texture", …) carried alongside the value so importers can
  // distinguish e.g. `"blackbody L" [5500]` from `"rgb L" [a b c]`. Keys
  // mirror `values`.
  std::map<std::string, std::string> types;

  float getFloat(const std::string &name, float def = 0.f) const;
  int getInt(const std::string &name, int def = 0) const;
  std::string getString(
      const std::string &name, const std::string &def = "") const;
  bool getBool(const std::string &name, bool def = false) const;
  const std::vector<float> &getFloats(const std::string &name) const;
  const std::vector<int> &getInts(const std::string &name) const;
  std::string getType(const std::string &name) const;
  bool has(const std::string &name) const;
};

// 4x4 column-major transform matrix.
struct Transform
{
  float m[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

  static Transform identity();
  static Transform translate(float x, float y, float z);
  static Transform scale(float x, float y, float z);
  static Transform rotate(float angleDeg, float ax, float ay, float az);
  static Transform lookAt(float ex,
      float ey,
      float ez,
      float lx,
      float ly,
      float lz,
      float ux,
      float uy,
      float uz);

  Transform operator*(const Transform &rhs) const;
};

struct Shape
{
  std::string type;
  ParamList params;
  Transform objectToWorld;
  std::string materialName;
  std::string areaLightType;
  ParamList areaLightParams;
  std::string interiorMedium;
  std::string exteriorMedium;
  bool reverseOrientation = false;
};

struct MediumDef
{
  std::string type;
  ParamList params;
};

struct MaterialDef
{
  std::string type;
  ParamList params;
};

struct TextureDef
{
  std::string name;
  std::string colorType;
  std::string implType;
  ParamList params;
};

struct LightDef
{
  std::string type;
  ParamList params;
  Transform lightToWorld;
};

struct CameraDef
{
  std::string type = "perspective";
  ParamList params;
  Transform cameraToWorld;
};

struct FilmDef
{
  std::string type = "rgb";
  int xResolution = 1920;
  int yResolution = 1080;
  std::string filename;
  ParamList params;
};

struct ObjectDef
{
  std::string name;
  std::vector<Shape> shapes;
  std::vector<LightDef> lights;
};

struct ObjectInstance
{
  std::string name;
  Transform instanceToWorld;
};

struct Scene
{
  CameraDef camera;
  FilmDef film;
  std::vector<Shape> shapes;
  std::vector<LightDef> lights;
  std::vector<ObjectInstance> instances;
  std::map<std::string, MaterialDef> namedMaterials;
  std::map<std::string, TextureDef> textures;
  std::map<std::string, ObjectDef> objects;
  std::map<std::string, MediumDef> namedMedia;
  std::map<std::string, Transform> coordinateSystems;
};

} // namespace pbrt
