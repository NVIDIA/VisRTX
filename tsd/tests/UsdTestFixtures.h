// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// Fixtures and lookups shared by the test_UsdImport*.cpp suites. Include this
// only from inside a `#if TSD_USE_USD` guard -- it needs OpenUSD-dependent
// declarations.

#pragma once

// catch
#include "catch.hpp"
// tsd
#include "tsd/animation/AnimationManager.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/scene/Scene.hpp"
#include "tsd/scene/objects/Material.hpp"
// std
#include <filesystem>
#include <fstream>
#include <random>
#include <string>

// A directory of this process' own, removed when the test binary exits.
struct ScopedFixtureDirectory
{
  ScopedFixtureDirectory();
  ~ScopedFixtureDirectory();

  std::filesystem::path path;
};

// Fixture files live in a directory unique to this process, so two concurrent
// runs of the test binary cannot collide on a name while relative asset
// references between fixtures still resolve.
inline const std::filesystem::path &fixtureDirectory()
{
  static const ScopedFixtureDirectory directory;
  return directory.path;
}

// Writes a text-format Stage to a temporary path for the lifetime of one
// scenario. USD's text format keeps every fixture readable next to the
// assertion it supports and keeps binary assets out of the repository.
struct StageFixture
{
  StageFixture(const char *name, const std::string &contents);
  ~StageFixture();

  std::string path() const;

 private:
  std::filesystem::path m_path;
};

// A Stage written to disk and imported into a Scene: what nearly every
// scenario needs before it can assert anything. Holding the Scene, the
// AnimationManager and the Import Report together lets a scenario open with
// one line of setup and go straight to its assertions.
struct ImportedStage
{
  ImportedStage(const char *name,
      const std::string &contents,
      const tsd::io::UsdImportOptions &options = {});

  std::string path() const;

  tsd::scene::Scene scene;
  tsd::animation::AnimationManager animMgr{&scene};
  tsd::io::UsdImportReport report;

 private:
  StageFixture m_stage;
};

// A real, decodable texture for the lifetime of one scenario. The import binds
// samplers by loading these, so a stand-in file with arbitrary bytes would not
// exercise the path -- this is a 1x1 uncompressed true-colour TGA, the
// smallest thing the image loader accepts that can be written by hand.
struct TextureFixture
{
  explicit TextureFixture(const char *name);
  ~TextureFixture();

  std::string path() const;

 private:
  std::filesystem::path m_path;
};

// Depth-first search for the first node whose name matches.
inline tsd::scene::LayerNodeRef findNode(
    tsd::scene::Layer *layer, const char *name)
{
  tsd::scene::LayerNodeRef found;
  layer->traverse(layer->root(), [&](auto &node, int) {
    if (!found && node->name() == name)
      found = layer->at(node.index());
    return true;
  });
  return found;
}

// The converted object a prim produced, found by the prim path the importer
// names it after.
template <typename T>
tsd::core::ObjectPoolRef<T> findObject(
    tsd::scene::Scene &scene, anari::DataType type, const char *name)
{
  for (size_t i = 0; i < scene.numberOfObjects(type); ++i) {
    auto object = scene.getObject<T>(i);
    if (object && object->name() == name)
      return object;
  }
  return {};
}

inline tsd::scene::GeometryRef findGeometry(
    tsd::scene::Scene &scene, const char *name)
{
  return findObject<tsd::scene::Geometry>(scene, ANARI_GEOMETRY, name);
}

// The material a Surface actually uses, rather than whatever happens to sit at
// index 0 of the pool (which is the Scene's own default material).
inline tsd::scene::Material *boundMaterial(tsd::scene::Scene &scene)
{
  auto surface = scene.getObject<tsd::scene::Surface>(0);
  REQUIRE(surface);
  auto *material = surface->parameterValueAsObject<tsd::scene::Material>(
      tsd::scene::tokens::surface::material);
  REQUIRE(material != nullptr);
  return material;
}

inline constexpr const char *QUAD_MESH_BODY = R"(
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
)";

// Inlined definitions ////////////////////////////////////////////////////////

inline ScopedFixtureDirectory::ScopedFixtureDirectory()
{
  // create_directory reports whether it was this process that made the
  // directory, so retrying on a taken name is what makes the choice safe
  // rather than merely unlikely.
  std::random_device entropy;
  const auto root = std::filesystem::temp_directory_path();
  do {
    path = root / ("tsd_test_usd_" + std::to_string(entropy()));
  } while (!std::filesystem::create_directory(path));
}

inline ScopedFixtureDirectory::~ScopedFixtureDirectory()
{
  std::error_code ec;
  std::filesystem::remove_all(path, ec);
}

inline StageFixture::StageFixture(const char *name, const std::string &contents)
    : m_path(fixtureDirectory() / name)
{
  std::ofstream file(m_path);
  file << contents;
}

inline StageFixture::~StageFixture()
{
  std::error_code ec;
  std::filesystem::remove(m_path, ec);
}

inline std::string StageFixture::path() const
{
  return m_path.string();
}

inline ImportedStage::ImportedStage(const char *name,
    const std::string &contents,
    const tsd::io::UsdImportOptions &options)
    : m_stage(name, contents)
{
  report = tsd::io::import_USD(scene, animMgr, path().c_str(), {}, options);
}

inline std::string ImportedStage::path() const
{
  return m_stage.path();
}

inline TextureFixture::TextureFixture(const char *name)
    : m_path(fixtureDirectory() / name)
{
  // clang-format off
  const unsigned char tga[] = {
      0, // no image ID
      0, // no colour map
      2, // uncompressed true-colour
      0, 0, 0, 0, 0, // empty colour map spec
      0, 0, 0, 0, // origin
      1, 0, // width
      1, 0, // height
      24, // bits per pixel
      0, // descriptor
      0x20, 0x40, 0x60 // one BGR pixel
  };
  // clang-format on
  std::ofstream file(m_path, std::ios::binary);
  file.write(reinterpret_cast<const char *>(tga), sizeof(tga));
}

inline TextureFixture::~TextureFixture()
{
  std::error_code ec;
  std::filesystem::remove(m_path, ec);
}

inline std::string TextureFixture::path() const
{
  return m_path.string();
}
