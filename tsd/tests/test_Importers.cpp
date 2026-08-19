// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/animation/AnimationManager.hpp"
#include "tsd/core/TSDMath.hpp"
#include "tsd/io/exporters.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
#include "tsd/scene/Scene.hpp"
// std
#include <cmath>
#include <filesystem>
#include <fstream>
#include <system_error>

SCENARIO("Splitting a path names its file and its directory", "[Importers]")
{
  // Callers concatenate the two back together to reach a file's sibling, so
  // whatever else the split does, it has to be reversible. On Windows that
  // rules out answering with the platform's own separator: these paths carry
  // '/', which Windows accepts, and handing back '/a/b\\volume.raw' would not
  // be the path that was passed in.
  auto rejoins = [](const char *path) {
    return tsd::io::pathOf(path) + tsd::io::fileOf(path) == path;
  };

  GIVEN("A path with a directory component")
  {
    THEN("The two halves rejoin into the original")
    {
      REQUIRE(tsd::io::fileOf("/a/b/volume.raw") == "volume.raw");
      REQUIRE(tsd::io::pathOf("/a/b/volume.raw") == "/a/b/");
      REQUIRE(rejoins("/a/b/volume.raw"));
      REQUIRE(tsd::io::fileOf("b/volume.raw") == "volume.raw");
      REQUIRE(tsd::io::pathOf("b/volume.raw") == "b/");
      REQUIRE(rejoins("b/volume.raw"));
    }
  }

  GIVEN("A path written with the separator the host prefers")
  {
    THEN("The two halves still rejoin")
    {
      REQUIRE(rejoins((std::filesystem::temp_directory_path() / "volume.raw")
              .string()
              .c_str()));
    }
  }

  GIVEN("A bare filename, as typed relative to the working directory")
  {
    THEN("It is the file, and there is no directory")
    {
      // Importers guard on fileOf() being non-empty before doing any work, so
      // answering "no file" here made every one of them a silent no-op.
      REQUIRE(tsd::io::fileOf("volume.raw") == "volume.raw");
      REQUIRE(tsd::io::pathOf("volume.raw").empty());
      REQUIRE(rejoins("volume.raw"));
    }
  }

  GIVEN("A path that names a directory rather than a file")
  {
    THEN("There is no file")
    {
      REQUIRE(tsd::io::fileOf("/a/b/").empty());
      REQUIRE(tsd::io::pathOf("/a/b/") == "/a/b/");
      REQUIRE(tsd::io::fileOf("").empty());
      REQUIRE(tsd::io::pathOf("").empty());
    }
  }

  GIVEN("A file directly under the root")
  {
    THEN("The directory is the root, and is not doubled")
    {
      REQUIRE(tsd::io::fileOf("/volume.raw") == "volume.raw");
      REQUIRE(tsd::io::pathOf("/volume.raw") == "/");
      REQUIRE(rejoins("/volume.raw"));
    }
  }
}

SCENARIO("A volume imports under a name relative to the working directory",
    "[Importers]")
{
  // The dimensions and voxel type come out of the filename, so an importer
  // needs the file half of the path whether or not a directory was given.
  const auto directory = std::filesystem::temp_directory_path();
  const char *name = "tsd_test_relative_2x2x2_uint8.raw";
  {
    std::ofstream file(directory / name, std::ios::binary);
    const unsigned char voxels[8] = {0, 32, 64, 96, 128, 160, 192, 255};
    file.write(reinterpret_cast<const char *>(voxels), sizeof(voxels));
  }

  tsd::scene::Scene scene;

  WHEN("The file is named without any directory")
  {
    const auto previous = std::filesystem::current_path();
    std::filesystem::current_path(directory);
    auto field = tsd::io::import_spatial_field(scene, name);
    std::filesystem::current_path(previous);

    THEN("It reads, and carries the name it was asked for")
    {
      REQUIRE(field);
      REQUIRE(field->name() == name);
    }
  }

  std::filesystem::remove(directory / name);
}

SCENARIO(
    "Volume transfer functions reject missing control points", "[Importers]")
{
  tsd::scene::Scene scene;
  auto volume = scene.createObject<tsd::scene::Volume>(
      tsd::scene::tokens::volume::transferFunction1D);
  tsd::core::TransferFunction transferFunction;

  WHEN("An empty transfer function is applied")
  {
    tsd::io::applyTransferFunction(scene, volume, transferFunction);

    THEN("The volume keeps its default scalar color")
    {
      REQUIRE(volume->parameterValueAsObject<tsd::scene::Array>("color")
          == nullptr);
    }
  }
}

SCENARIO(
    "Single volume file import uses a default transfer function", "[Importers]")
{
  const auto path =
      std::filesystem::temp_directory_path() / "tsd_test_1x1x1_uint8.raw";
  {
    std::ofstream file(path, std::ios::binary);
    const unsigned char voxel = 255;
    file.write(reinterpret_cast<const char *>(&voxel), sizeof(voxel));
  }

  tsd::scene::Scene scene;
  tsd::animation::AnimationManager animMgr(&scene);

  WHEN("A volume is imported through the single-file dispatcher")
  {
    tsd::io::import_file(
        scene, animMgr, {tsd::io::ImporterType::VOLUME, path.string()});

    THEN("The imported volume has a sampled color array")
    {
      REQUIRE(scene.numberOfObjects(ANARI_VOLUME) == 1);
      auto volume = scene.getObject<tsd::scene::Volume>(0);
      REQUIRE(volume);
      auto *color = volume->parameterValueAsObject<tsd::scene::Array>("color");
      REQUIRE(color != nullptr);
      REQUIRE(color->size() == 256);
    }
  }

  std::filesystem::remove(path);
}

SCENARIO("The spatial field dispatcher reads NanoVDB under both of its names",
    "[Importers]")
{
  // tsdVolumeToNanoVDB documents its output as '.vdb', so a NanoVDB grid
  // reaches TSD under that name as often as under '.nvdb', and both have to
  // find the same reader.
  const auto rawPath =
      std::filesystem::temp_directory_path() / "tsd_test_2x2x2_uint8.raw";
  {
    std::ofstream file(rawPath, std::ios::binary);
    const unsigned char voxels[8] = {0, 32, 64, 96, 128, 160, 192, 255};
    file.write(reinterpret_cast<const char *>(voxels), sizeof(voxels));
  }

  const auto vdbPath =
      std::filesystem::temp_directory_path() / "tsd_test_roundtrip.vdb";

  tsd::scene::Scene scene;

  GIVEN("A NanoVDB grid written out under a '.vdb' name")
  {
    auto source =
        tsd::io::import_spatial_field(scene, rawPath.string().c_str());
    REQUIRE(source);
    tsd::io::export_StructuredVolumeToNanoVDB(source.data(), vdbPath.string());
    REQUIRE(std::filesystem::exists(vdbPath));

    WHEN("The '.vdb' file is dispatched")
    {
      auto field =
          tsd::io::import_spatial_field(scene, vdbPath.string().c_str());

      THEN("The NanoVDB reader loads it")
      {
        REQUIRE(field);
        REQUIRE(field->subtype() == tsd::scene::tokens::spatial_field::nanovdb);
      }
    }
  }

  std::filesystem::remove(rawPath);
  std::filesystem::remove(vdbPath);
}

SCENARIO("The NanoVDB reader rejects a file it cannot be holding a grid",
    "[Importers]")
{
  // nanovdb::io::readGrid never returns on a file too short to hold a header,
  // so a stray '.vdb' -- an interrupted download, an empty placeholder, an
  // actual OpenVDB grid -- would hang whatever asked for it. If this scenario
  // ever times out rather than failing, that guard is gone.
  const auto path =
      std::filesystem::temp_directory_path() / "tsd_test_not_a_grid.vdb";

  auto writeBytes = [&](const void *bytes, size_t numBytes) {
    std::ofstream file(path, std::ios::binary);
    file.write(static_cast<const char *>(bytes), numBytes);
  };

  tsd::scene::Scene scene;

  GIVEN("An empty file under a '.vdb' name")
  {
    writeBytes(nullptr, 0);

    THEN("No field arrives")
    {
      REQUIRE(!tsd::io::import_spatial_field(scene, path.string().c_str()));
    }
  }

  GIVEN("A file too short to hold a header")
  {
    const unsigned char bytes[4] = {'N', 'a', 'n', 'o'};
    writeBytes(bytes, sizeof(bytes));

    THEN("No field arrives")
    {
      REQUIRE(!tsd::io::import_spatial_field(scene, path.string().c_str()));
    }
  }

  GIVEN("An OpenVDB grid under a '.vdb' name")
  {
    // The magic OpenVDB writes, then nothing that follows it.
    const unsigned char bytes[8] = {0x20, 0x42, 0x44, 0x56, 0, 0, 0, 0};
    writeBytes(bytes, sizeof(bytes));

    THEN("No field arrives")
    {
      REQUIRE(!tsd::io::import_spatial_field(scene, path.string().c_str()));
    }
  }

  std::filesystem::remove(path);
}

namespace {

// A 1x1 uncompressed grey+alpha 8-bit TIFF. Two channels is the case where
// stb's rule -- an even channel count ends in alpha, an odd one is all colour
// -- diverges from "the first three channels are colour", so it is the only
// shape that catches alpha being gamma-corrected.
struct GreyAlphaTiffFixture
{
  explicit GreyAlphaTiffFixture(const char *name)
      : m_path(std::filesystem::temp_directory_path() / name)
  {
    const unsigned char tiff[] = {
        // clang-format off
        'I', 'I', 0x2a, 0x00, 0x08, 0x00, 0x00, 0x00, // header, IFD at 8
        0x0a, 0x00, // 10 IFD entries
        0x00, 0x01, 0x03, 0x00, 0x01, 0, 0, 0, 0x01, 0x00, 0, 0, // width = 1
        0x01, 0x01, 0x03, 0x00, 0x01, 0, 0, 0, 0x01, 0x00, 0, 0, // height = 1
        0x02, 0x01, 0x03, 0x00, 0x02, 0, 0, 0, 0x08, 0x00, 0x08, 0x00, // bits
        0x03, 0x01, 0x03, 0x00, 0x01, 0, 0, 0, 0x01, 0x00, 0, 0, // no compress
        0x06, 0x01, 0x03, 0x00, 0x01, 0, 0, 0, 0x01, 0x00, 0, 0, // black-is-0
        0x11, 0x01, 0x04, 0x00, 0x01, 0, 0, 0, 0x86, 0x00, 0, 0, // strip @ 134
        0x15, 0x01, 0x03, 0x00, 0x01, 0, 0, 0, 0x02, 0x00, 0, 0, // 2 samples
        0x16, 0x01, 0x03, 0x00, 0x01, 0, 0, 0, 0x01, 0x00, 0, 0, // 1 row/strip
        0x17, 0x01, 0x04, 0x00, 0x01, 0, 0, 0, 0x02, 0x00, 0, 0, // 2 bytes
        0x52, 0x01, 0x03, 0x00, 0x01, 0, 0, 0, 0x02, 0x00, 0, 0, // unassoc a
        0x00, 0x00, 0x00, 0x00, // no next IFD
        0x60, 0x40 // one grey + alpha texel
        // clang-format on
    };
    std::ofstream file(m_path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(tiff), sizeof(tiff));
  }

  ~GreyAlphaTiffFixture()
  {
    std::error_code ec;
    std::filesystem::remove(m_path, ec);
  }

  std::string path() const
  {
    return m_path.string();
  }

 private:
  std::filesystem::path m_path;
};

// A 1x1 uncompressed RGB8 TIFF, little-endian, written by hand: stb has no
// TIFF decoder, so the fixture has to be a genuinely decodable file for the
// OpenImageIO branch to be exercised at all. Layout is header(8) + a 9-entry
// IFD(114) + the BitsPerSample triple(6) + one contiguous RGB texel(3).
struct TiffFixture
{
  explicit TiffFixture(const char *name)
      : m_path(std::filesystem::temp_directory_path() / name)
  {
    const unsigned char tiff[] = {
        // clang-format off
        'I', 'I', 0x2a, 0x00, 0x08, 0x00, 0x00, 0x00, // header, IFD at 8
        0x09, 0x00, // 9 IFD entries
        0x00, 0x01, 0x03, 0x00, 0x01, 0, 0, 0, 0x01, 0x00, 0, 0, // width = 1
        0x01, 0x01, 0x03, 0x00, 0x01, 0, 0, 0, 0x01, 0x00, 0, 0, // height = 1
        0x02, 0x01, 0x03, 0x00, 0x03, 0, 0, 0, 0x7a, 0x00, 0, 0, // bits @ 122
        0x03, 0x01, 0x03, 0x00, 0x01, 0, 0, 0, 0x01, 0x00, 0, 0, // no compress
        0x06, 0x01, 0x03, 0x00, 0x01, 0, 0, 0, 0x02, 0x00, 0, 0, // RGB
        0x11, 0x01, 0x04, 0x00, 0x01, 0, 0, 0, 0x80, 0x00, 0, 0, // strip @ 128
        0x15, 0x01, 0x03, 0x00, 0x01, 0, 0, 0, 0x03, 0x00, 0, 0, // 3 samples
        0x16, 0x01, 0x03, 0x00, 0x01, 0, 0, 0, 0x01, 0x00, 0, 0, // 1 row/strip
        0x17, 0x01, 0x04, 0x00, 0x01, 0, 0, 0, 0x03, 0x00, 0, 0, // 3 bytes
        0x00, 0x00, 0x00, 0x00, // no next IFD
        0x08, 0x00, 0x08, 0x00, 0x08, 0x00, // BitsPerSample = [8, 8, 8]
        0x60, 0x40, 0x20 // one RGB texel
        // clang-format on
    };
    std::ofstream file(m_path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(tiff), sizeof(tiff));
  }

  ~TiffFixture()
  {
    std::error_code ec;
    std::filesystem::remove(m_path, ec);
  }

  std::string path() const
  {
    return m_path.string();
  }

 private:
  std::filesystem::path m_path;
};

} // namespace

SCENARIO("TIFF textures decode into float texel arrays", "[Importers]")
{
  GIVEN("A 1x1 RGB8 TIFF file")
  {
    TiffFixture tiff("tsd_test_1x1_rgb8.tif");

    tsd::scene::Scene scene;
    tsd::io::ImageCache cache(&scene);

    WHEN("It is imported as a linear texture")
    {
      auto sampler =
          tsd::io::importTexture(cache, tiff.path(), /*isLinear=*/true);

#if TSD_USE_OIIO
      THEN("The sampler carries the file's texels untransformed")
      {
        REQUIRE(sampler);
        auto *image =
            sampler->parameterValueAsObject<tsd::scene::Array>("image");
        REQUIRE(image != nullptr);
        REQUIRE(image->elementType() == ANARI_FLOAT32_VEC3);
        REQUIRE(image->size() == 1);
        const auto *texels = image->dataAs<tsd::core::math::float3>();
        REQUIRE(texels[0].x == Approx(0x60 / 255.f));
        REQUIRE(texels[0].y == Approx(0x40 / 255.f));
        REQUIRE(texels[0].z == Approx(0x20 / 255.f));
      }
#else
      THEN("No sampler is produced, because no decoder is available")
      {
        REQUIRE(!sampler);
      }
#endif
    }

#if TSD_USE_OIIO
    WHEN("It is imported as an sRGB texture")
    {
      auto sampler =
          tsd::io::importTexture(cache, tiff.path(), /*isLinear=*/false);

      THEN("The texels are decoded to linear, matching the stb-backed paths")
      {
        REQUIRE(sampler);
        auto *image =
            sampler->parameterValueAsObject<tsd::scene::Array>("image");
        REQUIRE(image != nullptr);
        const auto *texels = image->dataAs<tsd::core::math::float3>();
        REQUIRE(texels[0].x == Approx(std::pow(0x60 / 255.f, 2.2f)));
        REQUIRE(texels[0].y == Approx(std::pow(0x40 / 255.f, 2.2f)));
        REQUIRE(texels[0].z == Approx(std::pow(0x20 / 255.f, 2.2f)));
      }
    }
#endif
  }

#if TSD_USE_OIIO
  GIVEN("A 1x1 grey+alpha TIFF file")
  {
    GreyAlphaTiffFixture tiff("tsd_test_1x1_greyalpha.tif");

    tsd::scene::Scene scene;
    tsd::io::ImageCache cache(&scene);

    WHEN("It is imported as an sRGB texture")
    {
      auto sampler =
          tsd::io::importTexture(cache, tiff.path(), /*isLinear=*/false);

      THEN("Only the grey channel is gamma-decoded, leaving alpha linear")
      {
        REQUIRE(sampler);
        auto *image =
            sampler->parameterValueAsObject<tsd::scene::Array>("image");
        REQUIRE(image != nullptr);
        REQUIRE(image->elementType() == ANARI_FLOAT32_VEC2);
        const auto *texels = image->dataAs<tsd::core::math::float2>();
        REQUIRE(texels[0].x == Approx(std::pow(0x60 / 255.f, 2.2f)));
        REQUIRE(texels[0].y == Approx(0x40 / 255.f));
      }
    }
  }
#endif
}
