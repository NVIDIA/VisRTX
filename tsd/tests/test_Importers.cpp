// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/animation/AnimationManager.hpp"
#include "tsd/core/TSDMath.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
#include "tsd/scene/Scene.hpp"
// std
#include <cmath>
#include <filesystem>
#include <fstream>
#include <system_error>

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

namespace {

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
    tsd::io::TextureCache cache;

    WHEN("It is imported as a linear texture")
    {
      auto sampler = tsd::io::importTexture(
          scene, tiff.path(), cache, /*isLinear=*/true);

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
      auto sampler = tsd::io::importTexture(
          scene, tiff.path(), cache, /*isLinear=*/false);

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
}
