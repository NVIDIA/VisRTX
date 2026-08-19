// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// UsdVol Volume prims and the spatial fields they reference.

#if TSD_USE_USD

// catch
#include "catch.hpp"
// tsd_tests
#include "UsdTestFixtures.h"
// tsd
#include "tsd/core/Logging.hpp"
#include "tsd/scene/objects/Array.hpp"
#include "tsd/scene/objects/Volume.hpp"
// std
#include <fstream>
#include <string>
#include <vector>

namespace {

// `valueRange` is held as an ANARI_FLOAT32_BOX1, which does not round-trip
// through Object::parameterValueAs<>() -- that asks the Any for its C++ type
// alone and a box1 is a float2 by another name.
tsd::math::float2 valueRangeOf(tsd::scene::Volume *volume)
{
  auto *parameter = volume->parameter("valueRange");
  REQUIRE(parameter != nullptr);
  return parameter->value().getAs<tsd::math::float2>(ANARI_FLOAT32_BOX1);
}

// A field file next to the Stage, so a `filePath` asset reference on a field
// prim resolves the way it does in a real asset. The name carries the volume's
// dimensions and voxel type, which is how import_RAW learns its layout.
struct RawFieldFixture
{
  explicit RawFieldFixture(const char *name) : m_path(fixtureDirectory() / name)
  {
    const unsigned char voxels[8] = {0, 32, 64, 96, 128, 160, 192, 255};
    std::ofstream file(m_path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(voxels), sizeof(voxels));
  }

  ~RawFieldFixture()
  {
    std::error_code ec;
    std::filesystem::remove(m_path, ec);
  }

 private:
  std::filesystem::path m_path;
};

// A field file next to the Stage that holds nothing any importer can read.
// Which importer an extension reaches is the question; the file only has to
// exist, so that `filePath` resolves to a real path the way it does in an
// asset rather than staying the bare name the Stage authored.
struct UnreadableFieldFixture
{
  explicit UnreadableFieldFixture(const char *name)
      : m_path(fixtureDirectory() / name)
  {
    std::ofstream file(m_path, std::ios::binary);
  }

  ~UnreadableFieldFixture()
  {
    std::error_code ec;
    std::filesystem::remove(m_path, ec);
  }

 private:
  std::filesystem::path m_path;
};

// Collects log messages for the lifetime of one scenario. Which importer a
// file extension reaches is otherwise invisible when the file named is a
// stand-in that no importer can actually read.
struct LogCapture
{
  LogCapture()
  {
    tsd::core::setLoggingCallback(
        [this](tsd::core::LogLevel, std::string message) {
          messages.push_back(std::move(message));
        });
  }

  ~LogCapture()
  {
    // No callback is the state the test binary starts in.
    tsd::core::setNoLogging();
  }

  bool sawMessageContaining(const char *text) const
  {
    for (const auto &message : messages) {
      if (message.find(text) != std::string::npos)
        return true;
    }
    return false;
  }

  std::vector<std::string> messages;
};

} // namespace

SCENARIO("A UsdVol Volume imports the field it references", "[UsdImport]")
{
  GIVEN("A Volume prim whose field names a RAW file next to the Stage")
  {
    RawFieldFixture field("tsd_test_usd_field_2x2x2_uint8.raw");
    ImportedStage stage("tsd_test_usd_volume.usda", R"(#usda 1.0

def Volume "Vol"
{
    rel field:density = </Vol/Density>

    def "Density"
    {
        asset filePath = @tsd_test_usd_field_2x2x2_uint8.raw@
    }
}
)");

    WHEN("The Stage is imported")
    {
      THEN("The volume carries the field and a default color map")
      {
        REQUIRE(stage.scene.numberOfObjects(ANARI_VOLUME) == 1);
        auto volume =
            findObject<tsd::scene::Volume>(stage.scene, ANARI_VOLUME, "/Vol");
        REQUIRE(volume);
        REQUIRE(
            volume->parameterValueAsObject<tsd::scene::SpatialField>("value")
            != nullptr);
        auto *color =
            volume->parameterValueAsObject<tsd::scene::Array>("color");
        REQUIRE(color != nullptr);
        REQUIRE(color->size() == 256);
      }
    }
  }
}

SCENARIO("The anari: volume annotations override the field's own range",
    "[UsdImport]")
{
  GIVEN("A Volume prim annotated with a value range and unit distance")
  {
    RawFieldFixture field("tsd_test_usd_annotated_2x2x2_uint8.raw");
    ImportedStage stage("tsd_test_usd_volume_annotated.usda", R"(#usda 1.0

def Volume "Vol"
{
    float2 anari:valueRange = (3, 7)
    float anari:unitDistance = 2.5
    rel field:density = </Vol/Density>

    def "Density"
    {
        asset filePath = @tsd_test_usd_annotated_2x2x2_uint8.raw@
    }
}
)");

    WHEN("The Stage is imported")
    {
      THEN("Both annotations reach the volume")
      {
        auto volume =
            findObject<tsd::scene::Volume>(stage.scene, ANARI_VOLUME, "/Vol");
        REQUIRE(volume);

        const auto range = valueRangeOf(volume.data());
        REQUIRE(range.x == Approx(3.f));
        REQUIRE(range.y == Approx(7.f));

        const auto unitDistance =
            volume->parameterValueAs<float>("unitDistance");
        REQUIRE(unitDistance.has_value());
        REQUIRE(*unitDistance == Approx(2.5f));
      }
    }
  }
}

SCENARIO("A colormap authored on a Volume becomes its transfer function",
    "[UsdImport]")
{
  GIVEN("A Volume prim with a child Shader carrying colormap points")
  {
    RawFieldFixture field("tsd_test_usd_colormapped_2x2x2_uint8.raw");
    ImportedStage stage("tsd_test_usd_volume_colormap.usda", R"(#usda 1.0

def Volume "Vol"
{
    rel field:density = </Vol/Density>

    def "Density"
    {
        asset filePath = @tsd_test_usd_colormapped_2x2x2_uint8.raw@
    }

    def Shader "Colormap"
    {
        float4[] rgbaPoints = [(1, 0, 0, 0), (0, 0, 1, 1)]
        float[] xPoints = [0, 1]
        float2 domain = (10, 20)
    }
}
)");

    WHEN("The Stage is imported")
    {
      THEN("The colormap's colors and domain reach the volume")
      {
        auto volume =
            findObject<tsd::scene::Volume>(stage.scene, ANARI_VOLUME, "/Vol");
        REQUIRE(volume);

        auto *color =
            volume->parameterValueAsObject<tsd::scene::Array>("color");
        REQUIRE(color != nullptr);
        REQUIRE(color->size() == 256);
        const auto *texels = color->dataAs<tsd::math::float4>();
        REQUIRE(texels[0].x == Approx(1.f));
        REQUIRE(texels[0].w == Approx(0.f));
        REQUIRE(texels[255].z == Approx(1.f));
        REQUIRE(texels[255].w == Approx(1.f));

        const auto range = valueRangeOf(volume.data());
        REQUIRE(range.x == Approx(10.f));
        REQUIRE(range.y == Approx(20.f));
      }
    }
  }
}

SCENARIO("A Volume whose field cannot be loaded is reported", "[UsdImport]")
{
  GIVEN("A Volume prim naming a field file that is not there")
  {
    ImportedStage stage("tsd_test_usd_volume_missing_field.usda", R"(#usda 1.0

def Volume "Vol"
{
    rel field:density = </Vol/Density>

    def "Density"
    {
        asset filePath = @tsd_test_usd_absent_2x2x2_uint8.raw@
    }
}
)");

    WHEN("The Stage is imported")
    {
      THEN("No volume arrives")
      {
        REQUIRE(stage.scene.numberOfObjects(ANARI_VOLUME) == 0);
      }

      THEN("The prim is named in the Import Report")
      {
        REQUIRE(stage.report.skipped.size() == 1);
        REQUIRE(stage.report.skipped[0].primPath == "/Vol");
        REQUIRE(stage.report.skipped[0].reason
            == tsd::io::UsdSkipReason::FIELD_LOAD_FAILED);
        REQUIRE(stage.report.skipped[0].detail.find(
                    "tsd_test_usd_absent_2x2x2_uint8.raw")
            != std::string::npos);
      }

      THEN("It leaves a disabled Placeholder Node where it belongs")
      {
        auto *layer = stage.scene.defaultLayer();
        auto vol = findNode(layer, "Vol");
        REQUIRE(vol);
        REQUIRE((*vol)->isEmpty());
        REQUIRE_FALSE((*vol)->isEnabled());
      }
    }
  }
}

SCENARIO("Volume fields go through the shared spatial-field dispatcher",
    "[UsdImport]")
{
  GIVEN("A Volume prim whose field names a Silo file")
  {
    UnreadableFieldFixture field("tsd_test_usd_field.silo");
    LogCapture log;
    ImportedStage stage("tsd_test_usd_volume_silo.usda", R"(#usda 1.0

def Volume "Vol"
{
    rel field:density = </Vol/Density>

    def "Density"
    {
        asset filePath = @tsd_test_usd_field.silo@
    }
}
)");

    WHEN("The Stage is imported")
    {
      THEN("The Silo importer is the one asked for the field")
      {
        // The extension chain the USD importer used to hand-roll knew nothing
        // of Silo, so this file fell off the end of it. Only import_SILO logs
        // under this prefix, in either build configuration, so seeing it is
        // proof the dispatcher routed the extension; not seeing the fallback
        // is proof nothing routed it by accident.
        REQUIRE(log.sawMessageContaining("[import_SILO]"));
        REQUIRE(!log.sawMessageContaining("no loader for file type"));
      }
    }
  }

  GIVEN("A Volume prim whose field names a FLASH-in-HDF5 file")
  {
    UnreadableFieldFixture field("tsd_test_usd_field.hdf5");
    LogCapture log;
    ImportedStage stage("tsd_test_usd_volume_hdf5.usda", R"(#usda 1.0

def Volume "Vol"
{
    rel field:density = </Vol/Density>

    def "Density"
    {
        asset filePath = @tsd_test_usd_field.hdf5@
    }
}
)");

    WHEN("The Stage is imported")
    {
      THEN("The FLASH importer is the one asked for the field")
      {
        REQUIRE(log.sawMessageContaining("[import_FLASH]"));
        REQUIRE(!log.sawMessageContaining("no loader for file type"));
      }
    }
  }
}

#endif // TSD_USE_USD
