// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// helium
#include <helium/helium_math.h>
// tsd
#include "tsd/animation/AnimationManager.hpp"
#include "tsd/core/TSDMath.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
#include "tsd/scene/Scene.hpp"
// std
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

// These tests characterize where a decoded image's rows land and which way an
// importer's `v` runs, because the two only produce a correct picture when
// they agree. See docs/tsd-io-image-import.md for the survey they came from.
//
// Fixtures are synthesized into the temp directory rather than checked in,
// following tests/test_UsdImport.cpp's TextureFixture and
// tests/test_Importers.cpp's TiffFixture.

namespace {

using namespace tsd::core::math;

// The two rows of every fixture image. Chosen far apart in every channel so
// no colour-space handling on any decode path can confuse one for the other.
constexpr float3 TOP_ROW_COLOR(1.f, 0.f, 0.f);
constexpr float3 BOTTOM_ROW_COLOR(0.f, 0.f, 1.f);

bool isTopRowColor(const float3 &c)
{
  return c.x > c.z;
}

bool isBottomRowColor(const float3 &c)
{
  return c.z > c.x;
}

// A file that exists for the lifetime of one scenario and removes itself
// after. Every fixture below is one of these.
struct TempFile
{
  TempFile(const char *name, const std::string &contents)
      : m_path(std::filesystem::temp_directory_path() / name)
  {
    std::ofstream file(m_path, std::ios::binary);
    file.write(contents.data(), std::streamsize(contents.size()));
  }

  ~TempFile()
  {
    std::error_code ec;
    std::filesystem::remove(m_path, ec);
  }

  std::string path() const
  {
    return m_path.string();
  }

  TempFile(const TempFile &) = delete;
  TempFile &operator=(const TempFile &) = delete;

 private:
  std::filesystem::path m_path;
};

// A 1x2 uncompressed true-colour TGA: red on top, blue on the bottom. TGA is
// the format test_UsdImport.cpp already hand-writes, and it is the one every
// stb-backed path in the tree can decode, so one fixture serves the OBJ,
// glTF, USD, and PBRT importers alike.
//
// A TGA with descriptor bit 5 clear stores its rows bottom-up, so the first
// texel in the file is the *bottom* row. That is deliberate: it means this
// fixture only reports "row 0 is the top row" if the import layer actually
// establishes the contract, rather than passing a decoder's byte order
// through and happening to agree with it.
std::string tgaFixtureContents()
{
  const unsigned char tga[] = {
      // clang-format off
      0, // no image ID
      0, // no colour map
      2, // uncompressed true-colour
      0, 0, 0, 0, 0, // empty colour map spec
      0, 0, 0, 0, // origin
      1, 0, // width = 1
      2, 0, // height = 2
      24, // bits per pixel
      0, // descriptor: origin lower-left, so rows are stored bottom-up
      0xff, 0x00, 0x00, // bottom row, BGR: blue
      0x00, 0x00, 0xff // top row, BGR: red
      // clang-format on
  };
  return std::string(reinterpret_cast<const char *>(tga), sizeof(tga));
}

// An 8x8 BC1 DDS: the top half of every block is red, the bottom half blue.
// Block-compressed texels are the one case the import layer cannot reorder, so
// this is the fixture that exercises the sampler-side compensation instead.
std::string ddsFixtureContents()
{
  // One BC1 block: color0 = red, color1 = blue, then four rows of 2-bit
  // indices, top row first -- rows 0 and 1 pick color0, rows 2 and 3 color1.
  const unsigned char block[] = {
      0x00, 0xf8, 0x1f, 0x00, 0x00, 0x00, 0x55, 0x55};

  std::string dds;
  auto u32 = [&dds](std::uint32_t v) {
    dds.append(reinterpret_cast<const char *>(&v), sizeof(v));
  };

  dds += "DDS ";
  u32(124); // header size
  u32(0x1 | 0x2 | 0x4 | 0x1000
      | 0x80000); // CAPS|HEIGHT|WIDTH|PIXELFORMAT|LINEARSIZE
  u32(8); // height
  u32(8); // width
  u32(4 * sizeof(block)); // linear size: four 4x4 blocks
  u32(0); // depth
  u32(1); // mip levels
  for (int i = 0; i < 11; ++i)
    u32(0); // reserved
  u32(32); // pixel format size
  u32(0x4); // DDPF_FOURCC
  dds += "DXT1";
  for (int i = 0; i < 5; ++i)
    u32(0); // bit counts and masks, unused for a fourCC format
  u32(0x1000); // DDSCAPS_TEXTURE
  for (int i = 0; i < 4; ++i)
    u32(0); // caps2..4, reserved2

  for (int i = 0; i < 4; ++i)
    dds.append(reinterpret_cast<const char *>(block), sizeof(block));
  return dds;
}

// A 1x2 Radiance HDR: red on top, blue on the bottom. The scanline data is
// flat RGBE rather than run-length encoded, which stb takes for any image
// under eight texels wide.
std::string hdrFixtureContents()
{
  // Mantissa plus a shared exponent of 129, so the bright channel reads a
  // little under 2.0 and the others are zero.
  const unsigned char scanlines[] = {
      0xff,
      0x00,
      0x00,
      0x81, // top row: red
      0x00,
      0x00,
      0xff,
      0x81 // bottom row: blue
  };

  std::string hdr = "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 1\n";
  hdr.append(reinterpret_cast<const char *>(scanlines), sizeof(scanlines));
  return hdr;
}

// Texel readers ///////////////////////////////////////////////////////////////

// Importers do not agree on an element type -- the shared path expands to
// ANARI_FLOAT32_*, glTF keeps the file's 8-bit type and asks for an sRGB
// format -- so orientation assertions have to read either. helium reads any
// of them, applying the element stride and the sRGB decode itself; a type it
// does not know reads back as (0, 0, 0), which no row predicate accepts.
float3 texelAsFloat3(const tsd::scene::Array *image, size_t index)
{
  const auto texel = helium::readAsAttributeValueFlat(
      image->data(), image->elementType(), index);
  return float3(texel.x, texel.y, texel.z);
}

// The fixture image a scene imported, found by its shape rather than by the
// material parameter it hangs off, which differs per importer.
const tsd::scene::Array *fixtureImage(tsd::scene::Scene &scene)
{
  for (size_t i = 0; i < scene.numberOfObjects(ANARI_SAMPLER); ++i) {
    auto sampler = scene.getObject<tsd::scene::Sampler>(i);
    if (!sampler)
      continue;
    auto *image = sampler->parameterValueAsObject<tsd::scene::Array>("image");
    if (image && image->dim(0) == 1 && image->dim(1) == 2)
      return image;
  }
  return nullptr;
}

// Resolve a texture coordinate the way ANARI resolves it: `v` runs down the
// picture, and row 0 is the picture's top row, so `v = 0` addresses row 0.
float3 sampleAsAnari(const tsd::scene::Array *image, const float2 &uv)
{
  const auto height = image->dim(1);
  auto row = size_t(uv.y * float(height));
  if (row >= height)
    row = height - 1;
  return texelAsFloat3(image, row * image->dim(0));
}

// The texture coordinate the top corner of the fixture quad carries. Every
// fixture below is the same unit quad in the XY plane, so the vertex with the
// greatest `y` is unambiguously the one at the top of the picture.
float2 uvAtTopOfQuad(tsd::scene::Scene &scene)
{
  for (size_t i = 0; i < scene.numberOfObjects(ANARI_GEOMETRY); ++i) {
    auto geometry = scene.getObject<tsd::scene::Geometry>(i);
    if (!geometry)
      continue;
    auto *positions =
        geometry->parameterValueAsObject<tsd::scene::Array>("vertex.position");
    auto *uvs = geometry->parameterValueAsObject<tsd::scene::Array>(
        "vertex.attribute0");
    if (!positions || !uvs || uvs->size() != positions->size())
      continue;

    const auto *p = positions->dataAs<float3>();
    const auto *t = uvs->dataAs<float2>();
    size_t top = 0;
    for (size_t v = 1; v < positions->size(); ++v) {
      if (p[v].y > p[top].y)
        top = v;
    }
    return t[top];
  }
  FAIL("no textured geometry in the imported scene");
  return float2(0.f);
}

// Fixture scenes //////////////////////////////////////////////////////////////

// A unit quad in the XY plane with the fixture texture as its base colour.
// The `v` each format assigns to the quad's top corner differs, because the
// formats' `v` conventions differ; each fixture spells its own out.

std::string objContents(const std::string &mtlName)
{
  // OBJ `vt` is v-up per the spec, so the top of the quad carries v = 1.
  return "mtllib " + mtlName
      + "\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 1 1 0\n"
        "v 0 1 0\n"
        "vt 0 0\n"
        "vt 1 0\n"
        "vt 1 1\n"
        "vt 0 1\n"
        "usemtl textured\n"
        "f 1/1 2/2 3/3\n"
        "f 1/1 3/3 4/4\n";
}

std::string mtlContents(const std::string &textureName)
{
  return "newmtl textured\n"
         "Kd 1 1 1\n"
         "map_Kd "
      + textureName + "\n";
}

std::string gltfContents(
    const std::string &binName, const std::string &textureName)
{
  // glTF's `v` runs down the image per the spec, so the top of the quad
  // carries v = 0.
  return R"({
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [{"mesh": 0}],
  "meshes": [{"primitives": [{
    "attributes": {"POSITION": 0, "TEXCOORD_0": 1},
    "indices": 2,
    "material": 0
  }]}],
  "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
  "textures": [{"source": 0}],
  "images": [{"uri": ")"
      + textureName + R"("}],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
     "min": [0, 0, 0], "max": [1, 1, 0]},
    {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC2"},
    {"bufferView": 2, "componentType": 5123, "count": 6, "type": "SCALAR"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 48},
    {"buffer": 0, "byteOffset": 48, "byteLength": 32},
    {"buffer": 0, "byteOffset": 80, "byteLength": 12}
  ],
  "buffers": [{"byteLength": 92, "uri": ")"
      + binName + R"("}]
})";
}

std::string gltfBufferContents()
{
  const float positions[] = {
      0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 1.f, 0.f};
  const float texCoords[] = {0.f, 1.f, 1.f, 1.f, 1.f, 0.f, 0.f, 0.f};
  const uint16_t indices[] = {0, 1, 2, 0, 2, 3};

  std::string bytes;
  bytes.append(reinterpret_cast<const char *>(positions), sizeof(positions));
  bytes.append(reinterpret_cast<const char *>(texCoords), sizeof(texCoords));
  bytes.append(reinterpret_cast<const char *>(indices), sizeof(indices));
  return bytes;
}

std::string usdContents(const std::string &textureName)
{
  // UsdPreviewSurface's `st` is v-up, so the top of the quad carries v = 1.
  return R"(#usda 1.0

def Xform "World"
{
    def Material "Textured"
    {
        token outputs:surface.connect = </World/Textured/PBR.outputs:surface>

        def Shader "PBR"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor.connect = </World/Textured/Tex.outputs:rgb>
            token outputs:surface
        }

        def Shader "Tex"
        {
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @)"
      + textureName + R"(@
            float2 inputs:st.connect = </World/Textured/Reader.outputs:result>
            float3 outputs:rgb
        }

        def Shader "Reader"
        {
            uniform token info:id = "UsdPrimvarReader_float2"
            token inputs:varname = "st"
            float2 outputs:result
        }
    }

    def Mesh "Quad" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
            interpolation = "vertex"
        )
        rel material:binding = </World/Textured>
    }
}
)";
}

std::string pbrtContents(const std::string &textureName)
{
  // PBRT's `v` is v-up, matching OBJ, so the top of the quad carries v = 1.
  return R"(WorldBegin
Texture "fixture" "spectrum" "imagemap" "string filename" [")"
      + textureName + R"("]
AttributeBegin
Material "diffuse" "texture reflectance" "fixture"
Shape "trianglemesh"
  "integer indices" [0 1 2 0 2 3]
  "point3 P" [0 0 0  1 0 0  1 1 0  0 1 0]
  "point2 uv" [0 0  1 0  1 1  0 1]
AttributeEnd
WorldEnd
)";
}

} // namespace

// Decoder-level contract //////////////////////////////////////////////////////

SCENARIO("Decoded images are stored in ANARI orientation", "[ImageImport]")
{
  GIVEN("A 1x2 image, red on top and blue on the bottom")
  {
    TempFile texture("tsd_test_orient.tga", tgaFixtureContents());

    tsd::scene::Scene scene;
    tsd::io::ImageCache cache(&scene);

    WHEN("It is imported through the shared texture path")
    {
      auto sampler =
          tsd::io::importTexture(cache, texture.path(), /*isLinear=*/true);

      THEN("Row 0 of the array is the top row of the picture")
      {
        REQUIRE(sampler);
        auto *image =
            sampler->parameterValueAsObject<tsd::scene::Array>("image");
        REQUIRE(image != nullptr);
        REQUIRE(image->dim(0) == 1);
        REQUIRE(image->dim(1) == 2);
        REQUIRE(isTopRowColor(texelAsFloat3(image, 0)));
        REQUIRE(isBottomRowColor(texelAsFloat3(image, 1)));
      }
    }
  }
}

// An hdri light's radiance is mapped over the sphere by the light rather than
// addressed by an image sampler, so the top-left origin the sampler path is
// stored for does not reach it. It keeps the decoder's bottom-up rows.
SCENARIO("An imported HDRI's radiance runs bottom-up", "[ImageImport]")
{
  GIVEN("A 1x2 Radiance HDR, red on top and blue on the bottom")
  {
    TempFile hdri("tsd_test_orient.hdr", hdrFixtureContents());

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("It is imported as a dome light")
    {
      tsd::io::import_HDRI(scene, animMgr, hdri.path().c_str());

      THEN("Row 0 of the radiance array is the bottom row of the picture")
      {
        REQUIRE(scene.numberOfObjects(ANARI_LIGHT) == 1);
        auto light = scene.getObject<tsd::scene::Light>(0);
        REQUIRE(light);
        auto *radiance =
            light->parameterValueAsObject<tsd::scene::Array>("radiance");
        REQUIRE(radiance != nullptr);
        REQUIRE(radiance->dim(0) == 1);
        REQUIRE(radiance->dim(1) == 2);
        REQUIRE(isBottomRowColor(texelAsFloat3(radiance, 0)));
        REQUIRE(isTopRowColor(texelAsFloat3(radiance, 1)));
      }
    }
  }
}

SCENARIO("Block-compressed images are bound as the file authored them",
    "[ImageImport]")
{
  // BC blocks are 4x4, so the texels can never be reordered. A DDS decodes
  // top-down, which is the order a sampled image is stored in, so nothing has
  // to be compensated for -- but a consumer asking for the other order gets
  // the compensation in the sampler's uv transform instead. The assertion is
  // on where a coordinate lands, not on the matrix.
  auto fetchedV = [](tsd::scene::SamplerRef sampler, float v) {
    auto *transform = sampler->parameter("inTransform");
    auto *offset = sampler->parameter("inOffset");
    REQUIRE(transform != nullptr);
    REQUIRE(offset != nullptr);
    const auto uv =
        tsd::core::math::mul(transform->value().get<tsd::core::math::mat4>(),
            float4(0.f, v, 0.f, 1.f))
        + offset->value().get<float4>();
    return uv.y;
  };

  GIVEN("An 8x8 BC1 DDS, red on top and blue on the bottom")
  {
    TempFile texture("tsd_test_orient.dds", ddsFixtureContents());

    tsd::scene::Scene scene;
    tsd::io::ImageCache cache(&scene);

    WHEN("It is imported with no uv transform of its own")
    {
      auto sampler = tsd::io::importTexture(cache, texture.path());

      THEN("The sampler leaves the coordinates alone")
      {
        REQUIRE(sampler);
        REQUIRE(sampler->subtype()
            == tsd::scene::tokens::sampler::compressedImage2D);
        // The block format and the picture's dimensions describe an Array
        // whose own shape is a flat byte run, so the sampler has to carry
        // them; nothing else tells the device how to read the blocks.
        auto *format = sampler->parameter("format");
        REQUIRE(format != nullptr);
        REQUIRE(format->value().getString() == "BC1_RGB");
        auto *size = sampler->parameter("size");
        REQUIRE(size != nullptr);
        REQUIRE(size->value().type() == ANARI_UINT64_VEC2);
        const auto *extent =
            static_cast<const std::uint64_t *>(size->value().data());
        REQUIRE(extent[0] == 8);
        REQUIRE(extent[1] == 8);
        REQUIRE(fetchedV(sampler, 1.f) == Approx(1.f).margin(1e-5));
        REQUIRE(fetchedV(sampler, 0.f) == Approx(0.f).margin(1e-5));
      }
    }

    WHEN("It is imported by a caller that authored its own uv transform")
    {
      // Half-scale in v, as USD's uvTransform or PBRT's vscale would give.
      tsd::io::SamplerSettings settings;
      settings.uvTransform =
          tsd::io::UvTransform{tsd::core::math::mat4(float4(1.f, 0.f, 0.f, 0.f),
              float4(0.f, 0.5f, 0.f, 0.f),
              float4(0.f, 0.f, 1.f, 0.f),
              float4(0.f, 0.f, 0.f, 1.f))};

      auto sampler = tsd::io::importTexture(
          cache, texture.path(), /*isLinear=*/false, settings);

      THEN("That transform reaches the sampler unchanged")
      {
        REQUIRE(sampler);
        REQUIRE(fetchedV(sampler, 1.f) == Approx(0.5f).margin(1e-5));
        REQUIRE(fetchedV(sampler, 0.f) == Approx(0.f).margin(1e-5));
      }
    }

    WHEN("It is acquired for a consumer that wants the opposite row order")
    {
      auto image = cache.acquire({texture.path(),
          tsd::io::ColorSpace::SRGB,
          tsd::io::RowOrder::BOTTOM_UP});
      auto sampler = tsd::io::makeImageSampler(cache, image, texture.path());

      THEN("The sampler reverses v, since the texels could not be")
      {
        REQUIRE(sampler);
        REQUIRE(fetchedV(sampler, 1.f) == Approx(0.f).margin(1e-5));
        REQUIRE(fetchedV(sampler, 0.f) == Approx(1.f).margin(1e-5));
      }
    }
  }
}

// Importer-level contract /////////////////////////////////////////////////////

// The property that has to hold whatever the storage convention is: the corner
// of the mesh at the top of the picture must address the picture's top row.

SCENARIO("An imported quad's top corner addresses the image's top row",
    "[ImageImport]")
{
  GIVEN("An OBJ quad textured with the fixture image")
  {
    TempFile texture("tsd_test_orient.tga", tgaFixtureContents());
    TempFile mtl("tsd_test_orient.mtl", mtlContents("tsd_test_orient.tga"));
    TempFile obj("tsd_test_orient.obj", objContents("tsd_test_orient.mtl"));

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("It is imported")
    {
      tsd::io::import_OBJ(scene, animMgr, obj.path().c_str());

      THEN("The top corner samples the top row")
      {
        const auto *image = fixtureImage(scene);
        REQUIRE(image != nullptr);
        REQUIRE(isTopRowColor(sampleAsAnari(image, uvAtTopOfQuad(scene))));
      }
    }
  }

#if TSD_USE_ASSIMP
  // Through the glTF fixture rather than the OBJ one: ASSIMP reports a
  // GL-style shading model for OBJ, and that branch of the material importer
  // binds no textures at all, so an OBJ would assert nothing here.
  GIVEN("The same glTF quad, read through ASSIMP")
  {
    TempFile texture("tsd_test_orient.tga", tgaFixtureContents());
    TempFile bin("tsd_test_orient.bin", gltfBufferContents());
    TempFile gltf("tsd_test_orient.gltf",
        gltfContents("tsd_test_orient.bin", "tsd_test_orient.tga"));

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("It is imported")
    {
      tsd::io::import_ASSIMP(scene, animMgr, gltf.path().c_str());

      THEN("The top corner samples the top row")
      {
        const auto *image = fixtureImage(scene);
        REQUIRE(image != nullptr);
        REQUIRE(isTopRowColor(sampleAsAnari(image, uvAtTopOfQuad(scene))));
      }
    }
  }
#endif

  GIVEN("A glTF quad textured with the fixture image")
  {
    TempFile texture("tsd_test_orient.tga", tgaFixtureContents());
    TempFile bin("tsd_test_orient.bin", gltfBufferContents());
    TempFile gltf("tsd_test_orient.gltf",
        gltfContents("tsd_test_orient.bin", "tsd_test_orient.tga"));

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("It is imported")
    {
      tsd::io::import_GLTF(scene, animMgr, gltf.path().c_str());

      THEN("The top corner samples the top row")
      {
        const auto *image = fixtureImage(scene);
        REQUIRE(image != nullptr);
        REQUIRE(isTopRowColor(sampleAsAnari(image, uvAtTopOfQuad(scene))));
      }
    }
  }

  GIVEN("A PBRT quad textured with the fixture image")
  {
    TempFile texture("tsd_test_orient.tga", tgaFixtureContents());
    TempFile pbrt("tsd_test_orient.pbrt", pbrtContents("tsd_test_orient.tga"));

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("It is imported")
    {
      tsd::io::import_PBRT(scene, animMgr, pbrt.path().c_str());

      THEN("The top corner samples the top row")
      {
        const auto *image = fixtureImage(scene);
        REQUIRE(image != nullptr);
        REQUIRE(isTopRowColor(sampleAsAnari(image, uvAtTopOfQuad(scene))));
      }
    }
  }

#if TSD_USE_USD
  GIVEN("A USD quad textured with the fixture image")
  {
    TempFile texture("tsd_test_orient.tga", tgaFixtureContents());
    TempFile stage("tsd_test_orient.usda", usdContents("tsd_test_orient.tga"));

    tsd::scene::Scene scene;
    tsd::animation::AnimationManager animMgr(&scene);

    WHEN("It is imported")
    {
      tsd::io::import_USD(scene, animMgr, stage.path().c_str());

      THEN("The top corner samples the top row")
      {
        const auto *image = fixtureImage(scene);
        REQUIRE(image != nullptr);
        REQUIRE(isTopRowColor(sampleAsAnari(image, uvAtTopOfQuad(scene))));
      }
    }
  }
#endif
}
