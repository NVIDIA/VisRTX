// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/procedural.hpp"
#include "tsd/io/serialization.hpp"
#include "tsd/scene/Scene.hpp"
#include "tsd/scripting/LuaBindings.hpp"
#include "tsd/scripting/Sol2Helpers.hpp"

#include <fmt/format.h>
#include <sol/sol.hpp>

namespace tsd::scripting {

static bool isNumericVector(const sol::table &t, size_t &len)
{
  len = t.size();
  if (len < 2 || len > 4)
    return false;
  for (size_t i = 1; i <= len; i++) {
    if (t[i].get_type() != sol::type::number)
      return false;
  }
  return true;
}

static constexpr ANARIDataType vecTypes[] = {
    ANARI_UNKNOWN, // 0
    ANARI_UNKNOWN, // 1
    ANARI_FLOAT32_VEC2, // 2
    ANARI_FLOAT32_VEC3, // 3
    ANARI_FLOAT32_VEC4, // 4
};

static void setNodeFromVec(core::DataNode &node, const sol::object &value)
{
  if (value.is<math::float2>()) {
    auto v = value.as<math::float2>();
    node.setValue(ANARI_FLOAT32_VEC2, &v);
  } else if (value.is<math::float3>()) {
    auto v = value.as<math::float3>();
    node.setValue(ANARI_FLOAT32_VEC3, &v);
  } else if (value.is<math::float4>()) {
    auto v = value.as<math::float4>();
    node.setValue(ANARI_FLOAT32_VEC4, &v);
  } else if (value.is<math::mat4>()) {
    auto v = value.as<math::mat4>();
    node.setValue(ANARI_FLOAT32_MAT4, &v);
  }
}

static void copyTableToNode(const sol::table &table, core::DataNode &node)
{
  for (auto &[key, value] : table) {
    auto name = key.as<std::string>();
    auto &child = node[name];
    // Check vec/mat userdata before sol::table — sol2 treats userdata with
    // __index as table-like, so is<sol::table>() can match float3 etc.
    if (value.is<math::float2>() || value.is<math::float3>()
        || value.is<math::float4>() || value.is<math::mat4>()) {
      setNodeFromVec(child, value);
    } else if (value.is<sol::table>()) {
      sol::table t = value.as<sol::table>();
      size_t len = 0;
      if (isNumericVector(t, len)) {
        float v[4];
        for (size_t i = 0; i < len; i++)
          v[i] = t[i + 1].get<float>();
        child.setValue(vecTypes[len], v);
      } else {
        copyTableToNode(t, child);
      }
    } else if (value.is<bool>()) {
      child = value.as<bool>();
    } else if (value.is<int>()) {
      child = value.as<int>();
    } else if (value.is<double>()) {
      child = static_cast<float>(value.as<double>());
    } else if (value.is<std::string>()) {
      child = value.as<std::string>();
    }
  }
}

#define TSD_LUA_IMPORT_WRAP(import_call, filename)                             \
  try {                                                                        \
    import_call;                                                               \
  } catch (const std::exception &e) {                                          \
    throw std::runtime_error(                                                  \
        fmt::format("Failed to import '{}': {}", filename, e.what()));         \
  }

#define TSD_LUA_IMPORT_WRAP_RETURN(import_call, filename)                      \
  try {                                                                        \
    return import_call;                                                        \
  } catch (const std::exception &e) {                                          \
    throw std::runtime_error(                                                  \
        fmt::format("Failed to import '{}': {}", filename, e.what()));         \
  }

void registerIOBindings(sol::state &lua)
{
  sol::table tsd = lua["tsd"];
  sol::table io = tsd["io"];

  // Importers - geometry/scene formats
  io["importOBJ"] = sol::overload(
      [](scene::Scene &s, const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_OBJ(s, f.c_str()), f);
      },
      [](scene::Scene &s, const std::string &f, scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_OBJ(s, f.c_str(), loc), f);
      },
      [](scene::Scene &s,
          const std::string &f,
          scene::LayerNodeRef loc,
          bool useDefaultMat) {
        TSD_LUA_IMPORT_WRAP(
            tsd::io::import_OBJ(s, f.c_str(), loc, useDefaultMat), f);
      });

  io["importGLTF"] = sol::overload(
      [](scene::Scene &s, const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_GLTF(s, f.c_str()), f);
      },
      [](scene::Scene &s, const std::string &f, scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_GLTF(s, f.c_str(), loc), f);
      });

  io["importPLY"] = sol::overload(
      [](scene::Scene &s, const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_PLY(s, f.c_str()), f);
      },
      [](scene::Scene &s, const std::string &f, scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_PLY(s, f.c_str(), loc), f);
      });

  io["importHDRI"] = sol::overload(
      [](scene::Scene &s, const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_HDRI(s, f.c_str()), f);
      },
      [](scene::Scene &s, const std::string &f, scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_HDRI(s, f.c_str(), loc), f);
      });

  io["importUSD"] = sol::overload(
      [](scene::Scene &s, const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_USD(s, f.c_str()), f);
      },
      [](scene::Scene &s, const std::string &f, scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_USD(s, f.c_str(), loc), f);
      });

  io["importPDB"] = sol::overload(
      [](scene::Scene &s, const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_PDB(s, f.c_str()), f);
      },
      [](scene::Scene &s, const std::string &f, scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_PDB(s, f.c_str(), loc), f);
      });

  io["importSWC"] = sol::overload(
      [](scene::Scene &s, const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_SWC(s, f.c_str()), f);
      },
      [](scene::Scene &s, const std::string &f, scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_SWC(s, f.c_str(), loc), f);
      });

  io["importAGX"] = sol::overload(
      [](scene::Scene &s, const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_AGX(s, f.c_str()), f);
      },
      [](scene::Scene &s, const std::string &f, scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_AGX(s, f.c_str(), loc), f);
      });

  io["importASSIMP"] = sol::overload(
      [](scene::Scene &s, const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_ASSIMP(s, f.c_str()), f);
      },
      [](scene::Scene &s, const std::string &f, scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_ASSIMP(s, f.c_str(), loc), f);
      },
      [](scene::Scene &s,
          const std::string &f,
          scene::LayerNodeRef loc,
          bool flatten) {
        TSD_LUA_IMPORT_WRAP(
            tsd::io::import_ASSIMP(s, f.c_str(), loc, flatten), f);
      });

  io["importAXYZ"] = sol::overload(
      [](scene::Scene &s, const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_AXYZ(s, f.c_str()), f);
      },
      [](scene::Scene &s, const std::string &f, scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_AXYZ(s, f.c_str(), loc), f);
      });

  io["importDLAF"] = sol::overload(
      [](scene::Scene &s, const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_DLAF(s, f.c_str()), f);
      },
      [](scene::Scene &s, const std::string &f, scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_DLAF(s, f.c_str(), loc), f);
      },
      [](scene::Scene &s,
          const std::string &f,
          scene::LayerNodeRef loc,
          bool useDefaultMat) {
        TSD_LUA_IMPORT_WRAP(
            tsd::io::import_DLAF(s, f.c_str(), loc, useDefaultMat), f);
      });

  io["importE57XYZ"] = sol::overload(
      [](scene::Scene &s, const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_E57XYZ(s, f.c_str()), f);
      },
      [](scene::Scene &s, const std::string &f, scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_E57XYZ(s, f.c_str(), loc), f);
      });

  io["importENSIGHT"] = sol::overload(
      [](scene::Scene &s, const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_ENSIGHT(s, f.c_str()), f);
      },
      [](scene::Scene &s, const std::string &f, scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_ENSIGHT(s, f.c_str(), loc), f);
      },
      [](scene::Scene &s,
          const std::string &f,
          scene::LayerNodeRef loc,
          sol::table fields) {
        std::vector<std::string> fs;
        for (size_t i = 1; i <= fields.size(); i++)
          fs.push_back(fields[i].get<std::string>());
        TSD_LUA_IMPORT_WRAP(tsd::io::import_ENSIGHT(s, f.c_str(), loc, fs), f);
      },
      [](scene::Scene &s,
          const std::string &f,
          scene::LayerNodeRef loc,
          sol::table fields,
          int timestep) {
        std::vector<std::string> fs;
        for (size_t i = 1; i <= fields.size(); i++)
          fs.push_back(fields[i].get<std::string>());
        TSD_LUA_IMPORT_WRAP(
            tsd::io::import_ENSIGHT(s, f.c_str(), loc, fs, timestep), f);
      });

  io["importHSMESH"] = sol::overload(
      [](scene::Scene &s, const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_HSMESH(s, f.c_str()), f);
      },
      [](scene::Scene &s, const std::string &f, scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_HSMESH(s, f.c_str(), loc), f);
      });

  io["importNBODY"] = sol::overload(
      [](scene::Scene &s, const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_NBODY(s, f.c_str()), f);
      },
      [](scene::Scene &s, const std::string &f, scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_NBODY(s, f.c_str(), loc), f);
      },
      [](scene::Scene &s,
          const std::string &f,
          scene::LayerNodeRef loc,
          bool useDefaultMat) {
        TSD_LUA_IMPORT_WRAP(
            tsd::io::import_NBODY(s, f.c_str(), loc, useDefaultMat), f);
      });

  io["importPOINTSBIN"] = sol::overload(
      [](scene::Scene &s, sol::table filepaths) {
        std::vector<std::string> paths;
        for (size_t i = 1; i <= filepaths.size(); i++)
          paths.push_back(filepaths[i].get<std::string>());
        TSD_LUA_IMPORT_WRAP(tsd::io::import_POINTSBIN(s, paths), "POINTSBIN");
      },
      [](scene::Scene &s, sol::table filepaths, scene::LayerNodeRef loc) {
        std::vector<std::string> paths;
        for (size_t i = 1; i <= filepaths.size(); i++)
          paths.push_back(filepaths[i].get<std::string>());
        TSD_LUA_IMPORT_WRAP(
            tsd::io::import_POINTSBIN(s, paths, loc), "POINTSBIN");
      });

  io["importPT"] = sol::overload(
      [](scene::Scene &s, const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_PT(s, f.c_str()), f);
      },
      [](scene::Scene &s, const std::string &f, scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_PT(s, f.c_str(), loc), f);
      });

  io["importSilo"] = sol::overload(
      [](scene::Scene &s, const std::string &f, scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_SILO(s, f.c_str(), loc), f);
      });

  io["importSMESH"] = sol::overload(
      [](scene::Scene &s, const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_SMESH(s, f.c_str()), f);
      },
      [](scene::Scene &s, const std::string &f, scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_SMESH(s, f.c_str(), loc), f);
      },
      [](scene::Scene &s,
          const std::string &f,
          scene::LayerNodeRef loc,
          bool isAnimation) {
        TSD_LUA_IMPORT_WRAP(
            tsd::io::import_SMESH(s, f.c_str(), loc, isAnimation), f);
      });

  io["importTRK"] = sol::overload(
      [](scene::Scene &s, const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_TRK(s, f.c_str()), f);
      },
      [](scene::Scene &s, const std::string &f, scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_TRK(s, f.c_str(), loc), f);
      });

  io["importUSD2"] = sol::overload(
      [](scene::Scene &s, const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_USD2(s, f.c_str()), f);
      },
      [](scene::Scene &s, const std::string &f, scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_USD2(s, f.c_str(), loc), f);
      });

  io["importXYZDP"] = sol::overload(
      [](scene::Scene &s, const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_XYZDP(s, f.c_str()), f);
      },
      [](scene::Scene &s, const std::string &f, scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_XYZDP(s, f.c_str(), loc), f);
      });

  // Volume importers
  io["importVolume"] = sol::overload(
      [](scene::Scene &s, const std::string &f) {
        TSD_LUA_IMPORT_WRAP_RETURN(tsd::io::import_volume(s, f.c_str()), f);
      },
      [](scene::Scene &s, const std::string &f, scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP_RETURN(
            tsd::io::import_volume(s, f.c_str(), loc), f);
      });

  io["importRAW"] = [](scene::Scene &s, const std::string &f) {
    TSD_LUA_IMPORT_WRAP_RETURN(tsd::io::import_RAW(s, f.c_str()), f);
  };

  io["importNVDB"] = [](scene::Scene &s, const std::string &f) {
    TSD_LUA_IMPORT_WRAP_RETURN(tsd::io::import_NVDB(s, f.c_str()), f);
  };

  io["importMHD"] = [](scene::Scene &s, const std::string &f) {
    TSD_LUA_IMPORT_WRAP_RETURN(tsd::io::import_MHD(s, f.c_str()), f);
  };

  io["importFLASH"] = [](scene::Scene &s, const std::string &f) {
    TSD_LUA_IMPORT_WRAP_RETURN(tsd::io::import_FLASH(s, f.c_str()), f);
  };

  io["importVTI"] = [](scene::Scene &s, const std::string &f) {
    TSD_LUA_IMPORT_WRAP_RETURN(tsd::io::import_VTI(s, f.c_str()), f);
  };

  io["importVTU"] = [](scene::Scene &s, const std::string &f) {
    TSD_LUA_IMPORT_WRAP_RETURN(tsd::io::import_VTU(s, f.c_str()), f);
  };

  // Procedural generators
  io["generateRandomSpheres"] =
      sol::overload([](scene::Scene &s) { tsd::io::generate_randomSpheres(s); },
          [](scene::Scene &s, scene::LayerNodeRef loc) {
            tsd::io::generate_randomSpheres(s, loc);
          },
          [](scene::Scene &s, scene::LayerNodeRef loc, bool useDefaultMat) {
            tsd::io::generate_randomSpheres(s, loc, useDefaultMat);
          });

  io["generateMaterialOrb"] =
      sol::overload([](scene::Scene &s) { tsd::io::generate_material_orb(s); },
          [](scene::Scene &s, scene::LayerNodeRef loc) {
            tsd::io::generate_material_orb(s, loc);
          });

  io["generateMonkey"] =
      sol::overload([](scene::Scene &s) { tsd::io::generate_monkey(s); },
          [](scene::Scene &s, scene::LayerNodeRef loc) {
            tsd::io::generate_monkey(s, loc);
          });

  io["generateCylinders"] =
      sol::overload([](scene::Scene &s) { tsd::io::generate_cylinders(s); },
          [](scene::Scene &s, scene::LayerNodeRef loc) {
            tsd::io::generate_cylinders(s, loc);
          },
          [](scene::Scene &s, scene::LayerNodeRef loc, bool useDefaultMat) {
            tsd::io::generate_cylinders(s, loc, useDefaultMat);
          });

  io["generateDefaultLights"] = [](scene::Scene &s) {
    tsd::io::generate_default_lights(s);
  };

  io["generateHdriDome"] =
      sol::overload([](scene::Scene &s) { tsd::io::generate_hdri_dome(s); },
          [](scene::Scene &s, scene::LayerNodeRef loc) {
            tsd::io::generate_hdri_dome(s, loc);
          });

  io["generateRtow"] =
      sol::overload([](scene::Scene &s) { tsd::io::generate_rtow(s); },
          [](scene::Scene &s, scene::LayerNodeRef loc) {
            tsd::io::generate_rtow(s, loc);
          });

  io["generateSphereSetVolume"] = sol::overload(
      [](scene::Scene &s) { tsd::io::generate_sphereSetVolume(s); },
      [](scene::Scene &s, scene::LayerNodeRef loc) {
        tsd::io::generate_sphereSetVolume(s, loc);
      });

  // Utilities
  io["makeDefaultColorMap"] = [](scene::Scene &s, sol::optional<size_t> size) {
    auto colors = core::makeDefaultColorMap(size.value_or(256));
    auto arr = s.createArray(ANARI_FLOAT32_VEC4, colors.size());
    arr->setData(colors.data());
    return arr;
  };

  // Serialization
  io["saveScene"] = sol::overload(
      [](scene::Scene &s, const std::string &filename) {
        core::DataTree tree;
        tsd::io::save_Scene(s, tree.root(), false);
        tree.save(filename.c_str());
      },
      [](scene::Scene &s, const std::string &filename, sol::table state) {
        core::DataTree tree;
        auto &root = tree.root();
        tsd::io::save_Scene(s, root["context"], false);
        copyTableToNode(state, root);
        tree.save(filename.c_str());
      });

  io["loadScene"] = [](scene::Scene &s, const std::string &filename) {
    core::DataTree tree;
    tree.load(filename.c_str());
    auto &root = tree.root();
    if (auto *c = root.child("context"); c != nullptr)
      tsd::io::load_Scene(s, *c);
    else
      tsd::io::load_Scene(s, root);
  };
}

#undef TSD_LUA_IMPORT_WRAP
#undef TSD_LUA_IMPORT_WRAP_RETURN

} // namespace tsd::scripting
