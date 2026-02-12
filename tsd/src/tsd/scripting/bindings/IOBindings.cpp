// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/scene/Scene.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/procedural.hpp"
#include "tsd/io/serialization.hpp"
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
      [](core::Scene &s, const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_OBJ(s, f.c_str()), f);
      },
      [](core::Scene &s, const std::string &f, core::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_OBJ(s, f.c_str(), loc), f);
      },
      [](core::Scene &s,
          const std::string &f,
          core::LayerNodeRef loc,
          bool useDefaultMat) {
        TSD_LUA_IMPORT_WRAP(
            tsd::io::import_OBJ(s, f.c_str(), loc, useDefaultMat), f);
      });

  io["importGLTF"] = sol::overload(
      [](core::Scene &s, const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_GLTF(s, f.c_str()), f);
      },
      [](core::Scene &s, const std::string &f, core::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_GLTF(s, f.c_str(), loc), f);
      });

  io["importPLY"] = sol::overload(
      [](core::Scene &s, const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_PLY(s, f.c_str()), f);
      },
      [](core::Scene &s, const std::string &f, core::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_PLY(s, f.c_str(), loc), f);
      });

  io["importHDRI"] = sol::overload(
      [](core::Scene &s, const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_HDRI(s, f.c_str()), f);
      },
      [](core::Scene &s, const std::string &f, core::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_HDRI(s, f.c_str(), loc), f);
      });

  io["importUSD"] = sol::overload(
      [](core::Scene &s, const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_USD(s, f.c_str()), f);
      },
      [](core::Scene &s, const std::string &f, core::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_USD(s, f.c_str(), loc), f);
      });

  io["importPDB"] = sol::overload(
      [](core::Scene &s, const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_PDB(s, f.c_str()), f);
      },
      [](core::Scene &s, const std::string &f, core::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_PDB(s, f.c_str(), loc), f);
      });

  io["importSWC"] = sol::overload(
      [](core::Scene &s, const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_SWC(s, f.c_str()), f);
      },
      [](core::Scene &s, const std::string &f, core::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_SWC(s, f.c_str(), loc), f);
      });

  // Volume importers
  io["importVolume"] = sol::overload(
      [](core::Scene &s, const std::string &f) {
        TSD_LUA_IMPORT_WRAP_RETURN(tsd::io::import_volume(s, f.c_str()), f);
      },
      [](core::Scene &s, const std::string &f, core::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP_RETURN(
            tsd::io::import_volume(s, f.c_str(), loc), f);
      });

  io["importRAW"] = [](core::Scene &s, const std::string &f) {
    TSD_LUA_IMPORT_WRAP_RETURN(tsd::io::import_RAW(s, f.c_str()), f);
  };

  io["importNVDB"] = [](core::Scene &s, const std::string &f) {
    TSD_LUA_IMPORT_WRAP_RETURN(tsd::io::import_NVDB(s, f.c_str()), f);
  };

  io["importMHD"] = [](core::Scene &s, const std::string &f) {
    TSD_LUA_IMPORT_WRAP_RETURN(tsd::io::import_MHD(s, f.c_str()), f);
  };

  // Procedural generators
  io["generateRandomSpheres"] =
      sol::overload([](core::Scene &s) { tsd::io::generate_randomSpheres(s); },
          [](core::Scene &s, core::LayerNodeRef loc) {
            tsd::io::generate_randomSpheres(s, loc);
          },
          [](core::Scene &s, core::LayerNodeRef loc, bool useDefaultMat) {
            tsd::io::generate_randomSpheres(s, loc, useDefaultMat);
          });

  io["generateMaterialOrb"] =
      sol::overload([](core::Scene &s) { tsd::io::generate_material_orb(s); },
          [](core::Scene &s, core::LayerNodeRef loc) {
            tsd::io::generate_material_orb(s, loc);
          });

  io["generateMonkey"] =
      sol::overload([](core::Scene &s) { tsd::io::generate_monkey(s); },
          [](core::Scene &s, core::LayerNodeRef loc) {
            tsd::io::generate_monkey(s, loc);
          });

  io["generateCylinders"] =
      sol::overload([](core::Scene &s) { tsd::io::generate_cylinders(s); },
          [](core::Scene &s, core::LayerNodeRef loc) {
            tsd::io::generate_cylinders(s, loc);
          },
          [](core::Scene &s, core::LayerNodeRef loc, bool useDefaultMat) {
            tsd::io::generate_cylinders(s, loc, useDefaultMat);
          });

  io["generateDefaultLights"] = [](core::Scene &s) {
    tsd::io::generate_default_lights(s);
  };

  io["generateHdriDome"] =
      sol::overload([](core::Scene &s) { tsd::io::generate_hdri_dome(s); },
          [](core::Scene &s, core::LayerNodeRef loc) {
            tsd::io::generate_hdri_dome(s, loc);
          });

  io["generateRtow"] =
      sol::overload([](core::Scene &s) { tsd::io::generate_rtow(s); },
          [](core::Scene &s, core::LayerNodeRef loc) {
            tsd::io::generate_rtow(s, loc);
          });

  io["generateSphereSetVolume"] = sol::overload(
      [](core::Scene &s) { tsd::io::generate_sphereSetVolume(s); },
      [](core::Scene &s, core::LayerNodeRef loc) {
        tsd::io::generate_sphereSetVolume(s, loc);
      });

  // Serialization
  io["saveScene"] = sol::overload(
      [](core::Scene &s, const std::string &filename) {
        core::DataTree tree;
        tsd::io::save_Scene(s, tree.root(), false);
        tree.save(filename.c_str());
      },
      [](core::Scene &s, const std::string &filename, sol::table state) {
        core::DataTree tree;
        auto &root = tree.root();
        tsd::io::save_Scene(s, root["context"], false);
        copyTableToNode(state, root);
        tree.save(filename.c_str());
      });

  io["loadScene"] = [](core::Scene &s, const std::string &filename) {
    core::DataTree tree;
    tree.load(filename.c_str());
    tsd::io::load_Scene(s, tree.root());
  };
}

#undef TSD_LUA_IMPORT_WRAP
#undef TSD_LUA_IMPORT_WRAP_RETURN

} // namespace tsd::scripting
