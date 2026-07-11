// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/animation/AnimationManager.hpp"
#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/io/archives/AnimationManagerArchive.hpp"
#include "tsd/io/archives/SceneArchive.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/procedural.hpp"
#include "tsd/scene/Scene.hpp"
#include "tsd/scripting/LuaBindings.hpp"
#include "tsd/scripting/Sol2Helpers.hpp"

#include <fmt/format.h>
#include <sol/sol.hpp>

namespace tsd::scripting {

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
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_OBJ(s, anim, f.c_str()), f);
      },
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f,
          scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_OBJ(s, anim, f.c_str(), loc), f);
      },
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f,
          scene::LayerNodeRef loc,
          bool useDefaultMat) {
        TSD_LUA_IMPORT_WRAP(
            tsd::io::import_OBJ(s, anim, f.c_str(), loc, useDefaultMat), f);
      });

  io["importGLTF"] = sol::overload(
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_GLTF(s, anim, f.c_str()), f);
      },
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f,
          scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_GLTF(s, anim, f.c_str(), loc), f);
      });

  io["importPLY"] = sol::overload(
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_PLY(s, anim, f.c_str()), f);
      },
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f,
          scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_PLY(s, anim, f.c_str(), loc), f);
      });

  io["importHDRI"] = sol::overload(
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_HDRI(s, anim, f.c_str()), f);
      },
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f,
          scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_HDRI(s, anim, f.c_str(), loc), f);
      });

  io["importUSD"] = sol::overload(
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_USD(s, anim, f.c_str()), f);
      },
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f,
          scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_USD(s, anim, f.c_str(), loc), f);
      });

  io["importPDB"] = sol::overload(
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_PDB(s, anim, f.c_str()), f);
      },
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f,
          scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_PDB(s, anim, f.c_str(), loc), f);
      });

  io["importPBRT"] = sol::overload(
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_PBRT(s, anim, f.c_str()), f);
      },
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f,
          scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_PBRT(s, anim, f.c_str(), loc), f);
      });

  io["importSWC"] = sol::overload(
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_SWC(s, anim, f.c_str()), f);
      },
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f,
          scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_SWC(s, anim, f.c_str(), loc), f);
      });

  io["importAGX"] = sol::overload(
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_AGX(s, anim, f.c_str()), f);
      },
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f,
          scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_AGX(s, anim, f.c_str(), loc), f);
      });

  io["importASSIMP"] = sol::overload(
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_ASSIMP(s, anim, f.c_str()), f);
      },
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f,
          scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_ASSIMP(s, anim, f.c_str(), loc), f);
      },
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f,
          scene::LayerNodeRef loc,
          bool flatten) {
        TSD_LUA_IMPORT_WRAP(
            tsd::io::import_ASSIMP(s, anim, f.c_str(), loc, flatten), f);
      });

  io["importAXYZ"] = sol::overload(
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_AXYZ(s, anim, f.c_str()), f);
      },
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f,
          scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_AXYZ(s, anim, f.c_str(), loc), f);
      });

  io["importDLAF"] = sol::overload(
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_DLAF(s, anim, f.c_str()), f);
      },
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f,
          scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_DLAF(s, anim, f.c_str(), loc), f);
      },
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f,
          scene::LayerNodeRef loc,
          bool useDefaultMat) {
        TSD_LUA_IMPORT_WRAP(
            tsd::io::import_DLAF(s, anim, f.c_str(), loc, useDefaultMat), f);
      });

  io["importE57XYZ"] = sol::overload(
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_E57XYZ(s, anim, f.c_str()), f);
      },
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f,
          scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_E57XYZ(s, anim, f.c_str(), loc), f);
      });

  io["importENSIGHT"] = sol::overload(
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_ENSIGHT(s, anim, f.c_str()), f);
      },
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f,
          scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(
            tsd::io::import_ENSIGHT(s, anim, f.c_str(), loc), f);
      },
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f,
          scene::LayerNodeRef loc,
          sol::table fields) {
        std::vector<std::string> fs;
        for (size_t i = 1; i <= fields.size(); i++)
          fs.push_back(fields[i].get<std::string>());
        TSD_LUA_IMPORT_WRAP(
            tsd::io::import_ENSIGHT(s, anim, f.c_str(), loc, fs), f);
      },
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f,
          scene::LayerNodeRef loc,
          sol::table fields,
          int timestep) {
        std::vector<std::string> fs;
        for (size_t i = 1; i <= fields.size(); i++)
          fs.push_back(fields[i].get<std::string>());
        TSD_LUA_IMPORT_WRAP(
            tsd::io::import_ENSIGHT(s, anim, f.c_str(), loc, fs, timestep), f);
      });

  io["importHSMESH"] = sol::overload(
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_HSMESH(s, anim, f.c_str()), f);
      },
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f,
          scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_HSMESH(s, anim, f.c_str(), loc), f);
      });

  io["importNBODY"] = sol::overload(
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_NBODY(s, anim, f.c_str()), f);
      },
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f,
          scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_NBODY(s, anim, f.c_str(), loc), f);
      },
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f,
          scene::LayerNodeRef loc,
          bool useDefaultMat) {
        TSD_LUA_IMPORT_WRAP(
            tsd::io::import_NBODY(s, anim, f.c_str(), loc, useDefaultMat), f);
      });

  io["importPOINTSBIN"] = sol::overload(
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          sol::table filepaths) {
        std::vector<std::string> paths;
        for (size_t i = 1; i <= filepaths.size(); i++)
          paths.push_back(filepaths[i].get<std::string>());
        TSD_LUA_IMPORT_WRAP(
            tsd::io::import_POINTSBIN(s, anim, paths), "POINTSBIN");
      },
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          sol::table filepaths,
          scene::LayerNodeRef loc) {
        std::vector<std::string> paths;
        for (size_t i = 1; i <= filepaths.size(); i++)
          paths.push_back(filepaths[i].get<std::string>());
        TSD_LUA_IMPORT_WRAP(
            tsd::io::import_POINTSBIN(s, anim, paths, loc), "POINTSBIN");
      });

  io["importPT"] = sol::overload(
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_PT(s, anim, f.c_str()), f);
      },
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f,
          scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_PT(s, anim, f.c_str(), loc), f);
      });

  io["importSilo"] = sol::overload([](scene::Scene &s,
                                       animation::AnimationManager &anim,
                                       const std::string &f,
                                       scene::LayerNodeRef loc) {
    TSD_LUA_IMPORT_WRAP(tsd::io::import_SILO(s, anim, f.c_str(), loc), f);
  });

  io["importSMESH"] = sol::overload(
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_SMESH(s, anim, f.c_str()), f);
      },
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f,
          scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_SMESH(s, anim, f.c_str(), loc), f);
      },
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f,
          scene::LayerNodeRef loc,
          bool isAnimation) {
        TSD_LUA_IMPORT_WRAP(
            tsd::io::import_SMESH(s, anim, f.c_str(), loc, isAnimation), f);
      });

  io["importTRK"] = sol::overload(
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_TRK(s, anim, f.c_str()), f);
      },
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f,
          scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_TRK(s, anim, f.c_str(), loc), f);
      });

  io["importXYZDP"] = sol::overload(
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_XYZDP(s, anim, f.c_str()), f);
      },
      [](scene::Scene &s,
          animation::AnimationManager &anim,
          const std::string &f,
          scene::LayerNodeRef loc) {
        TSD_LUA_IMPORT_WRAP(tsd::io::import_XYZDP(s, anim, f.c_str(), loc), f);
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

  io["generateEmissiveGeometries"] = sol::overload(
      [](scene::Scene &s) { tsd::io::generate_emissive_geometries(s); },
      [](scene::Scene &s, scene::LayerNodeRef loc) {
        tsd::io::generate_emissive_geometries(s, loc);
      });

  io["generateEmissiveMdlComparison"] = sol::overload(
      [](scene::Scene &s) { tsd::io::generate_emissive_mdl_comparison(s); },
      [](scene::Scene &s, scene::LayerNodeRef loc) {
        tsd::io::generate_emissive_mdl_comparison(s, loc);
      });

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

  // Archives
  io["saveSceneArchive"] = [](scene::Scene &s, const std::string &filename) {
    if (!tsd::io::save_SceneArchive(s, filename.c_str()))
      throw std::runtime_error("Failed to save Scene Archive");
  };
  io["loadSceneArchive"] = [](scene::Scene &s, const std::string &filename) {
    if (!tsd::io::load_SceneArchive(s, filename.c_str()))
      throw std::runtime_error("Failed to load Scene Archive");
  };
  io["saveAnimationManagerArchive"] = [](animation::AnimationManager &manager,
                                          const std::string &filename) {
    if (!tsd::io::save_AnimationManagerArchive(manager, filename.c_str()))
      throw std::runtime_error("Failed to save Animation Manager Archive");
  };
  io["loadAnimationManagerArchive"] = [](animation::AnimationManager &manager,
                                          const std::string &filename) {
    if (!tsd::io::load_AnimationManagerArchive(manager, filename.c_str()))
      throw std::runtime_error("Failed to load Animation Manager Archive");
  };
}

#undef TSD_LUA_IMPORT_WRAP
#undef TSD_LUA_IMPORT_WRAP_RETURN

} // namespace tsd::scripting
