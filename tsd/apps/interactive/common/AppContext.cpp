// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "AppContext.h"
#include "windows/Log.h"

namespace tsd_viewer {

static void statusFunc(const void *_appContext,
    ANARIDevice device,
    ANARIObject source,
    ANARIDataType sourceType,
    ANARIStatusSeverity severity,
    ANARIStatusCode code,
    const char *message)
{
  const char *typeStr = anari::toString(sourceType);
  const auto *ctx = (const AppContext *)_appContext;
  const bool verbose = ctx->logging.verbose;

  if (severity == ANARI_SEVERITY_FATAL_ERROR) {
    fprintf(stderr, "[ANARI][FATAL][%s][%p] %s", typeStr, source, message);
    std::exit(1);
  } else if (severity == ANARI_SEVERITY_ERROR)
    tsd::logError("[ANARI][ERROR][%s][%p] %s", typeStr, source, message);
  else if (severity == ANARI_SEVERITY_WARNING)
    tsd::logWarning("[ANARI][WARN ][%s][%p] %s", typeStr, source, message);
  else if (verbose && severity == ANARI_SEVERITY_PERFORMANCE_WARNING)
    tsd::logPerfWarning(
        "[ANARI][PERF ][%s][%p] %s", typeStr, source, message);
  else if (verbose && severity == ANARI_SEVERITY_INFO)
    tsd::logInfo("[ANARI][INFO ][%s][%p] %s", typeStr, source, message);
  else if (verbose && severity == ANARI_SEVERITY_DEBUG)
    tsd::logDebug("[ANARI][DEBUG][%s][%p] %s", typeStr, source, message);
}

static std::vector<std::string> parseLibraryList()
{
  const char *libsFromEnv = getenv("TSD_ANARI_LIBRARIES");

  auto splitString = [](const std::string &input,
                         const std::string &delim) -> std::vector<std::string> {
    std::vector<std::string> tokens;
    size_t pos = 0;
    while (true) {
      size_t begin = input.find_first_not_of(delim, pos);
      if (begin == input.npos)
        return tokens;
      size_t end = input.find_first_of(delim, begin);
      tokens.push_back(input.substr(
          begin, (end == input.npos) ? input.npos : (end - begin)));
      pos = end;
    }
  };

  auto libList = splitString(libsFromEnv ? libsFromEnv : "", ",");
  if (libList.empty()) {
    libList.push_back("helide");
    libList.push_back("visrtx");
    libList.push_back("visgl");
    if (getenv("ANARI_LIBRARY"))
      libList.push_back("environment");
  }

  libList.push_back("{none}");

  return libList;
}

// AppContext definitions /////////////////////////////////////////////////////

AppContext::AppContext()
{
  tsd.ctx.setUpdateDelegate(&anari.delegate);
}

AppContext::~AppContext()
{
  releaseAllDevices();
}

void AppContext::parseCommandLine(int argc, char *argv[])
{
  auto importerType = ImporterType::NONE;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-v" || arg == "--verbose")
      this->logging.verbose = true;
    else if (arg == "-e" || arg == "--echoOutput")
      this->logging.echoOutput = true;
    else if (arg == "--noDefaultLayout")
      this->commandLine.useDefaultLayout = false;
    else if (arg == "--debug" || arg == "-g")
      this->commandLine.enableDebug = true;
    else if (arg == "--trace" || arg == "-t")
      this->commandLine.traceDir = argv[++i];
    else if (arg == "-tsd") {
      importerType = ImporterType::TSD;
      this->commandLine.loadingContext = true;
    } else if (arg == "-hdri")
      importerType = ImporterType::HDRI;
    else if (arg == "-dlaf")
      importerType = ImporterType::DLAF;
    else if (arg == "-nbody")
      importerType = ImporterType::NBODY;
    else if (arg == "-obj")
      importerType = ImporterType::OBJ;
    else if (arg == "-assimp")
      importerType = ImporterType::ASSIMP;
    else if (arg == "-assimp_flat")
      importerType = ImporterType::ASSIMP_FLAT;
    else if (arg == "-ply")
      importerType = ImporterType::PLY;
    else if (arg == "-raw")
      importerType = ImporterType::RAW;
    else if (importerType != ImporterType::NONE)
      this->commandLine.filenames.push_back({importerType, arg});
  }

  this->commandLine.libraryList = parseLibraryList();
}

void AppContext::setupSceneFromCommandLine(bool hdriOnly)
{
  if (hdriOnly) {
    for (const auto &f : commandLine.filenames) {
      tsd::logStatus("...loading file '%s'", f.second.c_str());
      if (f.first == ImporterType::HDRI)
        tsd::import_HDRI(tsd.ctx, f.second.c_str());
    }
    return;
  }

  if (commandLine.filenames.empty()) {
#if 0
      tsd::logStatus("...generating noise volume");
      tsd::generate_noiseVolume(tsd.ctx);
#elif 0
    tsd::logStatus("...generating cylinders");
    tsd::generate_cylinders(tsd.ctx);
#elif 0
    tsd::logStatus("...generating rtow spheres");
    tsd::generate_rtow(tsd.ctx);
#elif 0
    tsd::logStatus("...generating monkey from embedded data");
    tsd::generate_monkey(tsd.ctx);
#elif 1
    tsd::logStatus("...generating material_orb from embedded data");
    tsd::generate_material_orb(tsd.ctx);
#else
    tsd::logStatus("...generating random spheres");
    tsd::generate_randomSpheres(tsd.ctx);
#endif
  } else {
    for (const auto &f : commandLine.filenames) {
      tsd::logStatus("...loading file '%s'", f.second.c_str());
      if (f.first == ImporterType::TSD)
        tsd::import_Context(tsd.ctx, f.second.c_str());
      else if (f.first == ImporterType::PLY)
        tsd::import_PLY(tsd.ctx, f.second.c_str());
      else if (f.first == ImporterType::OBJ)
        tsd::import_OBJ(tsd.ctx, f.second.c_str());
      else if (f.first == ImporterType::ASSIMP)
        tsd::import_ASSIMP(tsd.ctx, f.second.c_str(), false);
      else if (f.first == ImporterType::ASSIMP_FLAT)
        tsd::import_ASSIMP(tsd.ctx, f.second.c_str(), true);
      else if (f.first == ImporterType::DLAF)
        tsd::import_DLAF(tsd.ctx, f.second.c_str());
      else if (f.first == ImporterType::NBODY)
        tsd::import_NBODY(tsd.ctx, f.second.c_str());
      else if (f.first == ImporterType::HDRI)
        tsd::import_HDRI(tsd.ctx, f.second.c_str());
#if 0
      else if (f.first == ImporterType::RAW)
        tsd::import_RAW(tsd.ctx, f.second.c_str());
#endif
    }
  }
}

anari::Device AppContext::loadDevice(const std::string &libraryName)
{
  anari::Device dev = this->anari.loadedDevices[libraryName];

  if (libraryName.empty() || libraryName == "{none}")
    return nullptr;
  else if (dev) {
    anari::retain(dev, dev);
    return dev;
  }

  auto library = anari::loadLibrary(libraryName.c_str(), statusFunc, this);
  if (!library)
    throw std::runtime_error("Failed to load ANARI library");

  if (!this->commandLine.debug && this->commandLine.enableDebug)
    this->commandLine.debug = anari::loadLibrary("debug", statusFunc, this);

  dev = anari::newDevice(library, "default");

  anari::unloadLibrary(library);

  if (this->commandLine.enableDebug)
    anari::setParameter(dev, dev, "glDebug", true);

#ifdef USE_GLES2
  anari::setParameter(dev, dev, "glAPI", "OpenGL_ES");
#else
  anari::setParameter(dev, dev, "glAPI", "OpenGL");
#endif

  if (this->commandLine.debug) {
    anari::Device dbg = anari::newDevice(this->commandLine.debug, "debug");
    anari::setParameter(dbg, dbg, "wrappedDevice", dev);
    if (!this->commandLine.traceDir.empty()) {
      anari::setParameter(dbg, dbg, "traceDir", this->commandLine.traceDir);
      anari::setParameter(dbg, dbg, "traceMode", "code");
    }
    anari::commitParameters(dbg, dbg);
    anari::release(dev, dev);
    dev = dbg;
  }

  anari::commitParameters(dev, dev);

  this->anari.loadedDevices[libraryName] = dev;
  anari::retain(dev, dev);

  return dev;
}

tsd::RenderIndex *AppContext::acquireRenderIndex(anari::Device d)
{
  auto &liveIdx = this->anari.rIdxs[d];
  if (liveIdx.refCount == 0) {
#if 1
    liveIdx.idx = anari.delegate.emplace<tsd::RenderIndexTreeHierarchy>(d);
#else
    liveIdx.idx = anari.delegate.emplace<tsd::RenderIndexFlatRegistry>(d);
#endif
    liveIdx.idx->populate(this->tsd.ctx, false);
  }
  liveIdx.refCount++;
  return liveIdx.idx;
}

void AppContext::releaseRenderIndex(anari::Device d)
{
  auto &liveIdx = this->anari.rIdxs[d];
  if (liveIdx.refCount == 0)
    return;
  else if (liveIdx.refCount == 1)
    anari.delegate.erase(liveIdx.idx);
  liveIdx.refCount--;
}

void AppContext::releaseAllDevices()
{
  for (auto &d : anari.loadedDevices) {
    if (d.second)
      anari::release(d.second, d.second);
  }
  anari.loadedDevices.clear();
}

void AppContext::setSelectedObject(tsd::Object *o)
{
  tsd.selectedObject = o;
  anari.delegate.signalObjectFilteringChanged();
}

} // namespace tsd_viewer
