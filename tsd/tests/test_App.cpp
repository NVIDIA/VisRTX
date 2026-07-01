// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/app/ApplicationDump.h"
#include "tsd/app/Context.h"
#include "tsd/app/LegacyApplicationContext.h"
#include "tsd/io/archives/SceneArchive.hpp"
// std
#include <filesystem>
#include <fstream>
#include <string>
#include <variant>
#include <vector>

SCENARIO("Application Dumps embed required Archives without owning the root",
    "[App]")
{
  tsd::app::Context context;
  context.tsd.scene.addLayer("archived");

  tsd::core::DataTree tree;
  auto &root = tree.root();
  root["viewerState"]["theme"] = "dark";

  REQUIRE(tsd::app::serialize_ApplicationDump(context, root));

  auto *archives = root.child("archives");
  REQUIRE(archives != nullptr);
  REQUIRE(archives->child("scene") != nullptr);
  auto *animationManager = archives->child("animationManager");
  REQUIRE(animationManager != nullptr);
  REQUIRE(animationManager->child("objects") != nullptr);
  REQUIRE(animationManager->child("objects")->numChildren() == 0);
  REQUIRE(archives->child("scene")->child("animations") == nullptr);
  REQUIRE(root["viewerState"]["theme"].getValueAs<std::string>() == "dark");

  tsd::app::Context restored;
  REQUIRE(tsd::app::deserialize_ApplicationDump(restored, root));
  REQUIRE(root["viewerState"]["theme"].getValueAs<std::string>() == "dark");
}

SCENARIO("Application Dumps restore Scene before dependent animations", "[App]")
{
  tsd::app::Context source;
  auto archivedGeometry =
      source.tsd.scene.createObject<tsd::scene::Geometry>("sphere");
  archivedGeometry->setName("archived geometry");
  const float times[] = {0.f, 1.f};
  const float values[] = {1.f, 2.f};
  source.tsd.animationMgr.addAnimation("archived animation")
      .addObjectParameterBinding(
          archivedGeometry.data(), "radius", ANARI_FLOAT32, values, times, 2);

  tsd::core::DataTree tree;
  REQUIRE(tsd::app::serialize_ApplicationDump(source, tree.root()));

  tsd::app::Context target;
  auto staleGeometry =
      target.tsd.scene.createObject<tsd::scene::Geometry>("cone");
  target.tsd.animationMgr.addAnimation("stale animation")
      .addObjectParameterBinding(
          staleGeometry.data(), "radius", ANARI_FLOAT32, values, times, 2);
  target.tsd.animationMgr.play();

  REQUIRE(tsd::app::deserialize_ApplicationDump(target, tree.root()));
  REQUIRE(target.tsd.scene.numberOfObjects(ANARI_GEOMETRY) == 1);
  auto restoredGeometry = target.tsd.scene.getObject<tsd::scene::Geometry>(0);
  REQUIRE(restoredGeometry);
  REQUIRE(restoredGeometry->name() == "archived geometry");
  REQUIRE(target.tsd.animationMgr.animations().size() == 1);
  auto &restoredAnimation = target.tsd.animationMgr.animations().front();
  REQUIRE(restoredAnimation.name() == "archived animation");
  REQUIRE(restoredAnimation.objectParameterBindings().front().target()
      == restoredGeometry.data());
  REQUIRE_FALSE(target.tsd.animationMgr.isPlaying());
}

SCENARIO("Application Dumps remap animations to dense Scene Archive indices",
    "[App]")
{
  tsd::app::Context source;
  auto removed = source.tsd.scene.createObject<tsd::scene::Geometry>("sphere");
  auto retained =
      source.tsd.scene.createObject<tsd::scene::Geometry>("cylinder");
  retained->setName("retained geometry");
  source.tsd.scene.removeObject(removed.data());
  REQUIRE(retained.index() == 1);

  auto removedTransform = source.tsd.scene.insertChildTransformNode(
      source.tsd.scene.defaultLayer()->root(),
      tsd::math::IDENTITY_MAT4,
      "removed transform");
  auto retainedTransform = source.tsd.scene.insertChildTransformNode(
      source.tsd.scene.defaultLayer()->root(),
      tsd::math::IDENTITY_MAT4,
      "retained transform");
  source.tsd.scene.removeNode(removedTransform);
  REQUIRE(retainedTransform.index() == 2);

  const float times[] = {0.f, 1.f};
  const float values[] = {1.f, 2.f};
  auto &animation = source.tsd.animationMgr.addAnimation("sparse animation");
  animation.addObjectParameterBinding(
      retained.data(), "radius", ANARI_FLOAT32, values, times, 2);
  animation.addTransformBinding(retainedTransform);

  tsd::core::DataTree tree;
  REQUIRE(tsd::app::serialize_ApplicationDump(source, tree.root()));

  tsd::app::Context target;
  REQUIRE(tsd::app::deserialize_ApplicationDump(target, tree.root()));
  auto restored = target.tsd.scene.getObject<tsd::scene::Geometry>(0);
  REQUIRE(restored);
  REQUIRE(restored->name() == "retained geometry");
  REQUIRE(target.tsd.animationMgr.animations().size() == 1);
  auto &restoredAnimation = target.tsd.animationMgr.animations().front();
  REQUIRE(restoredAnimation.objectParameterBindings().front().target()
      == restored.data());
  auto restoredTransform = target.tsd.scene.defaultLayer()->at(1);
  REQUIRE(restoredTransform);
  REQUIRE((*restoredTransform)->name() == "retained transform");
  REQUIRE(restoredAnimation.transformBindings().front().target()
      == restoredTransform);
}

SCENARIO("Application Dumps round-trip only stable Context settings", "[App]")
{
  tsd::app::Context source;
  source.anari.setRenderIndexKind(tsd::app::RenderIndexKind::FLAT);
  source.offline.frame.width = 1920;
  source.offline.frame.height = 1080;
  source.offline.frame.colorFormat = ANARI_FLOAT32_VEC4;
  source.offline.frame.samples = 17;
  source.offline.frame.numFrames = 42;
  source.offline.frame.renderSubset = true;
  source.offline.frame.startFrame = 3;
  source.offline.frame.endFrame = 33;
  source.offline.frame.frameIncrement = 3;
  source.offline.camera.apertureRadius = 0.25f;
  source.offline.camera.focusDistance = 12.f;
  source.offline.camera.cameraIndex = 4;
  source.offline.renderer.activeRenderer = 2;
  source.offline.renderer.libraryName = "archived_library";
  auto &renderer = source.offline.renderer.rendererObjects.emplace_back(
      ANARI_RENDERER, "pathtracer");
  renderer.setName("archived renderer");
  renderer.setParameter("pixelSamples", 8);
  source.offline.output.outputDirectory = "/archived/output";
  source.offline.output.filePrefix = "beauty_";
  source.offline.aov.aovType = tsd::rendering::AOVType::DEPTH;
  source.offline.aov.depthMin = 2.f;
  source.offline.aov.depthMax = 20.f;
  source.offline.aov.edgeInvert = true;
  source.setLogVerbose(true);
  source.setLogEchoOutput(true);
  tsd::app::CameraPose pose;
  pose.name = "archived pose";
  pose.lookat = {1.f, 2.f, 3.f};
  pose.azeldist = {4.f, 5.f, 6.f};
  pose.fixedDist = 7.f;
  pose.upAxis = 2;
  pose.mode = 1;
  source.view.poses.push_back(pose);
  source.commandLine.stateFile = "source-only.tsd";
  source.tsd.sceneLoadComplete = true;

  tsd::core::DataTree tree;
  REQUIRE(tsd::app::serialize_ApplicationDump(source, tree.root()));
  REQUIRE(tree.root().child("commandLine") == nullptr);
  REQUIRE(tree.root().child("selectedNodes") == nullptr);
  REQUIRE(tree.root().child("sceneLoadComplete") == nullptr);
  REQUIRE(tree.root().child("context") == nullptr);

  tsd::app::Context target;
  target.commandLine.stateFile = "keep-me.tsd";
  target.tsd.sceneLoadComplete = false;
  REQUIRE(tsd::app::deserialize_ApplicationDump(target, tree.root()));

  REQUIRE(target.anari.renderIndexKind() == tsd::app::RenderIndexKind::FLAT);
  REQUIRE(target.offline.frame.width == 1920);
  REQUIRE(target.offline.frame.height == 1080);
  REQUIRE(target.offline.frame.colorFormat == ANARI_FLOAT32_VEC4);
  REQUIRE(target.offline.frame.samples == 17);
  REQUIRE(target.offline.frame.numFrames == 42);
  REQUIRE(target.offline.frame.renderSubset);
  REQUIRE(target.offline.frame.startFrame == 3);
  REQUIRE(target.offline.frame.endFrame == 33);
  REQUIRE(target.offline.frame.frameIncrement == 3);
  REQUIRE(target.offline.camera.apertureRadius == Approx(0.25f));
  REQUIRE(target.offline.camera.focusDistance == Approx(12.f));
  REQUIRE(target.offline.camera.cameraIndex == 4);
  REQUIRE(target.offline.renderer.activeRenderer == 2);
  REQUIRE(target.offline.renderer.libraryName == "archived_library");
  REQUIRE(target.offline.renderer.rendererObjects.size() == 1);
  REQUIRE(target.offline.renderer.rendererObjects.front().name()
      == "archived renderer");
  REQUIRE(target.offline.renderer.rendererObjects.front().subtype().str()
      == "pathtracer");
  REQUIRE(target.offline.renderer.rendererObjects.front().parameterValueAs<int>(
              "pixelSamples")
      == 8);
  REQUIRE(target.offline.output.outputDirectory == "/archived/output");
  REQUIRE(target.offline.output.filePrefix == "beauty_");
  REQUIRE(target.offline.aov.aovType == tsd::rendering::AOVType::DEPTH);
  REQUIRE(target.offline.aov.depthMin == Approx(2.f));
  REQUIRE(target.offline.aov.depthMax == Approx(20.f));
  REQUIRE(target.offline.aov.edgeInvert);
  REQUIRE(target.logVerbose());
  REQUIRE(target.logEchoOutput());
  REQUIRE(target.view.poses.size() == 1);
  REQUIRE(target.view.poses.front().name == "archived pose");
  REQUIRE(target.view.poses.front().lookat == tsd::math::float3(1.f, 2.f, 3.f));
  REQUIRE(
      target.view.poses.front().azeldist == tsd::math::float3(4.f, 5.f, 6.f));
  REQUIRE(target.view.poses.front().fixedDist == Approx(7.f));
  REQUIRE(target.view.poses.front().upAxis == 2);
  REQUIRE(target.view.poses.front().mode == 1);

  REQUIRE(target.commandLine.stateFile == "keep-me.tsd");
  REQUIRE_FALSE(target.tsd.sceneLoadComplete);
}

SCENARIO(
    "Application Dumps read legacy context payloads with animations", "[App]")
{
  tsd::app::Context source;
  auto geometry = source.tsd.scene.createObject<tsd::scene::Geometry>("sphere");
  geometry->setName("legacy geometry");
  const float times[] = {0.f, 1.f};
  const float values[] = {1.f, 2.f};
  source.tsd.animationMgr.addAnimation("legacy animation")
      .addObjectParameterBinding(
          geometry.data(), "radius", ANARI_FLOAT32, values, times, 2);

  tsd::core::DataTree tree;
  tsd::app::detail::serializeLegacyApplicationContext(
      source, tree.root()["context"]);
  REQUIRE(tree.root()["context"]["animations"]["objects"].numChildren() == 1);

  tsd::app::Context target;
  REQUIRE(tsd::app::deserialize_ApplicationDump(target, tree.root()));
  REQUIRE(target.tsd.scene.numberOfObjects(ANARI_GEOMETRY) == 1);
  REQUIRE(target.tsd.scene.getObject<tsd::scene::Geometry>(0)->name()
      == "legacy geometry");
  REQUIRE(target.tsd.animationMgr.animations().size() == 1);
  auto &animation = target.tsd.animationMgr.animations().front();
  REQUIRE(animation.name() == "legacy animation");
  REQUIRE(animation.objectParameterBindings().front().target()
      == target.tsd.scene.getObject<tsd::scene::Geometry>(0).data());
}

SCENARIO(
    "Application Dumps validate both Archives before replacing state", "[App]")
{
  tsd::app::Context source;
  source.tsd.scene.addLayer("archived");
  tsd::core::DataTree validTree;
  REQUIRE(tsd::app::serialize_ApplicationDump(source, validTree.root()));

  for (const char *missingArchive : {"scene", "animationManager"}) {
    DYNAMIC_SECTION("missing " << missingArchive)
    {
      tsd::core::DataTree tree;
      tree.root() = validTree.root();
      tree.root()["archives"].remove(missingArchive);

      tsd::app::Context target;
      target.tsd.scene.addLayer("preserved");
      target.tsd.animationMgr.addAnimation("preserved animation");
      target.tsd.animationMgr.play();

      REQUIRE_FALSE(tsd::app::deserialize_ApplicationDump(target, tree.root()));
      REQUIRE(target.tsd.scene.layer("preserved") != nullptr);
      REQUIRE(target.tsd.animationMgr.animations().size() == 1);
      REQUIRE(target.tsd.animationMgr.animations().front().name()
          == "preserved animation");
      REQUIRE(target.tsd.animationMgr.isPlaying());
    }
  }
}

SCENARIO(
    "Malformed Animation Manager Archives do not replace Application "
    "Dump state",
    "[App]")
{
  tsd::app::Context source;
  source.tsd.scene.addLayer("archived");
  source.anari.setRenderIndexKind(tsd::app::RenderIndexKind::FLAT);
  source.offline.frame.width = 1920;
  source.setLogVerbose(true);
  source.view.poses.push_back({});
  tsd::core::DataTree tree;
  REQUIRE(tsd::app::serialize_ApplicationDump(source, tree.root()));
  tree.root()["archives"]["animationManager"].remove("fps");

  tsd::app::Context target;
  target.tsd.scene.addLayer("preserved");
  target.tsd.animationMgr.addAnimation("preserved animation");
  target.tsd.animationMgr.setAnimationTime(0.25f);
  target.tsd.animationMgr.setAnimationIncrement(0.125f);
  target.tsd.animationMgr.setAnimationTotalFrames(41);
  target.tsd.animationMgr.setAnimationFPS(24.f);
  target.tsd.animationMgr.setLoop(false);
  target.tsd.animationMgr.play();
  target.anari.setRenderIndexKind(tsd::app::RenderIndexKind::ALL_LAYERS);
  target.offline.frame.width = 640;
  target.setLogVerbose(false);
  tsd::app::CameraPose preservedPose;
  preservedPose.name = "preserved pose";
  target.view.poses.push_back(preservedPose);

  REQUIRE_FALSE(tsd::app::deserialize_ApplicationDump(target, tree.root()));

  REQUIRE(target.tsd.scene.layer("preserved") != nullptr);
  REQUIRE(target.tsd.scene.layer("archived") == nullptr);
  REQUIRE(target.tsd.animationMgr.animations().size() == 1);
  REQUIRE(target.tsd.animationMgr.animations().front().name()
      == "preserved animation");
  REQUIRE(target.tsd.animationMgr.getAnimationTime() == Approx(0.25f));
  REQUIRE(target.tsd.animationMgr.getAnimationIncrement() == Approx(0.125f));
  REQUIRE(target.tsd.animationMgr.getAnimationTotalFrames() == 41);
  REQUIRE(target.tsd.animationMgr.getAnimationFPS() == Approx(24.f));
  REQUIRE_FALSE(target.tsd.animationMgr.isLoop());
  REQUIRE(target.tsd.animationMgr.isPlaying());
  REQUIRE(
      target.anari.renderIndexKind() == tsd::app::RenderIndexKind::ALL_LAYERS);
  REQUIRE(target.offline.frame.width == 640);
  REQUIRE_FALSE(target.logVerbose());
  REQUIRE(target.view.poses.size() == 1);
  REQUIRE(target.view.poses.front().name == "preserved pose");
}

SCENARIO("An empty Animation Manager Archive clears stale animations", "[App]")
{
  tsd::app::Context source;
  tsd::core::DataTree tree;
  REQUIRE(tsd::app::serialize_ApplicationDump(source, tree.root()));

  tsd::app::Context target;
  target.tsd.animationMgr.addAnimation("stale animation");
  target.tsd.animationMgr.play();
  REQUIRE(tsd::app::deserialize_ApplicationDump(target, tree.root()));
  REQUIRE(target.tsd.animationMgr.animations().empty());
  REQUIRE_FALSE(target.tsd.animationMgr.isPlaying());
}

SCENARIO("The TSD CLI records replacement and additive Scene inputs", "[App]")
{
  tsd::app::Context context;
  std::vector<std::string> args{
      "tsdViewer", "-tsd", "scene.tsd", "-obj", "mesh.obj"};

  context.parseCommandLine(args);

  REQUIRE(context.commandLine.sceneInputs.size() == 2);
  const auto &archive =
      std::get<tsd::app::SceneArchiveLoad>(context.commandLine.sceneInputs[0]);
  REQUIRE(archive.filename == "scene.tsd");
  const auto &foreignImport = std::get<tsd::app::ForeignSceneImport>(
      context.commandLine.sceneInputs[1]);
  REQUIRE(foreignImport.file.first == tsd::io::ImporterType::OBJ);
  REQUIRE(foreignImport.file.second == "mesh.obj;default");
}

SCENARIO("The TSD CLI rejects multiple replacement Scene Archives", "[App]")
{
  tsd::app::Context context;
  std::vector<std::string> args{
      "tsdViewer", "-tsd", "first.tsd", "-tsd", "second.tsd"};

  REQUIRE_THROWS_WITH(context.parseCommandLine(args),
      "Only one Scene Archive may be specified");
}

SCENARIO("The TSD CLI rejects conflicting native state inputs", "[App]")
{
  tsd::app::Context stateFirstContext;
  std::vector<std::string> stateFirstArgs{
      "tsdViewer", "viewer_state.tsd", "-tsd", "scene.tsd"};

  REQUIRE_THROWS_WITH(stateFirstContext.parseCommandLine(stateFirstArgs),
      "A Scene Archive cannot be combined with an application state file");

  tsd::app::Context archiveFirstContext;
  std::vector<std::string> archiveFirstArgs{
      "tsdViewer", "-tsd", "scene.tsd", "viewer_state.tsd"};

  REQUIRE_THROWS_WITH(archiveFirstContext.parseCommandLine(archiveFirstArgs),
      "A Scene Archive cannot be combined with an application state file");
}

SCENARIO(
    "The TSD CLI preserves animation grouping and layer selection", "[App]")
{
  tsd::app::Context context;
  std::vector<std::string> args{"tsdViewer",
      "--layer",
      "animated",
      "-pointsbin",
      "frame0.bin",
      "frame1.bin",
      "--layer",
      "mesh",
      "-obj",
      "mesh.obj"};

  context.parseCommandLine(args);

  REQUIRE(context.commandLine.animationFilenames.size() == 1);
  REQUIRE(context.commandLine.animationFilenames.front().first
      == tsd::io::ImporterType::POINTSBIN_MULTIFILE);
  REQUIRE(context.commandLine.animationFilenames.front().second
      == std::vector<std::string>{"frame0.bin", "frame1.bin"});
  REQUIRE(context.commandLine.animationLayerNames.size() == 1);
  REQUIRE(context.commandLine.animationLayerNames.front().str() == "animated");
  REQUIRE(context.commandLine.sceneInputs.size() == 1);
  const auto &foreignImport = std::get<tsd::app::ForeignSceneImport>(
      context.commandLine.sceneInputs.front());
  REQUIRE(foreignImport.file.first == tsd::io::ImporterType::OBJ);
  REQUIRE(foreignImport.file.second == "mesh.obj;mesh");
}

SCENARIO("Scene Archive CLI inputs are loaded by TSD App", "[App]")
{
  const auto archive =
      std::filesystem::temp_directory_path() / "tsd_app_scene_archive.tsd";
  std::filesystem::remove(archive);

  tsd::scene::Scene source;
  source.addLayer("archived");
  REQUIRE(tsd::io::save_SceneArchive(source, archive.string().c_str()));

  tsd::app::Context context;
  std::vector<std::string> args{"tsdViewer", "-tsd", archive.string()};
  context.parseCommandLine(args);
  context.setupSceneFromCommandLine();

  REQUIRE(context.tsd.scene.layer("archived") != nullptr);
  std::filesystem::remove(archive);
}

SCENARIO("Scene Archive CLI load failures are reported by TSD App", "[App]")
{
  const auto archive = std::filesystem::temp_directory_path()
      / "tsd_app_missing_scene_archive.tsd";
  std::filesystem::remove(archive);

  tsd::app::Context context;
  std::vector<std::string> args{"tsdViewer", "-tsd", archive.string()};
  context.parseCommandLine(args);

  REQUIRE_FALSE(context.loadCommandLineSceneInputs());
}

SCENARIO("Scene Archive replacement precedes additive CLI imports", "[App]")
{
  const auto temp = std::filesystem::temp_directory_path();
  const auto archive = temp / "tsd_app_mixed_scene_inputs.tsd";
  const auto obj = temp / "tsd_app_mixed_scene_inputs.obj";
  std::filesystem::remove(archive);
  std::filesystem::remove(obj);

  tsd::scene::Scene source;
  source.addLayer("archived");
  REQUIRE(tsd::io::save_SceneArchive(source, archive.string().c_str()));
  {
    std::ofstream file(obj);
    file << "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
  }

  tsd::app::Context context;
  context.tsd.scene.addLayer("stale");
  std::vector<std::string> args{
      "tsdViewer", "-obj", obj.string(), "-tsd", archive.string()};
  context.parseCommandLine(args);
  REQUIRE(context.loadCommandLineSceneInputs());

  REQUIRE(context.tsd.scene.layer("stale") == nullptr);
  REQUIRE(context.tsd.scene.layer("archived") != nullptr);
  REQUIRE(context.tsd.scene.objectDB().geometry.size() == 1);

  std::filesystem::remove(archive);
  std::filesystem::remove(obj);
}
