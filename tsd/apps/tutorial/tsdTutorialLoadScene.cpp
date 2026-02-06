// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd_core
#include <tsd/core/scene/Scene.hpp>
// tsd_io
#include <tsd/io/serialization.hpp>
// std
#include <cstdio>

int main(int argc, const char *argv[])
{
  if (argc < 2) {
    printf("usage: tsdLoadScene [file.tsd]\n");
    return 0;
  }

  tsd::core::Scene scene;
  tsd::io::load_Scene(scene, argv[1]);
  printf("-------------------TSD Scene Info---------------------\n\n");
  printf("%s\n", tsd::core::objectDBInfo(scene.objectDB()).c_str());

  printf("----------------------TSD Tree--------------------------\n\n");

  auto onNodeEntry = [&](auto &node, int level) {
    tsd::core::Object *obj = node->getObject();

    const char *typeText = "[-]";
    switch (node->type()) {
    case ANARI_FLOAT32_MAT4:
      typeText = "[T]";
      break;
    case ANARI_SURFACE:
      typeText = "[S]";
      break;
    case ANARI_VOLUME:
      typeText = "[V]";
      break;
    case ANARI_LIGHT:
      typeText = "[L]";
      break;
    default:
      break;
    }

    const char *nameText = "<unhandled UI node type>";
    if (!node->name().empty())
      nameText = node->name().c_str();
    else {
      switch (node->type()) {
      case ANARI_FLOAT32_MAT4:
        nameText = "xfm";
        break;
      case ANARI_SURFACE:
        nameText = obj ? obj->name().c_str() : "UNABLE TO FIND SURFACE";
        break;
      case ANARI_VOLUME:
        nameText = obj ? obj->name().c_str() : "UNABLE TO FIND VOLUME";
        break;
      case ANARI_LIGHT:
        nameText = obj ? obj->name().c_str() : "UNABLE TO FIND LIGHT";
        break;
      default:
        nameText = anari::toString(node->type());
        break;
      }
    }

    for (int i = 0; i < level; i++)
      printf("--");
    printf("%s | '%s'\n", typeText, nameText);

    return true;
  };

  scene.defaultLayer()->traverse(scene.defaultLayer()->root(), onNodeEntry);

  return 0;
}
