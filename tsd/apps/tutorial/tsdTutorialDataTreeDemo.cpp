// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd
#include <tsd/core/DataTree.hpp>
// std
#include <iostream>

int main()
{
  {
    tsd::core::DataTree tree;

    auto &root = tree.root();
    root["test"] = 500;
    root["test2"]["child1"] = std::string("yay").c_str();
    root["test2"]["child2"] = 3.14f;
    root["test2"]["child3"]["deep1"] = 1;
    root["test2"]["child3"]["deep2"] = 2;
    for (int i = 0; i < 3; i++)
      root["test2"]["child3"].append() = std::to_string(i);
    root["test2"]["child3"]["someInt"] = 55;
    root["test2"]["child4"] = std::string();

    auto &anon = root["anonymous"];
    for (int i = 0; i < 5; i++) {
      auto &a = anon.append();
      for (int i = 0; i < 3; i++)
        a.append() = std::to_string(i);
    }

    float arr[3] = {1.f, 2.f, 3.f};
    root["testArray"].setValueAsArray(arr, 3);

    root["testObject"].setValueObject(ANARI_OBJECT, 42);

    printf("====================================\n");
    tree.print();
    printf("====================================\n");

    printf("saving tree to 'test_tree.tsdx'\n");
    tree.save("test_tree.tsdx");
  }

  {
    tsd::core::DataTree tree;
    printf("loading fresh tree from 'test_tree.tsdx'\n");
    tree.load("test_tree.tsdx");

    printf("====================================\n");
    tree.print();
    printf("====================================\n");

    auto &root = tree.root();
    printf("test: %i\n", root["test"].getValueAs<int>());

    printf("test2/child3/deep1: %i\n",
        root["test2"]["child3"]["deep1"].getValueAs<int>());
    printf("test2/child3/deep2: %i\n",
        root["test2"]["child3"]["deep2"].getValueAs<int>());

    printf("test2/child1: '%s'\n",
        root["test2"]["child1"].getValueAs<std::string>().c_str());
    printf("test2/child2: %f\n", root["test2"]["child2"].getValueAs<float>());
    printf("test2/child4: '%s'\n",
        root["test2"]["child4"].getValueAs<std::string>().c_str());

    size_t arrSize = 0;
    float *arrPtr = nullptr;
    root["testArray"].getValueAsArray(&arrPtr, &arrSize);
    printf("testArray: %f, %f, %f\n", arrPtr[0], arrPtr[1], arrPtr[2]);

    printf("testObject: %s | %zu\n",
        anari::toString(root["testObject"].getValue().type()),
        root["testObject"].getValue().getAsObjectIndex());
  }

  return 0;
}
