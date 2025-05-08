// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd
#include <tsd/containers/DataTree.hpp>
// std
#include <iostream>

int main()
{
  {
    tsd::serialization::DataTree tree;

    auto &root = tree.root();
    root["test"] = 42;
    root["test2"]["child3"]["deep1"] = 1;
    root["test2"]["child3"]["deep2"] = 2;
    root["test2"]["child1"] = std::string("yaas").c_str();
    root["test2"]["child2"] = 3.14f;

    float arr[3] = {1.f, 2.f, 3.f};
    root["testArray"].setValueAsArray(arr, 3);

    printf("saving tree to 'test_tree.tsdx'\n");
    tree.save("test_tree.tsdx");
  }

  {
    tsd::serialization::DataTree tree;
    printf("loading fresh tree from 'test_tree.tsdx'\n");
    tree.load("test_tree.tsdx");

    auto &root = tree.root();
    printf("test: %i\n", root["test"].getValueAs<int>());

    printf("test2/child3/deep1: %i\n",
        root["test2"]["child3"]["deep1"].getValueAs<int>());
    printf("test2/child3/deep2: %i\n",
        root["test2"]["child3"]["deep2"].getValueAs<int>());

    printf("test2/child1: %s\n",
        root["test2"]["child1"].getValueAs<std::string>().c_str());
    printf("test2/child1: %f\n", root["test2"]["child2"].getValueAs<float>());

    size_t arrSize = 0;
    float *arrPtr = nullptr;
    root["testArray"].getValueAsArray(&arrPtr, &arrSize);
    printf("testArray: %f, %f, %f\n", arrPtr[0], arrPtr[1], arrPtr[2]);
  }

  return 0;
}
