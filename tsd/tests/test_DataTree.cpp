// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#define TSD_DATA_TREE_TEST_MODE
#include "tsd/core/DataTree.hpp"
// std
#include <algorithm>

SCENARIO("tsd::core::DataTree interface", "[DataTree]")
{
  GIVEN("A normally constructed DataTree")
  {
    tsd::core::DataTree tree;
    auto &root = tree.root();

    THEN("The root node is called 'root'")
    {
      REQUIRE(root.name() == "<root>");
    }

    THEN("The root node has no children")
    {
      REQUIRE(root.numChildren() == 0);
    }

    THEN("The root self() ref must be correct")
    {
      REQUIRE(root.self());
      REQUIRE(root.self().index() == 0);
    }

    WHEN("A child node of the root is accessed with child(i)")
    {
      auto *child = root.child(0);
      THEN("The returned pointer must be null")
      {
        REQUIRE(child == nullptr);
      }
      THEN("The root node still has no children")
      {
        REQUIRE(root.numChildren() == 0);
      }
    }

    WHEN("A child node of the root is accessed with child(name)")
    {
      auto *child = root.child("childNode");
      THEN("The returned pointer must be null")
      {
        REQUIRE(child == nullptr);
      }
      THEN("The root node still has no children")
      {
        REQUIRE(root.numChildren() == 0);
      }
    }

    WHEN("A child node of the root is accessed with operator[](name)")
    {
      auto &child = root["childNode"];
      THEN("The returned child must have the correct name")
      {
        REQUIRE(child.name() == "childNode");
      }
      THEN("The root node now has 1 child")
      {
        REQUIRE(root.numChildren() == 1);
      }

      WHEN("A value is set on the node")
      {
        child = 50;
        THEN("The value returned is the correct type")
        {
          REQUIRE(child.getValue().is<int>());
          REQUIRE(child.getValue().type() == ANARI_INT32);
        }
        THEN("The value returned is the correct value")
        {
          REQUIRE(child.getValueAs<int>() == 50);
        }
      }
    }

    WHEN("A blank child node is appended on the root")
    {
      auto &c1 = root.append();
      THEN("The root node now has 1 child")
      {
        REQUIRE(root.numChildren() == 1);
      }
#if 0 // no longer true until DataNode::append() uses numChildren()...
      THEN("The child name will have the name '<0>'")
      {
        REQUIRE(c1.name() == "<0>");
      }
#endif
      THEN("A loop of appending blank nodes creates the correct # of children")
      {
        for (int i = 0; i < 10; i++) {
          c1.append();
          REQUIRE(c1.numChildren() == (i + 1));
        }
      }

      WHEN("Appending a second blank node")
      {
        auto &c2 = root.append();
        THEN("The root node now has 2 children")
        {
          REQUIRE(root.numChildren() == 2);
        }
#if 0 // no longer true until DataNode::append() uses numChildren()...
        THEN("The second blank node will have the name '<1>'")
        {
          REQUIRE(c2.name() == "<1>");
        }
#endif
        THEN(
            "A loop of appending blank nodes creates the correct # of children")
        {
          for (int i = 0; i < 10; i++) {
            c2.append();
            REQUIRE(c2.numChildren() == (i + 1));
          }
        }
      }
    }

    THEN("A loop of appending blank nodes creates the correct # of children")
    {
      for (int i = 0; i < 10; i++) {
        root.append();
        REQUIRE(root.numChildren() == (i + 1));
      }
    }

    WHEN("Accessing multiple layers of nodes all in one go")
    {
      auto &child3a = root["child1a"]["child2a"]["child3a"];
      auto &child3b = root["child1b"]["child2b"]["child3b"];
      THEN("The deeper nodes do not query as null with node.child(i)")
      {
        REQUIRE(root["child1a"].child("child2a") != nullptr);
        REQUIRE(root["child1a"]["child2a"].child("child3a") != nullptr);
        REQUIRE(root["child1b"].child("child2b") != nullptr);
        REQUIRE(root["child1b"]["child2b"].child("child3b") != nullptr);
      }
      THEN("The names of the children are correct")
      {
        REQUIRE(root["child1a"].name() == "child1a");
        REQUIRE(root["child1b"].name() == "child1b");
        REQUIRE(root["child1a"]["child2a"].name() == "child2a");
        REQUIRE(root["child1b"]["child2b"].name() == "child2b");
        REQUIRE(root["child1a"]["child2a"]["child3a"].name() == "child3a");
        REQUIRE(root["child1b"]["child2b"]["child3b"].name() == "child3b");
      }
      THEN("The identity of the deepest nodes are the same")
      {
        REQUIRE(&root["child1a"]["child2a"]["child3a"] == &child3a);
        REQUIRE(&root["child1b"]["child2b"]["child3b"] == &child3b);
      }
      THEN("The root node now has 2 children")
      {
        REQUIRE(root.numChildren() == 2);
      }

      WHEN("Removing a child by name")
      {
        root.remove("child1a");
        THEN("Only the removed child should not exist anymore")
        {
          REQUIRE(root.child("child1a") == nullptr);
          REQUIRE(root.child("child1b") != nullptr);
        }
      }

      WHEN("Removing a child by reference")
      {
        root.remove(root["child1a"]);
        THEN("Only the removed child should not exist anymore")
        {
          REQUIRE(root.child("child1a") == nullptr);
          REQUIRE(root.child("child1b") != nullptr);
        }
      }

      WHEN("Setting a value on an intermediate node")
      {
        root["child1a"]["child2a"] = 100;
        THEN("The original leaf child should no longer exist")
        {
          REQUIRE(root["child1a"]["child2a"].child("child3a") == nullptr);
          REQUIRE(root["child1a"]["child2a"].getValueAs<int>() == 100);
        }
      }
    }

    WHEN("Setting an array as a value")
    {
      int values[5] = {1, 2, 3, 4, 5};

      auto &child = root["arrayChild"];
      child.setValueAsArray(values, 5);

      THEN("The node claims to hold an array")
      {
        REQUIRE(child.holdsArray());
      }

      THEN("The node array storage holds elements of the correct type")
      {
        REQUIRE(child.arrayType() == ANARI_INT32);
      }

      THEN("The node array storage holds the correct size + values")
      {
        int *checkedValues = nullptr;
        size_t size = 0;
        child.getValueAsArray(&checkedValues, &size);
        REQUIRE(size == 5);
        REQUIRE(checkedValues != nullptr);
        REQUIRE(std::equal(values, values + size, checkedValues));
      }
    }
  }
}
