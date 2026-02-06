// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/core/Forest.hpp"

SCENARIO("tsd::core::Forest<> interface", "[Forest]")
{
  GIVEN("A normally constructed Forest")
  {
    tsd::core::Forest<int> f{-1};

    THEN("The Forest is empty")
    {
      REQUIRE(f.empty());
    }

    THEN("An out-of-bounds node is invalid")
    {
      REQUIRE(!f.at(1));
    }

    THEN("The first node is the root")
    {
      REQUIRE(f.at(0)->isRoot());
      REQUIRE(*f.at(0) == f.root());
      REQUIRE(!f.root()->parent());
    }

    THEN("The first node is the root value")
    {
      REQUIRE(f[0] == -1);
      REQUIRE(f.root()->value() == -1);
    }

    THEN("The root node is a leaf node")
    {
      REQUIRE(f.root()->isLeaf());
    }

    WHEN("A child is pushed on the front of the root")
    {
      auto c = f.insert_first_child(f.root(), 3);

      THEN("The forest size is now 2")
      {
        REQUIRE(f.size() == 2);
      }

      THEN("The root node value is unchanged")
      {
        REQUIRE(f.root()->value() == -1);
      }

      THEN("The next node after root points back to the parent")
      {
        REQUIRE(c->parent() == f.root());
        REQUIRE(f.root()->next()->parent() == f.root());
      }

      THEN("The next node after root holds the new value")
      {
        REQUIRE(f.root()->next());
        REQUIRE(f.root()->next()->value() == 3);
      }

      THEN("The root node is no longer a leaf but is still the root")
      {
        REQUIRE(!f.root()->isLeaf());
        REQUIRE(f.root()->isRoot());
      }

      THEN("Erasing the new node returns the forest to being empty")
      {
        f.erase(c);
        REQUIRE(f.empty());
        REQUIRE(f.root()->isLeaf());
      }

      WHEN("Another child is prepended")
      {
        auto c2 = f.insert_first_child(f.root(), 5);
        THEN("The next node after root holds the new value")
        {
          REQUIRE(f.root()->next());
          REQUIRE(f.root()->next()->value() == 5);
        }

        THEN("The next node after the first child holds the new value")
        {
          REQUIRE(f.root()->next()->next());
          REQUIRE(f.root()->next()->next()->value() == 3);
        }

        THEN("The second child points back to the root")
        {
          REQUIRE(c2->parent() == f.root());
          REQUIRE(f.root()->next()->next()->next());
          REQUIRE(f.root() == f.root()->next()->next()->next());
        }

        THEN("Erasing all children from the root empties the forest")
        {
          f.root()->erase_subtree();
          REQUIRE(f.empty());
          REQUIRE(f.root()->isLeaf());
        }
      }

      WHEN("Another child is appended")
      {
        auto c3 = f.insert_last_child(f.root(), 5);
        THEN("The next node after root holds the new value")
        {
          REQUIRE(f.root()->next());
          REQUIRE(f.root()->next()->value() == 3);
        }

        THEN("The next node after the first child holds the new value")
        {
          REQUIRE(f.root()->next()->next());
          REQUIRE(f.root()->next()->next()->value() == 5);
        }

        THEN("The second child points back to the root")
        {
          REQUIRE(c3->parent() == f.root());
          REQUIRE(f.root()->next()->next()->next());
          REQUIRE(f.root() == f.root()->next()->next()->next());
        }
      }
    }
  }
}

SCENARIO("tsd::core::Forest<> reparent operations", "[Forest]")
{
  GIVEN("A Forest with a hierarchical structure")
  {
    tsd::core::Forest<int> f{0}; // root = 0
    auto child1 = f.insert_last_child(f.root(), 1);
    auto child2 = f.insert_last_child(f.root(), 2);
    auto grandchild1 = f.insert_last_child(child1, 11);
    auto grandchild2 = f.insert_last_child(child1, 12);

    THEN("Initial structure is correct")
    {
      REQUIRE(f.size() == 5);
      REQUIRE(child1->parent() == f.root());
      REQUIRE(child2->parent() == f.root());
      REQUIRE(grandchild1->parent() == child1);
      REQUIRE(grandchild2->parent() == child1);
    }

    WHEN("A leaf node is reparented")
    {
      f.move_subtree(grandchild1, child2);

      THEN("The leaf node has a new parent")
      {
        REQUIRE(grandchild1->parent() == child2);
      }

      THEN("The old parent no longer has this child")
      {
        REQUIRE(child1->next() == grandchild2);
      }

      THEN("The new parent has the child")
      {
        REQUIRE(child2->next() == grandchild1);
      }

      THEN("Forest size is unchanged")
      {
        REQUIRE(f.size() == 5);
      }
    }

    WHEN("A node with children is reparented")
    {
      f.move_subtree(child1, child2);

      THEN("The subtree moves together")
      {
        REQUIRE(child1->parent() == child2);
        REQUIRE(grandchild1->parent() == child1);
        REQUIRE(grandchild2->parent() == child1);
      }

      THEN("The new parent has the child")
      {
        REQUIRE(child2->next() == child1);
      }

      THEN("Forest size is unchanged")
      {
        REQUIRE(f.size() == 5);
      }
    }

    WHEN("Attempting to reparent to a descendant")
    {
      f.move_subtree(child1, grandchild1);

      THEN("The operation is prevented (circular dependency)")
      {
        REQUIRE(child1->parent() == f.root());
        REQUIRE(grandchild1->parent() == child1);
      }
    }

    WHEN("Attempting to reparent to the same parent")
    {
      f.move_subtree(child1, f.root());

      THEN("The operation is a no-op")
      {
        REQUIRE(child1->parent() == f.root());
        REQUIRE(f.size() == 5);
      }
    }

    WHEN("Attempting to reparent to itself")
    {
      f.move_subtree(child1, child1);

      THEN("The operation is prevented")
      {
        REQUIRE(child1->parent() == f.root());
        REQUIRE(f.size() == 5);
      }
    }

    WHEN("Attempting to reparent the root")
    {
      f.move_subtree(f.root(), child1);

      THEN("The operation is prevented")
      {
        REQUIRE(f.root()->isRoot());
        REQUIRE(!f.root()->parent());
      }
    }

    WHEN("Multiple nodes are reparented in sequence")
    {
      f.move_subtree(grandchild1, child2);
      f.move_subtree(grandchild2, child2);

      THEN("Both nodes have the new parent")
      {
        REQUIRE(grandchild1->parent() == child2);
        REQUIRE(grandchild2->parent() == child2);
      }

      THEN("The old parent has no children")
      {
        REQUIRE(child1->isLeaf());
      }

      THEN("Forest size is unchanged")
      {
        REQUIRE(f.size() == 5);
      }
    }
  }

  GIVEN("Edge case: null node refs")
  {
    tsd::core::Forest<int> f{0};
    auto child = f.insert_last_child(f.root(), 1);

    WHEN("Reparenting with null child")
    {
      tsd::core::ForestNodeRef<int> nullNode;
      f.move_subtree(nullNode, child);

      THEN("The operation is safely ignored")
      {
        REQUIRE(f.size() == 2);
      }
    }

    WHEN("Reparenting to null parent")
    {
      tsd::core::ForestNodeRef<int> nullNode;
      f.move_subtree(child, nullNode);

      THEN("The operation is safely ignored")
      {
        REQUIRE(child->parent() == f.root());
      }
    }
  }
}
