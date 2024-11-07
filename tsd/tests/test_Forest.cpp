// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/containers/Forest.hpp"

SCENARIO("tsd::utilty::Forest<> interface", "[Forest]")
{
  GIVEN("A normally constructed Forest")
  {
    tsd::utility::Forest<int> f{-1};

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
        f.insert_first_child(f.root(), 5);
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
          REQUIRE(f.root()->next()->next()->next());
          REQUIRE(f.root() == f.root()->next()->next()->next());
        }

        THEN("Erasing all children from the root empties the forest")
        {
          f.erase_subtree(f.root());
          REQUIRE(f.empty());
          REQUIRE(f.root()->isLeaf());
        }
      }

      WHEN("Another child is appended")
      {
        f.insert_last_child(f.root(), 5);
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
          REQUIRE(f.root()->next()->next()->next());
          REQUIRE(f.root() == f.root()->next()->next()->next());
        }
      }
    }
  }
}
