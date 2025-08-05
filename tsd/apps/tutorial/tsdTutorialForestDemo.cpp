// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd_core
#include <tsd/core/Forest.hpp>
// std
#include <iostream>

template <typename T>
void print(tsd::core::Forest<T> &f)
{
  std::cout << "----Forest----\n";
  f.traverse(f.root(), [](auto &node, int level) {
    for (int i = 0; i < level; i++)
      std::cout << "  ";
    std::cout << *node << " : " << level << std::endl;
    return true;
  });
  std::cout << "--------------\n";
}

int main()
{
  tsd::core::Forest<int> f(0);
  f.insert_last_child(f.root(), 1);

  auto l1r = f.insert_last_child(f.root(), 2);
  f.insert_last_child(l1r, 3);
  f.insert_last_child(l1r, 4);

  auto l2r = f.insert_last_child(l1r, 5);
  f.insert_last_child(l2r, 6);
  f.insert_last_child(l2r, 7);

  f.insert_last_child(f.root(), 8);

  print(f);

  auto s = tsd::core::find_first_child(l1r, [](auto &v) { return v == 4; });
  std::cout << "looking for '4' under '2', found " << **s << std::endl;

  std::cout << "erasing '5'" << std::endl;
  f.erase(l2r);

  print(f);

  int sum = 0;
  tsd::core::foreach_child(f.root(), [&](int &v) { sum += v; });
  std::cout << "sum of root node's children: " << sum << std::endl;

  sum = f.root()->value();
  tsd::core::forall_children(f.root(), [&](int &v) { sum += v; });
  std::cout << "sum of entire tree: " << sum << std::endl;

  return 0;
}
