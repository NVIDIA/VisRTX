// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd
#include <tsd/containers/Forest.hpp>
// std
#include <iostream>

template <typename T>
void print(tsd::utility::Forest<T> &f)
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
  tsd::utility::Forest<int> f(0);
  f.insert_last_child(f.root(), 1);

  auto l1r = f.insert_last_child(f.root(), 2);
  f.insert_last_child(l1r, 3);
  f.insert_last_child(l1r, 4);

  auto l2r = f.insert_last_child(l1r, 5);
  f.insert_last_child(l2r, 6);
  f.insert_last_child(l2r, 7);

  f.insert_last_child(f.root(), 8);

  print(f);

  std::cout << "erasing '5'" << std::endl;
  f.erase(l2r);

  print(f);

  int sum = 0;
  tsd::utility::foreach_child(f.root(), [&](int &v) { sum += v; });
  std::cout << "sum of root node's children: " << sum << std::endl;

  sum = f.root()->value();
  tsd::utility::forall_children(f.root(), [&](int &v) { sum += v; });
  std::cout << "sum of entire tree: " << sum << std::endl;

  return 0;
}
