// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared arithmetic-operator registration for the vector usertypes
// (float2/3/4). Each per-type TU includes this and applies it to its
// own sol::usertype<T>.

#include <sol/sol.hpp>

#include <stdexcept>

namespace tsd::scripting {

template <typename T>
void registerVecArithmetic(sol::usertype<T> &ut)
{
  ut[sol::meta_function::addition] = [](const T &a, const T &b) {
    return a + b;
  };
  ut[sol::meta_function::subtraction] = [](const T &a, const T &b) {
    return a - b;
  };
  ut[sol::meta_function::multiplication] =
      [](sol::object lhs, sol::object rhs) -> T {
    if (lhs.is<T>() && rhs.is<T>())
      return lhs.as<T>() * rhs.as<T>();
    if (lhs.is<T>() && rhs.is<double>())
      return lhs.as<T>() * static_cast<float>(rhs.as<double>());
    if (lhs.is<double>() && rhs.is<T>())
      return static_cast<float>(lhs.as<double>()) * rhs.as<T>();
    throw std::runtime_error("invalid operand types for *");
  };
  ut[sol::meta_function::division] =
      [](sol::object lhs, sol::object rhs) -> T {
    if (lhs.is<T>() && rhs.is<T>())
      return lhs.as<T>() / rhs.as<T>();
    if (lhs.is<T>() && rhs.is<double>())
      return lhs.as<T>() / static_cast<float>(rhs.as<double>());
    throw std::runtime_error("invalid operand types for /");
  };
  ut[sol::meta_function::unary_minus] = [](const T &a) { return -a; };
}

} // namespace tsd::scripting
