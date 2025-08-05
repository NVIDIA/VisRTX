// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// std
#include <string>
#include <string_view>

namespace tsd::core {

struct Token
{
  Token() = default;
  Token(const char *s);
  Token(const std::string &s);

  const char *c_str() const;
  const char *value() const;

  bool empty() const;
  operator bool() const;

  Token(const Token &) = default;
  Token &operator=(const Token &) = default;
  Token(Token &&) = default;
  Token &operator=(Token &&) = default;

 private:
  const char *m_value{nullptr};
};

bool operator==(const Token &t1, const Token &t2);
bool operator!=(const Token &t1, const Token &t2);

namespace literals {

Token operator""_t(const char *, size_t);

} // namespace literals
} // namespace tsd::core
