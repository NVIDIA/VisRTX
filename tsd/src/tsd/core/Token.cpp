// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/Token.hpp"
// std
#include <memory>
#include <unordered_set>

namespace tsd::core {

static std::unique_ptr<std::unordered_set<std::string>> g_tokenRegistry;

Token::Token(const char *s) : Token(std::string(s)) {}

Token::Token(const std::string &s)
{
  if (s.empty())
    return;
  else if (!g_tokenRegistry)
    g_tokenRegistry = std::make_unique<std::unordered_set<std::string>>();
  auto result = g_tokenRegistry->insert(s);
  m_value = result.first->c_str();
}

const char *Token::c_str() const
{
  return value();
}

const char *Token::value() const
{
  return m_value;
}

std::string Token::str() const
{
  return empty() ? std::string() : std::string(c_str());
}

bool Token::empty() const
{
  return value() == nullptr;
}

Token::operator bool() const
{
  return !empty();
}

bool operator==(const Token &t1, const Token &t2)
{
  return t1.value() == t2.value();
}

bool operator!=(const Token &t1, const Token &t2)
{
  return !(t1.value() == t2.value());
}

} // namespace tsd::core
