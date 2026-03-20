// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <set>
#include <string>
#include <vector>
#include "PbrtLexer.hpp"
#include "PbrtScene.hpp"

namespace pbrt {

// Join `relPath` against `basePath` and lexically normalize. Rejects absolute
// paths (a malicious scene cannot point at `/etc/passwd`); `..` traversal is
// allowed because real PBRT scene archives use sibling-directory references.
std::string resolveScenePath(
    const std::string &basePath, const std::string &relPath);

class Parser
{
 public:
  static Scene parseFile(const std::string &filename);
  static Scene parseString(
      const std::string &source, const std::string &basePath = "");

 private:
  Parser(const std::string &basePath);

  void parse(Lexer &lex);
  void parseDirective(Lexer &lex, const std::string &directive);

  // Directive handlers
  void parseFilm(Lexer &lex);
  void parseCamera(Lexer &lex);
  void parseShape(Lexer &lex);
  void parseMaterial(Lexer &lex);
  void parseMakeNamedMaterial(Lexer &lex);
  void parseNamedMaterial(Lexer &lex);
  void parseTexture(Lexer &lex);
  void parseLightSource(Lexer &lex);
  void parseAreaLightSource(Lexer &lex);
  void parseMakeNamedMedium(Lexer &lex);
  void parseMediumInterface(Lexer &lex);
  void parseObjectBegin(Lexer &lex);
  void parseObjectEnd(Lexer &lex);
  void parseObjectInstance(Lexer &lex);
  void parseInclude(Lexer &lex);
  void parseTransformDirective(Lexer &lex, const std::string &directive);

  // Parameter parsing
  ParamList parseParamList(Lexer &lex);
  std::string parseQuotedString(Lexer &lex);

  // Attribute stack
  struct GraphicsState
  {
    Transform ctm;
    std::string namedMaterialName;
    std::string areaLightType;
    ParamList areaLightParams;
    std::string interiorMedium;
    std::string exteriorMedium;
    bool reverseOrientation = false;
  };

  GraphicsState &currentState();
  void pushState();
  void popState();

  Scene m_scene;
  std::vector<GraphicsState> m_stateStack;
  std::string m_basePath;
  std::set<std::string> m_includeStack;
  int m_includeDepth = 0;
  bool m_inWorld = false;
  ObjectDef *m_currentObject = nullptr;
  int m_anonMaterialCounter = 0;
};

} // namespace pbrt
