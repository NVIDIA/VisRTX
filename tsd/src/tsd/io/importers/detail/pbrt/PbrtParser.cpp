// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "PbrtParser.hpp"
#include "tsd/core/Logging.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace pbrt {

namespace fs = std::filesystem;

// PBRT scenes routinely chain ~10 levels deep through Include; pick a bound
// well past that to catch pathological cycles without rejecting real scenes.
static constexpr int kMaxIncludeDepth = 32;

// Path policy /////////////////////////////////////////////////////////////////
//
// Real PBRT v4 scene archives organize assets into sibling directories
// (e.g. `<scene>/landscape/geometry/*.ply` referenced from
// `<scene>/main/scene.pbrt` as `../landscape/...`), so a strict scene-root
// containment check rejects legitimate scenes. We only block absolute paths
// here; the recursion-depth and include-cycle guards in `parseInclude` cover
// the DoS surface, and the importer treats `.pbrt` files as trusted input.

std::string resolveScenePath(
    const std::string &basePath, const std::string &relPath)
{
  if (relPath.empty())
    throw std::runtime_error("pbrt: empty relative path");
  fs::path rel(relPath);
  if (rel.is_absolute()) {
    throw std::runtime_error(
        "pbrt: absolute path not allowed in scene reference: " + relPath);
  }
  fs::path base = basePath.empty() ? fs::current_path() : fs::path(basePath);
  if (base.is_relative())
    base = fs::current_path() / base;
  return (base / rel).lexically_normal().string();
}

// Token validation helpers ////////////////////////////////////////////////////

static std::runtime_error tokenError(
    Lexer &lex, const Token &tok, const std::string &what)
{
  return std::runtime_error(lex.currentFile() + ":" + std::to_string(tok.line)
      + ": " + what + ", got '" + tok.text + "'");
}

static float expectFloat(Lexer &lex, const char *context)
{
  Token tok = lex.next();
  if (tok.type != TokenType::Number) {
    throw tokenError(
        lex, tok, std::string("expected number for ") + context);
  }
  return std::stof(tok.text);
}

static int expectInt(Lexer &lex, const char *context)
{
  Token tok = lex.next();
  if (tok.type != TokenType::Number) {
    throw tokenError(
        lex, tok, std::string("expected integer for ") + context);
  }
  return std::stoi(tok.text);
}

static std::string expectStringOrNumber(Lexer &lex, const char *context)
{
  Token tok = lex.next();
  if (tok.type != TokenType::String && tok.type != TokenType::Number
      && tok.type != TokenType::Identifier) {
    throw tokenError(
        lex, tok, std::string("expected value for ") + context);
  }
  return tok.text;
}

static void expectRBracket(Lexer &lex, const char *context)
{
  Token tok = lex.next();
  if (tok.type != TokenType::RBracket) {
    throw tokenError(
        lex, tok, std::string("expected ']' to close ") + context);
  }
}

// Loop guard for bracketed value lists: returns true while there's another
// value to consume; throws on Eof so a missing `]` produces a clear error
// instead of an infinite loop.
static bool moreInBracket(Lexer &lex, const char *context)
{
  TokenType t = lex.peek().type;
  if (t == TokenType::Eof) {
    throw std::runtime_error(lex.currentFile() + ":"
        + std::to_string(lex.currentLine()) + ": unterminated " + context
        + " (missing ']')");
  }
  return t != TokenType::RBracket;
}

// Parser public ///////////////////////////////////////////////////////////////

Scene Parser::parseFile(const std::string &filename)
{
  std::ifstream file(filename);
  if (!file.is_open())
    throw std::runtime_error("pbrt::Parser: cannot open file: " + filename);

  std::stringstream ss;
  ss << file.rdbuf();
  std::string contents = ss.str();

  // Extract base path from filename
  std::string basePath;
  auto lastSlash = filename.find_last_of("/\\");
  if (lastSlash != std::string::npos)
    basePath = filename.substr(0, lastSlash + 1);

  Parser parser(basePath);
  Lexer lex(contents, filename);
  parser.parse(lex);
  return std::move(parser.m_scene);
}

Scene Parser::parseString(
    const std::string &source, const std::string &basePath)
{
  Parser parser(basePath);
  Lexer lex(source, "<string>");
  parser.parse(lex);
  return std::move(parser.m_scene);
}

// Parser private //////////////////////////////////////////////////////////////

Parser::Parser(const std::string &basePath) : m_basePath(basePath)
{
  m_stateStack.push_back(GraphicsState{});
}

void Parser::parse(Lexer &lex)
{
  while (true) {
    Token tok = lex.peek();
    if (tok.type == TokenType::Eof)
      break;
    if (tok.type != TokenType::Identifier) {
      throw std::runtime_error(lex.currentFile() + ":"
          + std::to_string(tok.line) + ": expected directive, got '" + tok.text
          + "'");
    }
    lex.next();
    parseDirective(lex, tok.text);
  }
}

void Parser::parseDirective(Lexer &lex, const std::string &directive)
{
  // Transform directives
  if (directive == "Identity" || directive == "Transform"
      || directive == "ConcatTransform" || directive == "Translate"
      || directive == "Scale" || directive == "Rotate" || directive == "LookAt"
      || directive == "CoordinateSystem" || directive == "CoordSysTransform") {
    parseTransformDirective(lex, directive);
    return;
  }

  if (directive == "ActiveTransform") {
    lex.next(); // consume StartTime/EndTime/All
    return;
  }
  if (directive == "TransformTimes") {
    lex.next(); // start time
    lex.next(); // end time
    return;
  }

  // Scene-wide directives
  if (directive == "Film") {
    parseFilm(lex);
    return;
  }
  if (directive == "Camera") {
    parseCamera(lex);
    return;
  }

  // Consume-and-ignore directives (subtype + params)
  if (directive == "Sampler" || directive == "Integrator"
      || directive == "PixelFilter" || directive == "ColorSpace"
      || directive == "Accelerator") {
    parseQuotedString(lex);
    parseParamList(lex);
    return;
  }

  // World block
  if (directive == "WorldBegin") {
    m_inWorld = true;
    currentState().ctm = Transform::identity();
    return;
  }
  if (directive == "WorldEnd")
    return;

  // Attribute stack
  if (directive == "AttributeBegin" || directive == "TransformBegin") {
    pushState();
    return;
  }
  if (directive == "AttributeEnd" || directive == "TransformEnd") {
    popState();
    return;
  }

  // Object directives
  if (directive == "ObjectBegin") {
    parseObjectBegin(lex);
    return;
  }
  if (directive == "ObjectEnd") {
    parseObjectEnd(lex);
    return;
  }
  if (directive == "ObjectInstance") {
    parseObjectInstance(lex);
    return;
  }

  // Content directives
  if (directive == "Shape") {
    parseShape(lex);
    return;
  }
  if (directive == "Material") {
    parseMaterial(lex);
    return;
  }
  if (directive == "MakeNamedMaterial") {
    parseMakeNamedMaterial(lex);
    return;
  }
  if (directive == "NamedMaterial") {
    parseNamedMaterial(lex);
    return;
  }
  if (directive == "Texture") {
    parseTexture(lex);
    return;
  }
  if (directive == "LightSource") {
    parseLightSource(lex);
    return;
  }
  if (directive == "AreaLightSource") {
    parseAreaLightSource(lex);
    return;
  }

  if (directive == "MediumInterface") {
    parseMediumInterface(lex);
    return;
  }
  if (directive == "MakeNamedMedium") {
    parseMakeNamedMedium(lex);
    return;
  }

  // File inclusion
  if (directive == "Include" || directive == "Import") {
    parseInclude(lex);
    return;
  }

  // ReverseOrientation
  if (directive == "ReverseOrientation") {
    currentState().reverseOrientation = !currentState().reverseOrientation;
    return;
  }

  // Option (consume string + params and ignore)
  if (directive == "Option") {
    parseQuotedString(lex);
    parseParamList(lex);
    return;
  }

  tsd::core::logWarning("[pbrt] ignoring unknown directive '%s' at %s:%zu",
      directive.c_str(),
      lex.currentFile().c_str(),
      lex.currentLine());
  if (lex.peek().type == TokenType::String) {
    parseQuotedString(lex);
    parseParamList(lex);
  }
}

// Directive handlers //////////////////////////////////////////////////////////

void Parser::parseFilm(Lexer &lex)
{
  m_scene.film.type = parseQuotedString(lex);
  auto params = parseParamList(lex);
  if (params.has("xresolution"))
    m_scene.film.xResolution = params.getInt("xresolution");
  if (params.has("yresolution"))
    m_scene.film.yResolution = params.getInt("yresolution");
  if (params.has("filename"))
    m_scene.film.filename = params.getString("filename");
  m_scene.film.params = std::move(params);
}

void Parser::parseCamera(Lexer &lex)
{
  m_scene.camera.type = parseQuotedString(lex);
  m_scene.camera.params = parseParamList(lex);
  m_scene.camera.cameraToWorld = currentState().ctm;
}

void Parser::parseShape(Lexer &lex)
{
  Shape shape;
  shape.type = parseQuotedString(lex);
  shape.params = parseParamList(lex);
  shape.objectToWorld = currentState().ctm;
  shape.materialName = currentState().namedMaterialName;
  shape.areaLightType = currentState().areaLightType;
  shape.areaLightParams = currentState().areaLightParams;
  shape.interiorMedium = currentState().interiorMedium;
  shape.exteriorMedium = currentState().exteriorMedium;
  shape.reverseOrientation = currentState().reverseOrientation;

  if (m_currentObject)
    m_currentObject->shapes.push_back(std::move(shape));
  else
    m_scene.shapes.push_back(std::move(shape));
}

void Parser::parseMaterial(Lexer &lex)
{
  std::string type = parseQuotedString(lex);
  auto params = parseParamList(lex);

  std::string name = "__anon_" + std::to_string(m_anonMaterialCounter++);
  m_scene.namedMaterials[name] = MaterialDef{type, std::move(params)};
  currentState().namedMaterialName = name;
}

void Parser::parseMakeNamedMaterial(Lexer &lex)
{
  std::string name = parseQuotedString(lex);
  auto params = parseParamList(lex);

  std::string type = params.getString("type");
  m_scene.namedMaterials[name] = MaterialDef{type, std::move(params)};
}

void Parser::parseNamedMaterial(Lexer &lex)
{
  currentState().namedMaterialName = parseQuotedString(lex);
}

void Parser::parseTexture(Lexer &lex)
{
  std::string name = parseQuotedString(lex);
  std::string colorType = parseQuotedString(lex);
  std::string implType = parseQuotedString(lex);
  auto params = parseParamList(lex);

  TextureDef tex;
  tex.name = name;
  tex.colorType = colorType;
  tex.implType = implType;
  tex.params = std::move(params);
  m_scene.textures[name] = std::move(tex);
}

void Parser::parseMakeNamedMedium(Lexer &lex)
{
  std::string name = parseQuotedString(lex);
  auto params = parseParamList(lex);
  std::string type = params.getString("type");
  m_scene.namedMedia[name] = MediumDef{std::move(type), std::move(params)};
}

void Parser::parseMediumInterface(Lexer &lex)
{
  // PBRT v4: `MediumInterface "interior" "exterior"`. Either side may be
  // empty (""), meaning vacuum / the default surrounding medium.
  currentState().interiorMedium = parseQuotedString(lex);
  currentState().exteriorMedium = parseQuotedString(lex);
}

void Parser::parseLightSource(Lexer &lex)
{
  LightDef light;
  light.type = parseQuotedString(lex);
  light.params = parseParamList(lex);
  light.lightToWorld = currentState().ctm;

  if (m_currentObject)
    m_currentObject->lights.push_back(std::move(light));
  else
    m_scene.lights.push_back(std::move(light));
}

void Parser::parseAreaLightSource(Lexer &lex)
{
  currentState().areaLightType = parseQuotedString(lex);
  currentState().areaLightParams = parseParamList(lex);
}

void Parser::parseObjectBegin(Lexer &lex)
{
  std::string name = parseQuotedString(lex);
  // pbrt-v4: ObjectBegin pushes graphics state for scoping but does NOT
  // reset the CTM. Shapes inside inherit the CTM as set by any preceding
  // Transform directives, so it becomes part of shape.objectToWorld; the
  // CTM at ObjectInstance time is then composed on top as instanceToWorld.
  pushState();

  auto &obj = m_scene.objects[name];
  obj.name = name;
  m_currentObject = &obj;
}

void Parser::parseObjectEnd(Lexer & /*lex*/)
{
  popState();
  m_currentObject = nullptr;
}

void Parser::parseObjectInstance(Lexer &lex)
{
  ObjectInstance inst;
  inst.name = parseQuotedString(lex);
  inst.instanceToWorld = currentState().ctm;
  m_scene.instances.push_back(std::move(inst));
}

void Parser::parseInclude(Lexer &lex)
{
  std::string relPath = parseQuotedString(lex);
  std::string fullPath = resolveScenePath(m_basePath, relPath);

  if (m_includeStack.count(fullPath)) {
    throw std::runtime_error(lex.currentFile() + ":"
        + std::to_string(lex.currentLine())
        + ": pbrt: include cycle detected at '" + fullPath + "'");
  }
  if (m_includeDepth >= kMaxIncludeDepth) {
    throw std::runtime_error(lex.currentFile() + ":"
        + std::to_string(lex.currentLine())
        + ": pbrt: include depth limit (" + std::to_string(kMaxIncludeDepth)
        + ") exceeded at '" + fullPath + "'");
  }

  std::ifstream file(fullPath);
  if (!file.is_open())
    throw std::runtime_error("pbrt::Parser: cannot open include: " + fullPath);

  std::stringstream ss;
  ss << file.rdbuf();
  std::string contents = ss.str();

  std::string savedBasePath = m_basePath;
  auto lastSlash = fullPath.find_last_of("/\\");
  if (lastSlash != std::string::npos)
    m_basePath = fullPath.substr(0, lastSlash + 1);

  m_includeStack.insert(fullPath);
  ++m_includeDepth;

  Lexer subLex(contents, fullPath);
  try {
    parse(subLex);
  } catch (...) {
    m_includeStack.erase(fullPath);
    --m_includeDepth;
    m_basePath = savedBasePath;
    throw;
  }

  m_includeStack.erase(fullPath);
  --m_includeDepth;
  m_basePath = savedBasePath;
}

void Parser::parseTransformDirective(Lexer &lex, const std::string &directive)
{
  auto &ctm = currentState().ctm;

  if (directive == "Identity") {
    ctm = Transform::identity();
    return;
  }

  auto expectLBracket = [&](const char *what) {
    Token tok = lex.next();
    if (tok.type != TokenType::LBracket)
      throw tokenError(lex, tok, std::string("expected '[' after ") + what);
  };

  auto read16 = [&](const char *what, Transform &out) {
    expectLBracket(what);
    for (int i = 0; i < 16; ++i)
      out.m[i] = expectFloat(lex, what);
    expectRBracket(lex, what);
  };

  if (directive == "Transform") {
    Transform t;
    read16("Transform", t);
    ctm = t;
    return;
  }

  if (directive == "ConcatTransform") {
    Transform t;
    read16("ConcatTransform", t);
    ctm = ctm * t;
    return;
  }

  if (directive == "Translate") {
    float x = expectFloat(lex, "Translate x");
    float y = expectFloat(lex, "Translate y");
    float z = expectFloat(lex, "Translate z");
    ctm = ctm * Transform::translate(x, y, z);
    return;
  }

  if (directive == "Scale") {
    float x = expectFloat(lex, "Scale x");
    float y = expectFloat(lex, "Scale y");
    float z = expectFloat(lex, "Scale z");
    ctm = ctm * Transform::scale(x, y, z);
    return;
  }

  if (directive == "Rotate") {
    float angle = expectFloat(lex, "Rotate angle");
    float ax = expectFloat(lex, "Rotate axis x");
    float ay = expectFloat(lex, "Rotate axis y");
    float az = expectFloat(lex, "Rotate axis z");
    ctm = ctm * Transform::rotate(angle, ax, ay, az);
    return;
  }

  if (directive == "LookAt") {
    float ex = expectFloat(lex, "LookAt eye x");
    float ey = expectFloat(lex, "LookAt eye y");
    float ez = expectFloat(lex, "LookAt eye z");
    float lx = expectFloat(lex, "LookAt look x");
    float ly = expectFloat(lex, "LookAt look y");
    float lz = expectFloat(lex, "LookAt look z");
    float ux = expectFloat(lex, "LookAt up x");
    float uy = expectFloat(lex, "LookAt up y");
    float uz = expectFloat(lex, "LookAt up z");
    ctm = ctm * Transform::lookAt(ex, ey, ez, lx, ly, lz, ux, uy, uz);
    return;
  }

  if (directive == "CoordinateSystem") {
    std::string name = parseQuotedString(lex);
    m_scene.coordinateSystems[name] = ctm;
    return;
  }

  if (directive == "CoordSysTransform") {
    std::string name = parseQuotedString(lex);
    auto it = m_scene.coordinateSystems.find(name);
    if (it != m_scene.coordinateSystems.end())
      ctm = it->second;
    return;
  }
}

// Parameter parsing ///////////////////////////////////////////////////////////

ParamList Parser::parseParamList(Lexer &lex)
{
  ParamList params;

  while (true) {
    Token tok = lex.peek();
    if (tok.type != TokenType::String)
      break;

    // Typed param strings have "type name" format
    auto spacePos = tok.text.find(' ');
    if (spacePos == std::string::npos)
      break;

    lex.next(); // consume the param declaration string

    std::string paramType = tok.text.substr(0, spacePos);
    std::string paramName = tok.text.substr(spacePos + 1);

    // Check for bracketed values
    bool bracketed = (lex.peek().type == TokenType::LBracket);
    if (bracketed)
      lex.next(); // consume '['

    const std::string ctx = paramType + " " + paramName;

    // Preserve the type qualifier so importers can distinguish forms that
    // share a storage representation (e.g. `"blackbody L" [5500]` is one
    // float, `"rgb L" [r g b]` is three; both end up in vector<float>).
    params.types[paramName] = paramType;

    if (paramType == "string" || paramType == "texture") {
      std::vector<std::string> vals;
      if (bracketed) {
        while (moreInBracket(lex, ctx.c_str()))
          vals.push_back(parseQuotedString(lex));
        expectRBracket(lex, ctx.c_str());
      } else {
        vals.push_back(parseQuotedString(lex));
      }
      params.values[paramName] = std::move(vals);
    } else if (paramType == "bool") {
      std::vector<bool> vals;
      if (bracketed) {
        while (moreInBracket(lex, ctx.c_str()))
          vals.push_back(expectStringOrNumber(lex, ctx.c_str()) == "true");
        expectRBracket(lex, ctx.c_str());
      } else {
        vals.push_back(expectStringOrNumber(lex, ctx.c_str()) == "true");
      }
      params.values[paramName] = std::move(vals);
    } else if (paramType == "integer") {
      std::vector<int> vals;
      if (bracketed) {
        while (moreInBracket(lex, ctx.c_str()))
          vals.push_back(expectInt(lex, ctx.c_str()));
        expectRBracket(lex, ctx.c_str());
      } else {
        vals.push_back(expectInt(lex, ctx.c_str()));
      }
      params.values[paramName] = std::move(vals);
    } else {
      // float, point2, point3, vector2, vector3, normal3, rgb, spectrum, etc.
      // spectrum/blackbody params may reference named spectra (strings)
      bool valuesAreStrings = (lex.peek().type == TokenType::String);
      if (valuesAreStrings) {
        std::vector<std::string> vals;
        if (bracketed) {
          while (moreInBracket(lex, ctx.c_str()))
            vals.push_back(parseQuotedString(lex));
          expectRBracket(lex, ctx.c_str());
        } else {
          vals.push_back(parseQuotedString(lex));
        }
        params.values[paramName] = std::move(vals);
      } else {
        std::vector<float> vals;
        if (bracketed) {
          while (moreInBracket(lex, ctx.c_str()))
            vals.push_back(expectFloat(lex, ctx.c_str()));
          expectRBracket(lex, ctx.c_str());
        } else {
          vals.push_back(expectFloat(lex, ctx.c_str()));
        }
        params.values[paramName] = std::move(vals);
      }
    }
  }

  return params;
}

std::string Parser::parseQuotedString(Lexer &lex)
{
  Token tok = lex.next();
  if (tok.type != TokenType::String) {
    throw std::runtime_error(lex.currentFile() + ":" + std::to_string(tok.line)
        + ": expected quoted string, got '" + tok.text + "'");
  }
  return tok.text;
}

// Attribute stack /////////////////////////////////////////////////////////////

Parser::GraphicsState &Parser::currentState()
{
  return m_stateStack.back();
}

void Parser::pushState()
{
  m_stateStack.push_back(m_stateStack.back());
}

void Parser::popState()
{
  if (m_stateStack.size() <= 1)
    throw std::runtime_error("pbrt::Parser: attribute stack underflow");
  m_stateStack.pop_back();
}

} // namespace pbrt
