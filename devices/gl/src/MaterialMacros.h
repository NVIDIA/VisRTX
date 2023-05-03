#pragma once

#define MATERIAL_COMMIT_ATTRIBUTE(PARAM, TYPE, INDEX)\
if(current.PARAM.type() == TYPE) {\
  std::array<float, 4> color = {0, 0, 0, 1};\
  current.PARAM.get(TYPE, color.data());\
  thisDevice->materials.set(material_index+INDEX, color);\
} else if(current.PARAM.type() == ANARI_SAMPLER) {\
  if(auto sampler = acquire<SamplerObjectBase*>(current.PARAM)) {\
    auto meta = sampler->metadata();\
    thisDevice->materials.setMem(material_index+INDEX, &meta);\
  }\
}

#define ALLOCATE_SAMPLERS(PARAM, SLOT)\
if(current.PARAM.type() == ANARI_SAMPLER) {\
  if(auto sampler = acquire<SamplerObjectBase*>(current.PARAM)) {\
    sampler->allocateResources(surf, SLOT);\
  }\
}

#define MATERIAL_DRAW_COMMAND(PARAM, SLOT)\
if(current.PARAM.type() == ANARI_SAMPLER) {\
  if(auto sampler = acquire<SamplerObjectBase*>(current.PARAM)) {\
    int index = surf->resourceIndex(SLOT);\
    sampler->drawCommand(index, command);\
  }\
}

#define MATERIAL_FRAG_DECL(PARAM, SLOT)\
if(current.PARAM.type() == ANARI_SAMPLER) {\
  if(auto sampler = acquire<SamplerObjectBase*>(current.PARAM)) {\
    int index = surf->resourceIndex(SLOT);\
    sampler->declare(index, shader);\
  }\
}

#define MATERIAL_FRAG_SAMPLE(VAR, PARAM, TYPE, INDEX, SLOT)\
shader.append("  vec4 " VAR " = ");\
if(current.PARAM.type() == TYPE) {\
  shader.append("materials[instanceIndices.y+" #INDEX "u];\n");\
} else if(current.PARAM.type() == ANARI_STRING) {\
  shader.append(current.PARAM.getString());\
  shader.append(semicolon);\
} else if(current.PARAM.type() == ANARI_SAMPLER) {\
  if(auto sampler = acquire<SamplerObjectBase*>(current.PARAM)) {\
    int index = surf->resourceIndex(SLOT);\
    sampler->sample(index, shader, "materials[instanceIndices.y+" #INDEX "u]\n");\
  } else {\
    shader.append("vec4(1.0, 0.0, 1.0, 1.0);\n");\
  }\
} else {\
  shader.append("vec4(1.0, 0.0, 1.0, 1.0);\n");\
}
