#pragma once

#include <scene/surface/geometry/Triangle.h>

#include <anari/anari_cpp.hpp>
#include <glm/fwd.hpp>

namespace visrtx {

void computeVertexNormals(glm::vec3 *normals, // Output vertex normals
    const glm::vec3 *positions, // Input vertex positions
    const glm::uvec3 *indices, // Input triangle indices
    unsigned int numTriangles, // Number of triangles
    unsigned int numNormals // Number of normals
);

template <typename TexCoord>
void computeVertexTangents(
    glm::vec4 *tangents, // Output tangent vectors with handedness (w component)
    const glm::vec3 *positions, // Input vertex positions
    const glm::vec3 *normals, // Input vertex normals
    const TexCoord *texCoords, // Input texture coordinates
    const glm::uvec3 *indices, // Input triangle indices
    unsigned int numTriangles, // Number of triangles
    unsigned int numNormals // Number of normals
);

void updateGeometryTangent(Triangle *triangle);

} // namespace visrtx
