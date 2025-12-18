#include "TriangleMesh.h"

// void TriangleMesh::addOctehedra(const glm::vec4 quat,
//     const glm::vec3 center, const glm::vec3 size, const float density, const glm::vec3 color)
// {
// }

void TriangleMesh::addOctehedra(const glm::mat4x3 &xfm, const float dirac, const glm::vec3 color)
{
  int firstVertexID = (int)vertex.size();
  const float s = 1;
  vertex.push_back(glm::vec3(xfm * glm::vec4( 0.f, 0.f, s,1.f)));

  vertex.push_back(glm::vec3(xfm * glm::vec4( 0.f, 1.f, 0.f,1.f)));
  vertex.push_back(glm::vec3(xfm * glm::vec4( 1.f, 0.f, 0.f,1.f)));
  vertex.push_back(glm::vec3(xfm * glm::vec4( 0.f,-1.f, 0.f,1.f)));
  vertex.push_back(glm::vec3(xfm * glm::vec4(-1.f, 0.f, 0.f,1.f)));

  vertex.push_back(glm::vec3(xfm * glm::vec4( 0.f, 0.f,-s,1.f)));
  vertex.push_back(glm::vec3(xfm * glm::vec4( 0.f, 0.f, 0.f,1.f)));


  int indices[] = {0,1,2, 0,2,3,
                   0,3,4, 0,4,1,
                   5,1,2, 5,2,3,
                   5,3,4, 5,4,1,
                   // middle triangles
                   // top fins
                   0,1,6, 0,2,6,
                   0,3,6, 0,4,6,
                   // bot fins
                   5,1,6, 5,2,6,
                   5,3,6, 5,4,6,
                   // mid fins
                   6,1,2, 6,2,3,
                   6,3,4, 6,4,1,
                   };
  for (int i=0;i<8;i++)
  {
    index.push_back(firstVertexID+glm::ivec3(
          indices[3*i+0], indices[3*i+1], indices[3*i+2]));
    face_dirac.push_back(glm::vec4(dirac, dirac*color.x, dirac*color.y, dirac*color.z));
  }
  // add middle triangles
  for (int i=8;i<8+12;i++)
  {
    index.push_back(firstVertexID+glm::ivec3(
          indices[3*i+0], indices[3*i+1], indices[3*i+2]));
    face_dirac.push_back(glm::vec4(dirac, dirac*color.x, dirac*color.y, dirac*color.z));
  }
  features.push_back(color.x);
  features.push_back(color.y);
  features.push_back(color.z);
}
