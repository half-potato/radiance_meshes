#pragma once
#include "glm/glm.hpp"
#include <vector>

struct TriangleMesh {
  void addOctehedra(const glm::mat4x3 &xfm, const float dirac, const glm::vec3 color);
  
  std::vector<glm::vec3> vertex;
  std::vector<glm::ivec3> index;
  std::vector<glm::vec4> face_dirac;
  std::vector<float> features;
  size_t feature_size = 3;
};


