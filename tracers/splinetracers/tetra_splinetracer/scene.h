#ifndef EXPERIMENTAL_GSPLINE_VIEWER_SCENE_H_
#define EXPERIMENTAL_GSPLINE_VIEWER_SCENE_H_

#include <array>
#include <vector>

namespace gspline {
constexpr int kMaxSphericalHarmonicsDegree = 3;
constexpr int kMaxSphericalHarmonicsElements =
    (kMaxSphericalHarmonicsDegree + 1) * (kMaxSphericalHarmonicsDegree + 1) * 3;

struct GaussianScene {
  std::vector<std::array<float, 3>> means;
  std::vector<std::array<float, 3>> scales;
  std::vector<std::array<float, 4>> rotations;
  std::vector<float> alphas;
  std::vector<std::array<float, kMaxSphericalHarmonicsElements>>
      spherical_harmonics;
  int num_elements;
  int d_spherical_harmonics;
  std::array<float, 3> bbox_min;
  std::array<float, 3> bbox_max;
};
typedef std::vector<gspline::GaussianScene *> GaussianScenes;
} // namespace gspline

#endif // EXPERIMENTAL_GSPLINE_VIEWER_SCENE_H_
