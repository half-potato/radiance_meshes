// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
