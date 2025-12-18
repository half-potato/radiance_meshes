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

#include "ply_file_loader.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "scene.h"

#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"

#include <fstream>
#include <streambuf>
#include <string>
#include <iostream>

namespace gspline {

namespace {
// Returns the index of the \n byte and the index that starts the next
// line.
std::pair<int, int> NextLineEnd(absl::string_view str) {
  const char *line_end_idx = std::find(str.begin(), str.end(), '\n');
  if (line_end_idx == str.end()) {
    throw std::invalid_argument(
        "Expected string to contain a line ending, but found none.");
  }
  const int newline_idx = std::distance(str.begin(), line_end_idx);
  int next_line_idx = newline_idx + 1;
  if (next_line_idx < str.size() && str[next_line_idx] == '\r') {
    next_line_idx++;
  }
  return std::make_pair(newline_idx, next_line_idx);
}

absl::string_view ExtractLine(absl::string_view &str) {
  std::pair<int, int> line_sep_data = NextLineEnd(str);
  auto [ln_end, next_ln_start] = line_sep_data;
  absl::string_view line = str.substr(0, ln_end);
  if (next_ln_start < str.size()) {
    str = str.substr(next_ln_start);
  } else {
    str = absl::string_view();
  }
  return line;
}

void ResizeForComponents(GaussianScene &scene, const int number_of_components) {
  scene.num_elements = number_of_components;
  scene.means.resize(number_of_components);
  scene.scales.resize(number_of_components);
  scene.rotations.resize(number_of_components);
  scene.alphas.resize(number_of_components);
  scene.spherical_harmonics.resize(number_of_components);
}

void SortComponentsToMortonOrder(GaussianScene &scene) {
  std::vector<int> indices(scene.num_elements);
  std::iota(indices.begin(), indices.end(), 0);
  std::vector<uint64_t> mcode(scene.num_elements);

  float xmin(std::numeric_limits<float>::max()),
      xmax(std::numeric_limits<float>::min()),
      ymin(std::numeric_limits<float>::max()),
      ymax(std::numeric_limits<float>::min()),
      zmin(std::numeric_limits<float>::max()),
      zmax(std::numeric_limits<float>::min());
  for (int cidx = 0; cidx < scene.num_elements; ++cidx) {
    auto [x, y, z] = scene.means[cidx];
    if (x < xmin)
      xmin = x;
    if (x > xmax)
      xmax = x;
    if (y < ymin)
      ymin = y;
    if (y > ymax)
      ymax = y;
    if (z < zmin)
      zmin = z;
    if (z > zmax)
      zmax = z;
  }

  for (int cidx = 0; cidx < scene.num_elements; ++cidx) {
    float relx = (scene.means[cidx][0] - xmin) / (xmax - xmin);
    float rely = (scene.means[cidx][1] - ymin) / (ymax - ymin);
    float relz = (scene.means[cidx][2] - zmin) / (zmax - zmin);

    int scaled_x = static_cast<int>(static_cast<float>((1 << 21) - 1) * relx);
    int scaled_y = static_cast<int>(static_cast<float>((1 << 21) - 1) * rely);
    int scaled_z = static_cast<int>(static_cast<float>((1 << 21) - 1) * relz);

    uint64_t code = 0;
    for (int b = 0; b < 21; ++b) {
      code |= (static_cast<uint64_t>(scaled_x & (1 << b))) << (2 * b);
      code |= (static_cast<uint64_t>(scaled_y & (1 << b))) << (2 * b + 1);
      code |= (static_cast<uint64_t>(scaled_z & (1 << b))) << (2 * b + 2);
    }
    mcode[cidx] = code;
  }
  auto sort_order = [&mcode](const int idx1, const int idx2) {
    return mcode[idx1] < mcode[idx2];
  };
  std::sort(indices.begin(), indices.end(), sort_order);

  // Reshuffle data according to the new order.
  decltype(scene.means) new_means = scene.means;
  decltype(scene.scales) new_scales = scene.scales;
  decltype(scene.rotations) new_rotations = scene.rotations;
  decltype(scene.alphas) new_alphas = scene.alphas;
  decltype(scene.spherical_harmonics) new_spherical_harmonics =
      scene.spherical_harmonics;
  for (int nidx = 0; nidx < scene.num_elements; ++nidx) {
    new_means[nidx] = scene.means[indices[nidx]];
    new_scales[nidx] = scene.scales[indices[nidx]];
    new_rotations[nidx] = scene.rotations[indices[nidx]];
    new_alphas[nidx] = scene.alphas[indices[nidx]];
    new_spherical_harmonics[nidx] = scene.spherical_harmonics[indices[nidx]];
  }
  scene.means = new_means;
  scene.scales = new_scales;
  scene.rotations = new_rotations;
  scene.alphas = new_alphas;
  scene.spherical_harmonics = new_spherical_harmonics;
}

float sigmoid(float x) { return 1.0f / (1 + std::exp(-x)); }
float inv_opacity(float y) {
  return fmax(0, -log(fmax(1 - y, 1e-10)));
}

float softplus(float x) {
  if (x > 5) {
    return x;
  } else {
    return std::log(1+std::exp(x));
  }
}


// Applies sigmoid function to alphas, normalizes rotations and exponentiates
// scales.
void NormalizeSceneData(GaussianScene &scene) {
  float SCALE = 1;
  for (int cidx = 0; cidx < scene.num_elements; ++cidx) {
    // Rotations.
    float len = 0;
    for (int q = 0; q < 4; ++q) {
      len += scene.rotations[cidx][q] * scene.rotations[cidx][q];
    }
    for (int q = 0; q < 4; ++q) {
      scene.rotations[cidx][q] /= len;
    }
    for (int s = 0; s < 3; ++s) {
      scene.means[cidx][s] = SCALE*scene.means[cidx][s];
    }
    // Scales.
    float norm = 9999;
    for (int s = 0; s < 3; ++s) {
      scene.scales[cidx][s] = SCALE*std::min(
          softplus(scene.scales[cidx][s]) + 3*1e-4, 25.0);
      norm = std::min(2 * scene.scales[cidx][s], norm);
    }
    // Alphas.
    float alpha = 0.99*sigmoid(scene.alphas[cidx]);
    scene.alphas[cidx] = inv_opacity(alpha) / norm;

  }
}

} // namespace

GaussianScene ReadSceneFromFile(absl::string_view filename, bool approximate_morton_order) {
  std::string file_contents;

  std::string fname = {filename.begin(), filename.end()};
  std::ifstream t(fname, std::ios::binary);
  std::stringstream buffer;
  buffer << t.rdbuf();

  int number_of_components = 0;
  file_contents = buffer.str();

  absl::string_view remaining_contents = file_contents;
  absl::string_view line = ExtractLine(remaining_contents);
  if (line != "ply") {
    throw std::invalid_argument("Expected file to have \"ply\" header.");
  }
  line = ExtractLine(remaining_contents);
  if (line != "format binary_little_endian 1.0") {
    throw std::invalid_argument(
        "Expected file to have \"format binary_little_endian 1.0\" header.");
  }
  line = ExtractLine(remaining_contents);
  std::vector<absl::string_view> line_parts = absl::StrSplit(line, ' ');
  if (line_parts.size() != 3 ||
      !absl::SimpleAtoi(line_parts[2], &number_of_components)) {
    throw std::invalid_argument(
        "Expected file to have \"element vertex X\" in the header, where X is "
        "the number of components.");
  }
  std::cout << "Reading gaussian splatting scene with " << number_of_components
            << " components.";
  // Read property lines until "end_header" line.
  // pair<type, name>
  std::vector<std::pair<absl::string_view, absl::string_view>> properties;
  while (true) {
    line = ExtractLine(remaining_contents);
    if (line == "end_header") {
      break;
    }
    line_parts = absl::StrSplit(line, ' ');
    if (line_parts.size() != 3) {
      throw std::invalid_argument(
          "Expected property line have format \"property "
                          "<type> <name>\", got \"%s\"");
    }
    properties.push_back(std::make_pair(line_parts[1], line_parts[2]));
  }
  GaussianScene scene;
  ResizeForComponents(scene, number_of_components);
  scene.d_spherical_harmonics = kMaxSphericalHarmonicsDegree;

  // Main loading loop.
  // Assumes following order:
  //  [mX, mY, mZ]
  //  [nX, nY, nZ] <-- ignored.
  //  [4 * 4 * 3] Spherical Harmonics (first 3 components are DC).
  //  [alpha]
  //  [sX, sY, sZ]
  //  [qa, qb, qc, qd]
  constexpr int payload_size =
      sizeof(std::array<float, 3>) + sizeof(std::array<float, 3>) +
      sizeof(std::array<float, kMaxSphericalHarmonicsElements>) + sizeof(float) +
      sizeof(std::array<float, 3>) + sizeof(std::array<float, 4>);
  if (remaining_contents.size() < number_of_components * payload_size) {
    throw std::invalid_argument("File data is truncated, expected at least %d bytes, got %d.");
  }

  const char *data_begin = remaining_contents.data();
  const char *data = remaining_contents.data();

  float max_val = std::numeric_limits<float>::max();
  float min_val = std::numeric_limits<float>::min();
  std::array<float, 3> bbox_min{max_val, max_val, max_val};
  std::array<float, 3> bbox_max{min_val, min_val, min_val};

  for (int cidx = 0; cidx < number_of_components; ++cidx) {
    scene.means[cidx] = *reinterpret_cast<const std::array<float, 3> *>(data);
    data += sizeof(std::array<float, 3>);
    // Skip n.
    data += sizeof(std::array<float, 3>);
    scene.spherical_harmonics[cidx] = *reinterpret_cast<
        const std::array<float, kMaxSphericalHarmonicsElements> *>(data);

    data += sizeof(std::array<float, kMaxSphericalHarmonicsElements>);
    scene.alphas[cidx] = *reinterpret_cast<const float *>(data);
    data += sizeof(float);
    scene.scales[cidx] = *reinterpret_cast<const std::array<float, 3> *>(data);
    data += sizeof(std::array<float, 3>);
    scene.rotations[cidx] =
        *reinterpret_cast<const std::array<float, 4> *>(data);
    data += sizeof(std::array<float, 4>);
  }
  assert(data == data_begin + number_of_components * payload_size);

  NormalizeSceneData(scene);

  return scene;
}

} // namespace gspline
