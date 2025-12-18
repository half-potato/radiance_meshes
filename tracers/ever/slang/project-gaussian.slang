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

static const float PI = 3.14159265f;

[Differentiable]
float eradius(in float3 scales, in float3 xyz)
{
  float a = scales.x;
  float b = scales.y;
  float c = scales.z;
  float x = xyz.x;
  float y = xyz.y;
  float z = xyz.z;
  float c1 = b*b * x*x + a*a * y*y;
  float inner = c*c * c1 + a*a * b*b * z*z;
  return a * b * c / sqrt(inner);
}

[Differentiable]
float get_sigma(in float3 scales, in float3x3 rot, in float3 rayd) {
  float sigma1d = eradius(scales, mul(rot, rayd));
  return sigma1d;
}

[Differentiable]
float3 project_gaussian(
    float peak_density, float3 mu, float3 scales, float4 quat, 
    float3 rayo, float3 rayd) 
{
  let r = quat[0];
  let x = quat[1];
  let y = quat[2];
  let z = quat[3];

  // Compute rotation matrix from quaternion
  float3x3 Rt = {
      1.0 - 2.0 * (y * y + z * z),
      2.0 * (x * y - r * z),
      2.0 * (x * z + r * y),

      2.0 * (x * y + r * z),
      1.0 - 2.0 * (x * x + z * z),
      2.0 * (y * z - r * x),

      2.0 * (x * z - r * y),
      2.0 * (y * z + r * x),
      1.0 - 2.0 * (x * x + y * y)
  };

  let R = transpose(Rt);
  float3x3 invS = {
    1/(scales.x), 0, 0,
    0, 1/(scales.y), 0,
    0, 0, 1/(scales.z),
  };
  float3x3 invS2 = {
    1/(scales.x*scales.x), 0, 0,
    0, 1/(scales.y*scales.y), 0,
    0, 0, 1/(scales.z*scales.z),
  };


  // This is transposed for some reason
  let transform = mul(invS, Rt);
  let v = mul((mu - rayo), transform);
  let rrayd = mul(rayd, transform);

  let t = dot(v, rrayd) / dot(rrayd, rrayd);

  float3x3 S = {
    scales.x, 0, 0,
    0, scales.y, 0,
    0, 0, scales.z,
  };
  let M = mul(S, R);
  let sigma = mul(transpose(M), M);

  let sigma1d = get_sigma(scales, R, rayd);

  // next, compute marginal prob
  let invsigma = mul(R, mul(invS2, Rt));
  let mean = (t * rayd + rayo);
  let sdiff = mean - mu;
  let p2 = -0.5 * dot(mul(invsigma, sdiff), sdiff);
  let marginal_prob2D = exp(p2);
  let marginal_prob1D = 1;

  return float3(peak_density * marginal_prob2D / marginal_prob1D, t, sigma1d);
}
