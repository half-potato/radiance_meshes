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

static const float EPS = 1.1754944e-14;

struct SplineState : IDifferentiable
{
  // Spline state
  float t;
  float4 drgb;
  float4 d_drgb;

  // Volume Rendering State
  float logT;
  float d_spline;
  float3 avg_color;
  float area;
  float3 C;
};

struct ControlPoint : IDifferentiable
{
  float t;
  float4 dirac;
}

void bw_log1p(inout DifferentialPair<float> v, float.Differential R) {
  v = DifferentialPair<float>(v.p, 1/(1+v.p) * R);
}

[BackwardDerivative(bw_log1p)]
float _log1p(float v) {
  __intrinsic_asm "log1p(($0))";
}

void bw_expm1(inout DifferentialPair<float> v, float.Differential R) {
  v = DifferentialPair<float>(v.p, -exp(v.p) * R);
}

[BackwardDerivative(bw_expm1)]
float _expm1(float v) {
  __intrinsic_asm "expm1($0)";
}

[Differentiable]
float log1mexp(float x) {
  if (x > 0.30102999566) {
    return _log1p(-exp(-x));
  } else {
    return log(-_expm1(-x));
  }
}

[TorchEntryPoint]
[Differentiable]
SplineState update(
    in SplineState state,
    in ControlPoint ctrl_pt,
    in float t_min,
    in float t_max)
{
  const float t = ctrl_pt.t;
  const float dt = t - state.t;

  // This is the two cumsums
  SplineState new_state;
  new_state.drgb = max(state.drgb + state.d_drgb * dt, 0.0);
  new_state.d_drgb = state.d_drgb + ctrl_pt.dirac;

  new_state.t = t;

  // compute average color

  // Clamping
  float4 drgb = state.drgb;
  float4 new_drgb = new_state.drgb;
  if (t > t_max || state.t < t_min) {
    drgb = float4(0.0);
    new_drgb = float4(0.0);
  }

  // Integrate information
  const float d_avg = max(new_drgb.x, 0.0) / 2 + max(drgb.x, 0.0) / 2;
  const float area = d_avg * dt;

  const float3 new_rgb = max(float3(new_drgb.y, new_drgb.z, new_drgb.w), 0);
  const float3 rgb = max(float3(drgb.y, drgb.z, drgb.w), 0);
  const float3 rgb_avg = 0.5 * (rgb + new_rgb);
  const float3 rgb_norm = rgb_avg / max(d_avg, EPS);

  new_state.logT = area + state.logT;
  const float weight = exp(log1mexp(area) - state.logT);
  new_state.C = state.C + weight * rgb_norm;

  // debug
  new_state.area = area;
  new_state.d_spline = new_state.drgb.x;
  new_state.avg_color = rgb_norm;
  return new_state;
}


[TorchEntryPoint]
SplineState to_dual(in SplineState state, in ControlPoint ctrl_pt)
{
  SplineState dual_state = state;
  dual_state.d_drgb = state.d_drgb - ctrl_pt.dirac;
  return dual_state;
}

[TorchEntryPoint]
SplineState from_dual(in SplineState state, in ControlPoint ctrl_pt)
{
  SplineState dual_state = state;
  dual_state.d_drgb = state.d_drgb + ctrl_pt.dirac;
  return dual_state;
}

[TorchEntryPoint]
SplineState inverse_update_dual(
    SplineState new_state_dual,
    ControlPoint ctrl_pt,
    in float t_min,
    in float t_max)
{
  const float t = ctrl_pt.t;
  const float dt = new_state_dual.t - t;

  // This is the two cumsums
  SplineState state = {};
  state.drgb = max(new_state_dual.drgb - new_state_dual.d_drgb * dt, 0.0);
  state.d_drgb = new_state_dual.d_drgb - ctrl_pt.dirac;

  state.t = t;

  // compute average color

  // Clamping
  float4 drgb = state.drgb;
  float4 new_drgb = new_state_dual.drgb;
  if (t > t_max || state.t < t_min) {
    drgb = float4(0.0);
    new_drgb = float4(0.0);
  }

  // Integrate information
  const float d_avg = max(new_drgb.x, 0.0) / 2 + max(drgb.x, 0.0) / 2;
  const float area = d_avg * dt;

  const float3 new_rgb = max(float3(new_drgb.y, new_drgb.z, new_drgb.w), 0);
  const float3 rgb = max(float3(drgb.y, drgb.z, drgb.w), 0);
  const float3 rgb_avg = 0.5 * (rgb + new_rgb);
  const float3 rgb_norm = rgb_avg / max(d_avg, EPS);

  state.logT = new_state_dual.logT - area;
  const float weight = exp(log1mexp(area) - state.logT);

  state.C = new_state_dual.C - weight * rgb_norm;
  return state;
}
