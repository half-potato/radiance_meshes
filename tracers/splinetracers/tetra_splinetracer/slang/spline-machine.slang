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

#define EPS 1e-18
import safe_math;
#define PRE_MULTI 1000
#define LADDER_P -0.1

struct SplineState : IDifferentiable
{
  float2 distortion_parts;
  float2 cum_sum;
  float depth;
  float2 padding;
  // Spline state
  float t;
  float4 drgb;

  // Volume Rendering State
  float logT;
  float3 C;
};

SplineState make_empty_state() {
  return {
    float2(0.f),
    float2(0.f),
    0.f,
    float2(0.f),
    0.f,
    float4(0.0f),

    0.0,
    float3(0.0f),
  };
}

struct ControlPoint : IDifferentiable
{
  float t;
  float4 dirac;
}

bool AlmostEqualRelative(float A, float B,
                         float maxRelDiff = 1.2e-7f)
{
    return abs(A - B) <= max(abs(A), abs(B)) * maxRelDiff;
}

SplineState to_dual(in SplineState state, in ControlPoint ctrl_pt)
{
  SplineState dual_state = state;
  return dual_state;
}

SplineState from_dual(in SplineState state, in ControlPoint ctrl_pt)
{
  SplineState dual_state = state;
  return dual_state;
}

SplineState inverse_update_dual(
    in SplineState new_state,
    in ControlPoint new_ctrl_pt,
    in ControlPoint ctrl_pt,
    in float t_min,
    in float t_max)
{
  const float t = ctrl_pt.t;
  const float dt = max(new_state.t - t, 0.f);

  SplineState state = {};
  state.drgb = ctrl_pt.dirac;

  state.t = t;

  float4 drgb = state.drgb;

  let avg = drgb;
  float area = max(avg.x * dt, 0.f);
  let rgb_norm = safe_div(float3(avg.y, avg.z, avg.w), avg.x);

  state.logT = max(new_state.logT - area, 0.f);
  const float weight = clip((1-safe_exp(-area)) * safe_exp(-state.logT), 0.f, 1.f);

  state.C = new_state.C - weight * rgb_norm;
  const float AREA_THRESHOLD = 1e-6f;
  float segment_depth_val;
  float alpha = -safe_expm1(-area);
  if (avg.x < AREA_THRESHOLD) {
      segment_depth_val = alpha * t + (1-alpha) * state.t;
  } else {
      segment_depth_val = safe_div(1.f, avg.x) * (-safe_expm1(-area)) 
                        - (t + t_min) * safe_exp(-area) 
                        + (state.t + t_min);
  }

  const float segment_depth = max(segment_depth_val, 0.f);
  new_state.depth = state.depth - safe_exp(-state.logT) * segment_depth;

  // Distortion Loss
  // make sure this is the right t. I believe this is correct.
  let m = tukey_power_ladder((new_state.t+state.t)/2 * PRE_MULTI, LADDER_P);
  state.cum_sum.x = new_state.cum_sum.x - weight;
  state.cum_sum.y = new_state.cum_sum.y - weight * m;
  state.distortion_parts.x = new_state.distortion_parts.x - 2 * weight * m * state.cum_sum.x;
  state.distortion_parts.y = new_state.distortion_parts.y - 2 * weight * state.cum_sum.y;

  return state;
}


[Differentiable]
SplineState update(
    in SplineState state,
    in ControlPoint ctrl_pt,
    no_diff in float t_min,
    no_diff in float t_max)
{
  const float t = ctrl_pt.t;
  const float dt = max(t - state.t, 0.f);

  SplineState new_state;
  new_state.drgb = ctrl_pt.dirac;
  new_state.t = t;

  float4 drgb = state.drgb;

  let avg = drgb;
  let area = max(avg.x * dt, 0.f);

  let rgb_norm = safe_div(float3(avg.y, avg.z, avg.w), avg.x);

  new_state.logT = max(area + state.logT, 0.f);
  float alpha = -safe_expm1(-area);
  const float weight = clip(alpha * safe_exp(-state.logT), 0.f, 1.f);
  new_state.C = state.C + weight * rgb_norm;

  const float AREA_THRESHOLD = 1e-6f;
  float segment_depth_val;
  if (avg.x < AREA_THRESHOLD) {
      segment_depth_val = alpha * t + (1-alpha) * state.t;
  } else {
      segment_depth_val = safe_div(1.f, avg.x) * (-safe_expm1(-area)) 
                        - (t + t_min) * safe_exp(-area) 
                        + (state.t + t_min);
  }

  const float segment_depth = max(segment_depth_val, 0.f);
  new_state.depth = state.depth + safe_exp(-state.logT) * segment_depth;


  // Distortion Loss
  // make sure this is the right t. I believe this is correct.
  let m = tukey_power_ladder((new_state.t+state.t)/2 * PRE_MULTI, LADDER_P);
  new_state.distortion_parts.x = state.distortion_parts.x + 2 * weight * m * state.cum_sum.x;
  new_state.distortion_parts.y = state.distortion_parts.y + 2 * weight * state.cum_sum.y;
  new_state.cum_sum.x = state.cum_sum.x + weight;
  new_state.cum_sum.y = state.cum_sum.y + weight * m;

  return new_state;
}

struct SplineOutput: IDifferentiable {
  float3 C;
  float depth;
  float distortion_loss;
};

[Differentiable]
SplineOutput extract_color(in SplineState state) {
  // let opacity = 1-exp(-state.logT);
  return {
    state.C,
    state.depth,
    state.distortion_parts.x - state.distortion_parts.y
  };
}

