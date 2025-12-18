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
  float3 base_color;
  float depth;

  float3 grd;
  float t;


  // Volume Rendering State
  float3 C;
  float logT;

  float3 padding;
  float density;
};

SplineState make_empty_state() {
  return {
    float3(0.f),
    0.f,
    float3(0.f),
    0.f,
    float3(0.0f),
    0.f,
    float3(0.0f),
    0.0,
  };
}

struct ControlPoint : IDifferentiable
{
  float t;
  float density;
  float3 base_color;
  float3 grd;
}

bool AlmostEqualRelative(float A, float B,
                         float maxRelDiff = 1.2e-7f)
{
    return abs(A - B) <= max(abs(A), abs(B)) * maxRelDiff;
}

[Differentiable]
inline float3 linear_color(float3 grad, float3 v)
{
    float f = dot(grad, v);
    return {f, f, f};
}

[Differentiable]
float phi(float x) {
    if (abs(x) < 1e-6) {
        return 1.0f - x / 2.0f;
    }
    // return (1.0f - exp(-x)) / x;
    return -(safe_expm1(-x)) / x;
}


[Differentiable]
float3 compute_integral(float3 c0, float3 c1, float ddt) {
    float alpha = safe_exp(-ddt);
    
    float phi_val = phi(ddt);
    
    float w0 = phi_val - alpha;
    float w1 = 1.0f - phi_val;
    
    return w0 * c0 + w1 * c1;
}

SplineState inverse_update_dual(
    in float3 ray_o,
    in float3 ray_d,
    in SplineState new_state,
    in ControlPoint new_ctrl_pt,
    in ControlPoint ctrl_pt,
    in float t_min,
    in float t_max)
{
  const float t = ctrl_pt.t;
  const float dt = max(new_state.t - t, 0.f);

  SplineState state = {};
  state.density = ctrl_pt.density;
  state.base_color = ctrl_pt.base_color;
  state.grd = ctrl_pt.grd;
  state.t = t;

  let area = max(state.density * dt, 0.f);

  new_state.logT = max(area + state.logT, 0.f);
  
  float3 c_enter = max(linear_color(state.grd, ray_d) + state.base_color, 0.f);
  float3 c_exit = max(linear_color(state.grd, ray_d) + state.base_color, 0.f);
  float3 final_color = compute_integral(c_exit, c_enter, dt*state.density);
  float alpha = max(-safe_expm1(-state.density * dt), 0.f);
  state.C = new_state.C - final_color;

  const float AREA_THRESHOLD = 1e-6f;
  float segment_depth_val;
  if (state.density < AREA_THRESHOLD) {
      segment_depth_val = alpha * t + (1-alpha) * state.t;
  } else {
      segment_depth_val = safe_div(1.f, state.density) * (-safe_expm1(-area)) 
                        - (t + t_min) * safe_exp(-area) 
                        + (state.t + t_min);
  }

  const float segment_depth = max(segment_depth_val, 0.f);
  new_state.depth = state.depth - safe_exp(-state.logT) * segment_depth;

  return state;
}


[Differentiable]
SplineState update(
    in float3 ray_o,
    in float3 ray_d,
    in SplineState state,
    in ControlPoint ctrl_pt,
    no_diff in float t_min,
    no_diff in float t_max)
{
  const float t = ctrl_pt.t;
  const float dt = max(t - state.t, 0.f);

  SplineState new_state;
  new_state.density = ctrl_pt.density;
  new_state.base_color = ctrl_pt.base_color;
  new_state.grd = ctrl_pt.grd;
  new_state.t = t;

  let area = max(state.density * dt, 0.f);

  new_state.logT = max(area + state.logT, 0.f);

  float3 c_enter = max(linear_color(state.grd, ray_o+ray_d*state.t) + state.base_color, 0.f);
  float3 c_exit = max(linear_color(state.grd, ray_o+ray_d*t) + state.base_color, 0.f);
  float3 final_color = compute_integral(c_exit, c_enter, dt*state.density);
  float alpha = max(-safe_expm1(-state.density * dt), 0.f);

  new_state.C = state.C + safe_exp(-state.logT)*final_color;

  const float AREA_THRESHOLD = 1e-6f;
  float segment_depth_val;
  if (state.density < AREA_THRESHOLD) {
      segment_depth_val = alpha * t + (1-alpha) * state.t;
  } else {
      segment_depth_val = safe_div(1.f, state.density) * (-safe_expm1(-area)) 
                        - (t + t_min) * safe_exp(-area) 
                        + (state.t + t_min);
  }

  const float segment_depth = max(segment_depth_val, 0.f);
  new_state.depth = state.depth + safe_exp(-state.logT) * segment_depth;

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
    0.f
  };
}

