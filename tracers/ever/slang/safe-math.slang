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

// Found experimentally idk where it happens in slang
static const float TINY_VAL = 1.0754944e-20;
static const float MIN_VAL = -1e+20;
static const float MAX_VAL = 1e+20;

float nextafter(float x, float y) {
  __intrinsic_asm "nextafterf($0, $1)";
}

float l1(float3 v) {
  return abs(v.x) + abs(v.y) + abs(v.z);
}

float linf(float3 v) {
  return max(max(abs(v.x), abs(v.y)), abs(v.z));
}

bool is_nan(float3 x) {
    return !((x.x == x.x) && (x.y == x.y) && (x.z == x.z));
}

bool is_nan(float4 x) {
    return !((x.x == x.x) && (x.y == x.y) && (x.z == x.z) && (x.w == x.w));
}

bool is_nan(float x) {
    return (x.x != x.x);
}

[Differentiable]
float clip(float v, float minv, float maxv) {
  return max(min(v, maxv), minv);
}

[Differentiable]
float3 clip(float3 v, float minv, float maxv) {
  return {
    max(min(v.x, maxv), minv),
    max(min(v.y, maxv), minv),
    max(min(v.z, maxv), minv)
  };
}

[Differentiable]
float4 clip(float4 v, float minv, float maxv) {
  return {
    max(min(v.x, maxv), minv),
    max(min(v.y, maxv), minv),
    max(min(v.z, maxv), minv),
    max(min(v.w, maxv), minv)
  };
}

void bw_safe_div(inout DifferentialPair<float> a, inout DifferentialPair<float> b, float.Differential R) {
  if (abs(b.p) < TINY_VAL) {
    a = DifferentialPair<float>(a.p, clip(R / TINY_VAL, MIN_VAL, MAX_VAL));
  } else {
    a = DifferentialPair<float>(a.p, clip(R / b.p, MIN_VAL, MAX_VAL));
  }
  let b2 = b.p * b.p;
  if (b2 < TINY_VAL) {
    b = DifferentialPair<float>(b.p, clip(- a.p / TINY_VAL * R, MIN_VAL, MAX_VAL));
  } else {
    b = DifferentialPair<float>(b.p, clip(- a.p / b2 * R, MIN_VAL, MAX_VAL));
  }
}

//[TorchEntryPoint]
//[Differentiable]
[BackwardDerivative(bw_safe_div)]
float safe_div(in float a, in float b) {
  if (abs(b) < TINY_VAL) {
    return clip(a / TINY_VAL, MIN_VAL, MAX_VAL);
  } else {
    return clip(a / b, MIN_VAL, MAX_VAL);
  }
}

[Differentiable]
float3 safe_div(in float3 a, in float b) {
  return {
    safe_div(a.x, b),
    safe_div(a.y, b),
    safe_div(a.z, b),
  };
}

[Differentiable]
float4 safe_div(in float4 a, in float b) {
  return {
    safe_div(a.x, b),
    safe_div(a.y, b),
    safe_div(a.z, b),
    safe_div(a.w, b),
  };
}

[Differentiable]
float3 safe_div(in float3 a, in float3 b) {
  return {
    safe_div(a.x, b.x),
    safe_div(a.y, b.y),
    safe_div(a.z, b.z),
  };
}

//[TorchEntryPoint]
[Differentiable]
float safe_sqrt(in float a) {
  if (a < TINY_VAL) {
    return 0.0;
  } else {
    return sqrt(a);
  }
}

[Differentiable]
float4 l2_normalize(float4 x) {
  return safe_div(x, sqrt(max(dot(x, x), TINY_VAL)));
}

[Differentiable]
float3 l2_normalize(float3 x) {
  return safe_div(x, sqrt(max(dot(x, x), TINY_VAL)));
}

[Differentiable]
float safe_exp(float v) {
  return exp(clip(v, MIN_VAL, log(MAX_VAL)));
}

void bw_log1p(inout DifferentialPair<float> v, float.Differential R) {
  v = DifferentialPair<float>(v.p, safe_div(1, 1+v.p) * R);
}

[BackwardDerivative(bw_log1p)]
float _log1p(float v) {
  __intrinsic_asm "log1p(($0))";
}

void bw_log(inout DifferentialPair<float> v, float.Differential R) {
  v = DifferentialPair<float>(v.p, R / clip(v.p, MIN_VAL, MAX_VAL));
}

[BackwardDerivative(bw_log)]
float safe_log(float v) {
  return log(clip(v, MIN_VAL, MAX_VAL));
}

void bw_expm1(inout DifferentialPair<float> v, float.Differential R) {
  v = DifferentialPair<float>(v.p, exp(v.p) * R);
}

[BackwardDerivative(bw_expm1)]
float safe_expm1(float v) {
  __intrinsic_asm "expm1($0)";
}

void bw_log1mexp(inout DifferentialPair<float> v, float.Differential R) {
  v = DifferentialPair<float>(v.p, safe_div(R, safe_exp(v.p) - 1));
}

//[Differentiable]
[BackwardDerivative(bw_log1mexp)]
float log1mexp(float x) {
  if (x > 0.30102999566) {
    return _log1p(max(-exp(-x), -1+1e-5));
  } else {
    return log(max(-safe_expm1(-x), 1e-20));
  }
}

float4x4 inverse(float4x4 m) {
    float n11 = m[0][0], n12 = m[1][0], n13 = m[2][0], n14 = m[3][0];
    float n21 = m[0][1], n22 = m[1][1], n23 = m[2][1], n24 = m[3][1];
    float n31 = m[0][2], n32 = m[1][2], n33 = m[2][2], n34 = m[3][2];
    float n41 = m[0][3], n42 = m[1][3], n43 = m[2][3], n44 = m[3][3];

    float t11 = n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 - n23 * n32 * n44 + n22 * n33 * n44;
    float t12 = n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 + n13 * n32 * n44 - n12 * n33 * n44;
    float t13 = n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 - n13 * n22 * n44 + n12 * n23 * n44;
    float t14 = n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34;

    float det = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;
    float idet = 1.0f / det;

    float4x4 ret;

    ret[0][0] = t11 * idet;
    ret[0][1] = (n24 * n33 * n41 - n23 * n34 * n41 - n24 * n31 * n43 + n21 * n34 * n43 + n23 * n31 * n44 - n21 * n33 * n44) * idet;
    ret[0][2] = (n22 * n34 * n41 - n24 * n32 * n41 + n24 * n31 * n42 - n21 * n34 * n42 - n22 * n31 * n44 + n21 * n32 * n44) * idet;
    ret[0][3] = (n23 * n32 * n41 - n22 * n33 * n41 - n23 * n31 * n42 + n21 * n33 * n42 + n22 * n31 * n43 - n21 * n32 * n43) * idet;

    ret[1][0] = t12 * idet;
    ret[1][1] = (n13 * n34 * n41 - n14 * n33 * n41 + n14 * n31 * n43 - n11 * n34 * n43 - n13 * n31 * n44 + n11 * n33 * n44) * idet;
    ret[1][2] = (n14 * n32 * n41 - n12 * n34 * n41 - n14 * n31 * n42 + n11 * n34 * n42 + n12 * n31 * n44 - n11 * n32 * n44) * idet;
    ret[1][3] = (n12 * n33 * n41 - n13 * n32 * n41 + n13 * n31 * n42 - n11 * n33 * n42 - n12 * n31 * n43 + n11 * n32 * n43) * idet;

    ret[2][0] = t13 * idet;
    ret[2][1] = (n14 * n23 * n41 - n13 * n24 * n41 - n14 * n21 * n43 + n11 * n24 * n43 + n13 * n21 * n44 - n11 * n23 * n44) * idet;
    ret[2][2] = (n12 * n24 * n41 - n14 * n22 * n41 + n14 * n21 * n42 - n11 * n24 * n42 - n12 * n21 * n44 + n11 * n22 * n44) * idet;
    ret[2][3] = (n13 * n22 * n41 - n12 * n23 * n41 - n13 * n21 * n42 + n11 * n23 * n42 + n12 * n21 * n43 - n11 * n22 * n43) * idet;

    ret[3][0] = t14 * idet;
    ret[3][1] = (n13 * n24 * n31 - n14 * n23 * n31 + n14 * n21 * n33 - n11 * n24 * n33 - n13 * n21 * n34 + n11 * n23 * n34) * idet;
    ret[3][2] = (n14 * n22 * n31 - n12 * n24 * n31 - n14 * n21 * n32 + n11 * n24 * n32 + n12 * n21 * n34 - n11 * n22 * n34) * idet;
    ret[3][3] = (n12 * n23 * n31 - n13 * n22 * n31 + n13 * n21 * n32 - n11 * n23 * n32 - n12 * n21 * n33 + n11 * n22 * n33) * idet;

    return ret;
}

// This is for the conjugate quaternion
[Differentiable]
half3 rotate_vector(half3 v, half4 q)
{
  // Adapted from code by John D Cook.
  // https://www.johndcook.com/blog/2021/06/16/faster-quaternion-rotations/
  let t = 2.h * cross(-q.yzw, v);
  return v + q.x * t + cross(-q.yzw, t);
}

// This is for the conjugate quaternion
[Differentiable]
float3 rotate_vector(float3 v, float4 q)
{
  // Adapted from code by John D Cook.
  // https://www.johndcook.com/blog/2021/06/16/faster-quaternion-rotations/
  let t = 2 * cross(-q.yzw, v);
  return v + q.x * t + cross(-q.yzw, t);
}

[Differentiable]
float tukey_power_ladder(float x, float p) {
  // Compute sign(x) * |p - 1|/p * ((|x|/|p-1| + 1)^p - 1)
  let xp = abs(x);
  let xs = xp / max(TINY_VAL, abs(p - 1));
  let y = sign(x) * abs(p - 1) / p * (pow(xs + 1, p) - 1);
  return y;
}

[Differentiable]
float softplus(float x, float beta) {
  if (x * beta > 20) {
    return x;
  } else {
    return log(1 + exp(beta * x)) / beta;
  }
}

[Differentiable]
float3 softplus(float3 x, float beta) {
  return {
    softplus(x.x, beta),
    softplus(x.y, beta),
    softplus(x.z, beta)
  };
}

half hsign(half x) {
  return (x < 0.h) ? -1.h : 1.h;
}

half hsqrt(half x) {
  __intrinsic_asm "hsqrt($0)";
}

half float2half(float x) {
  __intrinsic_asm "__float2half($0)";
}

float half2float(half x) {
  __intrinsic_asm "__half2float($0)";
}

half3 float2half(float3 v) {
    return {
        float2half(v.x),
        float2half(v.y),
        float2half(v.z)
    };
}

float3 half2float(half3 v) {
    return {
        half2float(v.x),
        half2float(v.y),
        half2float(v.z)
    };
}

float4 half2float(half4 v) {
    return {
        half2float(v.x),
        half2float(v.y),
        half2float(v.z),
        half2float(v.w)
    };
}
