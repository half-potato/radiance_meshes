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

import optix_intrinsics;

uint mod(uint a, uint b) {
  __intrinsic_asm "$0 % $1";
}

float3 toSRGB(float3 c)
{
  const float invGamma = 1.f / 2.4f;
  const float3 powed = {pow(c.x, invGamma), pow(c.y, invGamma), pow(c.z, invGamma)};
  return {
    c.x < 0.0031308f ? 12.92f * c.x : 1.055f * powed.x - 0.055f,
    c.y < 0.0031308f ? 12.92f * c.y : 1.055f * powed.y - 0.055f,
    c.z < 0.0031308f ? 12.92f * c.z : 1.055f * powed.z - 0.055f
  };
}

uint make_color(float3 c) {
  float3 srgbC = c;

  const int r = int(255.99f*clamp(srgbC.x, 0, 1));
  const int g = int(255.99f*clamp(srgbC.y, 0, 1));
  const int b = int(255.99f*clamp(srgbC.z, 0, 1));

  // convert to 32-bit rgba value (we explicitly set alpha to 0xff
  // to make stb_image_write happy ...
  const uint rgba = 0x00000000
    | (r<<0) | (g<<8) | (b<<16) | (255u << 24);
  return rgba;
}

