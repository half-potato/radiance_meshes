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

#include <optix.h>
#include <cstdint>

union Union32 
{
    uint32_t u;
    int32_t i;
    float f;
};

__inline__ __device__ uint32_t F32_asuint(float f) { Union32 u; u.f = f; return u.u; }
__inline__ __device__ int32_t F32_asint(float f) { Union32 u; u.f = f; return u.i; }
__inline__ __device__ float U32_asfloat(uint32_t x) { Union32 u; u.u = x; return u.f; }

extern "C" __global__ void __anyhit__ah()
{
    float t_1 = optixGetRayTmax();
    uint _S21 = (optixGetAttribute_0());
    float other_t_0 = (U32_asfloat((_S21)));
    uint _S22 = optixGetPrimitiveIndex();
    uint hitkind_0 = (optixGetHitKind());
    float t_2;
    float other_t_1;
    if(hitkind_0 == 0U)
    {
        t_2 = other_t_0;
        other_t_1 = t_1;
    }
    else
    {
        t_2 = t_1;
        other_t_1 = other_t_0;
    }
    float _S23 = optixGetRayTmin();
    uint _S24 = 2U * _S22;
    uint _S25 = _S24 + 1U;

    float h_t, test_t;
    uint h_i;
    for(int n_0=0;n_0<2;n_0++)
    {
        
        
        if(n_0 == int(0))
        {
            h_t = t_2;
            h_i = _S25;
        }
        else
        {
            h_t = other_t_1;
            h_i = _S24;
        }
        if(h_t > _S23)
        {
            uint _S26 = (F32_asuint((h_t)));
            uint _S27 = (optixGetPayload_0());
            test_t = (U32_asfloat((_S27)));
            
            
            if(h_t < test_t)
            {
                optixSetPayload_0((_S26));
                uint _S28 = (optixGetPayload_1());
                optixSetPayload_1((h_i));
                h_t = test_t;
                h_i = _S28;
            }
            uint _S29 = (F32_asuint((h_t)));
            uint _S30 = (optixGetPayload_2());
            test_t = (U32_asfloat((_S30)));
            
            
            if(h_t < test_t)
            {
                optixSetPayload_2((_S29));
                uint _S31 = (optixGetPayload_3());
                optixSetPayload_3((h_i));
                h_t = test_t;
                h_i = _S31;
            }
            uint _S32 = (F32_asuint((h_t)));
            uint _S33 = (optixGetPayload_4());
            test_t = (U32_asfloat((_S33)));
            
            
            if(h_t < test_t)
            {
                optixSetPayload_4((_S32));
                uint _S34 = (optixGetPayload_5());
                optixSetPayload_5((h_i));
                h_t = test_t;
                h_i = _S34;
            }
            uint _S35 = (F32_asuint((h_t)));
            uint _S36 = (optixGetPayload_6());
            test_t = (U32_asfloat((_S36)));
            
            
            if(h_t < test_t)
            {
                optixSetPayload_6((_S35));
                uint _S37 = (optixGetPayload_7());
                optixSetPayload_7((h_i));
                h_t = test_t;
                h_i = _S37;
            }
            uint _S38 = (F32_asuint((h_t)));
            uint _S39 = (optixGetPayload_8());
            test_t = (U32_asfloat((_S39)));
            
            
            if(h_t < test_t)
            {
                optixSetPayload_8((_S38));
                uint _S40 = (optixGetPayload_9());
                optixSetPayload_9((h_i));
                h_t = test_t;
                h_i = _S40;
            }
            uint _S41 = (F32_asuint((h_t)));
            uint _S42 = (optixGetPayload_10());
            test_t = (U32_asfloat((_S42)));
            
            
            if(h_t < test_t)
            {
                optixSetPayload_10((_S41));
                uint _S43 = (optixGetPayload_11());
                optixSetPayload_11((h_i));
                h_t = test_t;
                h_i = _S43;
            }
            uint _S44 = (F32_asuint((h_t)));
            uint _S45 = (optixGetPayload_12());
            test_t = (U32_asfloat((_S45)));
            
            
            if(h_t < test_t)
            {
                optixSetPayload_12((_S44));
                uint _S46 = (optixGetPayload_13());
                optixSetPayload_13((h_i));
                h_t = test_t;
                h_i = _S46;
            }
            uint _S47 = (F32_asuint((h_t)));
            uint _S48 = (optixGetPayload_14());
            test_t = (U32_asfloat((_S48)));
            
            
            if(h_t < test_t)
            {
                optixSetPayload_14((_S47));
                uint _S49 = (optixGetPayload_15());
                optixSetPayload_15((h_i));
                h_t = test_t;
                h_i = _S49;
            }
            uint _S50 = (F32_asuint((h_t)));
            uint _S51 = (optixGetPayload_16());
            test_t = (U32_asfloat((_S51)));
            
            
            if(h_t < test_t)
            {
                optixSetPayload_16((_S50));
                uint _S52 = (optixGetPayload_17());
                optixSetPayload_17((h_i));
                h_t = test_t;
                h_i = _S52;
            }
            uint _S53 = (F32_asuint((h_t)));
            uint _S54 = (optixGetPayload_18());
            test_t = (U32_asfloat((_S54)));
            
            
            if(h_t < test_t)
            {
                optixSetPayload_18((_S53));
                uint _S55 = (optixGetPayload_19());
                optixSetPayload_19((h_i));
                h_t = test_t;
                h_i = _S55;
            }
            uint _S56 = (F32_asuint((h_t)));
            uint _S57 = (optixGetPayload_20());
            test_t = (U32_asfloat((_S57)));
            
            
            if(h_t < test_t)
            {
                optixSetPayload_20((_S56));
                uint _S58 = (optixGetPayload_21());
                optixSetPayload_21((h_i));
                h_t = test_t;
                h_i = _S58;
            }
            uint _S59 = (F32_asuint((h_t)));
            uint _S60 = (optixGetPayload_22());
            test_t = (U32_asfloat((_S60)));
            
            
            if(h_t < test_t)
            {
                optixSetPayload_22((_S59));
                uint _S61 = (optixGetPayload_23());
                optixSetPayload_23((h_i));
                h_t = test_t;
                h_i = _S61;
            }
            uint _S62 = (F32_asuint((h_t)));
            uint _S63 = (optixGetPayload_24());
            test_t = (U32_asfloat((_S63)));
            
            
            if(h_t < test_t)
            {
                optixSetPayload_24((_S62));
                uint _S64 = (optixGetPayload_25());
                optixSetPayload_25((h_i));
                h_t = test_t;
                h_i = _S64;
            }
            uint _S65 = (F32_asuint((h_t)));
            uint _S66 = (optixGetPayload_26());
            test_t = (U32_asfloat((_S66)));
            
            
            if(h_t < test_t)
            {
                optixSetPayload_26((_S65));
                uint _S67 = (optixGetPayload_27());
                optixSetPayload_27((h_i));
                h_t = test_t;
                h_i = _S67;
            }
            uint _S68 = (F32_asuint((h_t)));
            uint _S69 = (optixGetPayload_28());
            test_t = (U32_asfloat((_S69)));
            
            
            if(h_t < test_t)
            {
                optixSetPayload_28((_S68));
                uint _S70 = (optixGetPayload_29());
                optixSetPayload_29((h_i));
                h_t = test_t;
                h_i = _S70;
            }
            uint _S71 = (F32_asuint((h_t)));
            uint _S72 = (optixGetPayload_30());
            if(h_t < (U32_asfloat((_S72))))
            {
                optixSetPayload_30((_S71));
                uint _S73 = (optixGetPayload_31());
                optixSetPayload_31((h_i));
            }
        }
    }
    uint _S74 = (optixGetPayload_30());
    if(t_2 < (U32_asfloat((_S74))))
    {
        optixIgnoreIntersection();
    }
    return;
}
