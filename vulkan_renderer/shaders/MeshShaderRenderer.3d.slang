import MathUtils;
import Quaternion;
import TetrahedronScene;
import SortUtils;
import PackedTypes;

#include <Bitfield.h>

using namespace vkDelTet;
using namespace RoseEngine;

ParameterBlock<TetrahedronScene> scene;
StructuredBuffer<uint2> sortBuffer;
Buffer<float4> tetColors; // precomputed vertex SH

uniform float4x4 viewProjection;
uniform float3   rayOrigin;

struct v2f {
    float4 pos : SV_Position;

    [[vk::location(0)]] float3 rayDir;
    [[vk::location(1)]] float4 planeNumerators;
    [[vk::location(2)]] float4 planeDenominators;

    [[vk::location(3)]] nointerpolation float tetDensity;

    [[vk::location(4)]]  nointerpolation float3 c0;
    [[vk::location(5)]]  nointerpolation float3 c1;
    [[vk::location(6)]]  nointerpolation float3 c2;
    [[vk::location(7)]]  nointerpolation float3 c3;
    [[vk::location(8)]]  nointerpolation float3 v0;
    [[vk::location(9)]]  nointerpolation float3 v1;
    [[vk::location(10)]] nointerpolation float3 v2;
    [[vk::location(11)]] nointerpolation float3 v3;
};

#define GROUP_SIZE 32
#define TETS_PER_GROUP (GROUP_SIZE / 4)

#define softplus(x, beta) ( (1.0 / beta) * log(1.0 + exp(beta*x)) )

[shader("vertex")]
v2f vsmain(
    uint sv_vertexID : SV_VertexID
) {

    uint threadId = sv_vertexID / 4;
    uint tetVertexId = sv_vertexID % 4;

    v2f o = {};
    o.pos = 0;

    if (threadId < scene.numTets)
    {
        const uint tetId = sortBuffer[threadId].y;

        o.tetDensity = scene.load_tet_density(tetId);
        float3 tet_color = tetColors.Load(tetId).rgb;
        const uint vertexId = scene.load_index(tetId, tetVertexId);
        const float3 vertex = scene.load_vertex(vertexId);
        o.pos = mul(viewProjection, float4(vertex, 1));
        o.rayDir = vertex - rayOrigin;

        const uint baseLaneId = (WaveGetLaneIndex() / 4) * 4;

        #define WaveReadTetVar(x, id)        WaveReadLaneAt(x, baseLaneId + id)
        #define WaveReadTetVarRel(x, offset) WaveReadTetVar(x, (WaveGetLaneIndex() + offset) % 4)

        const uint3 tri = TetrahedronScene::kTetTriangles[tetVertexId];

        // outward facing normal
        const float3 n = cross(
            WaveReadTetVar(vertex, tri[2]) - WaveReadTetVar(vertex, tri[0]),
            WaveReadTetVar(vertex, tri[1]) - WaveReadTetVar(vertex, tri[0]));

        const float num = dot(n, WaveReadTetVar(vertex, tri[0]) - rayOrigin);

        o.planeNumerators = float4(
            WaveReadTetVar(num, 0),
            WaveReadTetVar(num, 1),
            WaveReadTetVar(num, 2),
            WaveReadTetVar(num, 3) );
        o.planeDenominators = float4(
            dot(WaveReadTetVar(n, 0), o.rayDir),
            dot(WaveReadTetVar(n, 1), o.rayDir),
            dot(WaveReadTetVar(n, 2), o.rayDir),
            dot(WaveReadTetVar(n, 3), o.rayDir) );

        float3 color = 0;
        color += scene.load_tet_vertex_color(tetId, tetVertexId).rgb;
        color += tet_color;
        color = softplus(color, 10);

        o.c0 = WaveReadTetVar(color, 0);
        o.c1 = WaveReadTetVar(color, 1);
        o.c2 = WaveReadTetVar(color, 2);
        o.c3 = WaveReadTetVar(color, 3);

        o.v0 = WaveReadTetVar(vertex, 0);
        o.v1 = WaveReadTetVar(vertex, 1);
        o.v2 = WaveReadTetVar(vertex, 2);
        o.v3 = WaveReadTetVar(vertex, 3);
    }

    return o;
}

/*
[shader("mesh")]
[outputtopology("triangle")]
[numthreads(GROUP_SIZE, 1, 1)]
void meshmain(
    in uint3 threadId: SV_DispatchThreadID,
    OutputVertices<v2f, GROUP_SIZE>  vertices,
    OutputIndices<uint3, GROUP_SIZE> triangles) {
        
    SetMeshOutputCounts(GROUP_SIZE, GROUP_SIZE);

    const uint meshVertexId = threadId.x % GROUP_SIZE;
    const uint tetVertexId  = threadId.x % 4;

    v2f o = {};
    o.pos = 0;

    if (threadId.x/4 < scene.numTets)
    {
        const uint tetId = sortBuffer[threadId.x/4].y;

        o.tetDensity = scene.load_tet_density(tetId);
        float3 tet_color = tetColors.Load(tetId).rgb;
        if (o.tetDensity > densityThreshold)
        {
            const uint vertexId = scene.load_index(tetId, tetVertexId);
            const float3 vertex = scene.load_vertex(vertexId);
            o.pos = mul(viewProjection, float4(vertex, 1));
            o.rayDir = vertex - rayOrigin;

            const uint baseLaneId = (WaveGetLaneIndex() / 4) * 4;

            #define WaveReadTetVar(x, id)        WaveReadLaneAt(x, baseLaneId + id)
            #define WaveReadTetVarRel(x, offset) WaveReadTetVar(x, (WaveGetLaneIndex() + offset) % 4)

            const uint3 tri = TetrahedronScene::kTetTriangles[tetVertexId];

            // outward facing normal
            const float3 n = cross(
                WaveReadTetVar(vertex, tri[2]) - WaveReadTetVar(vertex, tri[0]),
                WaveReadTetVar(vertex, tri[1]) - WaveReadTetVar(vertex, tri[0]));

            const float num = dot(n, WaveReadTetVar(vertex, tri[0]) - rayOrigin);

            o.planeNumerators = float4(
                WaveReadTetVar(num, 0),
                WaveReadTetVar(num, 1),
                WaveReadTetVar(num, 2),
                WaveReadTetVar(num, 3) );
            o.planeDenominators = float4(
                dot(WaveReadTetVar(n, 0), o.rayDir),
                dot(WaveReadTetVar(n, 1), o.rayDir),
                dot(WaveReadTetVar(n, 2), o.rayDir),
                dot(WaveReadTetVar(n, 3), o.rayDir) );

            float3 color = 0;
            color += scene.load_tet_vertex_color(tetId, tetVertexId).rgb;
            color += tet_color;
            color = softplus(color, 10);

            o.c0 = WaveReadTetVar(color, 0);
            o.c1 = WaveReadTetVar(color, 1);
            o.c2 = WaveReadTetVar(color, 2);
            o.c3 = WaveReadTetVar(color, 3);

            o.v0 = WaveReadTetVar(vertex, 0);
            o.v1 = WaveReadTetVar(vertex, 1);
            o.v2 = WaveReadTetVar(vertex, 2);
            o.v3 = WaveReadTetVar(vertex, 3);
        }
    }

    vertices[meshVertexId] = o;
    triangles[meshVertexId] = (meshVertexId/4)*4 + TetrahedronScene::kTetTriangles[tetVertexId];
}
*/

[shader("fragment")]
float4 fsmain(v2f v) : SV_Target {
    // normalize ray direction
    {
        const float d = length(v.rayDir);
        v.rayDir /= d;
        v.planeDenominators /= d;
    }

    // intersect every plane

    float opticalDepth = v.tetDensity;

    float2 t = float2(0, FLT_MAX);
    [ForceUnroll]
    for (uint i = 0; i < 4; i++) {
        if (!intersect_plane_convex_hull(v.planeNumerators[i], v.planeDenominators[i], t)) {
            opticalDepth = 0;
            t = -1;
            break;
        }
    }
    if (t.y <= 0) return float4(0,0,0,1); // this shouldnt be possible
    opticalDepth *= t.y - max(t.x, 0);

    float T = exp(-opticalDepth);

    // interpolated color

    float3x3 invB = inverse(transpose(float3x3(v.v0 - v.v3, v.v1 - v.v3, v.v2 - v.v3)));
    float3 uvw_enter = mul(invB, (rayOrigin - v.v3) + t.x * v.rayDir);
    float3 uvw_exit  = mul(invB, (rayOrigin - v.v3) + t.y * v.rayDir);

    float  w_enter = saturate(1 - uvw_enter.x - uvw_enter.y - uvw_enter.z);
    float4 b_enter = float4(uvw_enter, w_enter);
    float3 c_enter = v.c0 * b_enter.x + v.c1 * b_enter.y + v.c2 * b_enter.z + v.c3 * b_enter.w;
    
    float  w_exit  = saturate(1 - uvw_exit.x -  uvw_exit.y -  uvw_exit.z);
    float4 b_exit = float4(uvw_exit,  w_exit);
    float3 c_exit = v.c0 * b_exit.x  + v.c1 * b_exit.y  + v.c2 * b_exit.z  + v.c3 * b_exit.w;

    float3 base_color = ((-opticalDepth*T + 1 - T)*c_exit + (opticalDepth-1 + T)*c_enter) / opticalDepth;

    return float4(base_color, T);
}
