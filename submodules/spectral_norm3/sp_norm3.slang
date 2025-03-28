[Differentiable]
[AutoPyBindCUDA]
[CUDAKernel]
void sp_norm3(
    DiffTensorView matrices,
    DiffTensorView norm)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint prim_ind = dispatchIdx.x;
    if (prim_ind >= matrices.size(0)) return;

    // Load matrix elements
    float a00 = matrices[uint3(prim_ind, 0, 0)];
    float a01 = matrices[uint3(prim_ind, 0, 1)];
    float a02 = matrices[uint3(prim_ind, 0, 2)];
    float a10 = matrices[uint3(prim_ind, 1, 0)];
    float a11 = matrices[uint3(prim_ind, 1, 1)];
    float a12 = matrices[uint3(prim_ind, 1, 2)];
    float a20 = matrices[uint3(prim_ind, 2, 0)];
    float a21 = matrices[uint3(prim_ind, 2, 1)];
    float a22 = matrices[uint3(prim_ind, 2, 2)];

    // Compute symmetric matrix A^T A
    float b00 = a00*a00 + a10*a10 + a20*a20;
    float b01 = a00*a01 + a10*a11 + a20*a21;
    float b02 = a00*a02 + a10*a12 + a20*a22;
    float b11 = a01*a01 + a11*a11 + a21*a21;
    float b12 = a01*a02 + a11*a12 + a21*a22;
    float b22 = a02*a02 + a12*a12 + a22*a22;

    float p = b00 + b11 + b22;
    float q = b00*b11 + b11*b22 + b22*b00 - b01*b01 - b12*b12 - b02*b02;
    float r = b00*b11*b22 + 2.0f*b01*b12*b02 - b00*b12*b12 - b11*b02*b02 - b22*b01*b01;

    float m = (3.0f * q - p * p) / 3.0f;
    float n = (2.0f * p * p * p - 9.0f * p * q + 27.0f * r) / 27.0f;

    float discriminant = n * n * 0.25f + m * m * m / 27.0f;
    discriminant = min(discriminant, 0.0f);

    float sqrt_m = sqrt(-m / 3.0f);
    float phi = acos(min(max(-n / (2.0f * sqrt(-m * m * m / 27.0f)), -1.0f), 1.0f)) / 3.0f;

    float eig1 = p / 3.0f + 2.0f * sqrt_m * cos(phi);
    float eig2 = p / 3.0f + 2.0f * sqrt_m * cos(phi + 2.094395102f);  // +120 degrees
    float eig3 = p / 3.0f + 2.0f * sqrt_m * cos(phi + 4.188790205f);  // +240 degrees

    // float max_eig = max(eig1, max(eig2, eig3));
    float min_eig = min(eig1, min(eig2, eig3));

    norm[prim_ind] = sqrt(max(min_eig, 0.0f)); // numerical stability
    // norm[prim_ind] = sqrt(max(max_eig, 0.0f)); // numerical stability
}