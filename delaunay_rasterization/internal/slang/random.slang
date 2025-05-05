// ---------- Simple PCG‑style RNG (32‑bit state) ----------
struct RNG { uint state; };

void rng_seed(out RNG rng, uint seed)          // seed must be ≠ 0
{
    rng.state = seed | 1u;                     // force odd to avoid zeros
}

uint rng_u32(inout RNG rng)                    // 32‑bit random integer
{
    // PCG‑XSH‑RR 32/16 variant.
    rng.state = rng.state * 747796405u + 2891336453u;
    uint word  = ((rng.state >> ((rng.state >> 28u) + 4u)) ^ rng.state)
                 * 277803737u;
    return (word >> 22u) ^ word;
}

float rng_f32(inout RNG rng)                   // uniform [0,1)
{
    // Mantissa‑fill trick: 23 random bits → IEEE‑754 float in [1,2)
    return asfloat(0x3F800000u | (rng_u32(rng) >> 9)) - 1.0;
}

uint hash_wang(uint v)
{
    v = (v ^ 61u) ^ (v >> 16);
    v *= 9u * 0x27d4eb2d;                      // 0x27d4eb2d = 6665569 * 0x10001
    v ^= v >> 15;
    return v;
}
