// HLSL function to compute the power of the circumsphere of (A, B, C, D)
// with respect to point P, handling near-degenerate cases.
//
// Returns: (|P - O|^2 - R^2), where O,R is the circumsphere center & radius.
// If (A,B,C,D) is nearly degenerate, it falls back to a bounding sphere.
//
// Requires Shader Model 6.0+ for double types on many GPUs.
//
// Example usage in a Compute Shader or Pixel Shader function.

double3 CrossDouble(double3 a, double3 b)
{
    return double3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

double DotDouble(double3 a, double3 b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

// Return the absolute value of a double3
double3 AbsDouble3(double3 v)
{
    return double3(fabs(v.x), fabs(v.y), fabs(v.z));
}

// Simple helper to find distance squared
double DistSq(double3 p, double3 q)
{
    double3 d = p - q;
    return DotDouble(d, d);
}

// Compute the determinant of the 3x3 matrix formed by three double3 rows
double Determinant3x3(double3 r0, double3 r1, double3 r2)
{
    // | r0.x r0.y r0.z |
    // | r1.x r1.y r1.z |
    // | r2.x r2.y r2.z |
    return r0.x * (r1.y * r2.z - r1.z * r2.y)
         - r0.y * (r1.x * r2.z - r1.z * r2.x)
         + r0.z * (r1.x * r2.y - r1.y * r2.x);
}

// Invert a 3x3 matrix [rows: r0, r1, r2]. Outputs the inverse in inv0, inv1, inv2.
// Returns false if the determinant is near zero, true otherwise.
bool Invert3x3(
    double3 r0, double3 r1, double3 r2,
    out double3 inv0, out double3 inv1, out double3 inv2)
{
    double det = Determinant3x3(r0, r1, r2);
    const double eps = 1e-14; // Tweak threshold as needed

    if (fabs(det) < eps)
    {
        // Matrix is singular or near-singular.
        return false;
    }

    double invDet = 1.0 / det;

    // Adjoint / cofactor method
    inv0 = double3(
        (r1.y * r2.z - r1.z * r2.y) * invDet,
        (r0.z * r2.y - r0.y * r2.z) * invDet,
        (r0.y * r1.z - r0.z * r1.y) * invDet);

    inv1 = double3(
        (r1.z * r2.x - r1.x * r2.z) * invDet,
        (r0.x * r2.z - r0.z * r2.x) * invDet,
        (r0.z * r1.x - r0.x * r1.z) * invDet);

    inv2 = double3(
        (r1.x * r2.y - r1.y * r2.x) * invDet,
        (r0.y * r2.x - r0.x * r2.y) * invDet,
        (r0.x * r1.y - r0.y * r1.x) * invDet);

    return true;
}

// Compute volume * 6 = determinant(B-A, C-A, D-A)
double TetraVolume6(double3 A, double3 B, double3 C, double3 D)
{
    double3 AB = B - A;
    double3 AC = C - A;
    double3 AD = D - A;
    // The determinant is 6 * volume
    return Determinant3x3(AB, AC, AD);
}

// Fallback: returns center & radius of bounding sphere for points (A,B,C,D)
// by picking the farthest pair, then using midpoint and half-distance.
void BoundingSphereFor4Points(
    double3 A, double3 B, double3 C, double3 D,
    out double3 center, out double radius)
{
    // We'll brute force check all 6 pairs for the largest distance.
    double distAB = DistSq(A, B);
    double distAC = DistSq(A, C);
    double distAD = DistSq(A, D);
    double distBC = DistSq(B, C);
    double distBD = DistSq(B, D);
    double distCD = DistSq(C, D);

    // Track which pair is largest
    double bestDist = distAB;
    double3 bestP  = A, bestQ  = B;

    if (distAC > bestDist) { bestDist = distAC; bestP = A; bestQ = C; }
    if (distAD > bestDist) { bestDist = distAD; bestP = A; bestQ = D; }
    if (distBC > bestDist) { bestDist = distBC; bestP = B; bestQ = C; }
    if (distBD > bestDist) { bestDist = distBD; bestP = B; bestQ = D; }
    if (distCD > bestDist) { bestDist = distCD; bestP = C; bestQ = D; }

    // bestP, bestQ is the farthest pair
    center = 0.5 * (bestP + bestQ);
    radius = sqrt(bestDist) * 0.5;
}

// // Main function that returns the power of the circumsphere w.r.t. point P
// double power_of_circumsphere(
//     double3 A, double3 B, double3 C, double3 D,
//     double3 P)
// {
//     // 1) Check for near-degeneracy via tetra volume
//     double vol6 = TetraVolume6(A, B, C, D);
//     double absVol6 = fabs(vol6);
//     const double degenerateThreshold = 1e-12; // Adjust to your scale

//     if (absVol6 < degenerateThreshold)
//     {
//         // Fallback: bounding sphere for (A,B,C,D)
//         double3 center;
//         double radius;
//         BoundingSphereFor4Points(A, B, C, D, center, radius);

//         double distSqP  = DistSq(P, center);
//         double radSq    = radius * radius;
//         return distSqP - radSq; 
//     }
//     else
//     {
//         // 2) Non-degenerate: Solve for circumsphere center O via
//         //    (A-B)·O = (|A|^2 - |B|^2)/2
//         //    (A-C)·O = (|A|^2 - |C|^2)/2
//         //    (A-D)·O = (|A|^2 - |D|^2)/2

//         double a2 = DotDouble(A, A);
//         double b2 = DotDouble(B, B);
//         double c2 = DotDouble(C, C);
//         double d2 = DotDouble(D, D);

//         double3 Row0 = A - B;
//         double3 Row1 = A - C;
//         double3 Row2 = A - D;

//         double3 RHS;
//         RHS.x = 0.5 * (a2 - b2);
//         RHS.y = 0.5 * (a2 - c2);
//         RHS.z = 0.5 * (a2 - d2);

//         // Invert the 3x3
//         double3 inv0, inv1, inv2;
//         bool ok = Invert3x3(Row0, Row1, Row2, inv0, inv1, inv2);

//         if (!ok)
//         {
//             // If the matrix is near-singular here (which is odd if we passed the volume check),
//             // fall back to bounding sphere anyway:
//             double3 center;
//             double radius;
//             BoundingSphereFor4Points(A, B, C, D, center, radius);
//             return DistSq(P, center) - (radius * radius);
//         }

//         // Solve O = M^{-1} * RHS
//         double3 O;
//         O.x = DotDouble(inv0, RHS);
//         O.y = DotDouble(inv1, RHS);
//         O.z = DotDouble(inv2, RHS);

//         // Radius = distance(A, O)
//         // We can pick any of A,B,C,D; they should be the same ideally
//         double r = sqrt(DistSq(O, A));

//         // Finally, power = |P - O|^2 - r^2
//         double distSqP = DistSq(P, O);
//         double rSq      = r * r;
//         double power    = distSqP - rSq;

//         return power;
//     }
// }

[Differentiable]
double power_of_circumsphere(
    double3 A, double3 B, double3 C, double3 D,
    double3 P, )
{
    double3 center;
    double radius;
    // Compute vectors from A to other vertices
    double3 a = B - A;
    double3 b = C - A;
    double3 c = D - A;

    // Compute squares of lengths
    double aa = dot(a, a);
    double bb = dot(b, b);
    double cc = dot(c, c);

    // Compute cross products
    double3 cross_bc = cross(b, c);
    double3 cross_ca = cross(c, a);
    double3 cross_ab = cross(a, b);

    // Compute denominator for the circumcenter calculation
    double denominator = 2.0 * dot(a, cross_bc);

    if (abs(denominator) < 1e-7) {
        // Degenerate case: the points A, B, C, D are nearly coplanar
        // Fit a plane to the first three vertices A, B, C
        double3 plane_normal = cross(B - A, C - A);
        plane_normal = normalize(plane_normal);

        // Flip the normal so that the half-space contains the origin (0,0,0)
        // Check the sign of the distance from 0 to the plane defined by A and plane_normal
        if (dot(-A, plane_normal) < 0)
            plane_normal = -plane_normal;

        // Determine which side of the plane the point P lies on
        double side = dot(P - A, plane_normal);
        if (side > 0)
            return 1e20;
        else
            return -1e20;
    } else {
        // Compute the relative circumcenter position
        double3 relative_circumcenter = (
            aa * cross_bc +
                                         bb * cross_ca +
                                         cc * cross_ab
        ) / denominator;

        // Get the absolute position of the circumcenter and compute its radius
        center = A + relative_circumcenter;
        radius = length(relative_circumcenter);
    }

    double3 diff = center - P;
    return dot(diff, diff) - radius * radius;
}