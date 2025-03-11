// Helper function to compute the maximum pairwise distance among four points
[Differentiable]
inline double maxDistanceAmongPoints(const double3 p1,
                                     const double3 p2,
                                     const double3 p3,
                                     const double3 p4)
{
    double dAB = length(p2 - p1);
    double dAC = length(p3 - p1);
    double dAD = length(p4 - p1);
    double dBC = length(p3 - p2);
    double dBD = length(p4 - p2);
    double dCD = length(p4 - p3);
    return max(dAB, max(dAC, max(dAD, max(dBC, max(dBD, dCD)))));
}

[Differentiable]
double power_of_circumsphere(
    double3 A, double3 B, double3 C, double3 D,
    double3 P)
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

    if (fabs(denominator) < 1e-10)
    {
        // Check if the points form a "very small" tetrahedron
        double maxDist = maxDistanceAmongPoints(A, B, C, D);

        // If the primitive is extremely small, use the mean and minimal distance logic
        if (maxDist < 1e-7)
        {
            center = 0.25 * (A + B + C + D);

            double distA = length(A - center);
            double distB = length(B - center);
            double distC = length(C - center);
            double distD = length(D - center);

            // Use the minimum distance from center to a corner
            radius = min(distA, min(distB, min(distC, distD)));

            double3 diff = center - P;
            return dot(diff, diff) - radius * radius;
        }
        else
        {
            return -1e20;
            // Degenerate case: the points are nearly coplanar but not tiny
            // Fit a plane to the first three vertices A, B, C
            double3 plane_normal = cross(B - A, C - A);
            plane_normal = normalize(plane_normal);

            // Flip the normal so that the half-space contains the origin (0,0,0)
            if (dot(-A, plane_normal) < 0)
                plane_normal = -plane_normal;

            // Determine which side of the plane the point P lies on
            double side = dot(P - A, plane_normal);
            if (side > 0)
                return 1e20;
            else
                return -1e20;
        }
    }
    else
    {
        // Compute the relative circumcenter
        double3 relative_circumcenter = (
            aa * cross_bc +
            bb * cross_ca +
            cc * cross_ab
        ) / denominator;

        // Get the absolute position of the circumcenter and compute its radius
        center = A + relative_circumcenter;
        radius = length(relative_circumcenter);
    }

    // Return the power of the circumsphere
    double3 diff = center - P;
    return dot(diff, diff) - radius * radius;
}