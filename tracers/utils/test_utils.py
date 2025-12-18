from splinetracers import tetra_splinetracer
from splinetracers import quad

METHODS = [
    tetra_splinetracer,
]

SYM_METHODS = [
    tetra_splinetracer,
]

ALL_QUAD_PAIRS = [
    (tetra_splinetracer, quad.query_tetra),
]

QUAD_PAIRS = [
    (tetra_splinetracer, quad.query_tetra),
]

ALPHA_QUAD_PAIRS = [
]
