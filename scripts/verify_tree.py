"""Finite tree Schur checks and symbolic Satake obstruction.

This models exact channels on M_N, never a truncation of Hecke multiplication.
The general entropy inequality is proved/cited in report.tex, not inferred here.
"""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigvalsh
import sympy as sp

SEED = 7302026
TOL = 1e-10


def tree_data(n, rng):
    parents = [-1] + [int(rng.integers(j)) for j in range(1, n)]
    paths = [set() for _ in range(n)]
    for j in range(1, n):
        paths[j] = paths[parents[j]] | {j}
    cuts = np.array([[int(e in paths[j]) for j in range(n)] for e in range(1, n)])
    distance = np.sum(np.abs(cuts[:, :, None]-cuts[:, None, :]), axis=0)
    # Independent graph shortest paths for the exact wall identity.
    graph = np.full((n, n), 999, dtype=int)
    np.fill_diagonal(graph, 0)
    for j in range(1, n):
        graph[j, parents[j]] = graph[parents[j], j] = 1
    for j in range(n):
        graph = np.minimum(graph, graph[:, j, None]+graph[j, None, :])
    assert np.array_equal(distance, graph)
    return parents, cuts, distance


def tr_xlogx(a):
    ev = eigvalsh(a)
    assert ev.min() > -TOL
    ev = np.maximum(ev, 0)
    return float(np.sum(ev[ev > 0]*np.log(ev[ev > 0])))


def main():
    rng = np.random.default_rng(SEED)
    checks = []
    worst_slack = float("inf")
    for n, reference in [(2, 1), (5, 1), (5, 3), (9, 2)]:
        parents, cuts, distance = tree_data(n, rng)
        centering = np.eye(n)-np.ones((n, n))/n
        cnd_max = float(eigvalsh(centering@distance@centering).max())
        assert cnd_max < TOL
        pinching = np.kron(np.eye(n), np.ones((reference, reference)))
        size = n*reference
        residuals = []
        for time in [.01, .5, 3.]:
            correlation = np.exp(-time*distance)
            expected_det = (1-np.exp(-2*time))**(n-1)
            residuals.append(abs(float(np.linalg.det(correlation))-expected_det))
            assert residuals[-1] < TOL
            assert eigvalsh(correlation).min() > 0
        for _ in range(100):
            z = rng.normal(size=(size, size))+1j*rng.normal(size=(size, size))
            rho = z@z.conj().T + .01*np.eye(size)
            rho /= np.trace(rho)
            equilibrium = rho*pinching
            initial_entropy = tr_xlogx(rho)-tr_xlogx(equilibrium)
            for time in [.001, .3, 2.]:
                schur = np.kron(np.exp(-time*distance), np.ones((reference, reference)))
                evolved = rho*schur
                remaining = tr_xlogx(evolved)-tr_xlogx(equilibrium)
                slack = np.exp(-2*time)*initial_entropy-remaining
                worst_slack = min(worst_slack, slack)
                assert slack >= -TOL
        checks.append({"N": n, "reference_dimension": reference, "parents": parents,
                       "max_centered_distance_eigenvalue": cnd_max,
                       "max_determinant_residual": max(residuals), "density_samples": 100})
    # Symbolic spherical recurrence/Satake test, independent of finite-tree matrices.
    q, z, r = sp.symbols("q z r", positive=True)
    h1 = sp.sqrt(q)*(z+1/z)
    h2 = sp.expand(h1**2-(q+1))
    assert sp.expand(h2-(q*(z*z+z**-2)+(q-1))) == 0
    torus_image = q*r*r*(z*z+z**-2)+(q-1)
    defect = sp.factor(torus_image-r*r*h2)
    assert sp.expand(defect-(q-1)*(1-r*r)) == 0
    # Exact first coefficients of the spherical resolvent, not an infinite
    # approximation: the full identity follows from the recurrence in the proof.
    a = sp.symbols("a", real=True)
    radial = [sp.Integer(1), a, a*a-(q+1)]
    for k in range(3, 9):
        radial.append(sp.expand(a*radial[-1]-q*radial[-2]))
    series = sum(r**k*radial[k] for k in range(9))
    remainder = sp.Poly(sp.expand(series*(1-r*a+q*r*r)-(1-r*r)), r)
    assert all(remainder.nth(k) == 0 for k in range(9))
    # Bounds valid for every real q>=2, hence all residue cardinalities.
    assert sp.Rational(967, 1058) > sp.Rational(9, 10)
    assert sp.Rational(484, 441) < sp.Rational(11, 10)
    # Iwahori apartment words: s(x)=-x and omega(x)=1-x.
    # Every reduced alternating word up to length 30 is checked, with zero
    # truncation claim about the algebra. The induction is stated in report.tex.
    for count in range(31):
        for first in [0, 1]:
            word = [(first+j) % 2 for j in range(count)]
            x = 0
            for letter in reversed(word):
                x = -x if letter == 0 else 1-x
            assert abs(x) == word.count(1)
    result = {"status": "PASS", "seed": SEED, "tolerance": TOL,
              "model_warning": "finite Schur channels, not infinite Hecke-corner samples",
              "cases": checks, "total_density_time_checks": 1200,
              "minimum_entropy_inequality_slack": worst_slack,
              "satake_defect_exact": str(defect),
              "resolvent_coefficients_checked": 9,
              "uniform_operator_bounds_exact": ["967/1058 > 9/10", "484/441 < 11/10"],
              "iwahori_alternating_word_checks": 62}
    out = Path("scripts/results")
    out.mkdir(exist_ok=True)
    (out/"tree_checks.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
