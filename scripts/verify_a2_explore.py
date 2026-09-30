"""Exact certificates for failure of word-length positivity in H_q(S_3).

This is a genuine six-dimensional parabolic C*-subalgebra, not a truncation.
Normalized generators satisfy T_s^2 = 1 + (u-u^{-1}) T_s, q=u^2>0.
No floating-point positivity test is used for the mathematical certificate.
Run: uv run python scripts/verify_a2_explore.py
"""

from itertools import permutations
import json
from pathlib import Path

import sympy as sp


def zero(matrix):
    return all(sp.simplify(x) == 0 for x in matrix)


u, r, z = sp.symbols("u r z", positive=True)
q = u**2
c = u - 1/u
S = sp.diag(u, -1/u)
B = sp.sqrt(q*q+q+1)/(q+1)
T = sp.Matrix([[-1/(u*(q+1)), B], [B, u*q/(q+1)]])
assert zero(S*S-sp.eye(2)-c*S)
assert zero(T*T-sp.eye(2)-c*T)
assert zero(S*T*S-T*S*T)
assert S == S.T and T == T.T

numerator = sp.eye(2) + u*r*(S+T) + u*u*r*r*(S*T+T*S) + u**3*r**3*S*T*S
P = (1+q)*(1+q+q*q)
eig_minus = (1-r)*(1+q*r)*(1-u*r)
eig_plus = (1-r)*(1+q*r)*(1+u*r)
assert sp.factor(numerator.trace()-eig_minus-eig_plus) == 0
assert sp.factor(numerator.det()-eig_minus*eig_plus) == 0
assert sp.factor((z*sp.eye(2)-numerator).det()-(z-eig_minus)*(z-eig_plus)) == 0
assert zero(numerator.subs(r, 1))

# Independent exact regular representation at q=4 proves p_+ is positive:
# selfadjoint + idempotent in the faithful regular *-representation.
W = list(permutations(range(3)))
identity = tuple(range(3))
generators = [(1, 0, 2), (0, 2, 1)]
def mul(a, b):
    return tuple(a[b[i]] for i in range(3))
def length(a):
    return sum(a[i] > a[j] for i in range(3) for j in range(i+1, 3))
words = {identity: ()}
queue = [identity]
for w in queue:
    for i, s in enumerate(generators):
        v = mul(s, w)
        if v not in words:
            words[v] = (i,) + words[w]
            queue.append(v)
regular_generators = []
for s in generators:
    A = sp.zeros(6)
    for j, w in enumerate(W):
        v = mul(s, w)
        A[W.index(v), j] = 1
        if length(v) < length(w):
            A[j, j] += sp.Rational(3, 2)
    assert A == A.T
    regular_generators.append(A)
regular_basis = []
for w in W:
    A = sp.eye(6)
    for i in words[w]:
        A = A * regular_generators[i]
    regular_basis.append(A)
projection = sum((2**length(w)*A for w, A in zip(W, regular_basis)), sp.zeros(6))/105
assert projection == projection.T
assert projection*projection == projection
assert projection.rank() == 1

example = sp.simplify((numerator/P).subs({u: 2, r: sp.Rational(3, 4)}))
v = sp.Matrix([-sp.sqrt(21)/7, 1])
assert zero(example*v + v/210)
assert set(example.eigenvals()) == {sp.Rational(-1, 210), sp.Rational(1, 42)}
regular_image = sum(((sp.Rational(3, 2))**length(w)*A for w, A in zip(W, regular_basis)), sp.zeros(6))/105
assert set(regular_image.eigenvals()) >= {sp.Rational(-1, 210), sp.Rational(1, 42)}

# q -> q^{-1}: T_w(q^{-1}) -> (-1)^ell(w) T_w(q) is *-isomorphism,
# preserving diagonal multipliers. Verifies its quadratic coefficient identity.
assert sp.simplify((1/u-u)+c) == 0

result = {
    "status": "PASS",
    "arithmetic": "exact SymPy identities; no numerical tolerance",
    "finite_algebra": "H_q(S_3), genuine parabolic subalgebra of affine A2",
    "parameter": "q=u^2>0",
    "projection_normalizer": str(sp.factor(P)),
    "image_eigenvalues": [str(eig_minus/P), str(eig_plus/P)],
    "failure_for_q_gt_1": "1/sqrt(q) < r < 1, or 0<t<(log q)/2",
    "failure_for_q_ne_1": "0<t<abs(log q)/2 via sign twist",
    "q4_r3over4_eigenvalues": ["-1/210", "1/42"],
    "q4_projection": "exact selfadjoint rank-one projection in regular 6x6 model",
}
Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/a2_certificate.json").write_text(json.dumps(result, indent=2)+"\n")
Path("scripts/results").mkdir(exist_ok=True)
Path("scripts/results/a2_symbolic.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))
