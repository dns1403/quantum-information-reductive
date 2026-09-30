"""Independent exact finite A2 verification in the faithful left regular model.

No Hecke basis truncation occurs: all six elements of S3 are represented.
This script is independent of the two-dimensional symbolic certificate.
Run: uv run python scripts/verify_hecke.py
"""
from collections import deque
from itertools import permutations
import json
from pathlib import Path
import sympy as sp


def compose(a, b):
    return tuple(a[b[i]] for i in range(3))


def length(w):
    return sum(w[i] > w[j] for i in range(3) for j in range(i + 1, 3))


def regular_model(u):
    identity = (0, 1, 2)
    generators = [(1, 0, 2), (0, 2, 1)]
    elements = sorted(permutations(range(3)), key=lambda w: (length(w), w))
    index = {w: i for i, w in enumerate(elements)}
    matrices = []
    for s in generators:
        matrix = sp.zeros(6)
        for w in elements:
            sw = compose(s, w)
            matrix[index[sw], index[w]] = 1
            if length(sw) < length(w):
                matrix[index[w], index[w]] = u - 1 / u
        matrices.append(matrix)
    words = {identity: ()}
    queue = deque([identity])
    while queue:
        w = queue.popleft()
        for j, s in enumerate(generators):
            ws = compose(w, s)
            if ws not in words:
                words[ws] = words[w] + (j,)
                queue.append(ws)
    basis = []
    for w in elements:
        x = sp.eye(6)
        for j in words[w]:
            x = x * matrices[j]
        basis.append(x)
        assert x[:, 0] == sp.eye(6)[:, index[w]]
    return elements, matrices, basis


def main():
    u, r = sp.Integer(2), sp.Rational(3, 4)
    q = u**2
    elements, (s, t), basis = regular_model(u)
    identity = sp.eye(6)
    assert s == s.T and t == t.T
    assert s*s == identity + (u - 1/u)*s
    assert t*t == identity + (u - 1/u)*t
    assert s*t*s == t*s*t
    # tau is the identity coefficient, not the normalized matrix trace.
    for i, a in enumerate(basis):
        for j, b in enumerate(basis):
            assert (a.T*b)[0, 0] == int(i == j)
    denominator = (1+q)*(1+q+q*q)
    projection = sum((u**length(w)*a for w, a in zip(elements, basis)), sp.zeros(6))/denominator
    damped = sum(((u*r)**length(w)*a for w, a in zip(elements, basis)), sp.zeros(6))/denominator
    assert projection == projection.T
    assert projection*projection == projection
    assert s*projection == u*projection and t*projection == u*projection
    assert damped == damped.T
    eigenvalues = damped.eigenvals()
    assert eigenvalues[sp.Rational(-1, 210)] == 2
    witness = (damped + identity/210).nullspace()[0]
    quotient = sp.factor((witness.T*damped*witness)[0]/(witness.T*witness)[0])
    assert quotient == -sp.Rational(1, 210)
    # Faithful rank-one algebra: r id +(1-r)tau(.)1, all q>0, r in [0,1].
    # Check its classical transition matrix exactly at extreme unequal weights.
    for q0 in map(sp.Rational, ["1/100", "1/4", "1", "4", "100"]):
        weights = sp.Matrix([[1/(1+q0), q0/(1+q0)]])
        for r0 in [sp.Integer(0), sp.Rational(1, 2), sp.Integer(1)]:
            markov = r0*sp.eye(2)+(1-r0)*sp.ones(2, 1)*weights
            assert all(x >= 0 for x in markov)
            assert markov*sp.ones(2, 1) == sp.ones(2, 1)
            assert weights*markov == weights
    # Explicit modular averaging obstruction for C_p: N^2=pN.
    for prime in [2, 3, 5, 7]:
        cyclic_sum = sp.ones(prime)
        assert cyclic_sum*cyclic_sum == prime*cyclic_sum
        assert all(x % prime == 0 for x in cyclic_sum*cyclic_sum)
        assert any(x % prime != 0 for x in cyclic_sum)
    result = {
        "status": "PASS", "model": "full H_4(S3), faithful 6-dimensional left regular representation",
        "arithmetic": "exact rational; no numeric tolerance", "q": 4, "r": "3/4",
        "input_projection_rank": projection.rank(),
        "output_eigenvalues_multiplicity": {str(k): v for k, v in eigenvalues.items()},
        "negative_rayleigh_quotient": str(quotient),
        "exact_witness_in_ordered_basis": [str(x) for x in witness],
        "basis_permutations": [list(w) for w in elements],
        "trace_input": str(projection[0, 0]), "trace_output": str(damped[0, 0]),
        "rank_one_parameter_checks": 15, "modular_averaging_checks": 4,
    }
    out = Path("scripts/results")
    out.mkdir(exist_ok=True)
    (out/"hecke_exact.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
