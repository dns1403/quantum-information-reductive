"""Exact checks for the diagonal-expectation obstruction in affine A2.

The affine permutation model is infinite. Finite balls select input words
only; Hecke products retain ALL output terms. No truncated Hecke algebra is
used. The global theorems are proved in scripts/results/affine_phase2.txt.

Run: uv run python scripts/verify_affine_expectations.py
"""

from collections import Counter, deque
from fractions import Fraction
from functools import cache
from itertools import combinations
import json
from pathlib import Path


IDENTITY = (1, 2, 3)
GENERATORS = ((0, 2, 4), (2, 1, 3), (1, 3, 2))


def apply(w, k):
    """Affine permutation w(k+3)=w(k)+3, given its window w(1..3)."""
    period, position = divmod(k - 1, 3)
    return w[position] + 3 * period


def compose(w, v):
    return tuple(apply(w, k) for k in v)


def inverse(w):
    result = []
    for k in range(1, 4):
        for position, value in enumerate(w, start=1):
            if (k - value) % 3 == 0:
                result.append(position + k - value)
                break
    return tuple(result)


def length(w):
    return sum(abs((w[j] - w[i]) // 3) for i in range(3) for j in range(i + 1, 3))


@cache
def reduced_word(w):
    if w == IDENTITY:
        return ()
    for s, generator in enumerate(GENERATORS):
        v = compose(generator, w)
        if length(v) == length(w) - 1:
            return (s,) + reduced_word(v)
    raise AssertionError(f"No descent at {w}")


def ball(radius, allowed=(0, 1, 2)):
    words = {IDENTITY}
    queue = deque([IDENTITY])
    while queue:
        w = queue.popleft()
        if length(w) == radius:
            continue
        for s in allowed:
            v = compose(w, GENERATORS[s])
            if v not in words:
                words.add(v)
                queue.append(v)
    return sorted(words, key=lambda w: (length(w), w))


def left_generator(s, expansion, p):
    result = {}
    for w, value in expansion.items():
        sw = compose(GENERATORS[s], w)
        result[sw] = result.get(sw, 0) + value
        if length(sw) < length(w):
            result[w] = result.get(w, 0) + p * value
    return {w: value for w, value in result.items() if value != 0}


def left_word(word, expansion, p):
    for s in reversed(word):
        expansion = left_generator(s, expansion, p)
    return expansion


def multiply(w, v, p):
    return left_word(reduced_word(w), {v: Fraction(1)}, p)


def add(first, second, scale=1):
    result = dict(first)
    for w, value in second.items():
        result[w] = result.get(w, 0) + scale * value
    return {w: value for w, value in result.items() if value != 0}


def subsets(sequence):
    return [frozenset(c) for n in range(len(sequence) + 1) for c in combinations(sequence, n)]


def finite_parabolic_classification(p):
    """Exhaust all 32 unital basis subsets of the TRUE six-dimensional A2."""
    elements = ball(3, allowed=(1, 2))
    assert len(elements) == 6
    nonidentity = [w for w in elements if w != IDENTITY]
    products = {(w, v): multiply(w, v, p) for w in elements for v in elements}
    assert all(set(value) <= set(elements) for value in products.values())
    closed = []
    for subset in subsets(nonidentity):
        candidate = subset | {IDENTITY}
        if not all(inverse(w) in candidate for w in candidate):
            continue
        if all(set(products[w, v]) <= candidate for w in candidate for v in candidate):
            closed.append(candidate)
    standard = [frozenset(ball(3, allowed=tuple(j))) for j in subsets((1, 2))]
    if p != 0:
        assert set(closed) == set(standard)
    else:
        assert len(closed) == 6
        three_cycle = compose(GENERATORS[1], GENERATORS[2])
        nonstandard = frozenset((IDENTITY, three_cycle, inverse(three_cycle)))
        assert nonstandard in closed and nonstandard not in standard
    return len(closed)


def main():
    words = ball(8)
    counts = Counter(map(length, words))
    assert counts == {0: 1, **{n: 3 * n for n in range(1, 9)}}
    for w in words:
        assert compose(w, inverse(w)) == IDENTITY
        assert compose(inverse(w), w) == IDENTITY
        rebuilt = IDENTITY
        for s in reduced_word(w):
            rebuilt = compose(rebuilt, GENERATORS[s])
        assert rebuilt == w
        assert len(reduced_word(w)) == length(w)
        if length(w) >= 4:
            assert set(reduced_word(w)) == {0, 1, 2}

    descent_checks = 0
    relation_checks = 0
    largest_output_length = 0
    parameter_results = []
    for p in (Fraction(3, 2), Fraction(-3, 2), Fraction(0)):
        # q=4, q=1/4, and q=1, respectively; all arithmetic exact.
        for w in words:
            source = {w: Fraction(1)}
            for s in range(3):
                single = left_generator(s, source, p)
                double = left_generator(s, single, p)
                assert double == add(source, single, p)
                # (T_s-p)T_s=1, checked as an operator on each delta_w.
                assert add(double, single, -p) == source
                relation_checks += 2
            for s, t in combinations(range(3), 2):
                assert left_word((s, t, s), source, p) == left_word((t, s, t), source, p)
                relation_checks += 1
            product = multiply(w, inverse(w), p)
            assert product.get(IDENTITY, 0) == 1
            largest_output_length = max(largest_output_length, max(map(length, product)))
            for s in range(3):
                expected = p if length(compose(GENERATORS[s], w)) < length(w) else 0
                assert product.get(GENERATORS[s], 0) == expected
                descent_checks += 1
        parameter_results.append({
            "p": str(p),
            "finite_A2_unital_basis_subalgebras": finite_parabolic_classification(p),
        })

    # A concrete failure of the nonstandard q=1 subgroup at q!=1.
    st = compose(GENERATORS[1], GENERATORS[2])
    sts = compose(st, GENERATORS[1])
    p = Fraction(3, 2)
    assert multiply(st, inverse(st), p) == {
        IDENTITY: Fraction(1), GENERATORS[1]: p, sts: p
    }

    support_checks = 0
    for support in subsets((0, 1, 2)):
        for j in subsets((0, 1, 2)):
            assert int(not support <= j) <= sum(s not in j for s in support)
            support_checks += 1

    # A proper function passing ALL finite standard-parabolic cone tests,
    # yet incompatible with the multiplicative-domain consequence of fixing
    # all generators. Exact time t=log(2), not a numerical optimization.
    candidate=lambda w: max(length(w)-3,0)
    for j in subsets((0,1,2)):
        if len(j)<3:
            assert all(candidate(w)==0 for w in ball(3,allowed=tuple(j)))
    w4=next(w for w in words if length(w)==4)
    assert all(candidate(g)==0 for g in GENERATORS)
    assert Fraction(1,2)**candidate(w4)==Fraction(1,2)

    result = {
        "status": "PASS",
        "model": "full affine A2 affine permutations and exact faithful Hecke multiplication",
        "warning": "finite inputs are verification only; no output terms are truncated",
        "arithmetic": "exact fractions",
        "test_inputs": len(words),
        "max_input_length": 8,
        "max_encountered_product_length": largest_output_length,
        "words_at_each_length": dict(sorted(counts.items())),
        "operator_quadratic_inverse_braid_checks": relation_checks,
        "descent_coefficient_checks": descent_checks,
        "support_union_bound_cases": support_checks,
        "local_tests_not_sufficient": "a(w)=max(length(w)-3,0) is proper and zero on every finite standard parabolic, but fixes all generators while damping length-four words",
        "parameters": parameter_results,
        "global_proof": "scripts/results/affine_phase2.txt",
        "conclusion": "all diagonal Markov projections are standard parabolic at q!=1; their positive generator cone has no proper finite-valued rates in finite rank",
        "not_concluded": "existence or nonexistence of other proper diagonal QMS on affine A2",
    }
    destination = Path("scripts/results/affine_expectations_exact.json")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
