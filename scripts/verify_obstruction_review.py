"""Independent exact review of the A2 obstruction and block-Choi conventions.

The algebra is the complete six-dimensional normalized H_q(S3), never a
truncation of an infinite algebra. General-parameter proofs are recorded in
scripts/results/review_phase2.txt; this script checks their algebraic identities.
Run: uv run python scripts/verify_obstruction_review.py
"""

import sympy as sp


def is_zero(matrix):
    return all(sp.simplify(entry) == 0 for entry in matrix)


def regular_generators(u):
    """Full multiplication tables in order e,s,t,st,ts,sts."""
    c = u - 1/u
    s = sp.zeros(6)
    t = sp.zeros(6)
    for column, row in enumerate([1, 0, 3, 2, 5, 4]):
        s[row, column] = 1
    for column in [1, 3, 5]:
        s[column, column] = c
    for column, row in enumerate([2, 4, 0, 5, 1, 3]):
        t[row, column] = 1
    for column in [2, 4, 5]:
        t[column, column] = c
    return s, t


def basis(s, t):
    return [sp.eye(s.rows), s, t, s*t, t*s, s*t*s]


def combination(coefficients, matrices):
    return sum((c*b for c, b in zip(coefficients, matrices)),
               sp.zeros(matrices[0].rows))


def main():
    u, r = sp.symbols("u r", positive=True)
    q, c = u**2, u-1/u
    lengths = [0, 1, 1, 2, 2, 3]
    normalizer = (1+q)*(1+q+q*q)

    # Faithful regular model: every T_w sends delta_e to a distinct basis vector.
    regular_s, regular_t = regular_generators(u)
    regular = basis(regular_s, regular_t)
    assert is_zero(regular_s**2-sp.eye(6)-c*regular_s)
    assert is_zero(regular_t**2-sp.eye(6)-c*regular_t)
    assert is_zero(regular_s*regular_t*regular_s-regular_t*regular_s*regular_t)
    assert sp.Matrix.hstack(*(b[:, 0] for b in regular)) == sp.eye(6)
    projection = combination([u**k for k in lengths], regular)/normalizer
    assert is_zero(projection-projection.T)
    assert is_zero(projection*projection-projection)
    assert sp.simplify(projection[0, 0]-1/normalizer) == 0
    # Descent extraction underlying the global fixed-basis/idempotent theorem.
    for generator, left_descents in [(regular_s, {1, 3, 5}),
                                     (regular_t, {2, 4, 5})]:
        for index, element in enumerate(regular):
            coefficient = (generator*element*element.T)[0, 0]
            assert sp.simplify(coefficient-(c if index in left_descents else 0)) == 0
        assert is_zero(generator*(generator-c*sp.eye(6))-sp.eye(6))

    # Reconstructed two-dimensional irreducible *-representation.
    s = sp.diag(u, -1/u)
    t = sp.Matrix([[-1/u, sp.sqrt(q*q+q+1)],
                   [sp.sqrt(q*q+q+1), u*q]])/(1+q)
    assert is_zero(s*s-sp.eye(2)-c*s)
    assert is_zero(t*t-sp.eye(2)-c*t)
    assert is_zero(s*t*s-t*s*t)
    matrix_basis = basis(s, t)
    involution = s+t-c*sp.eye(2)
    assert is_zero(involution*involution-sp.eye(2))
    image = combination([(u*r)**k for k in lengths], matrix_basis)/normalizer
    factor = (1-r)*(1+q*r)/normalizer
    assert is_zero(image-factor*(sp.eye(2)+r*u*involution))
    exact_image = image.subs({u: 2, r: sp.Rational(3, 4)}).applyfunc(sp.simplify)
    vector = sp.Matrix([-sp.sqrt(21)/7, 1])
    assert is_zero(exact_image*vector+vector/210)
    assert exact_image.eigenvals() == {-sp.Rational(1, 210): 1, sp.Rational(1, 42): 1}

    # Parameter inversion includes (-1)^length and is spatial in regular models.
    signs = sp.diag(*[(-1)**k for k in lengths])
    reciprocal_regular = basis(*regular_generators(1/u))
    for k, original, reciprocal in zip(lengths, regular, reciprocal_regular):
        assert is_zero(signs*reciprocal*signs-(-1)**k*original)
    sign_projection = combination([(-1/u)**k for k in lengths], regular)
    reciprocal_normalizer = normalizer.subs(u, 1/u)
    sign_projection /= reciprocal_normalizer
    assert is_zero(sign_projection*sign_projection-sign_projection)
    assert is_zero(sign_projection-sign_projection.T)
    sign_image = combination([(-r/u)**k for k in lengths], regular)/reciprocal_normalizer
    reciprocal_image = combination([(r/u)**k for k in lengths], reciprocal_regular)/reciprocal_normalizer
    assert is_zero(sign_image-signs*reciprocal_image*signs)
    assert -sp.Rational(1, 210) in sign_image.subs({u: sp.Rational(1, 2), r: sp.Rational(3, 4)}).eigenvals()
    # Endpoints certify only the displayed witness, not positivity of the map.
    assert is_zero(image.subs(r, 1))
    assert sp.factor(image.det().subs(r, 1/u)) == 0
    assert is_zero(image.subs(r, 0)-sp.eye(2)/normalizer)

    # Exact trace weights and domain-first Fourier/Choi inversion.
    irreps = [
        [sp.Matrix([[u**k]]) for k in lengths],
        [sp.Matrix([[(-1/u)**k]]) for k in lengths],
        matrix_basis,
    ]
    weights = [1/normalizer, q**3/normalizer, q/(1+q+q*q)]
    dimensions = [1, 1, 2]
    assert sp.simplify(sum(d*w for d, w in zip(dimensions, weights))-1) == 0
    for output, output_basis in enumerate(irreps):
        for source, source_basis in enumerate(irreps):
            size = dimensions[output]*dimensions[source]
            choi = weights[source]*sum(
                (sp.kronecker_product(b.conjugate(), a)
                 for b, a in zip(source_basis, output_basis)), sp.zeros(size))
            expected = sp.zeros(size)
            if output == source:
                omega = sp.Matrix([int(i == j) for i in range(dimensions[source])
                                   for j in range(dimensions[source])])
                expected = omega*omega.T
            assert is_zero(choi-expected)

    # A generic Hermitian matrix verifies the conditional-Choi completion formula.
    entries = iter(sp.symbols("h:16", real=True))
    generic = sp.zeros(4)
    for i in range(4):
        generic[i, i] = next(entries)
    for i in range(4):
        for j in range(i+1, 4):
            real, imag = next(entries), next(entries)
            generic[i, j] = real+sp.I*imag
            generic[j, i] = real-sp.I*imag
    omega = sp.Matrix([1, 0, 0, 1])
    orthogonal_projection = sp.eye(4)-omega*omega.T/2
    remainder_vector = generic*omega/2-omega*(omega.T*generic*omega)[0]/8
    remainder = omega*remainder_vector.conjugate().T+remainder_vector*omega.T
    assert is_zero(generic-orthogonal_projection*generic*orthogonal_projection-remainder)
    coordinates = sp.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0]])
    assert coordinates.T*coordinates == sp.diag(2, 1, 1)

    # Independent exact symmetric-rate cone and sharp depolarizing repair.
    a, b, d, h, spectral = sp.symbols("a b d h spectral", real=True)
    symmetric_rates = [0, a, a, b, b, d]

    def generator_choi(output, source):
        return -weights[source]*sum(
            (rate*sp.kronecker_product(input_matrix.conjugate(), output_matrix)
             for rate, input_matrix, output_matrix in
             zip(symmetric_rates, irreps[source], irreps[output])),
            sp.zeros(dimensions[source]*dimensions[output]))

    plus_mean = a*(1-q)+b*q
    plus_difference = u*(a+b*(q-1)-d*q)
    minus_mean = a*(q-1)+b
    minus_difference = (a*q+b*(1-q)-d)/u
    assert is_zero(generator_choi(2, 0)-weights[0]*(plus_mean*sp.eye(2)-plus_difference*involution))
    assert is_zero(generator_choi(2, 1)-weights[1]/q*(minus_mean*sp.eye(2)+minus_difference*involution))
    for scalar in [0, 1]:
        assert is_zero(generator_choi(scalar, 2)
                       -weights[2]/weights[scalar]*generator_choi(2, scalar))
    first = 2*a-2*b+d
    second = a+b-d
    third = (q+1/q-1)*(b-a)+d
    assert is_zero(generator_choi(0, 1)-sp.Matrix([[weights[1]*first]]))
    assert is_zero(generator_choi(1, 0)-sp.Matrix([[weights[0]*first]]))
    orthonormal = sp.Matrix([[1, 0, 0], [0, 1, 1],
                            [0, 1, -1], [-1, 0, 0]])/sp.sqrt(2)
    assert orthonormal.T*orthonormal == sp.eye(3)
    compressed = orthonormal.T*generator_choi(2, 2)*orthonormal
    characteristic = (spectral*sp.eye(3)-compressed).det()
    expected_characteristic = sp.prod(spectral-weights[2]*value
                                      for value in [first, second, third])
    assert sp.factor(characteristic-expected_characteristic) == 0
    repair = {a: 1+h, b: 2+h, d: 3+h}
    for expression, expected in [
        (first, 1+h), (second, h), (third, q+1/q+2+h),
        (plus_mean, 1+q+h), (plus_difference, -u*(1+q)),
        (minus_mean, 1+q+q*h), (minus_difference, -(1+q)/u),
    ]:
        assert sp.simplify(expression.subs(repair)-expected) == 0
    threshold_q_ge_one = (u-1)*(1+q)
    threshold_q_le_one = (1/u-1)*(1+1/q)
    assert threshold_q_ge_one.subs(u, 2) == 5
    assert sp.simplify(threshold_q_ge_one.subs(u, sp.sqrt(2))-3*(sp.sqrt(2)-1)) == 0
    assert threshold_q_le_one.subs(u, sp.Rational(1, 2)) == 5

    print("PASS: symbolic regular projection, irreducible factorization, signed parameter inversion")
    print("PASS: exact q=4 and q=1/4 witness eigenvalue -1/210; witness endpoint identities")
    print("PASS: trace weights, all nine identity-Choi blocks, generic conditional-Choi completion")
    print("PASS: all seven symmetric-rate inequalities and sharp repair branches; h(4)=h(1/4)=5")
    print("PASS: all symbolic finite A2 descent coefficients and generator inverse identities")
    print("PROOF SCOPE: finite exact identities support the written proofs; no affine truncation tested")


if __name__ == "__main__":
    main()
