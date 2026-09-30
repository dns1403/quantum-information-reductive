"""Exact block-Choi derivation for the full four-rate H_q(S3) cone.

Run from project root: uv run python scripts/verify_generator_cone.py
All algebraic checks use SymPy exactly. No numerical SDP is a certificate.
"""
import json
from pathlib import Path
import sympy as s

u = s.symbols('u', positive=True)
x, y, z, v = s.symbols('x y z v', real=True)
q = u**2
D = q*q+q+1
P = (1+q)*D
S = s.diag(u, -1/u)
T = s.Matrix([[-1/u, s.sqrt(D)], [s.sqrt(D), u*q]])/(1+q)
words = [(), (0,), (1,), (0,1), (1,0), (0,1,0)]
lengths = [len(w) for w in words]
rates = [0, x, y, z, z, v]
reps = [[], [], []]
for w in words:
    reps[0].append(s.Matrix([[u**len(w)]]))
    reps[1].append(s.Matrix([[(-1/u)**len(w)]]))
    B=s.eye(2)
    for i in w:
        B=B*[S,T][i]
    reps[2].append(B.applyfunc(s.simplify))
weights = [1/P, q**3/P, q/D]
dims = [1,1,2]

def simp(M):
    return M.applyfunc(s.factor)

def zero(M):
    return all(s.simplify(e)==0 for e in M)

def choi(alpha,beta,multipliers):
    # Domain first; entries Tr(B*Eij)=conjugate(B_ij).
    return simp(weights[beta]*sum((m*s.kronecker_product(Bb.conjugate(),Ba)
        for m,Bb,Ba in zip(multipliers,reps[beta],reps[alpha])),
        s.zeros(dims[alpha]*dims[beta])))

V = s.Matrix([[1,0,0],[0,1,0],[0,0,1],[-1,0,0]])
U = s.Matrix([[1,0,0],[0,1,1],[0,1,-1],[-1,0,0]])/s.sqrt(2)
blocks = {}
for alpha in range(3):
    for beta in range(3):
        Cid=choi(alpha,beta,[1]*6)
        Omega=s.Matrix([int(i==j) for i in range(dims[alpha]) for j in range(dims[alpha])])
        assert zero(Cid-(Omega*Omega.T if alpha==beta else s.zeros(dims[alpha]*dims[beta])))
        C=choi(alpha,beta,[-a for a in rates])
        assert zero(C-C.T)
        if alpha!=beta:
            blocks[f'{alpha}{beta}']=C
        elif alpha==2:
            blocks['22c']=simp(V.T*C*V)

# Remove positive trace factors and duplicate reverse block constraints.
Fplus=simp(blocks['02']/weights[2])
Fminus=simp(blocks['12']/weights[2])
K=simp(U.T*choi(2,2,[-a for a in rates])*U/weights[2])
assert zero(blocks['20']-weights[0]*Fplus)
assert zero(blocks['21']-weights[1]*Fminus)
assert s.factor(blocks['01'][0]/weights[1]-(x+y-2*z+v))==0
assert s.factor(blocks['10'][0]/weights[0]-(x+y-2*z+v))==0

a,b,c,h,lam=s.symbols('a b c h lam',real=True)
sym_sub={x:a,y:a,z:b,v:c}
k=q+1/q-1
mplus=a*(1-q)+b*q
dplus=u*(a+b*(q-1)-c*q)
mminus=a*(q-1)+b
dminus=(a*q+b*(1-q)-c)/u
sym_eigs=[
    [mplus+dplus,mplus-dplus],
    [(mminus+dminus)/q,(mminus-dminus)/q],
    [2*a-2*b+c,a+b-c,k*(b-a)+c],
]
for M,expected in zip([Fplus,Fminus,K],sym_eigs):
    target=s.prod(lam-e for e in expected)
    assert s.factor((lam*s.eye(M.rows)-M.subs(sym_sub)).det()-target)==0

# Exact benchmarks. Depolarization shifts EVERY relevant block by c I.
for M in [Fplus,Fminus,K]:
    assert zero(M.subs({x:c,y:c,z:c,v:c})-c*s.eye(M.rows))
repair_eigs=[s.factor(ev.subs({a:1+h,b:2+h,c:3+h})) for es in sym_eigs for ev in es]
repair_expected=[h+(1+q)*(1-u),h+(1+q)*(1+u),
                 h+(1+1/q)*(1-1/u),h+(1+1/q)*(1+1/u),
                 1+h,h,k+3+h]
assert all(s.simplify(e-f)==0 for e,f in zip(repair_eigs,repair_expected))
# Parameter inversion leaves the complete cone unchanged by trace *-isomorphism.
# Sample exact extreme parameters to catch substitutions/denominators; the
# general sharp threshold is proved by the seven displayed symbolic branches.
for u0 in [s.Rational(1,10),s.Rational(1,2),s.Integer(1),s.Integer(2),s.Integer(10)]:
    Q=max(u0**2,u0**-2)
    h0=(s.sqrt(Q)-1)*(1+Q)
    values=[s.factor(e.subs({u:u0,h:h0})) for e in repair_eigs]
    assert all(e>=0 for e in values) and min(values)==0
    if h0>0:
        below=[s.factor(e.subs({u:u0,h:h0/2})) for e in repair_eigs]
        assert min(below)<0
# Non-depolarizing valid replacement, supplied by expectations.
for es in sym_eigs:
    for e in es:
        val=s.factor(e.subs({a:1,b:2,c:2}))
        # Cross branches reduce to positive factors or (u+-1)^2.
        assert all(val.subs(u,uu)>=0 for uu in [s.Rational(1,2),1,2,10])

# The q=1 group test has exactly the same Choi positivity, independently via
# the six-by-six Schoenberg kernel on S3. This is an exact benchmark, not an
# inference that group CND persists after deformation.
def group_ccp(rate_values):
    from itertools import permutations
    W=list(permutations(range(3)))
    def mul(g,h): return tuple(g[h[i]] for i in range(3))
    def inv(g): return tuple(g.index(i) for i in range(3))
    gens=[(1,0,2),(0,2,1)]
    rate={}
    for word,rv in zip(words,rate_values):
        w=(0,1,2)
        for i in word: w=mul(w,gens[i])
        rate[w]=rv
    A=s.Matrix([[rate[mul(inv(g),h)] for h in W] for g in W])
    V0=s.Matrix.vstack(s.eye(5),-s.ones(1,5))
    return -V0.T*A*V0

q1_tests=[[0,1,1,2,2,3],[0,1,1,2,2,2],[0,1,1,1,1,1],
          [0,1,2,3,3,4],[0,1,1,5,5,1]]
for rr in q1_tests:
    sub={u:s.Integer(1),x:rr[1],y:rr[2],z:rr[3],v:rr[5]}
    def psd_exact(M):
        from itertools import combinations
        return all(M.extract(ii,ii).det()>=0 for n in range(1,M.rows+1)
                   for ii in combinations(range(M.rows),n))
    cone=(rr[1]+rr[2]-2*rr[3]+rr[5]>=0 and
          all(psd_exact(M.subs(sub)) for M in [Fplus,Fminus,K]))
    assert cone==psd_exact(group_ccp(rr))

# Independent all-block Choi checks at one exact time for valid examples.
# This checks implementation of the exponential; all-time CP uses the theorem.
finite_time_cases=0
for u0 in [s.Rational(1,2),s.Integer(1),s.Integer(2)]:
    for rr in [[0,1,1,1,1,1],[0,1,2,3,3,3],[0,6,6,7,7,8]]:
        multipliers=[s.Rational(3,4)**n for n in rr]
        for alpha in range(3):
            for beta in range(3):
                assert psd_exact(simp(choi(alpha,beta,multipliers).subs(u,u0)))
                finite_time_cases+=1
bad=choi(2,0,[s.Rational(3,4)**n for n in lengths]).subs(u,2)
assert not psd_exact(bad)
# Initial physical state in the trivial sector gives negative matrix eigenvalue.
physical_bad=simp(bad*weights[2].subs(u,2)/weights[0].subs(u,2))
assert -s.Rational(2,21) in physical_bad.eigenvals()

# Trace weights and orthonormality establish exact Fourier inversion.
for i in range(6):
    for j in range(6):
        tr=sum(weights[a]*s.trace(reps[a][i].T*reps[a][j]) for a in range(3))
        assert s.simplify(tr-int(i==j))==0

def main():
    out=Path('scripts/results')
    out.mkdir(exist_ok=True)
    result={'status':'PASS', 'parameter':'q=u^2>0', 'arithmetic':'exact symbolic',
            'basis':['1','S','T','ST','TS','STS'], 'rates':['0','x','y','z','z','v'],
            'trace_weights':[str(w) for w in weights],
            'sympy_version':s.__version__,
            'full_cone':{k:[[str(e) for e in row] for row in M.tolist()] for k,M in blocks.items()},
            'reduced_pencils':{name:[[str(e) for e in row] for row in M.tolist()]
                               for name,M in zip(['Fplus','Fminus','K'],[Fplus,Fminus,K])},
            'symmetric_eigenvalues':[[str(s.factor(e)) for e in es] for es in sym_eigs],
            'repair_eigenvalues':[str(e) for e in repair_expected],
            'sharp_repair':'c_min=(sqrt(Q)-1)*(1+Q), Q=max(q,1/q)',
            'q4_repair_minimum':'5', 'q2_repair_minimum':'3*(sqrt(2)-1)',
            'q1_exact_CND_benchmarks':len(q1_tests),
            'exact_finite_time_Choi_blocks_checked':finite_time_cases,
            'invalid_length_physical_eigenvalue_q4_r3over4':'-2/21',
            'limitations':'General statements use the accompanying proofs, not sampled positivity.'}
    (out/'generator_cone.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Trace inversion and identity Choi checks: PASS')
    print('Full four-rate spectrahedron and symmetric characteristic polynomials: PASS')
    print('Exact sharp repair:',result['sharp_repair'])
    print('q4: c_min=5; q2: c_min=3*(sqrt(2)-1); q1: c_min=0')

if __name__=='__main__':
    main()
