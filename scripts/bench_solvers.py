"""How fast can this system actually be solved?

The matrices are real SIMP stiffnesses dumped from the design loop (1e3 contrast, not a uniform
field -- the contrast is what makes the system hard). They are symmetric to 5.6e-17, which is the
fact most of the headroom rests on: elasticity is SPD, so Cholesky and CG both apply and LU is
doing about twice the necessary work.

Two regimes are timed because topology optimisation lives in the second:
  COLD  -- symbolic + numeric factorisation + solve, what every iteration pays today.
  WARM  -- numeric refactorisation + solve with the symbolic analysis REUSED. The mesh
           connectivity never changes under node relocation, so the sparsity pattern is constant
           across all 400 iterations; only the values move. This is what a plan/factorise split
           (cuDSS, PARDISO phase 22, CHOLMOD's analyze/factor) buys.
"""

import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

PATH = sys.argv[1]
K = sp.load_npz(PATH).tocsc()
b = np.load(PATH.replace(".npz", "_b.npy"))
n, nnz = K.shape[0], K.nnz
print(f"n = {n}, nnz = {nnz} ({nnz / n:.1f}/row), symmetric to {abs(K - K.T).max():.1e}\n")

REP = 3
x_ref = spla.spsolve(K.tocsc(), b)
nrm = np.linalg.norm(b)


def report(name, cold, warm, x, extra=""):
    err = np.linalg.norm(K @ x - b) / nrm
    w = f"{warm * 1e3:8.1f}" if warm is not None else "       –"
    print(f"  {name:34} cold {cold * 1e3:8.1f} ms   warm {w} ms   |Kx-b|/|b| {err:.1e} {extra}")


def timed(fn, rep=REP):
    fn()
    t0 = time.time()
    for _ in range(rep):
        out = fn()
    return (time.time() - t0) / rep, out


# ---- scipy SuperLU, the current default route
t, _ = timed(lambda: spla.spsolve(K, b))
lu = spla.splu(K)
tw, _ = timed(lambda: spla.splu(K).solve(b))          # refactor, no symbolic reuse in scipy
report("scipy spsolve (SuperLU, LU)", t, tw, lu.solve(b))

# ---- CHOLMOD: analyze once, then numeric-only refactorisation. SPD only.
try:
    from sksparse.cholmod import analyze, cholesky
    t, _ = timed(lambda: cholesky(K.tocsc())(b))
    fac = analyze(K.tocsc())                            # SYMBOLIC analysis, done once ever

    def warm_chol():
        return fac.cholesky(K.tocsc())(b)               # numeric refactor + solve

    tw, x = timed(warm_chol)
    report("CHOLMOD Cholesky (SPD)", t, tw, x, "<- symbolic reused")
except Exception as e:
    print("  CHOLMOD failed:", type(e).__name__, str(e)[:60])

# ---- PARDISO
try:
    import pypardiso
    solver = pypardiso.PyPardisoSolver()
    t, _ = timed(lambda: pypardiso.spsolve(K, b))

    def warm_pardiso():
        solver.set_phase(23)                            # numeric factorise + solve, reuse analysis
        return solver.solve(K, b)

    solver.set_phase(12); solver.factorize(K)           # analysis + numeric, once
    tw, x = timed(warm_pardiso)
    report("MKL PARDISO (LU)", t, tw, x, "<- analysis reused")
except Exception as e:
    print("  PARDISO failed:", type(e).__name__, str(e)[:60])

# ---- iterative: smoothed-aggregation AMG preconditioned CG. No factorisation at all.
try:
    import pyamg
    def cold_amg():
        ml = pyamg.smoothed_aggregation_solver(K.tocsr(), max_coarse=500)
        r = []
        return ml.solve(b, tol=1e-8, accel="cg", residuals=r), len(r)

    t, (x, its) = timed(cold_amg, rep=2)
    ml = pyamg.smoothed_aggregation_solver(K.tocsr(), max_coarse=500)

    def warm_amg():
        r = []
        return ml.solve(b, tol=1e-8, accel="cg", residuals=r), len(r)

    tw, (xw, itw) = timed(warm_amg, rep=2)
    report("pyamg SA-AMG + CG", t, tw, x, f"<- {its} its cold / {itw} warm (hierarchy reused)")
except Exception as e:
    print("  pyamg failed:", type(e).__name__, str(e)[:70])

# ---- plain CG with Jacobi, for reference: what you get with no preconditioner worth the name
try:
    Dinv = 1.0 / K.diagonal()
    Mj = spla.LinearOperator((n, n), matvec=lambda v: Dinv * v)
    it = [0]
    def cb(_):
        it[0] += 1
    t0 = time.time()
    x, info = spla.cg(K, b, rtol=1e-8, M=Mj, maxiter=20000, callback=cb)
    report("Jacobi-CG", time.time() - t0, None, x, f"<- {it[0]} its, info={info}")
except Exception as e:
    print("  Jacobi-CG failed:", type(e).__name__, str(e)[:60])
