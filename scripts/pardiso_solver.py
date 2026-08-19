"""A minimal MKL PARDISO linear solver for `fem.solve(solve_fn=...)` / `integrate(fem, solver=...)`.

jNO's own `lu(backend="pardiso")` aborts the process (glibc `double free or corruption`) whenever a
P0 density carries `rho.constrain(d.patch_filter())`. That was chased hard and the cause was not
found -- but **standalone pypardiso solves the very same pathological operator in 2.5 ms with a
1.2e-12 residual**, so the fault lies in the backend's caching / phase-driving rather than in MKL.
This bypasses all of it: one `pure_callback` per solve, no cache, no manual phase control.

The trade is the symbolic analysis, which jNO's backend reuses across iterations and this does not.
Measured standalone, that is 2.5 ms cold against 2.1 ms warm -- ~20 %, against 46-125x over the
default backend. Worth losing.

The adjoint is written by hand because a `pure_callback` is opaque to AD. For `A x = b`:

    x_bar -> lambda = A^-T x_bar,    b_bar = lambda,    dC/dA_ij = -lambda_i x_j

so the cotangent of a COO value at ``(r, c)`` is ``-lambda[r] * x[c]``. That is one extra solve per
backward pass, on the transpose -- exactly what the default backend costs too.

    from scripts.pardiso_solver import pardiso_lu
    C = (E(rho) * a(eps(u), eps(u))).integrate(fem, solver=pardiso_lu)
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def _host(data, indices, rhs, transpose):
    """One PARDISO solve on concrete numpy arrays. Returns an array that OWNS its data."""
    import scipy.sparse as sp

    import pypardiso

    idx = np.asarray(indices)
    d = np.asarray(data, dtype=np.float64)
    b = np.asarray(rhs, dtype=np.float64).reshape(-1)
    n = b.size
    rows, cols = (idx[:, 1], idx[:, 0]) if transpose else (idx[:, 0], idx[:, 1])
    K = sp.coo_matrix((d, (rows, cols)), shape=(n, n)).tocsr()
    K.sum_duplicates()
    x = pypardiso.spsolve(K, b)
    # copy: the callback's result must own its buffer, or JAX frees what numpy also frees
    return np.array(np.asarray(x).reshape(-1), dtype=np.float64, copy=True)


def _call(data, indices, rhs, transpose):
    return jax.pure_callback(
        partial(_host, transpose=transpose),
        jax.ShapeDtypeStruct(rhs.shape, rhs.dtype),
        data,
        indices,
        rhs,
        vmap_method="sequential",
    )


@jax.custom_vjp
def _psolve(data, indices, b):
    return _call(data, indices, b, False)


def _psolve_fwd(data, indices, b):
    x = _call(data, indices, b, False)
    return x, (data, indices, x)


def _psolve_bwd(res, x_bar):
    data, indices, x = res
    lam = _call(data, indices, x_bar, True)  # A^T lambda = x_bar
    rows, cols = indices[:, 0], indices[:, 1]
    data_bar = -lam[rows] * x[cols]
    # `indices` is integer: JAX's cotangent for an integer primal is the empty float0 type.
    idx_bar = np.zeros(indices.shape, dtype=jax.dtypes.float0)
    return (data_bar, idx_bar, lam)


_psolve.defvjp(_psolve_fwd, _psolve_bwd)


def pardiso_lu(A, b):
    """``(A, b) -> x`` with the slot signature jNO's own linear solvers use."""
    shape = jnp.asarray(b).shape
    x = _psolve(A.data, A.indices, jnp.asarray(b).reshape(-1))
    return x.reshape(shape)
