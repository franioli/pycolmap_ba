"""How much of the true Hessian does Gauss-Newton throw away?

  C(x) = 0.5 ||f(x)||^2 ,  grad C = J^T f ,  Hess C = J^T J + sum_i f_i Hess f_i
The second term is obtained here as (numerical Hessian) - J^T J, with the
numerical Hessian from central differences of the exact gradient J^T f.
"""

import numpy as np
import pyceres
import docs.bundle_docs.toy as toy
from scipy.sparse import csr_matrix


def grad_and_H(prob, opts):
    crs = prob.evaluate_jacobian(opts)
    J = csr_matrix((crs.values, crs.cols, crs.rows), shape=(crs.num_rows, crs.num_cols))
    f = np.asarray(prob.evaluate_residuals(opts))
    return J.T @ f, (J.T @ J).toarray(), 0.5 * f @ f, f


def run(noise_px, label, solve_first=True):
    s = toy.build(n_cam=4, n_pt=25, noise_px=noise_px)
    prob, pts, poses = s["problem"], s["points"], s["poses"]
    opts = pyceres.EvaluateOptions()
    opts.set_parameter_blocks(pts + poses)
    if solve_first:
        o = pyceres.SolverOptions()
        o.max_num_iterations = 100
        o.linear_solver_type = pyceres.LinearSolverType.DENSE_SCHUR
        pyceres.solve(o, prob, pyceres.SolverSummary())
    n = 3 * len(pts)

    def setx(dx):
        for i, p in enumerate(pts):
            p += dx[3 * i : 3 * i + 3]
        for j, q in enumerate(poses):
            q[4:] += dx[n + 3 * j : n + 3 * j + 3]

    g0, GN, cost, f = grad_and_H(prob, opts)
    m = g0.size
    Hnum = np.zeros((m, m))
    for k in range(m):
        e = np.zeros(m)
        h = 1e-5
        e[k] = h
        setx(e)
        gp, *_ = grad_and_H(prob, opts)
        setx(-2 * e)
        gm, *_ = grad_and_H(prob, opts)
        setx(e)
        Hnum[:, k] = (gp - gm) / (2 * h)
    Hnum = 0.5 * (Hnum + Hnum.T)
    extra = Hnum - GN
    rms = np.sqrt(2 * cost / len(f))
    print(
        f"{label:28s} RMS residual {rms:8.3f} px   "
        f"||sum f_i Hess f_i||_F / ||J^T J||_F = "
        f"{np.linalg.norm(extra) / np.linalg.norm(GN):.3e}"
    )


run(0.5, "converged, 0.5 px noise")
run(8.0, "converged, 8 px noise")
run(0.5, "NOT converged (start point)", solve_first=False)
