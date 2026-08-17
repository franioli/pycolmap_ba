"""A hand-written LM trust-region loop on the toy problem, then Ceres for comparison."""

import numpy as np
import pyceres
import docs.bundle_docs.toy as toy
from scipy.sparse import csr_matrix


def linearise(prob, opts):
    crs = prob.evaluate_jacobian(opts)
    J = csr_matrix((crs.values, crs.cols, crs.rows), shape=(crs.num_rows, crs.num_cols))
    f = np.asarray(prob.evaluate_residuals(opts))
    return J, f


def state(pts, poses):
    return np.concatenate([np.concatenate(pts), np.concatenate([p[4:] for p in poses])])


def apply(pts, poses, dx):
    n = 3 * len(pts)
    for i, p in enumerate(pts):
        p += dx[3 * i : 3 * i + 3]
    for j, q in enumerate(poses):
        q[4:] += dx[n + 3 * j : n + 3 * j + 3]


s = toy.build(n_cam=5, n_pt=40)
prob, pts, poses = s["problem"], s["points"], s["poses"]
opts = pyceres.EvaluateOptions()
opts.set_parameter_blocks(pts + poses)

# --- damping extremes, at the starting point
J, f = linearise(prob, opts)
H = (J.T @ J).toarray()
g = J.T @ f
D2 = np.diag(np.diag(H))
gn = np.linalg.lstsq(H, -g, rcond=None)[0]
print("cos angle between the LM step and ...      -g        GN step")
for lam in [1e-8, 1e-3, 1.0, 1e3, 1e8]:
    d = np.linalg.solve(H + lam * D2, -g)
    sd = -np.linalg.solve(D2, g)  # scaled steepest descent
    c1 = d @ sd / (np.linalg.norm(d) * np.linalg.norm(sd))
    c2 = d @ gn / (np.linalg.norm(d) * np.linalg.norm(gn))
    print(
        f"  lambda = {lam:8.0e} : |dx| = {np.linalg.norm(d):9.4f}   {c1:8.5f}   {c2:8.5f}"
    )

# --- the accept / reject loop (Nocedal & Wright Alg. 4.1 in LM form)
lam = 1e-4
cost = 0.5 * f @ f
print("\nit   cost           rho      lambda     action")
for it in range(12):
    J, f = linearise(prob, opts)
    H = (J.T @ J).toarray()
    g = J.T @ f
    D2 = np.diag(np.diag(H))
    cost = 0.5 * f @ f
    dx = np.linalg.solve(H + lam * D2, -g)
    pred = -(g @ dx + 0.5 * dx @ H @ dx)
    apply(pts, poses, dx)
    f_new = np.asarray(prob.evaluate_residuals(opts))
    new_cost = 0.5 * f_new @ f_new
    rho = (cost - new_cost) / pred
    if rho > 1e-4:
        lam = max(lam * max(1 / 3, 1 - (2 * rho - 1) ** 3), 1e-12)
        act = "accept"
    else:
        apply(pts, poses, -dx)
        lam *= 4.0
        act = "REJECT"
    print(f"{it:2d}   {cost:12.5f}   {rho:7.4f}  {lam:9.2e}   {act}")
    if act == "accept" and (cost - new_cost) < 1e-8 * cost:
        print("   stop: relative cost decrease below function_tolerance=1e-8")
        break
print(
    f"final cost {0.5 * np.asarray(prob.evaluate_residuals(opts)) @ np.asarray(prob.evaluate_residuals(opts)):.5f}"
)

# --- what Ceres does with the same problem
s2 = toy.build(n_cam=5, n_pt=40)
o = pyceres.SolverOptions()
o.minimizer_type = pyceres.MinimizerType.TRUST_REGION
o.trust_region_strategy_type = pyceres.TrustRegionStrategyType.LEVENBERG_MARQUARDT
o.linear_solver_type = pyceres.LinearSolverType.DENSE_SCHUR
o.max_num_iterations = 50
o.minimizer_progress_to_stdout = False
summ = pyceres.SolverSummary()
pyceres.solve(o, s2["problem"], summ)
print("\nCeres:", summ.BriefReport())
