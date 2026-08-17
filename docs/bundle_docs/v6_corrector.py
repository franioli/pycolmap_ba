"""What Ceres' robustified residual/Jacobian actually are (the Corrector)."""

import numpy as np
import pyceres
import pycolmap
from pycolmap import cost_functions as cf

a = 2.0
camera = pycolmap.Camera.create_from_model_id(
    1, pycolmap.CameraModelId.SIMPLE_RADIAL, 1000.0, 1000, 800
)
camera.params = np.array([1000.0, 500.0, 400.0, -0.02])
X = np.array([0.7, -0.4, 10.0])
T = pycolmap.Rigid3d(
    pycolmap.Rotation3d(np.array([0.05, -0.03, 0.02])), np.array([0.2, -0.1, 0.05])
)

print(f"CauchyLoss(a={a}):  rho(s)=a^2 log(1+s/a^2)")
print(
    f"{'|f| px':>8} {'s=|f|^2':>9} {'|r_corr|^2':>11} {'rho(s)':>10} "
    f"{'rho_p(s)*s':>11} {'kappa=|r_c|^2/s':>15} {'rho_p(s)':>9}"
)
for err in [0.5, 2.0, 5.0, 20.0]:
    pt = X.copy()
    pose = np.array(T.params, float)
    cam = np.array(camera.params, float)
    uv = camera.img_from_cam(T * X) + np.array([err, 0.0])
    prob = pyceres.Problem()
    cost = cf.ReprojErrorCost(camera.model, uv)
    prob.add_residual_block(cost, pyceres.CauchyLoss(a), [pt, pose, cam])
    o = pyceres.EvaluateOptions()
    o.apply_loss_function = False  # pyceres/Ceres default is True
    plain = np.asarray(prob.evaluate_residuals(o))
    o2 = pyceres.EvaluateOptions()
    o2.apply_loss_function = True
    corr = np.asarray(prob.evaluate_residuals(o2))
    s = plain @ plain
    rho, rp, rpp = pyceres.CauchyLoss(a).evaluate(s)
    print(
        f"{np.linalg.norm(plain):8.3f} {s:9.3f} {corr @ corr:11.5f} {rho:10.5f} "
        f"{rp * s:11.5f} {(corr @ corr) / s:15.6f} {rp:9.6f}"
    )
