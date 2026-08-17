import numpy as np
import pyceres

a = 1.5  # loss scale


def analytic(name, s, a):
    if name == "Trivial":
        return np.array([s, 1.0, 0.0])
    if name == "Huber":
        b = a * a
        if s <= b:
            return np.array([s, 1.0, 0.0])
        r = np.sqrt(s)
        return np.array([2 * a * r - b, a / r, -0.5 * a / (s * r)])
    if name == "SoftLOne":
        b = a * a
        t = np.sqrt(1.0 + s / b)
        return np.array([2 * b * (t - 1.0), 1.0 / t, -0.5 / (b * t**3)])
    if name == "Cauchy":
        b = a * a
        t = 1.0 + s / b
        return np.array([b * np.log(t), 1.0 / t, -1.0 / (b * t * t)])


losses = {
    "Trivial": pyceres.TrivialLoss(),
    "Huber": pyceres.HuberLoss(a),
    "SoftLOne": pyceres.SoftLOneLoss(a),
    "Cauchy": pyceres.CauchyLoss(a),
}
print(f"scale a = {a};  s = ||f||^2;  r = ||f|| = sqrt(s)")
print(
    f"{'loss':9s} {'r':>6s} {'s':>9s} {'rho':>11s} {'rho_p':>9s} {'rho_pp':>11s}  {'max|err|':>9s}"
)
for name, L in losses.items():
    for r in [0.5, 1.5, 3.0, 10.0]:
        s = r * r
        got = np.asarray(L.evaluate(s))
        ref = analytic(name, s, a)
        print(
            f"{name:9s} {r:6.2f} {s:9.3f} {got[0]:11.6f} {got[1]:9.6f} "
            f"{got[2]:11.6f}  {np.max(np.abs(got - ref)):9.2e}"
        )
