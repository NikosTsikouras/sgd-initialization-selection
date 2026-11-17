"""Experiment 4: anisotropic Gaussian inputs X ~ N(0, Sigma) (EMPIRICAL - outside Theorems 1-2).

Teacher fixed: theta_* = e_1, beta = 1.  Sigma has eigenvalues log-spaced on [1, kappa],
then rescaled so that theta^T Sigma theta = 1 (teacher signal variance unchanged).  Three placements:
  top    : theta is the top eigenvector,
  bottom : theta is the bottom eigenvector,
  mixed  : theta = (q_max + q_min)/sqrt2, so Sigma theta is NOT parallel to theta
           (angle atan((kappa-1)/(kappa+1)) -> 45 deg).
Population loss is exact (Gaussian with X = Sigma^{1/2} Z).  We also track the max angle to the
sign-selected poles +-Sigma theta/|Sigma theta| (the direction of the early teacher force).
Fixed dataset M=2000, B=1, d=20, m=16, alpha=0.01, tau=0.003, 20 seeds, 40,000 steps.
Usage: python exp4_anisotropic.py
"""
import time
import numpy as np
import sgdlib as S
from common import RES, save_result, summarise, dump_json

d = 20
def make_sigma(kappa, case, seed=2718):
    lam = np.geomspace(kappa, 1.0, d)                    # lam[0] = max, lam[-1] = min
    rest = np.linalg.qr(np.random.default_rng(seed).standard_normal((d - 2, d - 2)))[0]
    Q = np.zeros((d, d))
    if case == "top":
        Q[0, 0] = 1.0; Q[1, d - 1] = 1.0
    elif case == "bottom":
        Q[1, 0] = 1.0; Q[0, d - 1] = 1.0
    else:  # mixed
        Q[:2, 0] = np.array([1, 1]) / np.sqrt(2); Q[:2, d - 1] = np.array([1, -1]) / np.sqrt(2)
    Q[2:, 1:d - 1] = rest
    Sigma = (Q * lam) @ Q.T
    Sigma /= Sigma[0, 0]                                  # theta^T Sigma theta = 1
    ev, V = np.linalg.eigh(Sigma)
    Lh = (V * np.sqrt(ev)) @ V.T
    return Sigma, Lh

if __name__ == "__main__":
    summ = []
    seeds = [11000 + i for i in range(20)]
    for case in ("top", "bottom", "mixed"):
        for kappa in (1, 3, 10, 30, 100, 300, 1000):
            if kappa == 1 and case != "top":
                continue
            t = time.time()
            Sigma, Lh = make_sigma(kappa, case)
            v = Sigma[:, 0]
            sp = S.Spec(d=d, m=16, seeds=seeds, mode="fixed", M=2000, L=Lh, init_dir="abs_half", max_steps=40_000,
                        check_every=20, stop_when_all_fit=False, train_loss_every=5, aux_dir=v,
                        label=f"aniso {case} kappa={kappa}")
            r = S.train(sp)
            name = f"exp4_{case}_kappa{kappa}"
            save_result(r, name)
            np.save(RES / f"{name}_Sigma.npy", Sigma)
            lamS = S.predicted_shares(r.a0, r.w0, r.signs, direction=v / np.linalg.norm(v))
            fitted = r.fit_step >= 0
            extra = dict(case=case, kappa=kappa,
                         angle_Sigma_theta_vs_theta_deg=float(np.degrees(np.arccos(v[0] / np.linalg.norm(v)))),
                         alloc_at_fit_SigmaTheta_prediction=S.allocation_metrics(lamS, r.shares_fit, r.signs, mask=fitted))
            s = summarise(r, extra); summ.append(s)
            print(name, f"{time.time()-t:.0f}s", s["n_fitted"], s["fit_step_q"], s["max_angle_at_fit_q"],
                  s["final_pop_loss_q"], s["alloc_at_fit"].get("median_log_err"),
                  extra["alloc_at_fit_SigmaTheta_prediction"].get("median_log_err"), flush=True)
    dump_json(summ, "exp4_summary")
