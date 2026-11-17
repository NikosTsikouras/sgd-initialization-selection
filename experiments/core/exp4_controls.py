"""Experiment 4 controls (EMPIRICAL).
(i)  'bottom' placement: the largest input variance is kappa, so the effective step size along those
     directions is ~tau*kappa.  Re-run with tau = 0.003/kappa (budget 40,000*kappa steps) to separate
     step-size instability from the geometry of anisotropy.
(ii) 'mixed' placement: longer budget (400,000 steps, same tau) to test whether failure at kappa>=30
     is slow rotation from +-Sigma theta to +-theta rather than a different end state.
"""
import time
import numpy as np
import sgdlib as S
from common import save_result, summarise, dump_json
from exp4_anisotropic import make_sigma

seeds = [11000 + i for i in range(20)]
summ = []
for case, kappa, tau, steps in [("bottom", 10, 0.0003, 400_000), ("bottom", 30, 0.0001, 1_200_000),
                                ("bottom", 100, 0.00003, 4_000_000),
                                ("mixed", 30, 0.003, 400_000), ("mixed", 100, 0.003, 400_000), ("mixed", 1000, 0.003, 400_000)]:
    t = time.time()
    Sigma, Lh = make_sigma(kappa, case)
    v = Sigma[:, 0]
    sp = S.Spec(d=20, m=16, seeds=seeds, mode="fixed", M=2000, L=Lh, tau=tau, init_dir="abs_half", max_steps=steps,
                check_every=max(20, steps // 2000), stop_when_all_fit=False, train_loss_every=10, aux_dir=v,
                label=f"control {case} kappa={kappa} tau={tau}")
    r = S.train(sp)
    name = f"exp4ctrl_{case}_kappa{kappa}_tau{tau}"
    save_result(r, name)
    lamS = S.predicted_shares(r.a0, r.w0, r.signs, direction=v / np.linalg.norm(v))
    fitted = r.fit_step >= 0
    obs = np.where(fitted[:, None], r.shares_fit, S.shares_from(r.a_end, r.w_end, r.signs))
    s = summarise(r, dict(case=case, kappa=kappa, tau=tau, steps=steps,
                          alloc_SigmaTheta_pred=S.allocation_metrics(lamS, obs, r.signs, mask=np.ones(20, bool)),
                          alloc_theta_pred_fit_or_end=S.allocation_metrics(r.lam, obs, r.signs, mask=np.ones(20, bool))))
    summ.append(s)
    print(name, f"{time.time()-t:.0f}s fitted {s['n_fitted']} fit {s['fit_step_q']} ang@fit {s['max_angle_at_fit_q']} "
          f"loss_end {s['final_pop_loss_q']} ang_end {s['final_max_angle_q']} "
          f"errθ {s['alloc_theta_pred_fit_or_end']['median_log_err']:.3f} errΣθ {s['alloc_SigmaTheta_pred']['median_log_err']:.3f}", flush=True)
dump_json(summ, "exp4_controls_summary")
