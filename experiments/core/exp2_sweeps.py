"""Experiment 2: initialisation-scale and learning-rate sweeps, fixed-dataset SGD.

Baseline (allocation setting): d=20, m=16, beta=1, M=2000 fixed Gaussian examples,
B=1 with replacement, alpha=0.01, tau=0.003, init |<u,theta>|<=1/2, balanced magnitudes.
Each sweep varies ONE quantity; 20 seeds per value; the same seed gives the same
initial directions and dataset across sweep values (paired design).
The active runner reproduces the fixed-dataset conditions in the final paper.
Usage: python exp2_sweeps.py alpha|tau
"""
import sys, time
import numpy as np
import sgdlib as S
from common import save_result, summarise, dump_json

which = sys.argv[1]
seeds = [5000 + i for i in range(20)]
summ = []
if which == "alpha":
    alphas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    for mode in ("fixed",):  # Final-paper conditions; per-condition seeds unchanged.
        for al in alphas:
            t = time.time()
            sp = S.Spec(d=20, m=16, seeds=seeds, alpha=al, tau=0.003, mode=mode, M=2000, max_steps=30_000,
                        check_every=20, stop_when_all_fit=False, train_loss_every=5, label=f"alpha={al} {mode}")
            r = S.train(sp)
            save_result(r, f"exp2_alpha{al}_{mode}")
            s = summarise(r, dict(alpha=al, tau=0.003)); summ.append(s)
            print(mode, al, f"{time.time()-t:.0f}s", s["n_fitted"], s["fit_step_q"], s["max_angle_at_fit_q"],
                  s["final_pop_loss_q"], s["alloc_at_fit"].get("median_log_err"), flush=True)
    dump_json(summ, "exp2_alpha_summary")
else:
    taus = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
    for mode in ("fixed",):  # Final-paper conditions; per-condition seeds unchanged.
        for tau in taus:
            t = time.time()
            steps = int(round(90 / tau))
            ce = max(1, int(round(0.06 / tau)))
            sp = S.Spec(d=20, m=16, seeds=seeds, alpha=0.01, tau=tau, mode=mode, M=2000, max_steps=steps,
                        check_every=ce, stop_when_all_fit=False, train_loss_every=5, label=f"tau={tau} {mode}")
            r = S.train(sp)
            save_result(r, f"exp2_tau{tau}_{mode}")
            s = summarise(r, dict(alpha=0.01, tau=tau, normalised_budget=90)); summ.append(s)
            print(mode, tau, f"{time.time()-t:.0f}s", s["n_fitted"], s["n_diverged"], s["fit_step_q"], s["max_angle_at_fit_q"],
                  s["final_pop_loss_q"], s["alloc_at_fit"].get("median_log_err"), flush=True)
    dump_json(summ, "exp2_tau_summary")
