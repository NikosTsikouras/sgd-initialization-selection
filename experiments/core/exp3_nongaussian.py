"""Experiment 3: centrally symmetric, isotropic, NON-Gaussian inputs (EMPIRICAL - outside Theorems 1-2).

Inputs: iid standardized coordinates, Cov(X)=I, X =_d -X:
  gaussian (reference), laplace (scale 1/sqrt2), uniform on [-sqrt3, sqrt3]^d,
  and *_rot variants: the same product law composed with a fixed random rotation, so that the
  teacher theta=e_1 is not a coordinate axis of the product structure.
Population loss: exact for Gaussian; otherwise Monte-Carlo on one fixed test set
(N=20,000 for d=20, 5,000 for d=2) shared by all runs.  All other settings as in Exp 1.
Usage: python exp3_nongaussian.py A|B|B16|B64
"""
import sys, time
import numpy as np
import sgdlib as S
from common import save_result, summarise, dump_json

def rotation(d, seed=31415):
    q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
    return q * np.sign(np.diag(r))

part = sys.argv[1]
summ = []
dists = [("gaussian", False), ("laplace", False), ("uniform", False), ("laplace", True), ("uniform", True)]
if part.startswith("B"):
    for m in ((16, 64) if part == "B" else (int(part[1:]),)):
        seeds = [9000 + 100 * m + i for i in range(20)]
        for dist, rot in dists:
            for mode in ("fixed",):  # Final-paper input-law conditions; per-condition seeds unchanged.
                t = time.time()
                sp = S.Spec(d=20, m=m, seeds=seeds, mode=mode, M=2000, dist=dist, L=rotation(20) if rot else None,
                            init_dir="abs_half", max_steps=15_000, check_every=20 if m == 16 else 40,
                            stop_when_all_fit=False, train_loss_every=5, n_test_mc=20_000,
                            label=f"{dist}{'_rot' if rot else ''} m={m} {mode}")
                r = S.train(sp)
                name = f"exp3B_m{m}_{dist}{'_rot' if rot else ''}_{mode}"
                save_result(r, name)
                s = summarise(r, dict(rotated=rot)); summ.append(s)
                print(name, f"{time.time()-t:.0f}s", s["n_fitted"], s["fit_step_q"], s["max_angle_at_fit_q"],
                      s["alloc_at_fit"].get("median_log_err"), s["alloc_at_fit"].get("log_corr"), flush=True)
    dump_json(summ, f"exp3{part}_summary")
else:
    seeds = list(range(17, 22))
    for dist, rot in dists:
        t = time.time()
        sp = S.Spec(d=2, m=1000, seeds=seeds, mode="fixed", M=1000, dist=dist, L=rotation(2) if rot else None,
                    init_dir="sign_half", max_steps=12_000, check_every=40, stop_when_all_fit=False,
                    train_loss_every=5, n_test_mc=5_000, label=f"A {dist}{'_rot' if rot else ''}")
        r = S.train(sp)
        name = f"exp3A_{dist}{'_rot' if rot else ''}_fixed_M1000"
        save_result(r, name)
        s = summarise(r, dict(rotated=rot)); summ.append(s)
        print(name, f"{time.time()-t:.0f}s", s["n_fitted"], s["fit_step_q"], s["align_step_q"], s["final_pop_loss_q"], flush=True)
    dump_json(summ, "exp3A_summary")
