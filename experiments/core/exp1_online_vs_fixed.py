"""Experiment 1: fresh-sample (online) SGD vs batch-size-one SGD with replacement
from ONE fixed Gaussian dataset (the algorithm of Theorems 1-2).

Part A  (Figure 2 / Appendix H setting): d=2, m=1000, beta=1, alpha=0.01, tau=0.003,
        B=1, init s_j<u_j,theta> >= -1/2, balanced magnitudes, 12,000 steps, check every 20.
Part B  (Figure 3 / allocation setting): d=20, m in {4,16,64}, 20 seeds, same alpha/tau/B,
        init |<u_j,theta>| <= 1/2 (as in the authors' allocation cell), max 20,000 steps.
        Active comparison: online B=1 versus fixed M=2000 B=1.
Usage: python exp1_online_vs_fixed.py A|B
"""
import sys, time
import numpy as np
import sgdlib as S
from common import RES, save_result, summarise, dump_json

part = sys.argv[1]
summ = []
if part == "A":
    seeds = list(range(17, 27))
    configs = [("online", None)] + [("fixed", M) for M in (20, 50, 200, 1000, 10000)]
    for mode, M in configs:
        t = time.time()
        sp = S.Spec(d=2, m=1000, seeds=seeds, mode=mode, M=M or 0, init_dir="sign_half", max_steps=12_000,
                    check_every=20, stop_when_all_fit=False, record_dirs_every=2,
                    train_loss_every=5, label=f"A_{mode}_{M}")
        r = S.train(sp)
        name = f"exp1A_{mode}" + (f"_M{M}" if M else "")
        save_result(r, name, keep_dirs=False)
        # keep the full direction history for the paper's seed (17) only, for the trajectory figure
        np.savez_compressed(RES / f"{name}_dirs_seed17.npz", dir_hist=r.dir_hist[:, 0], steps=r.dir_hist_steps,
                            signs=r.signs)
        s = summarise(r); summ.append(s)
        print(name, f"{time.time()-t:.0f}s", s["n_fitted"], s["fit_step_q"], s["align_step_q"], s["final_pop_loss_q"], flush=True)
    dump_json(summ, "exp1A_summary")
else:
    for m in (4, 16, 64):
        seeds = [1000 * m + i for i in range(20)]
        configs = [("online", None, 1), ("fixed", 2000, 1)]  # Conditions reported in the final sampling comparison.
        for mode, M, B in configs:
            t = time.time()
            sp = S.Spec(d=20, m=m, seeds=seeds, mode=mode, M=M or 0, batch=B, init_dir="abs_half",
                        max_steps=20_000, check_every=20, stop_when_all_fit=False,
                        train_loss_every=(10 if (M or 0) * m > 2e5 else 1), label=f"B_m{m}_{mode}_{M}_B{B}")
            r = S.train(sp)
            name = f"exp1B_m{m}_{mode}" + (f"_M{M}" if M else "") + f"_B{B}"
            save_result(r, name)
            s = summarise(r); summ.append(s)
            a = s["alloc_at_fit"]
            print(name, f"{time.time()-t:.0f}s fitted {s['n_fitted']}/20", "fit", s["fit_step_q"],
                  "err", round(a.get("median_log_err", np.nan), 4), "corr", round(a.get("log_corr", np.nan), 4), flush=True)
    dump_json(summ, "exp1B_summary")
