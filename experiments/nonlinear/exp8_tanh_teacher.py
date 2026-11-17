"""Experiment 8: ODD nonlinear teacher y = tanh(k theta^T x)/k (slope 1 at 0), ReLU student (EMPIRICAL).

Theory remark: for odd y and centrally symmetric X, E[y 1{w^T X>0} X] = E[yX]/2 for every w, so the
small-output dynamics are those of a linear teacher with beta_eff = E[g'(z)].  Hence Theorem 2's lambda
(unchanged formula) is the natural prediction.  tanh is not representable by a bias-free ReLU net,
so there is no exact fit: shares are tracked over time and reported at the loss plateau.
Protocol = Figure 3 (d=20, m in {4,16,64}, 20 seeds, alpha=0.01, tau=0.003); fixed M=2000 B=1.
Population loss: Monte Carlo on a fixed test set.  Usage: python exp8_tanh_teacher.py <k>
"""
import sys, time
import numpy as np
import sgdlib as S
from common import save_result, summarise, dump_json

k = sys.argv[1]; summ = []
steps = 40_000 if float(k) <= 1 else 80_000     # beta_eff = 0.61 (k=1) or 0.26 (k=3): slower early phase
snaps = [s for s in [2000, 4000, 6000, 8000, 10000, 15000, 20000, 30000, 40000, 60000, 80000] if s <= steps]
for m in (4, 16, 64):
    seeds = [1000 * m + i for i in range(20)]
    for mode, M, B in [("fixed", 2000, 1)]:  # Final-paper conditions; per-condition seeds unchanged.
        t0 = time.time()
        sp = S.Spec(d=20, m=m, seeds=seeds, mode=mode, M=M, batch=B, teacher=f"tanh:{k}", init_dir="abs_half",
                    max_steps=steps, check_every=50 if steps <= 40000 else 100, stop_when_all_fit=False, train_loss_every=10,
                    n_test_mc=20000 if m < 64 else 10000, record_shares=True, snapshot_steps=snaps,
                    label=f"tanh{k} m={m} {mode} B={B}")
        r = S.train(sp)
        nm = f"exp8_tanh{k}_m{m}_{mode}_B{B}"
        save_result(r, nm)
        s = summarise(r, dict(teacher=f"tanh(kz)/k, k={k}")); summ.append(s)
        print(nm, f"{time.time()-t0:.0f}s loss_end {s['final_pop_loss_q']} ang_end {s['final_max_angle_q']} "
              f"err_end {s['alloc_at_end'].get('median_log_err')} uni {s['alloc_at_end'].get('median_log_err_uniform_baseline')}", flush=True)
dump_json(summ, f"exp8_tanh{k}_summary")
