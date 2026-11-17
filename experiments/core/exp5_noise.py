"""Experiment 5: noisy labels, fixed-dataset SGD (EMPIRICAL - outside the theorems' noiseless assumption).

y_i = beta theta^T x_i + sigma xi_i, xi_i ~ N(0,1) drawn ONCE with the dataset.
Baseline otherwise identical: d=20, m=16, alpha=0.01, tau=0.003, B=1 with replacement.
For each seed the inputs X, the sampled index sequence and the initialisation are the
same across sigma (paired design), so sigma=0 is the exact noiseless twin.
Long training (200k steps, ~28x the noiseless fitting time) with snapshots to test drift.
Usage: python exp5_noise.py <M>
"""
import sys, time
import numpy as np
import sgdlib as S
from common import save_result, summarise, dump_json

M = int(sys.argv[1])
seeds = [7000 + i for i in range(20)]
sigmas = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]
snaps = [0, 5000, 7500, 10000, 15000, 20000, 30000, 50000, 100000, 150000, 200000]
summ = []
for sg in sigmas:
    t = time.time()
    sp = S.Spec(d=20, m=16, seeds=seeds, mode="fixed", M=M, noise_std=sg, max_steps=200_000, check_every=100,
                stop_when_all_fit=False, train_loss_every=2, record_shares=True, snapshot_steps=snaps,
                label=f"noise M={M} sigma={sg}")
    r = S.train(sp)
    save_result(r, f"exp5_M{M}_sigma{sg}", keep_data=True)
    s = summarise(r, dict(sigma=sg)); summ.append(s)
    print(M, sg, f"{time.time()-t:.0f}s", s["final_pop_loss_q"], s.get("final_train_loss_q"), s["final_max_angle_q"], flush=True)
dump_json(summ, f"exp5_M{M}_summary")
