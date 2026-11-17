"""Experiment 1C: separate SGD-noise error from finite-dataset error in the allocation prediction.
Same as Exp 1B (d=20, m=16, alpha=0.01, B=1, 20 seeds) but with a 10x smaller tau = 3e-4
(budget 80/tau steps), for online and fixed M in {500, 2000, 8000, 32000}."""
import time
import numpy as np
import sgdlib as S
from common import save_result, summarise, dump_json

tau = 3e-4; seeds = [1000 * 16 + i for i in range(20)]
summ = []
for mode, M in [("online", 0), ("fixed", 500), ("fixed", 2000), ("fixed", 8000), ("fixed", 32000)]:
    t = time.time()
    sp = S.Spec(d=20, m=16, seeds=seeds, mode=mode, M=M, tau=tau, max_steps=int(80 / tau), check_every=200,
                stop_when_all_fit=True, train_loss_every=10, label=f"C {mode} M={M}")
    r = S.train(sp)
    save_result(r, f"exp1C_tau3e-4_{mode}" + (f"_M{M}" if M else ""))
    s = summarise(r, dict(tau=tau)); summ.append(s)
    print(mode, M, f"{time.time()-t:.0f}s", s["n_fitted"], s["alloc_at_fit"].get("median_log_err"), s["alloc_at_fit"].get("log_corr"), flush=True)
dump_json(summ, "exp1C_summary")
