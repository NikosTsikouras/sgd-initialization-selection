"""Analysis + figures for Experiment 5 (noisy labels). EMPIRICAL: outside the theorems' noiseless setting."""
import json
import numpy as np
import matplotlib.pyplot as plt
import sgdlib as S
from common import RES, style, savefig, dump_json

style()
SIG = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]
CS = plt.cm.viridis(np.linspace(0, 0.9, len(SIG)))
SNAPS = [5000, 7500, 10000, 15000, 20000, 30000, 50000, 100000, 150000, 200000]
TREF = 10000

def med_abs_log(x, y):
    return np.median(np.abs(np.log(x / y)), axis=1)

def analyse(M):
    rows = []; ts = {}
    twin = np.load(RES / f"exp5_M{M}_sigma0.0.npz")
    for sg in SIG:
        d = np.load(RES / f"exp5_M{M}_sigma{sg}.npz")
        X, y, signs, lam = d["X"], d["y"], d["signs"], d["lam"]
        R = X.shape[0]
        th = np.stack([np.linalg.lstsq(X[r], y[r], rcond=None)[0] for r in range(R)])
        ols_train = np.array([0.5 * np.mean((X[r] @ th[r] - y[r]) ** 2) for r in range(R)])
        e1 = np.zeros(20); e1[0] = 1
        ols_clean = 0.5 * np.sum((th - e1) ** 2, 1)
        that = th / np.linalg.norm(th, axis=1, keepdims=True)
        ref = S.shares_from(d[f"snap_a_{TREF}"], d[f"snap_w_{TREF}"], signs)
        for t in SNAPS:
            a, w = d[f"snap_a_{t}"], d[f"snap_w_{t}"]
            at, wt = twin[f"snap_a_{t}"], twin[f"snap_w_{t}"]
            sh = S.shares_from(a, w, signs); sh0 = S.shares_from(at, wt, signs)
            clean = S.exact_gaussian_loss(a, w, 1.0)
            f = S.forward(a, w, X); train = 0.5 * np.mean((f - y) ** 2, 1)
            nrm = np.linalg.norm(w, axis=-1)
            ang_theta = np.arccos(np.clip(signs * w[..., 0] / nrm, -1, 1)).max(1)
            ang_hat = np.arccos(np.clip(signs * np.einsum("rmd,rd->rm", w, that) / nrm, -1, 1)).max(1)
            q = lambda v: [float(np.quantile(v, 0.1)), float(np.median(v)), float(np.quantile(v, 0.9))]
            rows.append(dict(M=M, sigma=sg, step=t,
                             clean_pop_loss=q(clean), train_loss=q(train), ols_train_loss=q(ols_train),
                             ols_clean_loss=q(ols_clean), train_minus_ols=q(train - ols_train),
                             clean_minus_ols=q(clean - ols_clean),
                             max_angle_to_theta=q(ang_theta), max_angle_to_ols_dir=q(ang_hat),
                             alloc_err_vs_pred=q(med_abs_log(sh, lam)), drift_vs_t10k=q(med_abs_log(sh, ref)),
                             diff_vs_noiseless_twin=q(med_abs_log(sh, sh0)),
                             alloc_log_corr=float(np.corrcoef(np.log(lam).ravel(), np.log(sh).ravel())[0, 1])))
        ts[sg] = dict(steps=d["steps"], pop=d["pop_loss"], train=d["train_loss"], ang=d["max_angle"],
                      share_hist=d["share_hist"], lam=lam, ols_clean=ols_clean, ols_train=ols_train)
    return rows, ts

def figure(M, rows, ts):
    fig, axs = plt.subplots(1, 4, figsize=(15.5, 3.6), constrained_layout=True)
    for c, sg in zip(CS, SIG):
        T = ts[sg]; st = T["steps"]
        axs[0].plot(st, np.nanmedian(np.maximum(T["pop"], 1e-14), 1), color=c, label=f"σ={sg}")
        if sg > 0:
            axs[0].axhline(np.median(T["ols_clean"]), color=c, ls=":", lw=1)
        tr = T["train"]; ok = np.isfinite(tr).all(1)
        if sg > 0:
            axs[1].plot(st[ok], np.median(tr[ok] / T["ols_train"][None], 1), color=c, label=f"σ={sg}")
        axs[2].plot(st, np.median(T["ang"], 1), color=c, label=f"σ={sg}")
        # share drift relative to the first check after TREF
        i0 = np.searchsorted(st, TREF)
        H = T["share_hist"]
        drift = np.median(np.abs(np.log(H[i0:] / H[i0][None])), axis=2)       # (T', R)
        err = np.median(np.abs(np.log(H / T["lam"][None])), axis=2)
        axs[3].plot(st[i0:], np.median(drift, 1), color=c, lw=1.4)
        axs[3].plot(st, np.median(err, 1), color=c, lw=0.9, ls="--")
    for ax in axs: ax.set_xscale("log"); ax.set_xlim(1000, 2.1e5)
    axs[0].set_yscale("log"); axs[0].set(xlabel="SGD step", ylabel="clean population loss",
                                         title="Clean population loss\n(dotted: OLS on the same noisy data)")
    axs[1].axhline(1, color="k", lw=0.8); axs[1].set_yscale("log"); axs[1].set_xlim(6000, 2.1e5)
    axs[1].set_ylim(top=1.6)
    axs[1].set(xlabel="SGD step", ylabel="train loss / OLS train loss",
               title="Training loss vs best linear fit\n(<1 ⇒ fitting noise beyond linear)")
    axs[2].set_yscale("log"); axs[2].set(xlabel="SGD step", ylabel="max angle to ±θ⋆ (rad)", title="Worst-neuron angle to teacher poles")
    axs[3].set_yscale("log"); axs[3].set(xlabel="SGD step", ylabel="median |log share ratio|",
                                         title=f"Shares: drift since step {TREF:,} (solid)\nerror vs Theorem-2 prediction (dashed)")
    axs[0].legend(frameon=False, fontsize=7, ncol=2)
    p = 16 * 21
    fig.suptitle(f"Exp. 5 — noisy labels, fixed dataset M={M} (m=16 ⇒ {p} parameters; d=20, α=0.01, τ=0.003, B=1; medians of 20 seeds)  [EMPIRICAL]",
                 fontweight="bold", fontsize=10.5)
    savefig(fig, f"exp5_noise_M{M}")

if __name__ == "__main__":
    allrows = []
    for M in (2000, 200):
        rows, ts = analyse(M); allrows += rows; figure(M, rows, ts)
    dump_json(allrows, "exp5_analysis")
    for r in allrows:
        if r["step"] in (10000, 50000, 200000):
            f = lambda k: r[k][1]
            print(f"M={r['M']:5d} σ={r['sigma']:<5} t={r['step']:6d} clean {f('clean_pop_loss'):.2e} (ols {f('ols_clean_loss'):.2e}) "
                  f"train {f('train_loss'):.2e} (ols {f('ols_train_loss'):.2e}) angθ {f('max_angle_to_theta'):.3f} angθ̂ {f('max_angle_to_ols_dir'):.3f} "
                  f"err {f('alloc_err_vs_pred'):.3f} drift {f('drift_vs_t10k'):.4f} twin {f('diff_vs_noiseless_twin'):.4f}")
