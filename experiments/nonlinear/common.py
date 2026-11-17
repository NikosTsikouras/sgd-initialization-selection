"""Shared helpers: saving results, summaries, plotting style."""
import json, math, os
from pathlib import Path
import numpy as np
import sgdlib as S

ROOT = Path(__file__).resolve().parent
RES = Path(os.environ.get("SGD_OUTPUT_DIR", str(ROOT / "results"))).resolve(); FIG = RES / "diagnostic_figures"
RES.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

def save_result(r: S.Result, name: str, keep_dirs=False, keep_data=False):
    d = dict(steps=r.steps, pop_loss=r.pop_loss, train_loss=r.train_loss, max_angle=r.max_angle,
             wmean_angle=r.wmean_angle, class_mass=r.class_mass, fit_step=r.fit_step, align_step=r.align_step,
             shares_fit=r.shares_fit, angles_fit=r.angles_fit, lam=r.lam, signs=r.signs, a0=r.a0, w0=r.w0,
             a_fit=r.a_fit, w_fit=r.w_fit, a_end=r.a_end, w_end=r.w_end, diverged=r.diverged,
             seeds=np.asarray(r.spec.seeds), alpha=np.asarray(r.spec.alpha), tau=np.asarray(r.spec.tau))
    if r.share_hist is not None: d["share_hist"] = r.share_hist
    if r.aux_max_angle is not None: d["aux_max_angle"] = r.aux_max_angle
    if keep_dirs and r.dir_hist is not None:
        d["dir_hist"] = r.dir_hist; d["dir_hist_steps"] = r.dir_hist_steps
    if keep_data and r.X is not None:
        d["X"] = r.X; d["y"] = r.y
    for t, (a, w) in (r.snapshots or {}).items():
        d[f"snap_a_{t}"] = a; d[f"snap_w_{t}"] = w
    np.savez_compressed(RES / f"{name}.npz", **d)

def summarise(r: S.Result, extra=None):
    fitted = r.fit_step >= 0
    met = S.allocation_metrics(r.lam, r.shares_fit, r.signs, mask=fitted)
    end_shares = S.shares_from(r.a_end, r.w_end, r.signs)
    ok = ~r.diverged
    met_end = S.allocation_metrics(r.lam, end_shares, r.signs, mask=ok) if ok.any() else {}
    fin = lambda x: [float(v) for v in x]
    out = dict(
        label=r.spec.label, mode=r.spec.mode, M=r.spec.M if r.spec.mode == "fixed" else None, d=r.spec.d, m=r.spec.m,
        batch=r.spec.batch, dist=r.spec.dist, n_runs=len(r.spec.seeds), n_fitted=int(fitted.sum()),
        n_diverged=int(r.diverged.sum()), steps_run=int(r.steps[-1]),
        fit_step_q=fin(np.quantile(r.fit_step[fitted], [0, .5, 1])) if fitted.any() else None,
        align_step_q=fin(np.quantile(r.align_step[r.align_step >= 0], [0, .5, 1])) if (r.align_step >= 0).any() else None,
        n_aligned_005=int((r.align_step >= 0).sum()),
        align_before_fit=int(((r.align_step >= 0) & fitted & (r.align_step <= r.fit_step)).sum()),
        max_angle_at_fit_q=fin(np.nanquantile(np.nanmax(r.angles_fit, 1), [0, .5, 1])) if fitted.any() else None,
        final_pop_loss_q=fin(np.nanquantile(r.pop_loss[-1], [0, .5, 1])) if ok.any() else None,
        final_max_angle_q=fin(np.nanquantile(r.max_angle[-1][ok], [0, .5, 1])) if ok.any() else None,
        alloc_at_fit=met, alloc_at_end=met_end,
    )
    if r.spec.mode == "fixed" and np.isfinite(r.train_loss[-1]).any():
        out["final_train_loss_q"] = fin(np.nanquantile(r.train_loss[-1], [0, .5, 1]))
    if extra: out.update(extra)
    return out

def dump_json(obj, name):
    def conv(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(type(o))
    (RES / f"{name}.json").write_text(json.dumps(obj, indent=1, default=conv))

def style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5, "legend.fontsize": 8,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 120, "savefig.dpi": 220,
                         "axes.grid": True, "grid.alpha": 0.18})

def savefig(fig, name):
    import matplotlib.pyplot as plt
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)

def band(ax, x, Y, color, label, lw=1.6, ls="-", log=False):
    """Median line with inter-quartile band across runs (Y: T x R)."""
    med = np.nanmedian(Y, 1); lo = np.nanquantile(Y, .25, 1); hi = np.nanquantile(Y, .75, 1)
    ax.plot(x, med, color=color, lw=lw, ls=ls, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0)
