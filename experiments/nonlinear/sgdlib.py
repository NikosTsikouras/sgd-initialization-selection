"""Vectorised joint-SGD engine for the ICLR-2027 rebuttal experiments.

Model, normalisation and initialisation follow the authors' notebook
(SGD_aligns.ipynb, cells 0 and 4) and Appendix H of the submission:

    f(x) = m^{-1/2} sum_j a_j ReLU(w_j^T x),   y(x) = beta theta_*^T x,
    theta_* = e_1,  p0 = beta alpha^2 / sqrt(m),  |a_j(0)| = ||w_j(0)|| = sqrt(p0),
    signs: first m/2 neurons +1, the rest -1,
    eta = tau sqrt(m) / beta, simultaneous update of both layers,
    no weight decay / clipping / normalisation / rebalancing.

Two data models are supported:

    mode="online": a fresh i.i.d. mini-batch at every step (Appendix H as run);
    mode="fixed":  one dataset of M examples is drawn once; every step uses one
                   index drawn uniformly *with replacement* (Theorems 1-2).

Many independent runs (different seeds and/or hyper-parameters) are stacked
along a leading "run" axis R and advanced together.  Every run owns three
independent random streams derived from its seed: initialisation, data, and
index sampling.  Therefore an online run and a fixed-dataset run with the same
seed start from *identical* parameters (paired comparison).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

SQRT3 = math.sqrt(3.0)


# -----------------------------------------------------------------------------
# Input distributions (all centrally symmetric with identity covariance before
# the optional linear map L, so that Cov(X) = L L^T)
# -----------------------------------------------------------------------------

def sample_inputs(rng: np.random.Generator, shape, dist: str) -> np.ndarray:
    if dist == "gaussian":
        return rng.standard_normal(shape)
    if dist == "laplace":  # iid standardized Laplace coordinates
        return rng.laplace(0.0, 1.0 / math.sqrt(2.0), size=shape)
    if dist == "uniform":  # iid standardized uniform coordinates (cube)
        return rng.uniform(-SQRT3, SQRT3, size=shape)
    raise ValueError(dist)


@dataclass
class Spec:
    """One stacked experiment: R runs with common (d, m, mode, ...)."""

    d: int
    m: int
    seeds: Sequence[int]
    alpha: object = 0.01          # scalar or length-R array
    tau: object = 0.003           # scalar or length-R array
    beta: float = 1.0
    batch: int = 1
    mode: str = "fixed"           # "online" or "fixed"
    M: int = 2000                 # dataset size for mode="fixed"
    dist: str = "gaussian"
    L: Optional[np.ndarray] = None   # covariance square root (d x d), Gaussian only for exact loss
    noise_std: float = 0.0        # label noise, drawn once with the dataset
    init_dir: str = "abs_half"    # "abs_half": |<u,theta>|<=1/2 ; "sign_half": s<u,theta> >= -1/2
    max_steps: int = 20_000
    check_every: int = 20
    fit_thr: float = 1e-6
    align_thr: float = 0.05
    stop_when_all_fit: bool = True
    extra_after_fit: int = 0      # keep training this many steps after the last run fits
    n_test_mc: int = 20_000       # Monte-Carlo test set for non-Gaussian population loss
    record_shares: bool = False
    record_dirs_every: int = 0    # >0: store all directions every k checks (d=2 plots)
    label: str = ""
    act: str = "relu"             # "relu" or "tanh"
    teacher: tuple = (1.0, -1.0)  # y = beta*(p*relu(z) + q*relu(-z)); (1,-1) is the linear teacher
    train_loss_every: int = 1           # evaluate the full training loss every k checks
    snapshot_steps: Sequence[int] = ()   # store (a, w) at these steps
    aux_dir: Optional[np.ndarray] = None  # also track max angle to +-aux_dir (sign-selected)


def _per_run(x, R):
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(R, float(arr))
    assert arr.shape == (R,)
    return arr


# -----------------------------------------------------------------------------
# Initialisation (identical to the notebook up to vectorisation)
# -----------------------------------------------------------------------------

def initialise(spec: Spec):
    R, m, d = len(spec.seeds), spec.m, spec.d
    alpha = _per_run(spec.alpha, R)
    signs = np.ones(m)
    signs[m // 2:] = -1.0
    a = np.empty((R, m))
    w = np.empty((R, m, d))
    for r, seed in enumerate(spec.seeds):
        rng = np.random.default_rng([int(seed), 0])
        dirs = np.empty((m, d))
        for j in range(m):
            while True:
                c = rng.standard_normal(d)
                c /= np.linalg.norm(c)
                ok = (abs(c[0]) <= 0.5) if spec.init_dir == "abs_half" else (signs[j] * c[0] >= -0.5)
                if ok:
                    dirs[j] = c
                    break
        p0 = spec.beta * alpha[r] ** 2 / math.sqrt(m)
        rad = math.sqrt(p0)
        a[r] = signs * rad
        w[r] = rad * dirs
    return a, w, signs


def predicted_shares(a0, w0, signs, direction=None):
    """lambda_j = (G_j^0)^2 / sum_{i in S_{s_j}} (G_i^0)^2 with
    G_j^0 = (|a_j(0)| + s_j <direction, w_j(0)>)/2  (direction = theta_* = e_1 by default)."""
    if direction is None:
        proj = w0[..., 0]
    else:
        proj = w0 @ direction
    G = 0.5 * (np.abs(a0) + signs[None, :] * proj)
    return class_normalise(G ** 2, signs)


def class_normalise(x, signs):
    out = np.empty_like(x)
    for s in (1.0, -1.0):
        idx = signs == s
        out[:, idx] = x[:, idx] / x[:, idx].sum(axis=1, keepdims=True)
    return out


# -----------------------------------------------------------------------------
# Population loss
# -----------------------------------------------------------------------------

def _kern(rad_w, cos):
    cos = np.clip(cos, -1.0, 1.0); ang = np.arccos(cos); sin = np.sqrt(np.maximum(1 - cos * cos, 0.0))
    return rad_w * (sin + (np.pi - ang) * cos) / (2 * np.pi)

def teacher_values(z, beta, teacher):
    if isinstance(teacher, str):          # "tanh:k"  ->  y = beta * tanh(k z) / k   (odd, slope beta at 0)
        k = float(teacher.split(":")[1])
        return beta * np.tanh(k * z) / k
    p, q = teacher
    return beta * (p * np.maximum(z, 0.0) + q * np.maximum(-z, 0.0))

def exact_gaussian_loss(a, w, beta, L=None, teacher=(1.0, -1.0)):
    """Exact 1/2 E[(f(X) - beta X_1)^2] for X ~ N(0, L L^T); vectorised over runs."""
    R, m, d = w.shape
    theta = np.zeros(d)
    theta[0] = 1.0
    if L is not None:
        w = w @ L           # w'_j = L^T w_j  (row-vector convention)
        theta = L.T @ theta
    radii = np.linalg.norm(w, axis=-1)
    safe = np.maximum(radii, 1e-300)
    u = w / safe[..., None]
    cos = np.clip(u @ u.transpose(0, 2, 1), -1.0, 1.0)
    ang = np.arccos(cos)
    sin = np.sqrt(np.maximum(1.0 - cos * cos, 0.0))
    ker = radii[:, :, None] * radii[:, None, :] * (sin + (math.pi - ang) * cos) / (2 * math.pi)
    student = np.einsum("ri,rj,rij->r", a, a, ker) / (2.0 * m)
    if tuple(teacher) == (1.0, -1.0):
        cross = -beta * np.einsum("rj,rj->r", a, w @ theta) / (2.0 * math.sqrt(m))
        tterm = 0.5 * beta ** 2 * float(theta @ theta)
    else:
        assert L is None, "nonlinear teacher implemented for isotropic inputs"
        p, q = teacher
        c = (w @ theta) / np.maximum(radii, 1e-300)
        cross_e = p * _kern(radii, c) + q * _kern(radii, -c)          # E[y sigma(w^T X)] / beta
        cross = -beta * np.einsum("rj,rj->r", a, cross_e) / math.sqrt(m)
        tterm = 0.5 * beta ** 2 * (p * p + q * q) / 2.0
    return np.maximum(student + cross + tterm, 0.0)


def forward(a, w, X, actf=None):
    """X: (R, N, d) -> f: (R, N)."""
    m = a.shape[1]
    actf = actf or (lambda z: np.maximum(z, 0.0))
    act = actf(X @ w.transpose(0, 2, 1))
    return np.einsum("rnm,rm->rn", act, a) / math.sqrt(m)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

@dataclass
class Result:
    spec: Spec
    signs: np.ndarray
    a0: np.ndarray
    w0: np.ndarray
    lam: np.ndarray
    steps: np.ndarray
    pop_loss: np.ndarray        # (T, R)
    train_loss: np.ndarray      # (T, R) (nan for online)
    max_angle: np.ndarray       # (T, R)
    wmean_angle: np.ndarray     # (T, R) mass-weighted mean angle
    class_mass: np.ndarray      # (T, R, 2)  (P_+, P_-)
    fit_step: np.ndarray        # (R,) -1 if never
    align_step: np.ndarray      # (R,) -1 if never
    shares_fit: np.ndarray      # (R, m) nan if never
    angles_fit: np.ndarray      # (R, m)
    a_fit: np.ndarray
    w_fit: np.ndarray
    a_end: np.ndarray
    w_end: np.ndarray
    diverged: np.ndarray        # (R,) bool
    share_hist: Optional[np.ndarray] = None   # (T, R, m)
    dir_hist: Optional[np.ndarray] = None
    dir_hist_steps: Optional[np.ndarray] = None
    X: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    aux_max_angle: Optional[np.ndarray] = None   # (T, R)
    snapshots: Optional[dict] = None             # step -> (a, w)


def angles_to_poles(w, signs):
    nrm = np.maximum(np.linalg.norm(w, axis=-1), 1e-300)
    cos = np.clip(signs[None, :] * w[..., 0] / nrm, -1.0, 1.0)
    return np.arccos(cos)


def masses(a, w):
    return np.abs(a) * np.linalg.norm(w, axis=-1)


def train(spec: Spec, verbose: bool = False) -> Result:
    R, m, d, B = len(spec.seeds), spec.m, spec.d, spec.batch
    beta = spec.beta
    sqm = math.sqrt(m)
    tau = _per_run(spec.tau, R)
    eta = tau * sqm / beta
    a, w, signs = initialise(spec)
    a0, w0 = a.copy(), w.copy()
    lam = predicted_shares(a0, w0, signs)
    L = spec.L
    exact = spec.dist == "gaussian" and spec.act == "relu" and not isinstance(spec.teacher, str)
    _act = (lambda z: np.maximum(z, 0.0)) if spec.act == "relu" else np.tanh

    data_rngs = [np.random.default_rng([int(s), 1]) for s in spec.seeds]
    idx_rngs = [np.random.default_rng([int(s), 2]) for s in spec.seeds]

    def draw(rng, shape):
        z = sample_inputs(rng, shape, spec.dist)
        return z @ L.T if L is not None else z

    X = y = None
    if spec.mode == "fixed":
        X = np.stack([draw(g, (spec.M, d)) for g in data_rngs])            # (R, M, d)
        clean = teacher_values(X[..., 0], beta, spec.teacher)
        noise = np.stack([g.standard_normal(spec.M) for g in data_rngs]) * spec.noise_std
        y = clean + noise
    X_test = None
    if not exact:
        trng = np.random.default_rng(987654321)
        X_test = draw(trng, (spec.n_test_mc, d))[None]                     # shared test set

    def pop_loss(a_, w_):
        if exact:
            return exact_gaussian_loss(a_, w_, beta, L, spec.teacher)
        f = forward(a_, w_, np.broadcast_to(X_test, (R,) + X_test.shape[1:]), _act)
        return 0.5 * np.mean((f - teacher_values(X_test[..., 0], beta, spec.teacher)) ** 2, axis=1)

    def train_loss(a_, w_):
        if X is None:
            return np.full(R, np.nan)
        f = forward(a_, w_, X, _act)
        return 0.5 * np.mean((f - y) ** 2, axis=1)

    # records
    steps, PL, TL, MA, WA, CM, SH, DH, DHS, AUX = [], [], [], [], [], [], [], [], [], []
    snaps = {}
    snap_set = set(int(t) for t in spec.snapshot_steps)
    if 0 in snap_set:
        snaps[0] = (a.copy(), w.copy())
    fit_step = -np.ones(R, dtype=int)
    align_step = -np.ones(R, dtype=int)
    shares_fit = np.full((R, m), np.nan)
    angles_fit = np.full((R, m), np.nan)
    a_fit = np.full_like(a, np.nan)
    w_fit = np.full_like(w, np.nan)
    diverged = np.zeros(R, dtype=bool)
    n_checks = 0

    def record(step):
        nonlocal n_checks
        pl = pop_loss(a, w)
        pl[diverged] = np.nan
        tl = train_loss(a, w) if n_checks % spec.train_loss_every == 0 else np.full(R, np.nan)
        ang = angles_to_poles(w, signs)
        p = masses(a, w)
        ma = ang.max(axis=1)
        wa = (p * ang).sum(1) / np.maximum(p.sum(1), 1e-300)
        cm = np.stack([p[:, signs > 0].sum(1), p[:, signs < 0].sum(1)], axis=1) / sqm
        sh = class_normalise(np.maximum(p, 1e-300), signs)
        if spec.aux_dir is not None:
            v = spec.aux_dir / np.linalg.norm(spec.aux_dir)
            nrm = np.maximum(np.linalg.norm(w, axis=-1), 1e-300)
            AUX.append(np.arccos(np.clip(signs[None, :] * (w @ v) / nrm, -1, 1)).max(1))
        steps.append(step); PL.append(pl); TL.append(tl); MA.append(ma); WA.append(wa); CM.append(cm)
        if spec.record_shares:
            SH.append(sh)
        if spec.record_dirs_every and n_checks % spec.record_dirs_every == 0:
            DH.append(w / np.maximum(np.linalg.norm(w, axis=-1, keepdims=True), 1e-300))
            DHS.append(step)
        newly_al = (align_step < 0) & (ma <= spec.align_thr) & ~diverged
        align_step[newly_al] = step
        newly = (fit_step < 0) & (pl <= spec.fit_thr) & ~diverged
        if newly.any():
            fit_step[newly] = step
            shares_fit[newly] = sh[newly]
            angles_fit[newly] = ang[newly]
            a_fit[newly] = a[newly]
            w_fit[newly] = w[newly]
        n_checks += 1

    record(0)
    chunk = 2000
    stop_at = spec.max_steps
    step = 0
    r_idx = np.arange(R)[:, None]
    while step < stop_at:
        n = min(chunk, stop_at - step)
        if spec.mode == "online":
            xs = np.stack([draw(g, (n, B, d)) for g in data_rngs], axis=1)   # (n, R, B, d)
            ys = teacher_values(xs[..., 0], beta, spec.teacher)
        else:
            ids = np.stack([g.integers(0, spec.M, size=(n, B)) for g in idx_rngs], axis=1)  # (n,R,B)
        for k in range(n):
            if spec.mode == "online":
                x, yt = xs[k], ys[k]
            else:
                x = X[r_idx, ids[k]]            # (R, B, d)
                yt = y[r_idx, ids[k]]           # (R, B)
            pre = x @ w.transpose(0, 2, 1)      # (R, B, m)
            if spec.act == "relu":
                act = np.maximum(pre, 0.0)
                dact = (pre > 0.0)
            else:
                act = np.tanh(pre)
                dact = 1.0 - act * act
            res = (act @ a[:, :, None])[..., 0] / sqm - yt      # (R, B)
            ga = np.einsum("rbm,rb->rm", act, res) / (B * sqm)
            gate = dact * res[:, :, None]
            gw = (gate.transpose(0, 2, 1) @ x) * a[:, :, None] / (B * sqm)
            a -= eta[:, None] * ga
            w -= eta[:, None, None] * gw
            step += 1
            if step in snap_set:
                snaps[step] = (a.copy(), w.copy())
            if step % spec.check_every == 0:
                bad = ~(np.isfinite(a).all(1) & np.isfinite(w).all((1, 2))) | (np.abs(a).max(1) > 1e8)
                if bad.any():
                    diverged |= bad
                    a[bad] = 0.0
                    w[bad] = 0.0
                record(step)
                done = (fit_step >= 0) | diverged
                if spec.stop_when_all_fit and done.all() and stop_at == spec.max_steps:
                    stop_at = min(spec.max_steps, step + spec.extra_after_fit)
                    if stop_at <= step:
                        break
        if verbose:
            print(f"  [{spec.label}] step {step}: fitted {int((fit_step>=0).sum())}/{R}", flush=True)

    T = lambda lst: np.asarray(lst)
    return Result(
        spec=spec, signs=signs, a0=a0, w0=w0, lam=lam, steps=T(steps), pop_loss=T(PL), train_loss=T(TL),
        max_angle=T(MA), wmean_angle=T(WA), class_mass=T(CM), fit_step=fit_step, align_step=align_step,
        shares_fit=shares_fit, angles_fit=angles_fit, a_fit=a_fit, w_fit=w_fit, a_end=a.copy(), w_end=w.copy(),
        diverged=diverged, share_hist=T(SH) if SH else None, dir_hist=T(DH) if DH else None,
        dir_hist_steps=T(DHS) if DHS else None, X=X, y=y,
        aux_max_angle=T(AUX) if AUX else None, snapshots=snaps,
    )


# -----------------------------------------------------------------------------
# Allocation metrics (same definitions as the notebook / paper)
# -----------------------------------------------------------------------------

def allocation_metrics(lam, obs, signs, mask=None):
    """lam, obs: (R, m). Returns pooled metrics over runs in `mask`."""
    if mask is None:
        mask = np.isfinite(obs).all(1)
    if mask.sum() == 0:
        return dict(n_runs=0)
    L_ = lam[mask].ravel()
    O_ = obs[mask].ravel()
    logerr = np.abs(np.log(O_ / L_))
    rel = np.abs(O_ / L_ - 1.0)
    # uniform-allocation baseline: predict 1/|S_s| for every neuron
    m = lam.shape[1]
    uni = np.where(signs > 0, 1.0 / (signs > 0).sum(), 1.0 / (signs < 0).sum())
    U_ = np.broadcast_to(uni, lam.shape)[mask].ravel()
    uni_err = np.abs(np.log(O_ / U_))
    per_run = np.median(np.abs(np.log(obs[mask] / lam[mask])), axis=1)
    corr = np.corrcoef(np.log(L_), np.log(O_))[0, 1] if L_.std() > 0 and O_.std() > 0 else np.nan
    return dict(
        n_runs=int(mask.sum()),
        median_log_err=float(np.median(logerr)),
        p90_rel_err=float(np.quantile(rel, 0.9)),
        max_log_err=float(np.max(logerr)),
        log_corr=float(corr),
        median_log_err_uniform_baseline=float(np.median(uni_err)),
        per_run_median_log_err_q=[float(q) for q in np.quantile(per_run, [0.0, 0.5, 1.0])],
    )


def shares_from(a, w, signs):
    return class_normalise(np.maximum(masses(a, w), 1e-300), signs)
