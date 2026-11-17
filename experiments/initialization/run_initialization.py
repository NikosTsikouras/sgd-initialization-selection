
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import os, sys
from pathlib import Path
OUT = Path(os.environ.get("SGD_OUTPUT_DIR", str(Path(__file__).resolve().parents[2]/"results/initialization"))).resolve()
OUT.mkdir(parents=True, exist_ok=True)
device = os.environ.get("SGD_DEVICE", "cpu")  # CPU reproduces the archived reported experiment.
if device == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA requested but unavailable")
torch.set_num_threads(2)
torch.manual_seed(2027)

# -----------------------------------------------------------
# Settings: paper-like baseline from Appendix H
# -----------------------------------------------------------
D = 20
WIDTH = 16
BETA = 1.0
ALPHA = 0.01
TAU = 0.003
ETA = TAU * math.sqrt(WIDTH) / BETA

N_SEEDS = int(os.environ.get("N_SEEDS", 20))              # Increase to 20 for reported results
DATASET_SIZE = 8192      # One fixed dataset per seed
STEPS = int(os.environ.get("SGD_STEPS", 15000))
CHECK_EVERY = int(os.environ.get("SGD_CHECK_EVERY", 100))
FIT_LOSS = 1e-6          # Population loss = 0.5 * MSE

m = WIDTH
sqrt_m = math.sqrt(m)

# Each configuration changes one factor at a time.
configs = []

def add(axis, setting, scale=1.0, angle="valid", mag="balanced"):
    configs.append(dict(axis=axis, setting=setting,
                        scale=scale, angle=angle, mag=mag))

add("baseline", "paper-like")

# A. Initialization-scale sweep
for factor in [0.25, 0.5, 2, 5, 20, 100]:
    add("scale", f"{factor:g}x", scale=factor)

# B. Angular condition sweep
add("angle", "unrestricted", angle="unrestricted")
add("angle", "25% violating", angle="bad25")
add("angle", "100% violating", angle="bad100")

# C. Layer-magnitude distribution sweep
add("magnitude", "bounded independent", mag="bounded")
add("magnitude", "wide layer ratios", mag="wide_ratio")
add("magnitude", "Gaussian output", mag="gaussian_output")

# Combined stress test: outside theorem assumptions.
# ARCHIVED control retained to preserve the original random stream.
# The final-paper Xavier-normal condition is experiments/xavier/run_xavier_normal.py.
add("stress_test", "Xavier-style", mag="xavier")

# NEW (v2): genuinely i.i.d. Gaussian output weights, NO sign balancing.
# a_j = sqrt(p0) * g_j, g_j ~ N(0,1); s_j = sign(g_j); |w_j(0)| = sqrt(p0).
add("magnitude", "iid Gaussian output", mag="iid_gauss_out", angle="own_valid")
add("magnitude", "iid Gaussian output, unrestricted dirs", mag="iid_gauss_out", angle="unrestricted")
# Same Gaussian law for BOTH layers but at the paper's small scale (separates scale from distribution):
# a_j = sqrt(p0) g_j,  w_j = sqrt(p0) z_j / sqrt(D),  g_j ~ N(0,1), z_j ~ N(0, I_D).
add("stress_test", "Gaussian both layers, small scale", mag="gauss_both_small")

# -----------------------------------------------------------
# Fixed datasets and common initial directions
# -----------------------------------------------------------
X_bank = torch.randn(
    N_SEEDS, DATASET_SIZE, D, device=device
)

# Exactly balanced sign classes at baseline
base_signs = torch.cat([
    torch.ones(N_SEEDS, m // 2, device=device),
    -torch.ones(N_SEEDS, m - m // 2, device=device)
], dim=1)
perm = torch.rand(N_SEEDS, m, device=device).argsort(dim=1)
base_signs = base_signs.gather(1, perm)

# Teacher theta_* = e_1
theta = torch.zeros(D, device=device)
theta[0] = 1.0

raw_dirs = F.normalize(
    torch.randn(N_SEEDS, m, D, device=device), dim=-1
)

# Uniform sphere directions conditioned on
# s_j <u_j, theta_*> >= -1/2
valid_dirs = raw_dirs.clone()
for _ in range(100):
    bad = base_signs * valid_dirs[..., 0] < -0.5
    if not bad.any().item():
        break
    replacement = F.normalize(
        torch.randn_like(valid_dirs), dim=-1
    )
    valid_dirs = torch.where(
        bad[..., None], replacement, valid_dirs
    )
else:
    raise RuntimeError("Angular rejection sampler failed.")

def make_initialization(cfg):
    if cfg["mag"] == "xavier":
        # Xavier-style stress test in our 1/sqrt(m)
        # output normalization, NOT an exact Tinaz reproduction.
        a = torch.randn(N_SEEDS, m, device=device)
        w = torch.randn(N_SEEDS, m, D, device=device) / math.sqrt(D)
        return a, w

    u = (raw_dirs if cfg["angle"] == "unrestricted"
         else valid_dirs).clone()

    if cfg["angle"] in ["bad25", "bad100"]:
        fraction = 0.25 if cfg["angle"] == "bad25" else 1.0
        k = round(fraction * m)

        chosen = torch.rand(
            N_SEEDS, m, device=device
        ).argsort(dim=1)[:, :k]

        mask = torch.zeros(
            N_SEEDS, m, dtype=torch.bool, device=device
        )
        mask.scatter_(1, chosen, True)

        # Force selected-pole cosines into [-0.9, -0.6],
        # violating the theorem's -1/2 lower bound.
        q = -0.9 + 0.3 * torch.rand(
            N_SEEDS, m, device=device
        )

        transverse = valid_dirs.clone()
        transverse[..., 0] = 0
        transverse = F.normalize(transverse, dim=-1)

        forced = (
            (base_signs * q)[..., None] * theta
            + torch.sqrt(1 - q.square())[..., None] * transverse
        )
        u = torch.where(mask[..., None], forced, u)

    alpha = ALPHA * cfg["scale"]
    p0 = BETA * alpha**2 / sqrt_m
    root = math.sqrt(p0)

    if cfg["mag"] == "iid_gauss_out":
        g = torch.randn(N_SEEDS, m, device=device)
        sg = torch.where(g >= 0, 1.0, -1.0)
        uu = F.normalize(torch.randn(N_SEEDS, m, D, device=device), dim=-1)
        if cfg["angle"] == "own_valid":   # s_j <u_j, theta> >= -1/2 w.r.t. the drawn sign
            for _ in range(200):
                bad = sg * uu[..., 0] < -0.5
                if not bad.any().item():
                    break
                uu = torch.where(bad[..., None], F.normalize(torch.randn_like(uu), dim=-1), uu)
        return root * g, root * uu
    if cfg["mag"] == "gauss_both_small":
        return (root * torch.randn(N_SEEDS, m, device=device),
                root * torch.randn(N_SEEDS, m, D, device=device) / math.sqrt(D))

    b = torch.full((N_SEEDS, m), root, device=device)
    r = torch.full((N_SEEDS, m), root, device=device)

    if cfg["mag"] == "bounded":
        # Both layer magnitudes in [sqrt(p0), 1.25 sqrt(p0)]
        b *= 1 + 0.25 * torch.rand_like(b)
        r *= 1 + 0.25 * torch.rand_like(r)

    elif cfg["mag"] == "wide_ratio":
        # Preserve the effective mass b*r = p0, but vary
        # how it is split between the two layers.
        z = 0.7 * torch.randn_like(b)
        b *= torch.exp(z)
        r *= torch.exp(-z)

    elif cfg["mag"] == "gaussian_output":
        # Folded-Gaussian magnitudes, with balanced signs.
        # Changes individual effective masses as well.
        b *= torch.randn_like(b).abs()

    a = base_signs * b
    w = r[..., None] * u
    return a, w

initial = [make_initialization(c) for c in configs]
A0 = torch.cat([v[0] for v in initial], dim=0)
W0 = torch.cat([v[1] for v in initial], dim=0)

R = A0.shape[0]
seed_ids = torch.arange(
    N_SEEDS, device=device
).repeat(len(configs))
row_ids = torch.arange(R, device=device)

initial_signs = torch.where(A0 >= 0, 1.0, -1.0)

# -----------------------------------------------------------
# Allocation prediction from Theorem 2
# -----------------------------------------------------------
def within_class_shares(mass, signs):
    positive = signs > 0
    p_plus = (mass * positive).sum(dim=1, keepdim=True)
    p_minus = (mass * (~positive)).sum(dim=1, keepdim=True)
    denominator = torch.where(positive, p_plus, p_minus)
    return mass / denominator.clamp_min(1e-30)

G0 = 0.5 * (
    A0.abs() + initial_signs * W0[..., 0]
)
predicted_shares = within_class_shares(
    G0.square(), initial_signs
)

# Exact Gaussian population loss, rather than
# a Monte Carlo estimate on a validation dataset.
def population_loss(a, w):
    a = a.double()
    w = w.double()

    norms = w.norm(dim=-1).clamp_min(1e-30)
    u = w / norms[..., None]

    cosine = torch.bmm(
        u, u.transpose(1, 2)
    ).clamp(-1.0, 1.0)
    angle = torch.acos(cosine)

    kernel = (
        torch.sin(angle)
        + (math.pi - angle) * cosine
    ) / (2 * math.pi)

    weights = a * norms
    ef2 = (
        weights[:, :, None]
        * weights[:, None, :]
        * kernel
    ).sum(dim=(1, 2)) / m

    cross_twice = (
        BETA / sqrt_m
    ) * (a * w[..., 0]).sum(dim=1)

    mse = (ef2 - cross_twice + BETA**2).clamp_min(0)
    return 0.5 * mse

def diagnostics(a, w):
    norms = w.norm(dim=-1).clamp_min(1e-30)
    masses = a.abs() * norms
    observed = within_class_shares(
        masses, initial_signs
    )

    log_error = (
        torch.log(observed.clamp_min(1e-30))
        - torch.log(predicted_shares.clamp_min(1e-30))
    ).abs()

    # Angles to poles selected by INITIAL output signs.
    cos_to_pole = (
        initial_signs * w[..., 0] / norms
    ).clamp(-1, 1)
    max_angle = torch.rad2deg(
        torch.acos(cos_to_pole)
    ).amax(dim=1)

    flips = (
        (torch.sign(a) != initial_signs)
        .float().mean(dim=1)
    )

    # Theorem 2 assumes no sign flips; do not treat a
    # sign-changing run as a valid allocation-law test.
    nan = torch.full_like(flips, float("nan"))
    mean_log_error = torch.where(
        flips == 0, log_error.mean(dim=1), nan
    )
    max_log_error = torch.where(
        flips == 0, log_error.amax(dim=1), nan
    )

    return max_angle, mean_log_error, max_log_error, flips

# NEW (v2): diagnostics that remain meaningful when signs flip.
EXTRA = ["flip_frac", "flip_mass_frac", "max_angle_cur", "wmean_angle_cur",
         "alloc_err_kept", "uniform_err_kept", "min_class_size"]
def extra_diagnostics(a, w):
    cur = torch.where(a >= 0, 1.0, -1.0)
    nrm = w.norm(dim=-1).clamp_min(1e-30)
    mass = (a.abs() * nrm).double()
    ang = torch.rad2deg(torch.acos((cur * w[..., 0] / nrm).clamp(-1, 1))).double()
    flipped = cur != initial_signs
    keep = ~flipped
    pos = initial_signs > 0
    # allocation among neurons that kept their sign, renormalised within the initial class;
    # classes with fewer than 2 kept neurons are excluded (a share of 1 is trivially "correct").
    n_pos = (keep & pos).sum(1, keepdim=True); n_neg = (keep & ~pos).sum(1, keepdim=True)
    ok = keep & torch.where(pos, n_pos, n_neg).ge(2)
    def norm_within(x):
        x = x * ok
        tp = (x * pos).sum(1, keepdim=True); tn = (x * ~pos).sum(1, keepdim=True)
        return x / torch.where(pos, tp, tn).clamp_min(1e-300)
    v = norm_within(mass); lam = norm_within(predicted_shares.double()); uni = norm_within(torch.ones_like(mass))
    L = lambda p: torch.log(p.clamp_min(1e-300))
    cnt = ok.sum(1).clamp_min(1)
    nan = torch.full((a.shape[0],), float("nan"), dtype=torch.float64, device=a.device)
    has = ok.sum(1) > 0
    return dict(
        flip_frac=flipped.double().mean(1),
        flip_mass_frac=(mass * flipped).sum(1) / mass.sum(1),
        max_angle_cur=ang.amax(1),
        wmean_angle_cur=(mass * ang).sum(1) / mass.sum(1),
        alloc_err_kept=torch.where(has, ((L(v) - L(lam)).abs() * ok).sum(1) / cnt, nan),
        uniform_err_kept=torch.where(has, ((L(v) - L(uni)).abs() * ok).sum(1) / cnt, nan),
        min_class_size=torch.minimum(pos.sum(1), (~pos).sum(1)).double(),
    )

fit_extra = {k: torch.full((A0.shape[0],), float("nan"), dtype=torch.float64, device=device) for k in EXTRA}

# -----------------------------------------------------------
# GPU training: batch-size-one SGD, fixed dataset
# All configurations and seeds run in parallel.
# No autograd, clipping, regularization, or rebalancing.
# -----------------------------------------------------------
a = A0.clone()
w = W0.clone()

fit_step = torch.full(
    (R,), -1, dtype=torch.int64, device=device
)
fit_angle = torch.full((R,), float("nan"), device=device)
fit_mean_error = fit_angle.clone()
fit_max_error = fit_angle.clone()
fit_flips = fit_angle.clone()

with torch.no_grad():
    for step in range(1, STEPS + 1):
        # Independent index sampled with replacement
        # from each run's fixed dataset.
        idx = torch.randint(
            DATASET_SIZE, (R,), device=device
        )
        x = X_bank[seed_ids, idx]

        preact = (w * x[:, None, :]).sum(dim=-1)
        activated = preact.clamp_min(0)

        prediction = (
            a * activated
        ).sum(dim=1) / sqrt_m
        target = BETA * x[:, 0]
        residual = prediction - target

        coefficient = (ETA / sqrt_m) * residual

        # Simultaneous updates using OLD parameters
        next_a = a - coefficient[:, None] * activated
        next_w = w - (
            coefficient[:, None, None]
            * (a * (preact > 0))[:, :, None]
            * x[:, None, :]
        )
        a, w = next_a, next_w

        if step % CHECK_EVERY != 0 and step != STEPS:
            continue

        loss = population_loss(a, w)
        angle, mean_err, max_err, flips = diagnostics(a, w)

        newly_fitted = (
            (fit_step < 0)
            & torch.isfinite(loss)
            & (loss <= FIT_LOSS)
        )

        fit_step = torch.where(
            newly_fitted, step, fit_step
        )
        fit_angle = torch.where(
            newly_fitted, angle, fit_angle
        )
        fit_mean_error = torch.where(
            newly_fitted, mean_err, fit_mean_error
        )
        fit_max_error = torch.where(
            newly_fitted, max_err, fit_max_error
        )
        fit_flips = torch.where(
            newly_fitted, flips, fit_flips
        )
        if bool(newly_fitted.any()):
            ex = extra_diagnostics(a, w)
            for k in EXTRA:
                fit_extra[k] = torch.where(newly_fitted, ex[k], fit_extra[k])

    final_loss = population_loss(a, w)
    final_angle, _, _, final_flips = diagnostics(a, w)
    final_extra = extra_diagnostics(a, w)

# -----------------------------------------------------------
# Results: individual seeds and aggregate summaries
# -----------------------------------------------------------
def to_numpy(t):
    return t.detach().cpu().numpy()

fs = to_numpy(fit_step)
fa = to_numpy(fit_angle)
fe = to_numpy(fit_mean_error)
fmax = to_numpy(fit_max_error)
ff = to_numpy(fit_flips)
fl = to_numpy(final_loss)
fang = to_numpy(final_angle)
fup = to_numpy(final_flips)

rows = []
for ci, cfg in enumerate(configs):
    for seed in range(N_SEEDS):
        k = ci * N_SEEDS + seed
        fitted = fs[k] >= 0
        rows.append(dict(
            axis=cfg["axis"],
            setting=cfg["setting"],
            seed=seed,
            fitted=fitted,
            fit_step=fs[k] if fitted else np.nan,
            max_angle_deg_at_fit=fa[k],
            mean_abs_log_share_error=fe[k],
            max_abs_log_share_error=fmax[k],
            sign_flips_at_fit=ff[k],
            stable_fit=bool(fitted and ff[k] == 0),
            final_population_loss=fl[k],
            final_max_angle_deg=fang[k],
            final_sign_flips=fup[k],
            **{f"fit_{e}": to_numpy(fit_extra[e])[k] for e in EXTRA},
            **{f"final_{e}": to_numpy(final_extra[e])[k] for e in EXTRA},
        ))

df = pd.DataFrame(rows)

summary = (
    df.groupby(["axis", "setting"], sort=False)
    .agg(
        fit_rate=("fitted", "mean"),
        stable_fit_rate=("stable_fit", "mean"),
        median_fit_step=("fit_step", "median"),
        median_angle_deg=("max_angle_deg_at_fit", "median"),
        median_mean_log_error=(
            "mean_abs_log_share_error", "median"
        ),
        median_max_log_error=(
            "max_abs_log_share_error", "median"
        ),
        median_final_loss=("final_population_loss", "median")
    )
    .reset_index()
)

pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 160)
print("Device:", device, "| N_SEEDS =", N_SEEDS)
print("\nSummary:")
print(summary.round(5).to_string(index=False))

df.to_csv(OUT / "reviewer2_v2_individual_runs.csv", index=False)
summary.to_csv(OUT / "reviewer2_v2_summary.csv", index=False)

