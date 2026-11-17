#!/usr/bin/env python3
"""Generate final-paper Tables 1--5 from saved or freshly trained results.

Standard library only. Defaults are resolved relative to the repository, not
its caller's working directory. Explicit relative paths use the caller's cwd.
Raw measurements are saved in JSON and CSV; the TeX uses the paper's rounding.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    with path.open() as stream:
        return json.load(stream)


def med(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    values = [value for value in values if math.isfinite(value)]
    return statistics.median(values) if values else None


def fitted(row):
    return str(row["fitted"]).lower() in ("true", "1")


def decimal(value, places=4):
    return "--" if value is None else f"{value:.{places}f}"


def error(record, key="alloc_at_fit", metric="median_log_err"):
    return record.get(key, {}).get(metric)


def check_runs(record):
    if record.get("n_runs", 20) != 20:
        raise ValueError("Paper tables require 20 seeds per condition: " + record.get("label", "unknown"))


def row(identifier, cells, values):
    return {"id": identifier, "cells": list(map(str, cells)), "values": values}


def table(number, filename, label, headers, rows, caption, align, breaks=()):
    return dict(number=number, filename=filename, label=label, headers=headers,
                rows=rows, caption=caption, align=align, breaks=list(breaks))


def sampling(data):
    rows = []
    base = [x for x in read_json(data / "core/exp1B_summary.json")
            if x["batch"] == 1 and (x["mode"] == "online" or x["M"] == 2000)]
    base.sort(key=lambda x: (x["m"], x["mode"] != "online"))
    small = read_json(data / "core/exp1C_summary.json")
    small.sort(key=lambda x: -1 if x["mode"] == "online" else x["M"])
    for block, records in (("baseline", base), ("small_step", small)):
        for x in records:
            check_runs(x)
            tau = x.get("tau", 0.003 if block == "baseline" else 0.0003)
            allocation = error(x)
            corr = error(x, metric="log_corr")
            dataset = x["M"] if x["mode"] == "fixed" else None
            rows.append(row(f"{block}_{x['mode']}_m{x['m']}_M{dataset}",
                            [x["m"], x["mode"].capitalize(), dataset if dataset else "--", f"{tau:g}",
                             f"{x['n_fitted']}/20", decimal(allocation), decimal(corr)],
                            dict(block=block, width=x["m"], sampling=x["mode"], dataset_size=dataset,
                                 tau=tau, n_fitted=x["n_fitted"], n_runs=20,
                                 pooled_log_error=allocation, log_correlation=corr)))
    return table(1, "sampling_dataset_size", "tab:h-sampling",
                 [r"$m$", "Sampling", r"$M$", r"$\tau$", "Fits", r"$E_{\rm pool}$", "Log corr."], rows,
                 r"Online and fixed-dataset SGD ($B=1$, 20 seeds per row). Allocation is evaluated at the first "
                 r"monitored loss at most $10^{-6}$. $E_{\rm pool}$ pools absolute log-share errors over neurons "
                 r"and fitted runs. The lower block is the separate dataset-size sweep at a smaller learning rate.",
                 "rlrrrrr", [len(base)])


INIT_ORDER = [
    ("paper-like", "Baseline"), ("0.25x", r"Scale $0.25\times$"),
    ("0.5x", r"Scale $0.5\times$"), ("2x", r"Scale $2\times$"),
    ("5x", r"Scale $5\times$"), ("20x", r"Scale $20\times$"),
    ("100x", r"Scale $100\times$"), ("xavier_normal", "Xavier normal"),
    ("unrestricted", "Unrestricted directions"),
    ("25% violating", r"25\% violating angle condition"),
    ("100% violating", r"100\% violating angle condition"),
    ("bounded independent", "Bounded independent magnitudes"),
    ("wide layer ratios", "Wide layer ratios"),
    ("iid Gaussian output", r"i.i.d. Gaussian $a$"),
    ("iid Gaussian output, unrestricted dirs", r"i.i.d. Gaussian $a$, unrestricted directions"),
    ("Gaussian both layers, small scale", r"Small Gaussian $a,w$"),
]
RETAINED = {name for name, _ in INIT_ORDER[-4:]}


def initialization(data):
    with (data / "initialization/reviewer2_v2_individual_runs.csv").open(newline="") as stream:
        individuals = list(csv.DictReader(stream))
    xavier = read_json(data / "xavier/summary.json")
    if xavier["protocol"]["n_seeds"] != 20:
        raise ValueError("Xavier table requires 20 seeds.")
    rows = []
    for condition, title in INIT_ORDER:
        retained = condition in RETAINED
        if condition == "xavier_normal":
            n_fit = xavier["n_fitted"]
            at_end = n_fit == 0
            angle = xavier["final" if at_end else "at_fit"]["max_angle_initial"]["median"]
            # The published control has no fits. If a changed experiment does fit,
            # do not silently combine its diagnostics with the archived ablation.
            if n_fit:
                raise ValueError("New Xavier runs fit: define fitting-time allocation and uniform diagnostics before publishing.")
            allocation = uniform = None
        else:
            group = [r for r in individuals if r["setting"] == condition]
            if len(group) != 20:
                raise ValueError(f"Expected 20 initialization rows for {condition!r}; found {len(group)}")
            fits = [r for r in group if fitted(r)]
            n_fit = len(fits)
            if not retained and any(float(r["fit_flip_frac"]) > 0 for r in fits):
                raise ValueError(f"Unexpected sign changes in {condition}: specify matched allocation/uniform subsets.")
            at_end = not fits
            angle_key = "fit_max_angle_cur" if retained and fits else (
                "max_angle_deg_at_fit" if fits else "final_max_angle_deg")
            if retained and at_end:
                angle_key = "final_max_angle_cur"
            angle = med(fits or group, angle_key) * math.pi / 180
            allocation = med(fits, "fit_alloc_err_kept" if retained else "mean_abs_log_share_error")
            uniform = med(fits, "fit_uniform_err_kept")
        angle_cell = decimal(angle, 2 if retained or at_end else 3)
        if at_end:
            angle_cell += r"$^{*}$"
        if retained:
            angle_cell += r"$^{\dagger}$"
        allocation_cell = decimal(allocation, 3 if retained else 4)
        uniform_cell = decimal(uniform, 2)
        if retained and allocation is not None:
            allocation_cell += r"$^{\dagger}$"
        if retained and uniform is not None:
            uniform_cell += r"$^{\dagger}$"
        rows.append(row(condition, [title, f"{n_fit}/20", angle_cell, allocation_cell, uniform_cell],
                        dict(condition=condition, n_fitted=n_fit, n_runs=20, angle_radians=angle,
                             angle_poles="current" if retained else "initial",
                             checkpoint="final" if at_end else "first_fit",
                             allocation_mean_error_median=allocation, uniform_mean_error_median=uniform,
                             retained_neurons=retained)))
    return table(2, "initialization", "tab:h-initialization",
                 ["Initialization", "Fits", r"\shortstack{Max. angle\\(rad)}",
                  r"\shortstack{Allocation\\error}", "Uniform"], rows,
                 r"Initialization ablations ($M=8192$, 20 seeds per condition, 15,000 updates). Angles are "
                 r"medians across runs of the maximum neuron angle to the initial sign-selected teacher pole, "
                 r"except $\dagger$ entries, which use current signs. Measurements are at the first fitting "
                 r"checkpoint, except $*$ entries, measured at the end of the budget. Allocation errors are "
                 r"medians across runs of the within-run mean absolute log-share error; ``Uniform'' uses the "
                 r"corresponding uniform prediction. For $\dagger$ allocation and uniform errors, only neurons "
                 r"retaining their initial sign are included, predictions and observations are renormalized "
                 r"within each retained sign class, and classes with fewer than two retained neurons are "
                 r"excluded. Dashes indicate no fit.", "lrrrr")


def input_records(data):
    """Read compact archived records, or derive them from fresh training summaries."""
    compact = data / "core/exp3_table.json"
    if compact.exists():
        return [x for x in read_json(compact) if x["mode"] == "fixed" and x["m"] in (16, 64)]
    result = []
    for width in (16, 64):
        for x in read_json(data / f"core/exp3B{width}_summary.json"):
            if x["mode"] != "fixed":
                continue
            check_runs(x)
            distribution = x["dist"] + ("_rot" if x.get("rotated") else "")
            result.append(dict(m=x["m"], mode=x["mode"], dist=distribution, n_fitted=x["n_fitted"],
                               align_before_fit=x["align_before_fit"],
                               max_angle_at_fit_median=x["max_angle_at_fit_q"][1] if x["n_fitted"] else None,
                               alloc_median_log_err=error(x),
                               uniform_guess_err=error(x, metric="median_log_err_uniform_baseline")))
    return result


def nongaussian(data):
    laws = [("gaussian", "Gaussian"), ("laplace", "Laplace"), ("uniform", "Uniform"),
            ("laplace_rot", "Rotated Laplace"), ("uniform_rot", "Rotated uniform")]
    rank = {law: index for index, (law, _) in enumerate(laws)}
    names = dict(laws)
    records = sorted(input_records(data), key=lambda x: (x["m"], rank[x["dist"]]))
    rows = []
    for x in records:
        if x["n_fitted"] != 20 or x["align_before_fit"] != 20:
            raise ValueError("Input-law results changed: update the all-runs-fit-and-align caption.")
        rows.append(row(f"m{x['m']}_{x['dist']}",
                        [x["m"], names[x["dist"]], f"{x['n_fitted']}/20",
                         decimal(x["max_angle_at_fit_median"]), decimal(x["alloc_median_log_err"]),
                         decimal(x["uniform_guess_err"], 3)],
                        dict(width=x["m"], input_law=x["dist"], n_fitted=x["n_fitted"], n_runs=20,
                             max_angle_median=x["max_angle_at_fit_median"], pooled_log_error=x["alloc_median_log_err"],
                             uniform_pooled_log_error=x["uniform_guess_err"])))
    return table(3, "nongaussian", "tab:h-nongaussian",
                 [r"$m$", "Input law", "Fits", "Max. angle (rad)", r"$E_{\rm pool}$", "Uniform"], rows,
                 r"Input-law comparison on fixed datasets ($M=2000$, $B=1$, 20 seeds per row). Angles are "
                 r"medians across runs; allocation errors pool neurons and fitted runs, and ``Uniform'' is "
                 r"the same error for uniform shares. For non-Gaussian inputs the loss is estimated on an "
                 r"independent test set. Every run reaches the alignment threshold by the fitting checkpoint.", "rlrrrr")


def anisotropic(data):
    base = read_json(data / "core/exp4_summary.json")
    controls = read_json(data / "core/exp4_controls_summary.json")
    rows = []
    for block, records in (("main", base), ("control", controls)):
        for x in records:
            check_runs(x)
            tau = x.get("tau", 0.003)
            budget = x.get("steps", 40000)
            original = error(x)
            if x["case"] in ("top", "bottom"):
                # Sigma is positive definite and theta is its eigenvector:
                # Sigma theta / ||Sigma theta|| = theta exactly. Consequently
                # both allocation predictors (and their fitting-only errors) agree,
                # including the bottom control where only 14 of 20 runs fitted.
                covariance = original
            elif "alloc_at_fit_SigmaTheta_prediction" in x:
                covariance = error(x, "alloc_at_fit_SigmaTheta_prediction")
            elif x["n_fitted"] == 20:
                covariance = error(x, "alloc_SigmaTheta_pred")
            elif x["n_fitted"] == 0:
                covariance = None
            else:
                raise ValueError("Mixed partial-fit covariance summary mixes fit/end checkpoints; recompute fitting-only metric.")
            rows.append(row(f"{block}_{x['case']}_k{x['kappa']}",
                            [x["case"].capitalize(), x["kappa"], f"{tau:.5f}".rstrip("0"),
                             f"{budget // 1000}k", f"{x['n_fitted']}/20", decimal(original), decimal(covariance)],
                            dict(block=block, teacher_placement=x["case"], kappa=x["kappa"], tau=tau,
                                 budget=budget, n_fitted=x["n_fitted"], n_runs=20,
                                 original_pooled_log_error=original, covariance_pooled_log_error=covariance)))
    return table(4, "anisotropic", "tab:h-anisotropic",
                 ["Teacher", r"$\kappa$", r"$\tau$", "Budget", "Fits", r"$E_\theta$", r"$E_{\Sigma\theta}$"], rows,
                 r"Anisotropic Gaussian inputs and budget/step-size controls (20 seeds per row; "
                 r"$1\mathrm{k}=1000$ updates). Top and bottom place the teacher along eigenvectors of the "
                 r"largest and smallest covariance eigenvalues; mixed uses their normalized sum. $E_\theta$ "
                 r"and $E_{\Sigma\theta}$ are pooled median absolute log-share errors at fitting for the "
                 r"original and covariance-based predictions. The lower block reports the controls. The two "
                 r"predictions coincide when $\theta_\star$ is an eigenvector. Dashes indicate no fit.",
                 "lrrrrrr", [len(base)])


def nonlinear(data):
    rows = []
    for steepness in (1, 3):
        records = [x for x in read_json(data / f"nonlinear/exp8_tanh{steepness}_summary.json")
                   if x["mode"] == "fixed" and x["M"] == 2000 and x["batch"] == 1]
        for x in sorted(records, key=lambda r: r["m"]):
            check_runs(x)
            if x["n_fitted"]:
                raise ValueError("Nonlinear run reached the fitting threshold; update the paper caption.")
            step = x["steps_run"]
            loss, angle = x["final_pop_loss_q"][1], x["final_max_angle_q"][1]
            allocation, uniform = error(x, "alloc_at_end"), error(x, "alloc_at_end", "median_log_err_uniform_baseline")
            rows.append(row(f"k{steepness}_m{x['m']}",
                            [steepness, x["m"], f"{step // 1000}k", decimal(loss, 5), decimal(angle),
                             decimal(allocation), decimal(uniform)],
                            dict(k=steepness, width=x["m"], step=step, test_loss_median=loss,
                                 max_angle_median=angle, pooled_log_error=allocation,
                                 uniform_pooled_log_error=uniform)))
    return table(5, "nonlinear_teachers", "tab:h-nonlinear",
                 [r"$k$", r"$m$", "Step", "Test loss", "Angle (rad)", r"$E_{\rm pool}$", "Uniform"], rows,
                 r"Nonlinear teachers $y(x)=\tanh(k\theta_\star^\top x)/k$ with a ReLU student, fixed "
                 r"$M=2000$ and $B=1$ (20 seeds per row). Loss and maximum angle are medians across runs; "
                 r"allocation errors pool neurons across runs. All measurements use the final checkpoint, "
                 r"and no run reaches loss $10^{-6}$.", "rrrrrrr")


def generate_tables(data):
    return [factory(data) for factory in (sampling, initialization, nongaussian, anisotropic, nonlinear)]


def tex_content(item):
    lines = ["% Generated from experiment results by scripts/make_tables.py.",
             r"\begin{table}[htbp]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{" + item["align"] + "}", r"\toprule",
             " & ".join(item["headers"]) + r" \\", r"\midrule"]
    for index, record in enumerate(item["rows"]):
        if index in item["breaks"]:
            lines.append(r"\midrule")
        lines.append(" & ".join(record["cells"]) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\caption{" + item["caption"] + "}",
                  r"\label{" + item["label"] + "}", r"\end{table}", ""])
    return "\n".join(lines)


def write_tables(tables, output):
    output.mkdir(parents=True, exist_ok=True)
    for item in tables:
        stem = output / item["filename"]
        stem.with_suffix(".tex").write_text(tex_content(item))
        stem.with_suffix(".json").write_text(json.dumps(item, indent=2, allow_nan=False) + "\n")
        keys = list(dict.fromkeys(key for record in item["rows"] for key in record["values"]))
        with stem.with_suffix(".csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["id", *keys])
            writer.writeheader()
            for record in item["rows"]:
                writer.writerow(dict(id=record["id"], **record["values"]))
    (output / "tables.json").write_text(json.dumps(tables, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "reference_results")
    parser.add_argument("--output", type=Path, default=ROOT / "paper/tables")
    args = parser.parse_args()
    tables = generate_tables(args.data_dir.resolve())
    write_tables(tables, args.output.resolve())
    print(f"Generated five tables ({sum(len(t['rows']) for t in tables)} rows) in {args.output.resolve()}")


if __name__ == "__main__":
    main()
