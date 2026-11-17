#!/usr/bin/env python3
"""Render the paper's vector figures from saved results, without retraining.

Run from the project root:
    python scripts/make_figures.py
    python scripts/make_figures.py --data-dir results --figures 2,3

The data directory must contain the main/ and/or core/ subdirectories for
selected figures. Only numpy and reportlab are required. No network access,
external font installation, LaTeX, or image renderer is needed.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def read_json(path):
    with Path(path).open(encoding='utf-8') as handle:
        return json.load(handle)

def new_canvas(path, size, title):
    result = canvas.Canvas(str(path), pagesize=size, invariant=1)
    result.setTitle(title)
    result.setAuthor('')
    result.setCreator('scripts/make_figures.py')
    return result

def draw_figure2(data_dir, output_dir):
    """Match the two original panels, including the enlarged 9 pt legends."""
    with np.load(data_dir / 'main/figure2_fixed_M2000_seed17.npz') as archive:
        z = {key: archive[key] for key in archive.files}
    summary = read_json(data_dir / 'main/figure2_fixed_summary.json')
    alignment = summary.get('alignment_step')
    if alignment is None or alignment < 0:
        raise ValueError('Figure 2 needs a recorded alignment checkpoint.')
    if not np.any(z['dir_hist_steps'] == alignment):
        raise ValueError('The Figure 2 alignment checkpoint has no saved directions.')
    blue = colors.HexColor('#235e93')
    red = colors.HexColor('#c44637')
    purple = colors.HexColor('#77449b')
    green = colors.HexColor('#2d876d')
    grey = colors.HexColor('#888888')

    def txt(x, y, s, size=8, col=colors.black, center=False):
        c.setFillColor(col)
        c.setFont('Helvetica', size)
        (c.drawCentredString if center else c.drawString)(x, y, s)

    def line(x1, y1, x2, y2, col=grey, width=0.5, dash=None):
        c.setStrokeColor(col)
        c.setLineWidth(width)
        c.setDash(dash or [])
        c.line(x1, y1, x2, y2)
        c.setDash()

    def curve(x, y, col, width=0.8):
        c.setStrokeColor(col)
        c.setLineWidth(width)
        p = c.beginPath()
        p.moveTo(float(x[0]), float(y[0]))
        for a, b in zip(x[1:], y[1:]):
            p.lineTo(float(a), float(b))
        c.drawPath(p)

    def point(x, y, col, filled):
        c.setStrokeColor(col)
        c.setFillColor(col if filled else colors.white)
        c.circle(float(x), float(y), 2, stroke=1, fill=1)
    c = new_canvas(output_dir / 'figure2a_fixed_dataset.pdf', (395, 305), 'Neuron direction trajectories')
    c.translate(47.5, 0)
    xc, yc, rad = (150, 178, 91)
    for tick in [-1, -0.5, 0, 0.5, 1]:
        xx = xc + rad * tick
        yy = yc + rad * tick
        txt(xx, yc - rad - 14, f'{tick:g}', 7, center=True)
        txt(xc - rad - 27, yy - 3, f'{tick:g}', 7)
    line(xc - rad, yc, xc + rad, yc, colors.HexColor('#dddddd'))
    line(xc, yc - rad, xc, yc + rad, colors.HexColor('#dddddd'))
    c.setStrokeColor(colors.HexColor('#bbbbbb'))
    c.circle(xc, yc, rad, stroke=1, fill=0)
    H = z['dir_hist']
    st = z['dir_hist_steps']
    al = summary['alignment_step']
    ai = int(np.where(st == al)[0][0])
    elig = np.flatnonzero(np.abs(H[0, :, 0]) <= 0.5)
    pick = np.random.default_rng(123).choice(elig, 30, replace=False)
    for j in pick:
        col = blue if z['signs'][j] > 0 else red
        xx = xc + rad * H[:ai + 1, j, 0]
        yy = yc + rad * H[:ai + 1, j, 1]
        curve(xx, yy, col, 0.65)
        point(xx[0], yy[0], col, False)
        point(xx[-1], yy[-1], col, True)
        k = max(2, ai // 3)
        vx = xx[k] - xx[k - 2]
        vy = yy[k] - yy[k - 2]
        norm = math.hypot(vx, vy)
        if norm > 0:
            ux, uy = (vx / norm, vy / norm)
            px, py = (xx[k], yy[k])
            line(px, py, px - 4 * ux + 1.7 * uy, py - 4 * uy - 1.7 * ux, col, 0.7)
            line(px, py, px - 4 * ux - 1.7 * uy, py - 4 * uy + 1.7 * ux, col, 0.7)
    for s in [-1, 1]:
        xx = xc + s * rad
        line(xx - 4, yc, xx + 4, yc, colors.black, 1)
        line(xx, yc - 4, xx, yc + 4, colors.black, 1)
    txt(150, 291, 'Neurons align before fitting', 10, center=True)
    txt(150, 54, 'Component along the teacher', 9, center=True)
    c.saveState()
    c.translate(17, 178)
    c.rotate(90)
    txt(0, 0, 'Perpendicular component', 9, center=True)
    c.restoreState()
    line(40, 34, 54, 34, blue, 1)
    txt(58, 31, 'positive sign', 9)
    line(135, 34, 149, 34, red, 1)
    txt(153, 31, 'negative sign', 9)
    point(44, 18, grey, False)
    txt(52, 15, 'initial direction', 9)
    point(139, 18, grey, True)
    txt(147, 15, 'end of alignment', 9)
    c.save()
    c = new_canvas(output_dir / 'figure2b_fixed_dataset.pdf', (395, 305), 'Alignment, growth and fitting')
    c.translate(-295, 0)
    if int(z['steps'][-1]) != 12000:
        raise ValueError('Figure 2 requires the full 12000-update trajectory.')
    x, y, w, h = (350, 91, 291, 177)
    steps = z['steps']
    xx = x + w * steps / 12000
    lo = -12
    hi = 0
    rightmax = 2.2
    yp = lambda q: y + h * (np.log10(np.maximum(q, 10.0 ** lo)) - lo) / (hi - lo)
    yr = lambda q: y + h * q / rightmax
    for tick in [0, 2000, 4000, 6000, 8000, 10000, 12000]:
        tx = x + w * tick / 12000
        line(tx, y, tx, y + h, colors.HexColor('#ededed'))
        txt(tx, y - 12, str(tick), 7, center=True)
    for exp in [0, -2, -4, -6, -8, -10, -12]:
        ty = y + h * (exp - lo) / (hi - lo)
        line(x, ty, x + w, ty, colors.HexColor('#ededed'))
        txt(x - 29, ty - 3, f'1e{exp}', 7)
    for v in [0, 0.5, 1, 1.5, 2]:
        txt(x + w + 6, yr(v) - 3, f'{v:g}', 7, purple)
    curve(xx, yp(z['pop_loss']), colors.black, 1.3)
    curve(xx, yr(z['max_angle']), purple, 1)
    c.setDash(4, 2)
    curve(xx, yr(z['class_mass'].mean(axis=1)), purple, 1)
    c.setDash()
    line(x, yp(np.array(1e-06)), x + w, yp(np.array(1e-06)), grey, 0.7, [2, 2])
    line(x, yr(1), x + w, yr(1), grey, 0.6, [1, 3])
    alx = x + w * al / 12000
    line(alx, y, alx, y + h, green, 0.9, [3, 2])
    txt(x + w / 2, 291, 'Alignment is followed by growth and fitting', 10, center=True)
    txt(x + w / 2, 54, 'SGD step', 9, center=True)
    c.saveState()
    c.translate(308, 179)
    c.rotate(90)
    txt(0, 0, 'Exact population loss (log scale)', 9, center=True)
    c.restoreState()
    c.saveState()
    c.translate(682, 179)
    c.rotate(90)
    txt(0, 0, 'Angle (rad) and average class mass', 8, purple, center=True)
    c.restoreState()
    line(334, 35, 350, 35, colors.black, 1.2)
    txt(354, 32, 'population loss', 9)
    line(442, 35, 458, 35, purple, 1)
    txt(462, 32, 'largest angle', 9)
    line(544, 35, 560, 35, purple, 1, [4, 2])
    txt(564, 32, 'average class mass', 9)
    line(334, 18, 350, 18, green, 1, [3, 2])
    txt(354, 15, 'first alignment checkpoint', 9)
    line(493, 18, 509, 18, grey, 0.7, [2, 2])
    txt(513, 15, 'fitting threshold (1e-6)', 9)
    c.save()

def draw_figure3(data_dir, output_dir):
    c = new_canvas(output_dir / 'figure3_fixed_dataset.pdf', (570, 235), 'Initialization predicts allocation under fixed-dataset SGD')
    blue = colors.HexColor('#235b7a')
    grey = colors.HexColor('#666666')
    run_counts = []
    c.setFont('Helvetica-Bold', 11)
    c.drawCentredString(285, 220, 'Initialization predicts allocation under fixed-dataset SGD')
    for idx, m in enumerate((4, 16, 64)):
        z = np.load(data_dir / 'main' / f'fixed_m{m}.npz')
        mask = z['fit_step'] >= 0
        n_runs = len(mask)
        n_fits = int(mask.sum())
        run_counts.append((n_fits, n_runs))
        xx = (m * z['lam'][mask] / 2).ravel()
        yy = (m * z['shares_fit'][mask] / 2).ravel()
        if not (np.all(np.isfinite(xx)) and np.all(np.isfinite(yy))):
            raise ValueError(f'Non-finite fitting-time shares at width {m}.')
        x, y, w, h = (45 + idx * 184, 48, 145, 140)
        top = math.ceil(max(xx.max(), yy.max()) * 2) / 2 if len(xx) else 1.0
        c.setFont('Helvetica-Bold', 9)
        c.drawCentredString(x + w / 2, 202, f'm = {m} | {n_fits}/{n_runs} fits')
        c.setStrokeColor(grey)
        c.setLineWidth(0.5)
        c.line(x, y, x + w, y)
        c.line(x, y, x, y + h)
        for t in np.linspace(0, top, 5):
            xp = x + w * t / top
            yp = y + h * t / top
            c.setStrokeColor(colors.HexColor('#dddddd'))
            c.line(xp, y, xp, y + h)
            c.line(x, yp, x + w, yp)
            c.setFillColor(grey)
            c.setFont('Helvetica', 7)
            c.drawCentredString(xp, y - 11, f'{t:g}')
            c.drawRightString(x - 5, yp - 2, f'{t:g}')
        c.setStrokeColor(grey)
        c.setDash(3, 3)
        c.line(x, y, x + w, y + h)
        c.setDash()
        c.setFillColor(blue)
        for a, b in zip(xx, yy):
            c.circle(x + w * a / top, y + h * b / top, 1.1, stroke=0, fill=1)
        z.close()
        c.setFillColor(colors.black)
        c.setFont('Helvetica', 8)
        c.drawCentredString(x + w / 2, 23, 'Predicted share / uniform share')
        c.saveState()
        c.translate(x - 30, y + h / 2)
        c.rotate(90)
        c.drawCentredString(0, 0, 'Observed share / uniform share')
        c.restoreState()
    footer = 'All neurons from 20 seeds per width' if all((a == b == 20 for a, b in run_counts)) else 'All neurons from fitted runs'
    c.setFont('Helvetica', 7)
    c.setFillColor(grey)
    c.drawCentredString(285, 6, footer + '; M = 2000, B = 1; shares at first population loss <= 1e-6.')
    c.save()

def polyline(c, pts, color, width=1.5, dash=None):
    c.setStrokeColor(HexColor(color))
    c.setLineWidth(width)
    c.setDash(dash or [])
    p = c.beginPath()
    p.moveTo(*pts[0])
    [p.lineTo(*q) for q in pts[1:]]
    c.drawPath(p)
    c.setDash([])

def axes(c, x, y, w, h, xlim, ylim, xticks, yticks, title, xlabel, ylabel='', logx=False):

    def xx(v):
        return x + w * ((math.log10(v) - math.log10(xlim[0])) / (math.log10(xlim[1]) - math.log10(xlim[0])) if logx else (v - xlim[0]) / (xlim[1] - xlim[0]))

    def yy(v):
        return y + h * (v - ylim[0]) / (ylim[1] - ylim[0])
    c.setFont('Helvetica', 8.5)
    for v, lab in yticks:
        c.setStrokeColor(HexColor('#e3e7eb'))
        c.setLineWidth(0.5)
        c.line(x, yy(v), x + w, yy(v))
        c.setFillColor(HexColor('#343a40'))
        c.drawRightString(x - 6, yy(v) - 3, lab)
    for v, lab in xticks:
        c.setFillColor(HexColor('#343a40'))
        c.drawCentredString(xx(v), y - 13, lab)
    c.setStrokeColor(HexColor('#687480'))
    c.line(x, y, x + w, y)
    c.line(x, y, x, y + h)
    c.setFillColor(HexColor('#202830'))
    c.setFont('Helvetica-Bold', 10)
    c.drawString(x, y + h + 12, title)
    c.setFont('Helvetica', 9)
    c.drawCentredString(x + w / 2, y - 29, xlabel)
    if ylabel:
        c.saveState()
        c.translate(x - 34, y + h / 2)
        c.rotate(90)
        c.drawCentredString(0, 0, ylabel)
        c.restoreState()
    return (xx, yy)

def draw_figure4(data_dir, output_dir):
    a = [x for x in read_json(data_dir / 'core/exp2_alpha_summary.json') if x['mode'] == 'fixed']
    t = [x for x in read_json(data_dir / 'core/exp2_tau_summary.json') if x['mode'] == 'fixed']
    c = new_canvas(output_dir / 'appendix_scale_rate.pdf', (600, 410), 'Initialization-scale and learning-rate sweeps')
    for col, (data, key, title) in enumerate([(a, 'alpha', 'Initialization scale'), (t, 'tau', 'Learning rate')]):
        xs = [x[key] for x in data]
        x0 = 55 + col * 300
        n_runs = max((x['n_runs'] for x in data))
        for row in [0, 1]:
            y = 245 if row == 0 else 62
            xx, yy = axes(c, x0, y, 235, 115, (min(xs), max(xs)), (0, n_runs * 1.05) if row == 0 else (0, 0.35), [(v, f'{v:g}') for v in xs], [(v, f'{v:g}') for v in (np.linspace(0, n_runs, 5) if row == 0 else [0, 0.1, 0.2, 0.3])], f"({'abcd'[row * 2 + col]}) " + title + (' - fits' if row == 0 else ' - allocation'), 'alpha' if key == 'alpha' else 'tau', f'Runs (out of {n_runs})' if row == 0 else 'Pooled median log error', True)
            vals = [x['n_fitted'] if row == 0 else err(x) for x in data]
            pts = [(xx(v), yy(z)) for v, z in zip(xs, vals) if z is not None]
            polyline(c, pts, '#1984b5')
            for q in pts:
                c.setFillColor(HexColor('#1984b5'))
                c.circle(*q, 2.6, stroke=0, fill=1)
            if row == 0:
                av = [x['align_before_fit'] for x in data]
                polyline(c, [(xx(v), yy(z)) for v, z in zip(xs, av)], '#df8f20', 1.3, [3, 2])
                if col == 0:
                    c.setFont('Helvetica', 8.5)
                    c.setFillColor(HexColor('#1984b5'))
                    c.drawString(x0 + 10, y + 37, 'Fit')
                    c.setFillColor(HexColor('#9c630d'))
                    c.drawString(x0 + 10, y + 23, 'Aligned by fit')
            else:
                for v, z in zip(xs, vals):
                    if z is None:
                        c.setFillColor(HexColor('#666666'))
                        c.setFont('Helvetica', 7)
                        c.saveState()
                        c.translate(xx(v), y + 15)
                        c.rotate(90)
                        c.drawString(0, 0, 'no fit')
                        c.restoreState()
    c.save()
NOISE_COLORS = ['#343a40', '#7a5daf', '#1984b5', '#209b73', '#df8f20', '#ce4757']

def err(record):
    return (record.get('alloc_at_fit') or {}).get('median_log_err')

def draw_figure5(data_dir, output_dir):
    noise = read_json(data_dir / 'core/exp5_analysis.json')
    c = new_canvas(output_dir / 'appendix_noise.pdf', (660, 420), 'Fixed label noise and longer training')
    for j, s in enumerate([0, 0.01, 0.03, 0.1, 0.3, 1]):
        c.setFillColor(HexColor(NOISE_COLORS[j]))
        c.circle(54 + j * 102, 403, 3, fill=1, stroke=0)
        c.setFont('Helvetica', 10)
        c.drawString(62 + j * 102, 399, f'sigma = {s:g}')
    for row, M in enumerate([2000, 200]):
        for col, (key, title, lim) in enumerate([('max_angle_to_theta', 'Maximum angle (rad)', 2), ('alloc_err_vs_pred', 'Allocation error', 0.5), ('drift_vs_t10k', 'Drift from step 10k', 0.5)]):
            x = 47 + col * 220
            y = 245 if row == 0 else 58
            xx, yy = axes(c, x, y, 165, 107, (10, 200), (0, lim), [(v, str(v)) for v in [10, 50, 100, 150, 200]], [(v, f'{v:g}') for v in ([0, 0.5, 1, 1.5, 2] if lim == 2 else [0, 0.1, 0.2, 0.3, 0.4, 0.5])], f"({'abcdef'[row * 3 + col]}) " + title, 'Updates (thousands)')
            c.setFont('Helvetica', 9)
            c.setFillColor(HexColor('#343a40'))
            c.drawString(x + 7, y + 95, f'M = {M}')
            for j, s in enumerate([0, 0.01, 0.03, 0.1, 0.3, 1]):
                d = sorted([z for z in noise if z['M'] == M and z['sigma'] == s and (z['step'] >= 10000)], key=lambda z: z['step'])
                if not d:
                    raise ValueError(f'Missing noise snapshots for M={M}, sigma={s}.')
                pts = [(xx(z['step'] / 1000), yy(z[key][1])) for z in d]
                polyline(c, pts, NOISE_COLORS[j], 1.4, [4, 2] if j < 2 else None)
                for q in pts:
                    c.setFillColor(HexColor(NOISE_COLORS[j]))
                    c.circle(*q, 1.6, stroke=0, fill=1)
    c.save()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'reference_results', help='Results root containing main/ and core/ (default: reference_results).')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'paper/figures', help='Directory for vector PDFs (default: paper/figures).')
    parser.add_argument('--figures', default='all', help='all, or a comma-separated subset of 2,3,4,5.')
    args = parser.parse_args()
    selected = ['2', '3', '4', '5'] if args.figures == 'all' else args.figures.split(',')
    selected = list(dict.fromkeys((value.strip() for value in selected)))
    if not selected or any((value not in {'2', '3', '4', '5'} for value in selected)):
        parser.error('--figures must be all or a comma-separated subset of 2,3,4,5')
    args.output.mkdir(parents=True, exist_ok=True)
    renderers = {'2': draw_figure2, '3': draw_figure3, '4': draw_figure4, '5': draw_figure5}
    try:
        for number in selected:
            renderers[number](args.data_dir, args.output)
            print(f'Created Figure {number}' + (' panels a and b' if number == '2' else ''))
    except (FileNotFoundError, ValueError, KeyError) as exc:
        parser.exit(1, f'Figure generation failed: {exc}\n')
if __name__ == '__main__':
    main()
