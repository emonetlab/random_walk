"""Figure style and shared figures."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .analysis import local_slope

# One fixed color and line pattern per model, used in every figure.
COLORS = {"brownian": "#2a78d6", "biased": "#eb6834", "run_tumble": "#1baf7a", "ou": "#d55181", "theory": "0.1"}
STYLE = {"brownian":   dict(color=COLORS["brownian"], ls="-"),
         "biased":     dict(color=COLORS["biased"], ls=(0, (6, 2))),
         "run_tumble": dict(color=COLORS["run_tumble"], ls="-"),
         "ou":         dict(color=COLORS["ou"], ls=(0, (6, 2, 1.5, 2)))}
GRAY = dict(color="0.45", ls=(0, (3, 2)))
THEORY_KW = dict(color="0.1", lw=2)


def set_style():
    sns.set_style("white")
    sns.set_context("talk")
    plt.rcParams.update({"legend.frameon": False, "figure.dpi": 80})


def plot_tracks(tracks, fps, color, title, n_show=5, windows=(2, 60), lims=(80, 800)):
    """Example tracks (array n_tracks x n_frames x 2, starting at the origin) over a short and a long window."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, w, L in zip(axes, windows, lims):
        k = int(round(w * fps)) + 1
        for tr in tracks[:n_show]:
            ax.plot(tr[:k, 0], tr[:k, 1], color=color, lw=1.2, alpha=0.85)
            ax.plot(tr[k - 1, 0], tr[k - 1, 1], "o", ms=6, color=color)
        ax.plot(0, 0, "+", color="k", ms=14, mew=2)
        ax.set(xlim=(-L, L), ylim=(-L, L), aspect="equal", xlabel="x (µm)", ylabel="y (µm)",
               title=f"first {w} s  (+ start, ● end)")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def msd_figure(curves, theory=(), vlines=(), alpha_lines=(), slopes=(), title="", ylim_alpha=(-0.1, 2.3)):
    """Left: MSD on log-log axes. Right: local exponent alpha(tau) = d log MSD / d log tau.

    curves:      list of (tau, msd, style, label), style = dict(color=..., ls=...).
                 Earlier curves are drawn thicker and underneath, so overlapping curves stay visible.
    theory:      list of (tau, msd, label), thin black dashed / dotted lines on top.
    vlines:      list of (tau, label), for example a crossover time.
    alpha_lines: extra horizontal reference lines on the alpha panel, list of (alpha, label).
    slopes:      reference segments on the MSD panel, list of (alpha, tau_start, tau_end[, offset]),
                 drawn next to the first curve (offset 0.3 = a bit below it, the default; offset > 1 = above it).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.2))
    for i, (tau, m, style, label) in enumerate(curves):
        kw = dict(lw=max(2.0, 5.5 - 1.2 * i), alpha=0.9, zorder=2 + i, **style)
        ax1.loglog(tau, m, label=label, **kw)
        ax2.semilogx(tau, local_slope(tau, m), **kw)
    for i, (tau, m, label) in enumerate(theory):
        ls = ["--", ":"][i % 2]
        ax1.loglog(tau, m, ls=ls, label=label, zorder=10, **THEORY_KW)
        ax2.semilogx(tau, local_slope(tau, m), ls=ls, zorder=10, **THEORY_KW)
    if slopes:
        tau0, m0 = curves[0][0], curves[0][1]
        for seg in slopes:
            a, t1, t2 = seg[:3]
            offset = seg[3] if len(seg) > 3 else 0.3
            y1 = offset * np.exp(np.interp(np.log(t1), np.log(tau0), np.log(m0)))
            y2 = y1 * (t2 / t1) ** a
            ax1.plot([t1, t2], [y1, y2], color="0.25", lw=2.5, solid_capstyle="butt", zorder=11)
            below = offset < 1
            ax1.text(np.sqrt(t1 * t2), np.sqrt(y1 * y2) * (0.4 if below else 2.5), f"slope {a:g}",
                     ha="left" if below else "right", va="top" if below else "bottom",
                     fontsize="x-small", color="0.25")
    for a, lab in [(1, "diffusive (α = 1)"), (2, "ballistic (α = 2)")] + list(alpha_lines):
        ax2.axhline(a, color="0.75", lw=1.2, zorder=1)
        ax2.text(0.01, a + 0.04, lab, transform=ax2.get_yaxis_transform(), fontsize="x-small", color="0.35")
    for x, lab in vlines:
        for ax in (ax1, ax2):
            ax.axvline(x, color="0.55", lw=1.2, ls=":", zorder=1)
        ax2.text(x * 1.1, ylim_alpha[0] + 0.2, lab, ha="left", fontsize="x-small", color="0.3")
    ax1.set(xlabel="lag τ (s)", ylabel="MSD (µm²)", title="MSD(τ)")
    ax2.set(xlabel="lag τ (s)", ylabel="α(τ) = d log MSD / d log τ", ylim=ylim_alpha, title="local exponent α(τ)")
    ax1.legend(fontsize="x-small", loc="upper left")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
