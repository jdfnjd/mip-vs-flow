"""Plots for the controllable-multimodality study."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDER = ["regression", "mip", "flow_z0", "flow"]
LABEL = {
    "regression": "Regression (RCP)",
    "mip": "MIP (2-step)",
    "flow_z0": "Flow, z=0",
    "flow": "Flow, z~N(0,I)",
}
COLOR = {"regression": "#8c8c8c", "mip": "#2a6fdb", "flow_z0": "#e3a21a", "flow": "#c2372e"}
FIG = Path("figures")


def load(name):
    return pd.read_json(f"results/{name}.jsonl", lines=True)


def agg(df, by, col):
    g = df.groupby(by + ["method"])[col]
    return g.mean().unstack("method"), g.std().unstack("method")


def lines(ax, df, x, col, title, xlabel, logx=False):
    m, s = agg(df, [x], col)
    for meth in ORDER:
        if meth not in m:
            continue
        ax.errorbar(m.index, m[meth], yerr=s[meth], label=LABEL[meth], color=COLOR[meth],
                    marker="o", capsize=3, lw=2)
    if logx:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.3)


def plot_alpha_n():
    df = load("alpha_n")
    Ns = sorted(df.n_demos.unique())
    fig, axes = plt.subplots(2, len(Ns), figsize=(4.2 * len(Ns), 7), sharey="row")
    for j, n in enumerate(Ns):
        d = df[df.n_demos == n]
        lines(axes[0, j], d, "alpha", "success", f"N = {n} demos", r"mode ambiguity $\alpha$")
        lines(axes[1, j], d, "alpha", "tv", "", r"mode ambiguity $\alpha$")
    axes[0, 0].set_ylabel("closed-loop success")
    axes[1, 0].set_ylabel("mode TV distance (successes)\nlower = matches expert")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("When does distribution learning matter? (K=2 gaps)")
    fig.tight_layout()
    fig.savefig(FIG / "alpha_n.png", dpi=150)

    # phase diagram: flow minus mip success
    m, _ = agg(df, ["alpha", "n_demos"], "success")
    gap = (m["flow"] - m["mip"]).unstack("n_demos")
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(gap.values, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower", aspect="auto")
    ax.set_xticks(range(len(gap.columns)), gap.columns)
    ax.set_yticks(range(len(gap.index)), gap.index)
    ax.set_xlabel("N demos")
    ax.set_ylabel(r"mode ambiguity $\alpha$")
    for (i, jj), v in np.ndenumerate(gap.values):
        ax.text(jj, i, f"{v:+.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, label="success(Flow) - success(MIP)")
    ax.set_title("Advantage of distribution learning")
    fig.tight_layout()
    fig.savefig(FIG / "phase_flow_minus_mip.png", dpi=150)


def plot_exec():
    df = load("exec")
    alphas = sorted(df.alpha.unique())
    fig, axes = plt.subplots(2, len(alphas), figsize=(4.2 * len(alphas), 7), sharey="row")
    for j, a in enumerate(alphas):
        d = df[df.alpha == a]
        lines(axes[0, j], d, "n_action_exec", "success", rf"$\alpha$ = {a}", "steps executed per replan")
        lines(axes[1, j], d, "n_action_exec", "tv", "", "steps executed per replan")
        axes[1, j].axhline(a / 2, color="k", ls=":", lw=1)
        axes[1, j].text(1, a / 2, " fully collapsed", va="bottom", fontsize=8)
        for ax in axes[:, j]:
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 2, 4, 8], [1, 2, 4, 8])
    axes[0, 0].set_ylabel("closed-loop success")
    axes[1, 0].set_ylabel("mode TV distance to expert\nlower = matches expert")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Replanning frequency: feedback rescues regression but erases flow's mode diversity (N=250)")
    fig.tight_layout()
    fig.savefig(FIG / "exec.png", dpi=150)


def plot_commit():
    df = load("commit")
    df["geometry"] = df.apply(lambda r: f"runway {-r.start_y:g}\nstart ±{r.start_x_range:g}", axis=1)
    order = (df[["start_y", "start_x_range", "geometry"]].drop_duplicates()
             .sort_values(["start_x_range", "start_y"], ascending=[False, True]).geometry.tolist())
    alphas = sorted(df.alpha.unique())
    fig, axes = plt.subplots(1, len(alphas), figsize=(6.5 * len(alphas), 3.8), sharey=True)
    for ax, a in zip(np.atleast_1d(axes), alphas):
        m, s = agg(df[df.alpha == a], ["geometry"], "success")
        m, s = m.loc[order], s.loc[order]
        w, x = 0.2, np.arange(len(order))
        for i, meth in enumerate(ORDER):
            ax.bar(x + (i - 1.5) * w, m[meth], w, yerr=s[meth], label=LABEL[meth], color=COLOR[meth], capsize=2)
        ax.set_xticks(x, order, fontsize=8)
        ax.set_title(rf"$\alpha$ = {a}")
        ax.grid(alpha=0.3, axis="y")
    np.atleast_1d(axes)[0].set_ylabel("closed-loop success")
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="lower left", ncol=4, framealpha=0.95)
    fig.suptitle("Time to commit: shorter runway / more symmetric starts (N=250)")
    fig.tight_layout()
    fig.savefig(FIG / "commit.png", dpi=150)


def plot_hard():
    df = load("hard")
    alphas = sorted(df.alpha.unique())
    fig, axes = plt.subplots(1, len(alphas), figsize=(4.2 * len(alphas), 3.8), sharey=True)
    for ax, a in zip(axes, alphas):
        lines(ax, df[df.alpha == a], "n_action_exec", "success", rf"$\alpha$ = {a}", "steps executed per replan")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8], [1, 2, 4, 8])
    axes[0].set_ylabel("closed-loop success")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Hard corner: short runway (0.5), near-symmetric starts (±0.1), N=250")
    fig.tight_layout()
    fig.savefig(FIG / "hard.png", dpi=150)


def plot_multimodality():
    import json

    df = pd.DataFrame(json.loads(Path("results/multimodality.json").read_text()))
    df["pct"] = 100 * df.frac_mm
    short = {"pusht": "push-t", "tool_hang_ph": "tool-hang\nph", "tool_hang_good_policy": "tool-hang\npolicy",
             "tool_hang_delay_policy": "tool-hang\n+delay", "tool_hang_noisydelay_policy": "tool-hang\n+delay+noise"}
    names = [n for n in df.dataset.unique() if not n.startswith("toy")]
    labels = [short.get(n, n.replace("_", "\n")) for n in names]
    x = np.arange(len(names))

    def get(obs_steps=2, n_pc=1, inject=0.0, inject_dir="none", gripper="any"):
        d = df[(df.obs_steps == obs_steps) & (df.n_pc == n_pc) & (df.inject == inject) & (df.inject_dir == inject_dir)]
        if gripper != "any":  # Push-T has no gripper (None): it appears in both series
            d = d[d.gripper.isna() | (d.gripper == gripper)]
        return d.set_index("dataset").pct

    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(19, 4.6), gridspec_kw={"width_ratios": [1, 2.1, 2.1]})
    toy = df[df.dataset.str.startswith("toy") & (df.inject == 0) & (df.n_pc == 1)].copy()
    toy["alpha"] = toy.dataset.str.replace("toy_alpha", "").astype(float)
    for steps, c in ((1, COLOR["flow"]), (2, COLOR["flow_z0"])):
        t = toy[toy.obs_steps == steps].sort_values("alpha")
        a0.plot(t.alpha, t.pct, "o-", color=c, lw=2, label=f"{steps} observation" + ("s" if steps > 1 else ""))
    a0.axhline(1, color="k", ls=":", lw=1)
    a0.set_xlabel(r"toy mode ambiguity $\alpha$")
    a0.set_ylabel("% of states flagged bimodal")
    a0.set_title("Toy: the score depends on\nhow much history defines a state")
    a0.legend(fontsize=8)
    a0.grid(alpha=0.3)

    wo, wi = get(gripper=False), get(gripper=True)
    a1.bar(x - 0.2, [wo[n] for n in names], 0.4, color=COLOR["regression"], label="arm motion only")
    a1.bar(x + 0.2, [wi[n] for n in names], 0.4, color=COLOR["mip"], label="with gripper command")
    a1.axhline(1, color="k", ls=":", lw=1)
    a1.text(-0.4, 1.15, "chance (1%)", fontsize=8)
    a1.set_xticks(x, labels, fontsize=8)
    a1.set_ylabel("% of states flagged bimodal")
    a1.set_title("Demonstration data (2 observations, 8-step chunks)")
    a1.legend(fontsize=8, loc="upper left")
    a1.grid(alpha=0.3, axis="y")

    ax_, rn_ = get(inject=1.0, inject_dir="axis", gripper=True), get(inject=1.0, inject_dir="random", gripper=True)
    a2.bar(x - 0.2, [ax_[n] for n in names], 0.4, color="#7a7a7a", label="modes along one action axis")
    a2.bar(x + 0.2, [rn_[n] for n in names], 0.4, color="#c9c9c9", label="modes along a random direction")
    a2.set_xticks(x, labels, fontsize=8)
    a2.set_ylim(0, 100)
    a2.set_ylabel("% of states flagged")
    a2.set_title(r"Positive control: injected modes 2$\sigma$ apart (detection power)")
    a2.legend(fontsize=8, loc="upper left")
    a2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "multimodality.png", dpi=150)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    if "multimodality" in sys.argv[1:]:
        plot_multimodality()
        print("plotted multimodality")
    which = sys.argv[1:] or ["alpha_n", "exec", "commit", "hard"]
    for w in which:
        if Path(f"results/{w}.jsonl").exists():
            globals()[f"plot_{w}"]()
            print("plotted", w)
