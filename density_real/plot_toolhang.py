"""Figure for the Tool-Hang sweep: python plot_toolhang.py results/toolhang_runs.csv figures/toolhang.png"""

import sys

import matplotlib.pyplot as plt
import pandas as pd

COLOR = {"flow": "#2a78d6", "mip": "#eb6834", "regression": "#1baf7a"}
LABEL = {"flow": "flow", "mip": "MIP", "regression": "regression"}
PAPER = {"flow": 73, "mip": 71, "regression": 59}  # Table 7, Chi-UNet, average of last five evaluations

df = pd.read_csv(sys.argv[1])
df["pct"] = 100 * df["final_zero_400"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True, gridspec_kw={"width_ratios": [1.2, 1]})

# left: N=200 against the paper
for i, m in enumerate(COLOR):
    for dx, tag, face in ((-0.12, "sweep", COLOR[m]), (0.12, "nocompile", "white")):
        v = df[(df.tag == tag) & (df.n_demos == 200) & (df.method == m)]["pct"]
        if len(v):
            a.scatter([i + dx] * len(v), v, s=28, facecolor=face, edgecolor=COLOR[m], linewidth=1.5, zorder=3)
            a.plot([i + dx - 0.08, i + dx + 0.08], [v.mean()] * 2, color=COLOR[m], linewidth=2)
    a.plot([i - 0.3, i + 0.3], [PAPER[m]] * 2, color="#52514e", linestyle="--", linewidth=1)
a.text(2.32, PAPER["regression"], "paper\naverage", color="#52514e", va="center", fontsize=8)
a.set_xticks(range(3), [LABEL[m] for m in COLOR])
a.set_xlim(-0.5, 2.8)
a.set_ylabel("success, 400 episodes (%)")
a.set_title("200 demos: filled = compiled, open = compile off", fontsize=9)

# right: 200 vs 25 demos, compiled sweep
for m in COLOR:
    s = df[(df.tag == "sweep") & (df.method == m)]
    means = s.groupby("n_demos")["pct"].mean()
    b.plot([0, 1], [means[25], means[200]], color=COLOR[m], linewidth=2)
    for x, n in ((0, 25), (1, 200)):
        b.scatter([x] * 3, s[s.n_demos == n]["pct"], s=18, color=COLOR[m], zorder=3)
    b.text(1.06, means[200], f"{LABEL[m]} {means[200]:.0f}", va="center", fontsize=8)
    b.text(-0.06, means[25], f"{means[25]:.0f}", va="center", ha="right", fontsize=8)
b.set_xticks([0, 1], ["25 demos", "200 demos"])
b.set_xlim(-0.35, 1.5)
b.set_title("demo count (dots = seeds, lines = means)", fontsize=9)

for ax in (a, b):
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
fig.suptitle("Tool-Hang PH state, Chi-UNet, official code, 300k steps, 3 seeds, final checkpoint", fontsize=9)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=200)
