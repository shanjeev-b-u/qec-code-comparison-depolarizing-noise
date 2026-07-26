import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("results.json") as f:
    R = json.load(f)

p = np.array(R["p"])
bf_sim = np.array(R["bitflip"]["sim"])
bf_exact = np.array(R["bitflip"]["exact"])
bf_ci = np.array(R["bitflip"]["ci95"])
sc_sim = np.array(R["surface"]["sim"])
sc_exact = np.array(R["surface"]["exact"])
sc_ci = np.array(R["surface"]["ci95"])

plt.rcParams.update({
    "font.size": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# ---------------- Figure 1: PL vs p, log-log ----------------
fig, ax = plt.subplots(figsize=(6.5, 5.2))
ax.plot(p, p, 'k--', label=r'No correction ($P_L=p$)', linewidth=1.5)
ax.errorbar(p, np.clip(bf_sim, 1e-7, None), yerr=bf_ci, fmt='o', color='#1f77b4',
            label='3-qubit bit-flip code (sim.)', capsize=3, markersize=5)
ax.errorbar(p, np.clip(sc_sim, 1e-7, None), yerr=sc_ci, fmt='s', color='#d62728',
            label='Distance-3 surface code (sim.)', capsize=3, markersize=5)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Physical error rate $p$')
ax.set_ylabel('Logical error rate $P_L$')
ax.set_title('Logical Error Rate vs. Physical Error Rate (Depolarizing Noise)')
ax.legend(loc='upper left', fontsize=9)
sc_pstar = R["sc_pstar"]
if sc_pstar:
    ax.axvline(sc_pstar, color='#d62728', linestyle=':', alpha=0.6, linewidth=1)
    ax.annotate(f"$p^*_{{SC}}\\approx{sc_pstar:.3f}$", xy=(sc_pstar, sc_pstar),
                xytext=(sc_pstar*1.3, sc_pstar*0.15), fontsize=8, color='#d62728')
fig.tight_layout()
fig.savefig("fig1_logical_error_vs_physical.png", dpi=300, bbox_inches="tight")
print("Saved fig1")

# ---------------- Figure 2: simulated vs analytical ----------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))

ax = axes[0]
ax.plot(p, bf_exact, '-', color='#1f77b4', label='Analytical (exact enumeration)', linewidth=2)
ax.errorbar(p, bf_sim, yerr=bf_ci, fmt='o', color='#08306b', label='Monte Carlo simulation',
            capsize=3, markersize=6)
ax.fill_between(p, np.clip(bf_sim - bf_ci, 0, None), bf_sim + bf_ci, color='#1f77b4', alpha=0.15)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Physical error rate $p$', fontsize=11); ax.set_ylabel('Logical error rate $P_L$', fontsize=11)
ax.set_title('Bit-flip code (bit-observable metric)', fontsize=12)
ax.legend(fontsize=10, loc='upper left')
ax.tick_params(labelsize=10)

ax = axes[1]
ax.plot(p, sc_exact, '-', color='#d62728', label='Analytical (exact enumeration)', linewidth=2)
ax.errorbar(p, sc_sim, yerr=sc_ci, fmt='s', color='#7f0000', label='Monte Carlo simulation',
            capsize=3, markersize=6)
ax.fill_between(p, np.clip(sc_sim - sc_ci, 0, None), sc_sim + sc_ci, color='#d62728', alpha=0.15)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Physical error rate $p$', fontsize=11); ax.set_ylabel('Logical error rate $P_L$', fontsize=11)
ax.set_title('Distance-3 surface code', fontsize=12)
ax.legend(fontsize=10, loc='upper left')
ax.tick_params(labelsize=10)

fig.suptitle('Simulated vs. Analytical Logical Error Rate Agreement', y=1.02, fontsize=13)
fig.tight_layout()
fig.savefig("fig2_sim_vs_analytical.png", dpi=300, bbox_inches="tight")
print("Saved fig2")

# ---------------- Figure 3: overhead-normalized comparison at p=0.001 ----------------
# Simplified per review feedback: two single-encoding bar panels instead of a
# dual-axis (bars + twin-axis line) mixed-encoding plot, for easier reading.
idx = list(p).index(0.001)
bf_pl_001 = bf_exact[idx]
sc_pl_001 = sc_exact[idx]
qubits = [3, 17]
pls = [bf_pl_001, sc_pl_001]
labels = ['3-qubit\nbit-flip code', 'Distance-3\nsurface code\n(9 data + 8 anc.)']
colors = ['#1f77b4', '#d62728']

fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.5, 4.2))
x = np.arange(2)

axL.bar(x, pls, width=0.5, color=colors, alpha=0.85)
axL.set_yscale('log')
axL.set_ylabel('Logical error rate $P_L$ at $p=0.001$')
axL.set_xticks(x); axL.set_xticklabels(labels, fontsize=9)
axL.set_title('Logical error rate', fontsize=11)
for xi, v in zip(x, pls):
    axL.text(xi, v * 1.5, f"{v:.2e}", ha='center', fontsize=9)

axR.bar(x, qubits, width=0.5, color=colors, alpha=0.85)
axR.set_ylabel('Physical qubits per logical qubit')
axR.set_xticks(x); axR.set_xticklabels(labels, fontsize=9)
axR.set_title('Physical qubit overhead', fontsize=11)
axR.set_ylim(0, 20)
for xi, v in zip(x, qubits):
    axR.text(xi, v + 0.6, f"{v}", ha='center', fontsize=9)

fig.suptitle('Overhead-Normalized Comparison at $p=0.001$', y=1.02)
fig.tight_layout()
fig.savefig("fig3_overhead_comparison.png", dpi=300, bbox_inches="tight")
print("Saved fig3")
