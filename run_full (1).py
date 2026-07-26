import numpy as np
import json
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simulate import (bitflip_PL_exact, bitflip_PL_montecarlo, bitflip_PL_Xonly,
                       bitflip_PL_full_exact, surface_PL_exact, surface_PL_montecarlo)

rng = np.random.default_rng(20260719)

p_values = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
NSHOTS = 200000  # per (p, code) data point

results = {"p": p_values, "bitflip": {}, "surface": {}}

bf_sim = []
bf_exact = []
bf_ci = []
for p in p_values:
    pl_mc, k = bitflip_PL_montecarlo(p, NSHOTS, rng)
    bf_sim.append(pl_mc)
    bf_exact.append(bitflip_PL_exact(p))
    # Wilson 95% CI
    z = 1.96
    n = NSHOTS
    phat = pl_mc
    denom = 1 + z**2/n
    center = (phat + z**2/(2*n)) / denom
    halfwidth = (z * np.sqrt(phat*(1-phat)/n + z**2/(4*n**2))) / denom
    bf_ci.append(halfwidth)

sc_sim = []
sc_exact = []
sc_ci = []
for p in p_values:
    pl_mc, k = surface_PL_montecarlo(p, NSHOTS, rng)
    sc_sim.append(pl_mc)
    sc_exact.append(surface_PL_exact(p))
    z = 1.96
    n = NSHOTS
    phat = pl_mc
    denom = 1 + z**2/n
    center = (phat + z**2/(2*n)) / denom
    halfwidth = (z * np.sqrt(phat*(1-phat)/n + z**2/(4*n**2))) / denom
    sc_ci.append(halfwidth)

results["bitflip"]["sim"] = bf_sim
results["bitflip"]["exact"] = bf_exact
results["bitflip"]["ci95"] = bf_ci
results["surface"]["sim"] = sc_sim
results["surface"]["exact"] = sc_exact
results["surface"]["ci95"] = sc_ci

print("p       BF_sim      BF_exact     SC_sim       SC_exact")
for i, p in enumerate(p_values):
    print(f"{p:<8g}{bf_sim[i]:<12.5g}{bf_exact[i]:<13.5g}{sc_sim[i]:<13.5g}{sc_exact[i]:<13.5g}")

# ---- Pseudo-thresholds: solve PL_exact(p) - p = 0 via brentq on a fine grid ----
def find_crossing(func, lo, hi):
    f = lambda p: func(p) - p
    flo, fhi = f(lo), f(hi)
    if flo == 0:
        return lo
    if flo * fhi > 0:
        return None
    return brentq(f, lo, hi, xtol=1e-6)

# bit-flip pseudo-threshold (depolarizing), search in (0, 0.5)
grid = np.linspace(1e-6, 0.5, 4000)
vals = np.array([bitflip_PL_exact(p) - p for p in grid])
sign_changes = np.where(np.diff(np.sign(vals)) != 0)[0]
bf_pstar = None
for idx in sign_changes:
    lo, hi = grid[idx], grid[idx+1]
    if lo > 0.001:  # skip trivial p=0 root region
        bf_pstar = find_crossing(bitflip_PL_exact, lo, hi)
        break
print("Bit-flip pseudo-threshold (exact, depolarizing):", bf_pstar)

# pure-X-only theoretical check (should be 0.5)
grid2 = np.linspace(1e-6, 0.9, 4000)
vals2 = np.array([bitflip_PL_Xonly(p) - p for p in grid2])
sc2 = np.where(np.diff(np.sign(vals2)) != 0)[0]
bf_pstar_Xonly = None
for idx in sc2:
    lo, hi = grid2[idx], grid2[idx+1]
    if lo > 0.001:
        bf_pstar_Xonly = find_crossing(bitflip_PL_Xonly, lo, hi)
        break
print("Bit-flip pseudo-threshold (X-only, analytic check vs 0.5):", bf_pstar_Xonly)

# surface code pseudo-threshold, search in (0, 0.2) at fine resolution
grid3 = np.concatenate([np.linspace(1e-5, 0.01, 40), np.linspace(0.01, 0.2, 100)])
vals3 = np.array([surface_PL_exact(p) - p for p in grid3])
sc3 = np.where(np.diff(np.sign(vals3)) != 0)[0]
sc_pstar = None
for idx in sc3:
    lo, hi = grid3[idx], grid3[idx+1]
    if lo > 1e-5:
        sc_pstar = find_crossing(surface_PL_exact, lo, hi)
        break
print("Surface code pseudo-threshold (exact, depolarizing):", sc_pstar)

results["bf_pstar"] = bf_pstar
results["bf_pstar_Xonly"] = bf_pstar_Xonly
results["sc_pstar"] = sc_pstar

# Diagnostic: full quantum-state-protection variant (includes phase leakage).
# Check whether PL_full(p) - p is ever negative for p in (0, 0.9)
grid4 = np.linspace(1e-4, 0.9, 2000)
diag = np.array([bitflip_PL_full_exact(p) - p for p in grid4])
results["bf_full_never_beneficial"] = bool(np.all(diag > 0))
results["bf_full_min_margin_p"] = float(grid4[np.argmin(diag)])
results["bf_full_min_margin_val"] = float(diag.min())
print("Full-protection variant: PL_full(p) > p for ALL p in (0,0.9)?", results["bf_full_never_beneficial"])
print("  minimum margin at p=", results["bf_full_min_margin_p"], " margin=", results["bf_full_min_margin_val"])

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved results.json")
