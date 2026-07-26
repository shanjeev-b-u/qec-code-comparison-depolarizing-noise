```markdown
# QEC Code Comparison Under Depolarizing Noise

Simulation code and data for:

> **Comparative Analysis of Logical Error Rate Scaling in the Three-Qubit Bit-Flip Code
> and the Distance-3 Surface Code Under Depolarizing Noise: A Simulation Study Using
> Qiskit Aer and a Validated Stabilizer-Formalism Engine**  
> Shanjeev B U, Department of Computer Science and Engineering, Meenakshi College of
> Engineering (Anna University), Chennai, India. July 2026.  
> ORCID: [0009-0001-7286-4428](https://orcid.org/0009-0001-7286-4428)

This repository reproduces every simulated number, table, and figure in the paper. All
random seeds are fixed, so re-running the scripts or notebooks reproduces the exact
values reported (e.g. `p*_SC = 0.08278398659713192`, Table 2 in the paper).

---

## What this study does

Two QEC codes are compared under an independent single-qubit depolarizing channel,
sweeping the physical error rate p ∈ [0.001, 0.30]:

- **Three-qubit bit-flip code** — simulated by **literal Qiskit Aer circuit execution**
  (encoding, injected depolarizing noise, dynamic classical feed-forward correction).
- **Distance-3 rotated surface code** (9 data + 8 ancilla = 17 qubits, "Surface-17") —
  simulated with a **custom vectorized stabilizer-formalism engine** implementing the
  same Aaronson–Gottesman algorithm underlying Aer's stabilizer backend. This is
  cross-validated directly against genuine Aer circuit execution (see
  `scripts/aer_bitflip_check.py` and Section 5.1 of the paper); no surface-code result
  in the paper comes from literal 17-qubit Aer execution.

The headline finding is that the bit-flip code's apparent performance is **entirely
dependent on the protection criterion** used to evaluate it — see Section 1.1 and 4.2
of the paper, and `pseudo_thresholds_summary.json` below.

| Code | Metric | Simulated / exact pseudo-threshold p* |
|---|---|---|
| 3-qubit bit-flip | bit-observable (protects a classical bit value) | no crossing in (0,1) — unconditionally beneficial |
| 3-qubit bit-flip | full quantum-state protection | ≈ 0.75 (physically irrelevant regime) |
| Distance-3 surface code (17 qubits) | — | ≈ 0.0828 (8.28%), idealized single-round finite-size result |

---

## Repository layout

```text
.
├── scripts/            Plain Python (.py) — the main simulation, decoder, and analysis code
│   ├── build_surface_code.py     Constructs & validates the distance-3 surface code
│   ├── simulate.py               Core engine: exact enumeration + Monte Carlo, both codes
│   ├── run_full.py               Runs the full p-sweep, solves pseudo-thresholds
│   ├── make_figures.py           Generates all 3 paper figures (300 DPI)
│   └── aer_bitflip_check.py      Real Qiskit Aer circuit run + cross-validation
│
├── notebooks/          Jupyter notebooks (.ipynb) — step-by-step, executed, with outputs
│   ├── 01_surface_code_construction.ipynb
│   ├── 02_qiskit_aer_bitflip_verification.ipynb
│   └── 03_full_sweep_simulation_and_figures.ipynb
│
├── data/                Raw simulation outputs (before figures/summary tables)
│   ├── surface_code_structure.npz     Hx, Hz, Lx, Lz, n_data
│   ├── results.json                   Full per-p sweep: MC + exact values, 95% CI, p*
│   ├── sweep_results.csv              Same sweep, flat CSV
│   └── pseudo_thresholds_summary.json Headline p* numbers only
│
├── figures/             fig1_logical_error_vs_physical.png
│                        fig2_sim_vs_analytical.png
│                        fig3_overhead_comparison.png
│
├── requirements.txt
└── README.md

```

---

## Quick start

```bash
git clone [https://github.com/shanjeev-b-u/qec-code-comparison-depolarizing-noise.git](https://github.com/shanjeev-b-u/qec-code-comparison-depolarizing-noise.git)
cd qec-code-comparison-depolarizing-noise
pip install -r requirements.txt

```

Reproduce everything from scratch:

```bash
cd scripts
python3 build_surface_code.py     # -> ../data/surface_code_structure.npz
python3 run_full.py               # -> ../data/results.json          (~1-2 min)
python3 make_figures.py           # -> ../figures/fig1/2/3.png (300 DPI)
python3 aer_bitflip_check.py      # independent Aer cross-validation sanity check

```

Or open the notebooks in order (`01` → `02` → `03`) for a narrated, step-by-step
walkthrough with the same code and inline explanations.

**Requirements:** Python 3.12, Qiskit 2.5.0, Qiskit Aer 0.17.2, NumPy, SciPy,
Matplotlib. A fixed random seed (`20260719`) is used throughout.

---

## Methodology summary

* **Noise model**: independent single-qubit depolarizing channel, `p ∈ {X: p/3, Y: p/3, Z: p/3, I: 1-p}`, swept over 12 values from 0.001 to 0.30.
* **Bit-flip code**: evaluated under two distinct, explicitly labeled protection criteria — *bit-observable* (does a definite computational-basis value survive?) and *full quantum-state protection* (does an arbitrary logical qubit state survive?). See `simulate.py::bitflip_PL_exact` vs. `bitflip_PL_full_exact`.
* **Surface code**: CSS decoding via an exact minimum-weight lookup table (built by exhaustive enumeration over all 2⁹ single-sector error patterns — exact at d=3), applied independently to the X and Z stabilizer sectors.
* **Statistics**: 200,000 Monte Carlo shots per (p, code) data point; uncertainty reported as the true Wilson score interval (not the simpler normal/Wald approximation) at 95% confidence.
* **Cross-validation**: the surface code's Pauli-frame/stabilizer-formalism engine is checked against literal Qiskit Aer circuit execution for the bit-flip code at a representative operating point (p=0.05) — the two agree to within statistical noise (`aer_bitflip_check.py`).

Full derivations, all equations, and the complete discussion of limitations and scope
are in the paper (Sections 4–7).

---

## Validation

Every number below was regenerated fresh from these scripts immediately before this
README was written, and matches the paper exactly:

* All three notebooks execute end-to-end with **zero errors**.
* `run_full.py` → `sc_pstar = 0.08278398659713192` (Table 2).
* `aer_bitflip_check.py` at p=0.05: Aer circuit P_L ≈ 0.0033–0.0035 vs. Pauli-frame exact P_L = 0.0032593 — consistent within Monte Carlo shot noise.

## Citation

If you use this code, please cite the accompanying paper (see repository description
for the full reference) and this repository.

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

```

```
