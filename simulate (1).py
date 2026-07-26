import numpy as np
import itertools, json

rng = np.random.default_rng(20260719)

# ---------------------------------------------------------------------
# 1. THREE-QUBIT BIT-FLIP CODE — exact enumeration (4^3 = 64 configs)
#    Pauli per qubit: I (1-p), X (p/3), Y (p/3), Z (p/3)
#    x-component present iff error in {X,Y}; z-component present iff error in {Z,Y}
#    Logical X-bar failure iff weight(x-components) >= 2  (majority-vote miscorrects)
#    Logical Z-bar failure iff weight(z-components) is odd (1 or 3)  (undetectable phase leakage)
#    Overall logical failure iff either condition holds.
# ---------------------------------------------------------------------
paulis = ['I', 'X', 'Y', 'Z']
xcomp = {'I': 0, 'X': 1, 'Y': 1, 'Z': 0}
zcomp = {'I': 0, 'X': 0, 'Y': 1, 'Z': 1}

# PRIMARY reported metric: "bit-observable" logical error rate. The 3-qubit code's
# stabilizers (Z0Z1, Z1Z2) and majority-vote decoder only ever act on the X-component of
# the error; Z-type components are invisible to this code by construction. This is the
# quantity Eq. (7)-(8) generalizes to full depolarizing noise: each qubit carries an
# X-component (from an X or Y error) with probability q = 2p/3.
def bitflip_PL_exact(p):
    q = 2 * p / 3
    return 3 * q**2 * (1 - q) + q**3

def bitflip_PL_montecarlo(p, nshots, rng):
    probs = [1 - p, p/3, p/3, p/3]
    draws = rng.choice(4, size=(nshots, 3), p=probs)  # 0=I,1=X,2=Y,3=Z
    xw = ((draws == 1) | (draws == 2)).sum(axis=1)
    fail = (xw >= 2)
    k = int(fail.sum())
    return k / nshots, k

# SECONDARY diagnostic: full quantum-state-protection logical error rate, which also
# counts the (always-undetected) phase-leakage failures from Z/Y components. Used only
# to test whether the code can serve as a general-purpose QEC code under full
# depolarizing noise (Section 7.2 discussion), not as the primary Table 2 metric.
def bitflip_PL_full_exact(p):
    total = 0.0
    prob = {'I': 1 - p, 'X': p/3, 'Y': p/3, 'Z': p/3}
    for combo in itertools.product(paulis, repeat=3):
        pr = 1.0
        for e in combo:
            pr *= prob[e]
        xw = sum(xcomp[e] for e in combo)
        zw = sum(zcomp[e] for e in combo)
        fail = (xw >= 2) or (zw % 2 == 1)
        if fail:
            total += pr
    return total

# pure-X-only theoretical curve (Eq 8 context)
def bitflip_PL_Xonly(p):
    return 3*p**2*(1-p) + p**3

# ---------------------------------------------------------------------
# 2. DISTANCE-3 SURFACE CODE — Monte Carlo with exact min-weight lookup decoder
# ---------------------------------------------------------------------
data = np.load("surface_code_structure.npz")
Hx, Hz, Lx, Lz = data['Hx'], data['Hz'], data['Lx'], data['Lz']
n = int(data['n_data'])

def build_lookup(H):
    """H: (m, n) parity check. Returns dict: syndrome(int) -> correction vector (min weight)."""
    m = H.shape[0]
    table = {}
    weight_table = {}
    for bits in itertools.product([0, 1], repeat=n):
        e = np.array(bits)
        syn = tuple((H @ e) % 2)
        syn_int = int(''.join(map(str, syn)), 2) if m > 0 else 0
        w = e.sum()
        if syn_int not in table or w < weight_table[syn_int]:
            table[syn_int] = e.copy()
            weight_table[syn_int] = w
    return table

lut_z = build_lookup(Hz)   # syndrome from Z-stabilizers -> correction for X errors
lut_x = build_lookup(Hx)   # syndrome from X-stabilizers -> correction for Z errors

def syn_to_int(bits):
    if len(bits) == 0:
        return 0
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out

def surface_PL_montecarlo(p, nshots, rng):
    probs = [1 - p, p/3, p/3, p/3]
    draws = rng.choice(4, size=(nshots, n), p=probs)  # 0=I,1=X,2=Y,3=Z
    ex = ((draws == 1) | (draws == 2)).astype(int)  # X-component per qubit
    ez = ((draws == 3) | (draws == 2)).astype(int)  # Z-component per qubit

    synZ = (ex @ Hz.T) % 2   # detects X errors  (nshots, n_Zstab)
    synX = (ez @ Hx.T) % 2   # detects Z errors  (nshots, n_Xstab)

    fail = np.zeros(nshots, dtype=bool)
    # vectorize decode via precomputed table indexed by integer syndrome
    # build syndrome->correction arrays
    nZ = Hz.shape[0]
    nX = Hx.shape[0]
    # correction lookup as arrays for speed
    max_synZ = 2**nZ
    max_synX = 2**nX
    corrZ_arr = np.zeros((max_synZ, n), dtype=int)
    for s, c in lut_z.items():
        corrZ_arr[s] = c
    corrX_arr = np.zeros((max_synX, n), dtype=int)
    for s, c in lut_x.items():
        corrX_arr[s] = c

    weightsZ = (2**np.arange(nZ - 1, -1, -1))
    weightsX = (2**np.arange(nX - 1, -1, -1))
    synZ_int = (synZ * weightsZ).sum(axis=1)
    synX_int = (synX * weightsX).sum(axis=1)

    corr_x = corrZ_arr[synZ_int]   # correction to X-error pattern (from Z-syndrome)
    corr_z = corrX_arr[synX_int]   # correction to Z-error pattern (from X-syndrome)

    resid_x = (ex + corr_x) % 2
    resid_z = (ez + corr_z) % 2

    # logical X-type failure: residual X pattern anticommutes with logical Z (overlap odd)
    logicalZ_fail = (resid_x @ Lz) % 2 == 1
    # logical Z-type failure: residual Z pattern anticommutes with logical X (overlap odd)
    logicalX_fail = (resid_z @ Lx) % 2 == 1

    fail = logicalZ_fail | logicalX_fail
    k = int(fail.sum())
    return k / nshots, k

# exact small-n check isn't feasible (4^9 too large for brute enumeration only if we want, but 4^9=262144, actually feasible!)
_nZ = Hz.shape[0]; _nX = Hx.shape[0]
_weightsZ = (2**np.arange(_nZ - 1, -1, -1))
_weightsX = (2**np.arange(_nX - 1, -1, -1))
_corrZ_arr = np.zeros((2**_nZ, n), dtype=int)
for s, c in lut_z.items():
    _corrZ_arr[s] = c
_corrX_arr = np.zeros((2**_nX, n), dtype=int)
for s, c in lut_x.items():
    _corrX_arr[s] = c

_ids = np.arange(4)
_grids = np.meshgrid(*([_ids] * n), indexing='ij')
_combo_ids = np.stack([g.ravel() for g in _grids], axis=1)

_ex_all = ((_combo_ids == 1) | (_combo_ids == 2)).astype(int)
_ez_all = ((_combo_ids == 3) | (_combo_ids == 2)).astype(int)

_synZ_all = (_ex_all @ Hz.T) % 2
_synX_all = (_ez_all @ Hx.T) % 2
_synZ_int_all = (_synZ_all * _weightsZ).sum(axis=1)
_synX_int_all = (_synX_all * _weightsX).sum(axis=1)

_corr_x_all = _corrZ_arr[_synZ_int_all]
_corr_z_all = _corrX_arr[_synX_int_all]
_resid_x_all = (_ex_all + _corr_x_all) % 2
_resid_z_all = (_ez_all + _corr_z_all) % 2

_logicalZ_fail_all = (_resid_x_all @ Lz) % 2 == 1
_logicalX_fail_all = (_resid_z_all @ Lx) % 2 == 1
_fail_all = _logicalZ_fail_all | _logicalX_fail_all

_nI = (_combo_ids == 0).sum(axis=1)
_n_err = n - _nI

def surface_PL_exact(p):
    if p <= 0:
        return 0.0
    if p >= 1:
        p = 0.999999999
    log_probs = _nI * np.log(1 - p) + _n_err * np.log(p / 3)
    probs = np.exp(log_probs)
    return float(probs[_fail_all].sum())

if __name__ == "__main__":
    print("Testing bit-flip code p=0.05: exact =", bitflip_PL_exact(0.05))
    mc, k = bitflip_PL_montecarlo(0.05, 200000, rng)
    print("  MC =", mc)

    print("Testing surface code p=0.01: exact =", surface_PL_exact(0.01))
    mc, k = surface_PL_montecarlo(0.01, 50000, rng)
    print("  MC =", mc)
