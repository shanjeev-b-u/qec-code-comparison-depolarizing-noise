"""
Programmatically construct the distance-3 rotated surface code
(9 data qubits + 8 ancilla qubits = 17 total), verify stabilizers commute,
and derive logical operators. This is the standard construction used in
e.g. Versluis et al. 2017 / Google 'Surface-17' experiments.
"""
import numpy as np
import itertools

d = 3
data_pos = {}
idx = 0
for y in range(d):
    for x in range(d):
        data_pos[(x, y)] = idx
        idx += 1
n_data = idx
print("n_data =", n_data)

# candidate plaquette centers
stabs = []  # list of (type, frozenset(data indices), (px,py))
for x in range(-1, d):
    for y in range(-1, d):
        px, py = x + 0.5, y + 0.5
        corners = [(x, y), (x+1, y), (x, y+1), (x+1, y+1)]
        touched = [c for c in corners if c in data_pos]
        if len(touched) == 4:
            ptype = 'X' if (x + y) % 2 == 0 else 'Z'
            stabs.append((ptype, frozenset(data_pos[c] for c in touched), (px, py)))
        elif len(touched) == 2:
            # natural type via same parity rule
            ntype = 'X' if (x + y) % 2 == 0 else 'Z'
            (c1, c2) = touched
            horizontal = (c1[1] == c2[1])  # same y -> horizontal pair -> top/bottom boundary
            vertical = (c1[0] == c2[0])    # same x -> vertical pair -> left/right boundary
            # convention: Z stabilizers only on top/bottom boundary (horizontal pairs)
            #             X stabilizers only on left/right boundary (vertical pairs)
            if ntype == 'Z' and horizontal:
                stabs.append((ntype, frozenset(data_pos[c] for c in touched), (px, py)))
            elif ntype == 'X' and vertical:
                stabs.append((ntype, frozenset(data_pos[c] for c in touched), (px, py)))
            # else discard (wrong-type boundary plaquette)

x_stabs = [s for s in stabs if s[0] == 'X']
z_stabs = [s for s in stabs if s[0] == 'Z']
print("n_X_stabs =", len(x_stabs), "n_Z_stabs =", len(z_stabs))
for t, supp, pos in stabs:
    print(t, sorted(supp), pos)

n_anc = len(x_stabs) + len(z_stabs)
print("total ancillas =", n_anc, " total physical qubits =", n_data + n_anc)

# Build parity-check (binary) matrices
Hx = np.zeros((len(x_stabs), n_data), dtype=int)  # X-stabilizers -> detect Z errors
for i, (_, supp, _) in enumerate(x_stabs):
    for q in supp:
        Hx[i, q] = 1

Hz = np.zeros((len(z_stabs), n_data), dtype=int)  # Z-stabilizers -> detect X errors
for i, (_, supp, _) in enumerate(z_stabs):
    for q in supp:
        Hz[i, q] = 1

# verify all X stabilizers commute with all Z stabilizers (even overlap)
overlaps = (Hx @ Hz.T) % 2
assert np.all(overlaps == 0), "Stabilizers do not commute!"
print("All stabilizers commute. OK.")

# Logical operators: Z-logical = a vertical column of Z's (commutes with all X stabilizers since
# each X stabilizer's support intersects any single column in an even number of qubits: 0 or 2)
# X-logical = a horizontal row of X's.
# Try column x=0 for Z-logical, row y=0 for X-logical.
Lz = np.zeros(n_data, dtype=int)
for y in range(d):
    Lz[data_pos[(0, y)]] = 1
Lx = np.zeros(n_data, dtype=int)
for x in range(d):
    Lx[data_pos[(x, 0)]] = 1

# verify Lz commutes with all X stabilizers
assert np.all((Hx @ Lz) % 2 == 0), "Lz does not commute with X-stabilizers"
# verify Lx commutes with all Z stabilizers
assert np.all((Hz @ Lx) % 2 == 0), "Lx does not commute with Z-stabilizers"
# verify Lx and Lz anticommute (overlap odd)
overlap_LxLz = int(np.sum(Lx * Lz)) % 2
print("Lx . Lz overlap parity (should be 1):", overlap_LxLz)
assert overlap_LxLz == 1

print("Lz support:", sorted(np.nonzero(Lz)[0]))
print("Lx support:", sorted(np.nonzero(Lx)[0]))

np.savez("surface_code_structure.npz", Hx=Hx, Hz=Hz, Lx=Lx, Lz=Lz, n_data=n_data)
print("Saved.")
