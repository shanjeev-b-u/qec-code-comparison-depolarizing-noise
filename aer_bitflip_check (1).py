import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error

# ---- Encoding + (noiseless) decode circuit for fidelity verification ----
def encode_circuit(input_state_prep):
    qc = QuantumCircuit(3)
    input_state_prep(qc, 0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    return qc

def decode_circuit(qc):
    # inverse of encoding: two CNOTs (self-inverse) recovers the data qubit onto qubit 0
    qc.cx(0, 2)
    qc.cx(0, 1)
    return qc

def run_fidelity_check(prep_name, prep_fn):
    qc = QuantumCircuit(3)
    prep_fn(qc, 0)
    ref_state = Statevector.from_instruction(qc)  # reference single-qubit-embedded state (for logging only)

    enc = QuantumCircuit(3)
    prep_fn(enc, 0)
    enc.cx(0, 1); enc.cx(0, 2)
    full_after_encode = Statevector.from_instruction(enc)

    dec = enc.copy()
    dec.cx(0, 2); dec.cx(0, 1)
    full_after_decode = Statevector.from_instruction(dec)

    # expected decoded state: |psi> on qubit0 tensor |00> on qubits 1,2
    expected = QuantumCircuit(3)
    prep_fn(expected, 0)
    expected_sv = Statevector.from_instruction(expected)

    fid = state_fidelity(full_after_decode, expected_sv)
    print(f"{prep_name}: noiseless encode->decode fidelity = {fid:.6f}")
    return fid

def prep0(qc, q): pass
def prep1(qc, q): qc.x(q)
def prepPlus(qc, q): qc.h(q)
def prepMinus(qc, q): qc.x(q); qc.h(q)

fids = {}
for name, fn in [("|0>", prep0), ("|1>", prep1), ("|+>", prepPlus), ("|->", prepMinus)]:
    fids[name] = run_fidelity_check(name, fn)

print("\nAll noiseless fidelities == 1.0:", all(abs(v - 1.0) < 1e-9 for v in fids.values()))

# ---- Noisy circuit-level cross-check against the Pauli-frame Monte Carlo result ----
# Full circuit: encode -> depolarizing noise on 3 data qubits -> 2-ancilla syndrome
# extraction -> classically-controlled X correction -> measure data qubits in Z basis.
def build_noisy_circuit(p):
    data = QuantumRegister(3, 'd')
    anc = QuantumRegister(2, 'a')
    creg_syn = ClassicalRegister(2, 'syn')
    creg_out = ClassicalRegister(3, 'out')
    qc = QuantumCircuit(data, anc, creg_syn, creg_out)

    # encode logical |0>
    qc.cx(data[0], data[1])
    qc.cx(data[0], data[2])

    # noisy identity gates (depolarizing channel) on each data qubit
    qc.id(data[0]); qc.id(data[1]); qc.id(data[2])

    # syndrome extraction: anc0 = Z0 xor Z1 parity, anc1 = Z1 xor Z2 parity
    qc.cx(data[0], anc[0]); qc.cx(data[1], anc[0])
    qc.cx(data[1], anc[1]); qc.cx(data[2], anc[1])
    qc.measure(anc[0], creg_syn[0])
    qc.measure(anc[1], creg_syn[1])

    # classically-controlled correction (majority vote via syndrome table)
    with qc.if_test((creg_syn, 1)):   # syn=01 -> qubit0 flipped
        qc.x(data[0])
    with qc.if_test((creg_syn, 3)):   # syn=11 -> qubit1 flipped
        qc.x(data[1])
    with qc.if_test((creg_syn, 2)):   # syn=10 -> qubit2 flipped
        qc.x(data[2])

    qc.measure(data[0], creg_out[0])
    qc.measure(data[1], creg_out[1])
    qc.measure(data[2], creg_out[2])
    return qc

def noise_model_for_p(p):
    nm = NoiseModel()
    err = pauli_error([('X', p/3), ('Y', p/3), ('Z', p/3), ('I', 1 - p)])
    nm.add_all_qubit_quantum_error(err, ['id'])
    return nm

sim = AerSimulator()
shots = 20000
p_test = 0.05
qc = build_noisy_circuit(p_test)
qc_t = qc  # AerSimulator can run if_test natively
nm = noise_model_for_p(p_test)
result = sim.run(qc_t, noise_model=nm, shots=shots, basis_gates=nm.basis_gates + ['id','cx','x','h']).result()
counts = result.get_counts()

# logical bit-flip failure = majority of the 3 output bits != 0 (after correction,
# should be 000 for success under the code's designed correction capability;
# residual X-type failure appears as all-ones 111 pattern; but decoder only fixes X part,
# so any pattern other than 000 counts as an X-observable logical/physical mismatch here.)
fail = 0
total = 0
for bitstring, c in counts.items():
    # bitstring format: "out syn" separated by space (Qiskit convention, out is leftmost)
    parts = bitstring.split()
    out_bits = parts[0]
    ones = out_bits.count('1')
    total += c
    if ones >= 2:  # majority-corrected value still reads logical 1 -> failure (bit-observable metric)
        fail += c

pl_aer = fail / total
print(f"\nAer noisy circuit simulation at p={p_test}: shots={total}, logical failures={fail}, PL={pl_aer:.5f}")

from simulate import bitflip_PL_exact, bitflip_PL_montecarlo
import numpy.random as npr
rng = npr.default_rng(1)
print("Cross-check vs. Pauli-frame exact:", bitflip_PL_exact(p_test))
mc, k = bitflip_PL_montecarlo(p_test, total, rng)
print("Cross-check vs. Pauli-frame Monte Carlo (same shot count):", mc)
