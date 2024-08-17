import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

def get_quantum_random_numbers(num_samples):
    qc = QuantumCircuit(2, 2)
    qc.h([0, 1])
    qc.measure([0, 1], [0, 1])

    simulator = Aer.get_backend('qasm_simulator')
    qc = transpile(qc, simulator)
    job = simulator.run(qc, shots=num_samples)
    result = job.result()
    counts = result.get_counts()
    
    quantum_random_numbers = []
    for outcome in counts:
        for _ in range(counts[outcome]):
            if outcome == '00':
                quantum_random_numbers.append((+1, -1))
            elif outcome == '01':
                quantum_random_numbers.append((+1, +1))
            elif outcome == '10':
                quantum_random_numbers.append((-1, -1))
            elif outcome == '11':
                quantum_random_numbers.append((-1, +1))
    
    return quantum_random_numbers

def simulate_random_walk(n_samples, steps):
    all_distances = np.zeros((n_samples, steps))
    
    for i in range(n_samples):
        x, y = 0, 0
        for j in range(steps):
            direction = get_quantum_random_numbers(1)[0]
            x += direction[0]
            y += direction[1]
            distance = np.sqrt(x**2 + y**2)
            all_distances[i, j] = distance
    
    return all_distances

def plot_average_distance(n_samples, steps):
    distances = simulate_random_walk(n_samples, steps)
    avg_distances = np.mean(distances, axis=0)
    
    plt.plot(avg_distances, label="Average Distance from Origin")
    plt.xlabel("Step")
    plt.ylabel("Distance from Origin")
    plt.title(f"Average Distance (from origin) over {n_samples} Random Walks")
    plt.legend()
    plt.grid(True)
    plt.show()

n_samples = 1000
steps = 10

plot_average_distance(n_samples, steps)
