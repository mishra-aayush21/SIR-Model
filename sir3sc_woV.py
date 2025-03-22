import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm  # For progress bars

# Agent class
class Agent:
    def __init__(self, agent_id):
        self.id = agent_id
        self.state = 'S'  # S, I, R

# Simulation class
class SIRSimulation:
    def __init__(self, N=100, beta=0.6, gamma=0.2, topology='all'):
        self.N = N
        self.beta = beta  # Infection rate
        self.gamma = gamma  # Recovery rate
        self.topology = topology  # Interaction network type
        self.agents = [Agent(i) for i in range(N)]
        self.time_steps = 100  # Simulate for 100 days
        self.initialize_states()
        self.initialize_network()

    def initialize_states(self):
        # Set initial states: one infected, rest susceptible
        for agent in self.agents:
            agent.state = 'S'
        self.agents[0].state = 'I'  # First agent is infected

    def initialize_network(self):
        # Create interaction network based on topology
        if self.topology == 'all':
            # All-to-all: every agent can interact with every other agent
            self.network = {i: [j for j in range(self.N) if j != i] for i in range(self.N)}
        elif self.topology == 'lattice':
            # 2D lattice: agents interact only with adjacent neighbors
            grid_size = int(np.sqrt(self.N))
            self.network = {}
            for i in range(grid_size):
                for j in range(grid_size):
                    neighbors = []
                    if i > 0: neighbors.append((i-1)*grid_size + j)  # Left neighbor
                    if i < grid_size-1: neighbors.append((i+1)*grid_size + j)  # Right neighbor
                    if j > 0: neighbors.append(i*grid_size + (j-1))  # Top neighbor
                    if j < grid_size-1: neighbors.append(i*grid_size + (j+1))  # Bottom neighbor
                    self.network[i*grid_size + j] = neighbors
        else:
            raise ValueError("Invalid topology. Choose 'all' or 'lattice'.")

    def step(self):
        # Infection phase
        new_infections = []
        for agent in self.agents:
            if agent.state == 'I':
                neighbors = self.network[agent.id]
                for neighbor_id in neighbors:
                    neighbor = self.agents[neighbor_id]
                    if neighbor.state == 'S' and random.random() < self.beta:
                        new_infections.append(neighbor.id)

        # Update states for new infections
        for agent_id in new_infections:
            self.agents[agent_id].state = 'I'

        # Recovery phase
        for agent in self.agents:
            if agent.state == 'I' and random.random() < self.gamma:
                agent.state = 'R'

    def run(self):
        S, I, R = [], [], []
        for _ in range(self.time_steps):
            self.step()
            S.append(sum(1 for a in self.agents if a.state == 'S'))
            I.append(sum(1 for a in self.agents if a.state == 'I'))
            R.append(sum(1 for a in self.agents if a.state == 'R'))
        return S, I, R

# Run simulations for different beta and gamma values
def run_scenarios(N=100, topology='all', num_runs=10):
    # Define 3 scenarios: (beta, gamma)
    scenarios = [
        (0.4, 0.1),  # Low infection rate, low recovery rate
        (0.6, 0.2),  # Medium infection rate, medium recovery rate
        (0.8, 0.3),  # High infection rate, high recovery rate
    ]

    # Store results for each scenario
    results = {}

    for beta, gamma in scenarios:
        print(f"Running simulation for beta={beta}, gamma={gamma}")
        S_avg, I_avg, R_avg = [], [], []
        for _ in tqdm(range(num_runs)):
            sim = SIRSimulation(N=N, beta=beta, gamma=gamma, topology=topology)
            S, I, R = sim.run()
            if not S_avg:
                S_avg, I_avg, R_avg = S, I, R
            else:
                S_avg = [s1 + s2 for s1, s2 in zip(S_avg, S)]
                I_avg = [i1 + i2 for i1, i2 in zip(I_avg, I)]
                R_avg = [r1 + r2 for r1, r2 in zip(R_avg, R)]
        S_avg = [s / num_runs for s in S_avg]
        I_avg = [i / num_runs for i in I_avg]
        R_avg = [r / num_runs for r in R_avg]
        results[(beta, gamma)] = (S_avg, I_avg, R_avg)

    return results

# Plot results for all scenarios
def plot_scenarios(results):
    plt.figure(figsize=(14, 8))

    for i, ((beta, gamma), (S, I, R)) in enumerate(results.items()):
        plt.subplot(3, 1, i + 1)
        plt.plot(S, label='Susceptible')
        plt.plot(I, label='Infected')
        plt.plot(R, label='Recovered')
        plt.xlabel('Time Steps (Days)')
        plt.ylabel('Population')
        plt.title(f'SIR Model (beta={beta}, gamma={gamma})')
        plt.legend()
        plt.grid()

    plt.tight_layout()
    plt.show()

# Main
if __name__ == "__main__":
    # Parameters
    N = 1000  # Number of agents
    topology = 'lattice'  # 'all' or 'lattice'
    num_runs = 10  # Number of runs for averaging

    # Run scenarios
    results = run_scenarios(N=N, topology=topology, num_runs=num_runs)

    # Plot results
    plot_scenarios(results)
