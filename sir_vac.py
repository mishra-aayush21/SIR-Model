import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm 

# Agent class
class Agent:
    def __init__(self, agent_id, strategy='L'):
        self.id = agent_id
        self.state = 'S'
        self.strategy = strategy  

# Simulation class
class SIRSimulation:
    def __init__(self, N=100, beta=0.6, gamma=0.2, vaccine_coverage=0.05, vaccine_efficacy=0.9, recovery_boost=1.5, topology='all'):
        self.N = N
        self.beta = beta  # Infection rate
        self.gamma = gamma  # Recovery rate
        self.vaccine_coverage = vaccine_coverage  # Fraction of population vaccinated
        self.vaccine_efficacy = vaccine_efficacy  # Vaccine efficacy (reduces infection probability)
        self.recovery_boost = recovery_boost  # Vaccine recovery boost
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

        # Assign vaccination strategies
        num_vaccinated = int(self.vaccine_coverage * self.N)
        vaccinated_agents = random.sample(self.agents, num_vaccinated)
        for agent in vaccinated_agents:
            agent.strategy = 'V'
            if random.random() < self.vaccine_efficacy:
                agent.state = 'V'  # Vaccinated and immune

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
                    if i > 0: neighbors.append((i-1)*grid_size + j)  
                    if i < grid_size-1: neighbors.append((i+1)*grid_size + j)  
                    if j > 0: neighbors.append(i*grid_size + (j-1)) 
                    if j < grid_size-1: neighbors.append(i*grid_size + (j+1)) 
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
                    if neighbor.state == 'S':
                        # Vaccinated agents have reduced infection probability
                        if neighbor.strategy == 'V':
                            infection_prob = self.beta * (1 - self.vaccine_efficacy)
                        else:
                            infection_prob = self.beta
                        if random.random() < infection_prob:
                            new_infections.append(neighbor.id)

        # Update states for new infections
        for agent_id in new_infections:
            self.agents[agent_id].state = 'I'

        # Recovery phase
        for agent in self.agents:
            if agent.state == 'I':
                # Vaccinated agents recover faster
                if agent.strategy == 'V':
                    recovery_prob = self.gamma * self.recovery_boost
                else:
                    recovery_prob = self.gamma
                if random.random() < recovery_prob:
                    agent.state = 'R'

    def run(self):
        S, I, R, V = [], [], [], []
        for _ in range(self.time_steps):
            self.step()
            S.append(sum(1 for a in self.agents if a.state == 'S'))
            I.append(sum(1 for a in self.agents if a.state == 'I'))
            R.append(sum(1 for a in self.agents if a.state == 'R'))
            V.append(sum(1 for a in self.agents if a.state == 'V'))
        return S, I, R, V

# Run simulations
def run_simulations(N=100, beta=0.6, gamma=0.2, vaccine_coverage=0.5, vaccine_efficacy=0.9, recovery_boost=1.5, topology='all', num_runs=10):
    S_avg, I_avg, R_avg, V_avg = [], [], [], []
    for _ in tqdm(range(num_runs)):
        sim = SIRSimulation(N=N, beta=beta, gamma=gamma, vaccine_coverage=vaccine_coverage, vaccine_efficacy=vaccine_efficacy, recovery_boost=recovery_boost, topology=topology)
        S, I, R, V = sim.run()
        if not S_avg:
            S_avg, I_avg, R_avg, V_avg = S, I, R, V
        else:
            S_avg = [s1 + s2 for s1, s2 in zip(S_avg, S)]
            I_avg = [i1 + i2 for i1, i2 in zip(I_avg, I)]
            R_avg = [r1 + r2 for r1, r2 in zip(R_avg, R)]
            V_avg = [v1 + v2 for v1, v2 in zip(V_avg, V)]
    S_avg = [s / num_runs for s in S_avg]
    I_avg = [i / num_runs for i in I_avg]
    R_avg = [r / num_runs for r in R_avg]
    V_avg = [v / num_runs for v in V_avg]
    return S_avg, I_avg, R_avg, V_avg

# Plot results
def plot_results(S, I, R, V, title):
    plt.figure(figsize=(10, 6))
    plt.plot(S, label='Susceptible')
    plt.plot(I, label='Infected')
    plt.plot(R, label='Recovered')
    plt.plot(V, label='Vaccinated')
    plt.xlabel('Time Steps (Days)')
    plt.ylabel('Number of Agents')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

# Main
if __name__ == "__main__":
    # Parameters
    N = 1000  # Number of agents
    beta = 0.6  # Infection rate
    gamma = 0.2  # Recovery rate
    vaccine_coverage = 0.2  # Fraction of population vaccinated
    vaccine_efficacy = 0.4  # Vaccine efficacy
    recovery_boost = 1.5  # Vaccine recovery boost
    topology = 'all'  # 'all' or 'lattice'
    num_runs = 10  # Number of runs for averaging

    # Run simulation
    S, I, R, V = run_simulations(N=N, beta=beta, gamma=gamma, vaccine_coverage=vaccine_coverage, vaccine_efficacy=vaccine_efficacy, recovery_boost=recovery_boost, topology=topology, num_runs=num_runs)

    # Plot results
    plot_results(S, I, R, V, title=f"SIR Model with Vaccination (Topology: {topology})")
