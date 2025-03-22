import pygame
import numpy as np
import random

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 800, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("SIR Model Visualization")
clock = pygame.time.Clock()

# Colors
COLORS = {
    'S': (0, 255, 0),    # Green for Susceptible
    'I': (255, 0, 0),    # Red for Infected
    'R': (0, 0, 255),    # Blue for Recovered
    'V': (255, 255, 0)   # Yellow for Vaccinated
}

# Font for displaying text
font = pygame.font.SysFont("Arial", 24)

# Agent class
class Agent:
    def __init__(self, agent_id, strategy='L'):
        self.id = agent_id
        self.state = 'S'  # S, I, R, V
        self.strategy = strategy  # 'V' for vaccinate, 'L' for loner
        self.x = random.randint(0, WIDTH)
        self.y = random.randint(0, HEIGHT)
        self.dx = random.choice([-1, 1])
        self.dy = random.choice([-1, 1])

    def move(self):
        # Move the agent within the screen boundaries
        self.x += self.dx
        self.y += self.dy
        if self.x <= 0 or self.x >= WIDTH:
            self.dx *= -1
        if self.y <= 0 or self.y >= HEIGHT:
            self.dy *= -1

    def draw(self):
        # Draw the agent as a colored dot
        color = COLORS[self.state]
        pygame.draw.circle(screen, color, (self.x, self.y), 5)

# Simulation class
class SIRSimulation:
    def __init__(self, N=100, beta=0.6, gamma=0.2, vaccine_coverage=0.5, vaccine_efficacy=0.9, recovery_boost=1.5):
        self.N = N
        self.beta = beta  # Infection rate
        self.gamma = gamma  # Recovery rate
        self.vaccine_coverage = vaccine_coverage  # Fraction of population vaccinated
        self.vaccine_efficacy = vaccine_efficacy  # Vaccine efficacy (reduces infection probability)
        self.recovery_boost = recovery_boost  # Vaccine recovery boost
        self.agents = [Agent(i) for i in range(N)]
        self.initialize_states()

    def initialize_states(self):
        # Set initial states: one infected, rest susceptible
        for agent in self.agents:
            agent.state = 'S'
        self.agents[0].state = 'I'  # First agent is infected
        self.agents[1].state = 'I'
        self.agents[2].state = 'I'
        self.agents[3].state = 'I'
        self.agents[4].state = 'I'

        # Assign vaccination strategies
        num_vaccinated = int(self.vaccine_coverage * self.N)
        vaccinated_agents = random.sample(self.agents, num_vaccinated)
        for agent in vaccinated_agents:
            agent.strategy = 'V'
            if random.random() < self.vaccine_efficacy:
                agent.state = 'V'  # Vaccinated and immune

    def step(self):
        # Infection phase
        new_infections = []
        for agent in self.agents:
            if agent.state == 'I':
                for other_agent in self.agents:
                    if other_agent.state == 'S':
                        # Check proximity for infection
                        distance = np.sqrt((agent.x - other_agent.x)**2 + (agent.y - other_agent.y)**2)
                        if distance < 20:  # Infection radius
                            # Vaccinated agents have reduced infection probability
                            if other_agent.strategy == 'V':
                                infection_prob = self.beta * (1 - self.vaccine_efficacy)
                            else:
                                infection_prob = self.beta
                            if random.random() < infection_prob:
                                new_infections.append(other_agent.id)

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

    def get_state_counts(self):
        # Count the number of agents in each state
        counts = {'S': 0, 'I': 0, 'R': 0, 'V': 0}
        for agent in self.agents:
            counts[agent.state] += 1
        return counts

    def run(self):
        running = True
        while running:
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Update agent states
            self.step()

            # Move agents
            for agent in self.agents:
                agent.move()

            # Draw agents
            screen.fill((0, 0, 0))  # Clear screen
            for agent in self.agents:
                agent.draw()

            # Display state counts
            counts = self.get_state_counts()
            text_s = font.render(f"S: {counts['S']}", True, (255, 255, 255))
            text_i = font.render(f"I: {counts['I']}", True, (255, 255, 255))
            text_r = font.render(f"R: {counts['R']}", True, (255, 255, 255))
            text_v = font.render(f"V: {counts['V']}", True, (255, 255, 255))
            screen.blit(text_s, (10, 10))  # Top-left corner
            screen.blit(text_i, (10, 40))
            screen.blit(text_r, (10, 70))
            screen.blit(text_v, (10, 100))

            pygame.display.flip()

            # Control frame rate
            clock.tick(30)

        pygame.quit()

# Main
if __name__ == "__main__":
    # Parameters
    N = 100  # Number of agents
    beta = 0.4  # Infection rate
    gamma = 0.01  # Recovery rate
    vaccine_coverage = 0.1  # Fraction of population vaccinated
    vaccine_efficacy = 0.9  # Vaccine efficacy
    recovery_boost = 1.5  # Vaccine recovery boost

    # Run simulation
    sim = SIRSimulation(N=N, beta=beta, gamma=gamma, vaccine_coverage=vaccine_coverage, vaccine_efficacy=vaccine_efficacy, recovery_boost=recovery_boost)
    sim.run()
