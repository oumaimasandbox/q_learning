import numpy as np
import random
import matplotlib.pyplot as plt
import sys
sys.stdout.reconfigure(encoding='utf-8')

class GridWorld:
    def __init__(self, gamma=0.9, alpha=0.1, epsilon=0.1, episodes=1000):
        self.grid = np.array([
            [-1, -1, -1, 10],
            [-1, -5, -1, -10],
            [-1, -1, -1, -5],
            [-1, -1, -1, -1],
        ])
        
        self.actions = [
            (-1, 0),  # Up
            (1, 0),   # Down
            (0, -1),  # Left
            (0, 1)    # Right
        ]
        
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.episodes = episodes
        self.q_table = np.zeros((*self.grid.shape, len(self.actions)))
        self.rewards_per_episode = []

    def step(self, state, action):
        y, x = state
        ay, ax = self.actions[action]
        new_y = max(0, min(y + ay, self.grid.shape[0] - 1))
        new_x = max(0, min(x + ax, self.grid.shape[1] - 1))
        return (new_y, new_x), self.grid[new_y, new_x]

    def is_final_state(self, state):
        y, x = state
        return self.grid[y, x] == 10 or self.grid[y, x] == -10

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, len(self.actions) - 1)   
        else:
            y, x = state
            return np.argmax(self.q_table[y, x])  

    def sarsa(self):
        for episode in range(self.episodes):
            state = (random.randint(0, self.grid.shape[0] - 1), random.randint(0, self.grid.shape[1] - 1))
            action = self.choose_action(state)  # Initial action
            total_reward = 0  # Track the cumulative reward for this episode
            
            while not self.is_final_state(state):
                next_state, reward = self.step(state, action)
                y, x = state
                ny, nx = next_state

                # Choose the next action
                next_action = self.choose_action(next_state)
                
                # Update Q-value using SARSA formula
                td_target = reward + self.gamma * self.q_table[ny, nx, next_action]
                td_error = td_target - self.q_table[y, x, action]
                self.q_table[y, x, action] += self.alpha * td_error

                # Accumulate reward
                total_reward += reward
                state = next_state
                action = next_action  # Move to the next state-action pair

            # Store the total reward for this episode
            self.rewards_per_episode.append(total_reward)

    def get_policy(self):
        """Extract the policy from the Q-table with arrows."""
        arrow_mapping = {0: '↑', 1: '↓', 2: '←', 3: '→'}
        policy = np.full(self.grid.shape, '', dtype=str)
        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                best_action = np.argmax(self.q_table[y, x])
                policy[y, x] = arrow_mapping[best_action]
        return policy

    def display_q_values(self):
        """Display the Q-values for each state-action pair."""
        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                print(f"State ({y}, {x}): {self.q_table[y, x]}")

    def plot_rewards(self):
        """Plot the total reward per episode to show convergence."""
        plt.plot(self.rewards_per_episode)
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.title('Reward over Episodes (SARSA)')
        plt.show()

if __name__ == "__main__":
    grid_world = GridWorld()
    grid_world.sarsa()  # Use SARSA instead of Q-learning
    policy = grid_world.get_policy()
    
    print("Learned policy with arrows (SARSA):")
    for row in policy:
        print(" ".join(row))
    
    grid_world.display_q_values()
    grid_world.plot_rewards()  # Plot the reward curve for SARSA
