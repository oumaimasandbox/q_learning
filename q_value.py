import numpy as np
import random

class GridWorld:
    def __init__(self, gamma=0.9, alpha=0.1, epsilon=0.1, episodes=1000):
        self.grid = np.array([
            [10, 0, 0, 0, 0],
            [0, 0, 0, -3, 0],
            [0, 0, -7, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, -10]
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

    def q_learning(self):
        
     for episode in range(self.episodes):
            state = (random.randint(0, self.grid.shape[0] - 1), random.randint(0, self.grid.shape[1] - 1))
            while not self.is_final_state(state):
                action = self.choose_action(state)
                next_state, reward = self.step(state, action)
                y, x = state
                ny, nx = next_state

                
                best_next_action = np.argmax(self.q_table[ny, nx])
                td_target = reward + self.gamma * self.q_table[ny, nx, best_next_action]
                td_error = td_target - self.q_table[y, x, action]
                self.q_table[y, x, action] += self.alpha * td_error

                 
                state = next_state

    def get_policy(self):
        """Extract the policy from the Q-table."""
        policy = np.zeros((self.grid.shape[0], self.grid.shape[1]), dtype=int)
        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                policy[y, x] = np.argmax(self.q_table[y, x])
        return policy

    def display_q_values(self):
        """Display the Q-values for each state-action pair."""
        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                print(f"State ({y}, {x}): {self.q_table[y, x]}")

if __name__ == "__main__":
    grid_world = GridWorld()
    grid_world.q_learning()
    policy = grid_world.get_policy()
    print("Learned policy:")
    print(policy)
    grid_world.display_q_values()
