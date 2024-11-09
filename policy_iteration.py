import numpy as np

class GridWorld:
    def __init__(self, gamma=0.9, theta=1e-6):
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
        self.theta = theta
        self.value_function = np.zeros_like(self.grid, dtype=float)

        # Initialize policy to the custom stochastic policy
        self.policy = np.zeros((self.grid.shape[0], self.grid.shape[1], len(self.actions)))
        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                if not self.is_final_state((y, x)):
                    self.policy[y, x] = [0.7 if action == 0 else 0.1 for action in range(len(self.actions))]

    def step(self, state, action):
        y, x = state
        ay, ax = self.actions[action]
        new_y = max(0, min(y + ay, self.grid.shape[0] - 1))
        new_x = max(0, min(x + ax, self.grid.shape[1] - 1))
        return (new_y, new_x), self.grid[new_y, new_x]

    def is_final_state(self, state):
        y, x = state
        return self.grid[y, x] == 10 or self.grid[y, x] == -10

    def policy_evaluation(self):
        """Evaluate the value function under the current stochastic policy."""
        while True:
            delta = 0
            new_value_function = np.copy(self.value_function)
            
            for y in range(self.grid.shape[0]):
                for x in range(self.grid.shape[1]):
                    state = (y, x)
                    if self.is_final_state(state):
                        continue
                    
                    v = 0
                    for action, action_prob in enumerate(self.policy[y, x]):
                        next_state, reward = self.step(state, action)
                        next_y, next_x = next_state
                        v += action_prob * (reward + self.gamma * self.value_function[next_y, next_x])
                    
                    new_value_function[y, x] = v
                    delta = max(delta, abs(self.value_function[y, x] - v))
            
            self.value_function = new_value_function
            if delta < self.theta:
                break

    def policy_improvement(self):
        """Improve the current policy using the custom stochastic rule."""
        policy_stable = True

        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                state = (y, x)
                if self.is_final_state(state):
                    continue

                old_action = np.argmax(self.policy[y, x])

               
                action_values = np.zeros(len(self.actions))
                for action in range(len(self.actions)):
                    next_state, reward = self.step(state, action)
                    next_y, next_x = next_state
                    action_values[action] = reward + self.gamma * self.value_function[next_y, next_x]

                
                self.policy[y, x] = [0.7 if action == 0 else 0.1 for action in range(len(self.actions))]

                best_action = np.argmax(self.policy[y, x])
                if old_action != best_action:
                    policy_stable = False

        return policy_stable

    def policy_iteration(self):
        """Perform policy iteration using stochastic policy improvement."""
        while True:
            self.policy_evaluation()
            if self.policy_improvement():
                break

        print("Converged Stochastic Value Function:")
        print(np.round(self.value_function, decimals=2))  # This is the stochastic value function

        print("\nStochastic Policy :")
        policy_symbols = {0: '↑', 1: '↓', 2: '←', 3: '→'}
        policy_display = np.array([[policy_symbols[np.argmax(self.policy[y, x])] if not self.is_final_state((y, x)) else 'T'
                                    for x in range(self.grid.shape[1])]
                                    for y in range(self.grid.shape[0])])
        print(policy_display)


if __name__ == "__main__":
    grid_world = GridWorld()
    grid_world.policy_iteration()