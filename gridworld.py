import numpy as np 
import random

class GridWorld:
    def __init__(self):
        self.grid = np.array([
            [0, 0, 10],
            [0, -1, 0],
            [0, 0, -10]])
        self.actions = np.array([
            [-1, 0],
            [1, 0],
            [0, -1],
            [0, 1]])
        self.x = random.randint(0, 2)
        self.y = random.randint(0, 2)

    def un_pas(self, action):
        # make sure to give the position of the agent 
        ay, ax = self.actions[action]
        # move the agent and check boundaries
        self.y = max(0, min(self.y + ay, 2))
        self.x = max(0, min(self.x + ax, 2))

        grid_value = self.grid[self.y, self.x]
        return self.y * 3 + self.x + 1, grid_value  # mapping

    def is_final_state(self):
        return self.grid[self.y, self.x] == 10 or self.grid[self.y, self.x] == -10

    def reinitialize(self):
        self.x = random.randint(0, 2)
        self.y = random.randint(0, 2)
        return self.y * 3 + self.x + 1


if __name__ == "__main__":
    # Create an instance of GridWorld
    grid_world = GridWorld()

    # Display the initial position of the agent
    print(f"Position initiale: {grid_world.y * 3 + grid_world.x + 1}")

    # Take an action (e.g., action 0)
    new_position, reward = grid_world.un_pas(0)
    print(f"Nouvelle position: {new_position}, Récompense: {reward}")

    # Check if the state is a final state
    if grid_world.is_final_state():
        print("État final atteint!")
    else:
        print("L'état n'est pas final.")

    # Reinitialize the agent's state
    position_reinitialisee = grid_world.reinitialize()
    print(f"Position réinitialisée: {position_reinitialisee}")