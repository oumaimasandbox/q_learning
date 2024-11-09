import numpy as np
import random

class GridWorld:
    def __init__(self, gamma=0.9):
         
        self.grid = np.array([
            [10, 0, 0,0,0],
            [0, 0, 0,-3,0],
            [0, 0, -7,0,0],
            [0,0,0,0,0],
            [0,0,0,0,-10]
        ])
        
        
        self.actions =  np.array([
            [10, 0, 0,0,0],
            [0, 0, 0,-3,0],
            [0, 0, -7,0,0],
            [0,0,0,0,0],              
            [0,0,0,0,-10]
        ])
        self.gamma = gamma  
         
        
        self.value_function =np.ze
    
    def step(self, state, action):
        
        y, x = state
        ay, ax = self.actions[action]
      
        new_y = max(0, min(y + ay, 2))
        new_x = max(0, min(x + ax, 2))
    
        return (new_y, new_x), self.grid[new_y, new_x]
    
    def is_final_state(self, state):
        
        y, x = state
        return self.grid[y, x] == 10 or self.grid[y, x] == -10

    def value_iteration(self, theta=1e-6):
        
        while True: 
            delta = 0
            new_value_function = np.copy(self.value_function)
 
            for y in range(self.grid.shape[0]):
                for x in range(self.grid.shape[1]):
                    state = (y, x)
                    
                    
                    if self.is_final_state(state):
                        continue

                    v_max = -float('inf') 
                    
                    
                    for action in range(len(self.actions)):
                        next_state, reward = self.step(state, action)
                        next_y, next_x = next_state
                        action_value = reward + self.gamma * self.value_function[next_y, next_x]
                        v_max = max(v_max, action_value)

        
                    new_value_function[y, x] = v_max
               
                    delta = max(delta, abs(self.value_function[y, x] - new_value_function[y, x]))
            
            self.value_function = new_value_function
            
        
            if delta < theta:
                break

         
        self.policy = np.zeros_like(self.grid, dtype=int)

        
        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                state = (y, x)

                if self.is_final_state(state):
                    continue

                v_max = -float('inf')
                best_action = None

                for action in range(len(self.actions)):
                    next_state, _ = self.step(state, action)
                    next_y, next_x = next_state

                    action_value = self.gamma * self.value_function[next_y, next_x]
                    if action_value > v_max:
                        v_max = action_value
                        best_action = action

                self.policy[y, x] = best_action
        
        print("Converged Value Function:")
        print(np.round(self.value_function, decimals=2))

        print("\nOptimal Policy:")
        policy_symbols = {0: '↑', 1: '↓', 2: '←', 3: '→'}
        policy_display = np.array([[policy_symbols[self.policy[y, x]] if not self.is_final_state((y, x)) else 'T'
                                    for x in range(self.grid.shape[1])]
                                    for y in range(self.grid.shape[0])])
        print(policy_display)
    

if __name__ == "__main__":
 
    grid_world = GridWorld()
     
    grid_world.value_iteration()
   
 