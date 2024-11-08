import random

class DynaQAgent:
    def __init__(self, grid_size, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.grid_size = grid_size
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = self._init_q_table()
        self.model = {}

    def _init_q_table(self):
        return { (x, y): {action: 0 for action in self.actions} 
                 for x in range(self.grid_size) for y in range(self.grid_size)}

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        else:
            return max(self.q_table[state], key=self.q_table[state].get)

    def update_model(self, state, action, next_state, reward):
        if state not in self.model:
            self.model[state] = {}
        self.model[state][action] = {"next_state": next_state, "reward": reward}

    def dyna_q_planning(self):
        for state in self.model:
            for action in self.model[state]:
                next_state = self.model[state][action]["next_state"]
                reward = self.model[state][action]["reward"]
                self.q_table[state][action] += self.alpha * (reward + self.gamma * max(self.q_table[next_state].values()) - self.q_table[state][action])

    def take_action(self, state, action):
        x, y = state
        if action == 'UP' and x > 0:
            return (x - 1, y), -1
        elif action == 'DOWN' and x < self.grid_size - 1:
            return (x + 1, y), -1
        elif action == 'LEFT' and y > 0:
            return (x, y - 1), -1
        elif action == 'RIGHT' and y < self.grid_size - 1:
            return (x, y + 1), -1
        return state, -1  # Tetap di tempat jika aksi tidak valid
