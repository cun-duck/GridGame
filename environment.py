import numpy as np

class GridEnvironment:
    def __init__(self, grid_size=5, goal_state=(4, 4), obstacle_state=(2, 2)):
        self.grid_size = grid_size
        self.goal_state = goal_state
        self.obstacle_state = obstacle_state

    def render(self, agent_position):
        img = np.ones((400, 400, 3), dtype=np.uint8) * 255  # Latar belakang putih
        # Gambar agen
        cv2.rectangle(img, (agent_position[1] * 80, agent_position[0] * 80), 
                      (agent_position[1] * 80 + 80, agent_position[0] * 80 + 80), 
                      (0, 0, 255), -1)
        # Gambar goal
        cv2.rectangle(img, (self.goal_state[1] * 80, self.goal_state[0] * 80), 
                      (self.goal_state[1] * 80 + 80, self.goal_state[0] * 80 + 80), 
                      (0, 255, 0), -1)
        # Gambar halangan
        if agent_position == self.obstacle_state:
            cv2.rectangle(img, (self.obstacle_state[1] * 80, self.obstacle_state[0] * 80),
                          (self.obstacle_state[1] * 80 + 80, self.obstacle_state[0] * 80 + 80), 
                          (255, 0, 0), -1)
        return img
