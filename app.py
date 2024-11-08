import random
import numpy as np
import streamlit as st
import time
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image

# Constants
GRID_SIZE = 5
PLANNING_STEPS = 50
GOAL_POSITION = (4, 4)
OBSTACLE_POSITIONS = [(1, 1), (2, 2), (3, 3)]  # Add obstacles here
ACTION_SPACE = ['up', 'down', 'left', 'right']

# Initialize Q-table and model
q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTION_SPACE)))
model = defaultdict(dict)

# Initialize agent position
agent_position = (0, 0)

# Helper functions
def get_max_action(state):
    """Return the action with the highest Q-value for a given state."""
    x, y = state
    return np.argmax(q_table[x, y])

def move_agent(position, action):
    """Return the next position based on the action."""
    x, y = position
    if action == 0:  # Up
        return max(x - 1, 0), y
    elif action == 1:  # Down
        return min(x + 1, GRID_SIZE - 1), y
    elif action == 2:  # Left
        return x, max(y - 1, 0)
    elif action == 3:  # Right
        return x, min(y + 1, GRID_SIZE - 1)

def update_q(state, action, reward, next_state):
    """Update Q-table using the Q-learning update rule."""
    x, y = state
    next_x, next_y = next_state
    alpha = 0.1  # Learning rate
    gamma = 0.9  # Discount factor
    max_future_q = np.max(q_table[next_x, next_y])
    q_table[x, y, action] = (1 - alpha) * q_table[x, y, action] + alpha * (reward + gamma * max_future_q)

def dyna_q_planning():
    """Perform Dyna-Q planning."""
    for _ in range(PLANNING_STEPS):
        state = random.choice(list(model.keys())) if model else None
        if state is None:
            continue

        if model[state]:
            action = random.choice(list(model[state].keys()))
            next_state = model[state][action]["next_state"]
            reward = model[state][action]["reward"]
            update_q(state, action, reward, next_state)

# Function to run the episode
def run_episode():
    global agent_position
    total_reward = 0
    agent_position = (0, 0)  # Reset agent position at start of each episode

    trajectory = [agent_position]  # Store agent's path for video visualization

    while agent_position != GOAL_POSITION:
        action = get_max_action(agent_position)  # Choose the action with the highest Q-value

        next_state = move_agent(agent_position, action)  # Simulate agent's next state

        # Check if the next state is an obstacle or out of bounds
        if next_state in OBSTACLE_POSITIONS or next_state[0] < 0 or next_state[0] >= GRID_SIZE or next_state[1] < 0 or next_state[1] >= GRID_SIZE:
            reward = -1  # Negative reward for hitting obstacle or boundary
            next_state = agent_position  # Don't move if it hits an obstacle
        else:
            reward = -0.1  # Small negative reward to encourage quicker completion

        # Update Q-table based on real experience
        update_q(agent_position, action, reward, next_state)

        # Save the transition in the model for planning
        if agent_position not in model:
            model[agent_position] = {}
        model[agent_position][action] = {
            "next_state": next_state,
            "reward": reward
        }

        # Perform Dyna-Q planning
        dyna_q_planning()

        # Move the agent
        agent_position = next_state
        trajectory.append(agent_position)
        total_reward += reward

    return total_reward, trajectory

# Function to create video
def create_video(trajectory, filename='agent_path.mp4'):
    """Generate a video of the agent's path."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec for mp4
    out = cv2.VideoWriter(filename, fourcc, 10.0, (400, 400))

    img = np.ones((400, 400, 3), dtype=np.uint8) * 255  # White canvas for each frame

    for position in trajectory:
        img = np.ones((400, 400, 3), dtype=np.uint8) * 255  # Reset to white
        x, y = position
        cv2.circle(img, (y * 80 + 40, x * 80 + 40), 20, (0, 0, 255), -1)  # Draw agent as red circle
        cv2.circle(img, (GOAL_POSITION[1] * 80 + 40, GOAL_POSITION[0] * 80 + 40), 20, (0, 255, 0), -1)  # Draw goal as green circle
        for obs in OBSTACLE_POSITIONS:
            cv2.rectangle(img, (obs[1] * 80, obs[0] * 80), (obs[1] * 80 + 80, obs[0] * 80 + 80), (0, 0, 0), -1)  # Draw obstacle
        out.write(img)  # Write frame to video

    out.release()  # Finalize video

# Streamlit UI
st.title("Dyna-Q Agent Training Visualization")

# Slider for number of episodes
episodes = st.slider("Number of Episodes", min_value=1, max_value=100, value=10, step=1)

# Checkbox for Q-learning
use_q_learning = st.checkbox("Use Q-learning", value=True)

# Start button
if st.button("Start Training"):
    st.text("Training the agent...")

    trajectory = []

    # Run the training process
    for _ in range(episodes):
        total_reward, episode_trajectory = run_episode()
        trajectory.extend(episode_trajectory)  # Append trajectory for video generation

    # Create video from trajectory
    create_video(trajectory)

    # Display video
    st.video('agent_path.mp4')
