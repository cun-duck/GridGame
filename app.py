# app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# Set grid size and positions
grid_size = 5
goal_position = (4, 4)
start_position = (0, 0)

# Q-table for demonstration purposes
Q_table = np.zeros((grid_size, grid_size, 4))  # 4 actions (up, down, left, right)
action_mapping = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}

# Display function
def display_grid(agent_position, display_area):
    grid = np.zeros((grid_size, grid_size))
    grid[goal_position] = 0.5  # Goal marked in a different color
    grid[agent_position] = 1  # Agent marked in another color

    fig, ax = plt.subplots()
    ax.imshow(grid, cmap="coolwarm", origin="upper")
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.grid(True)
    
    display_area.pyplot(fig)

# Move agent based on action
def move_agent(agent_position, action):
    x, y = agent_position
    if action == 0 and x > 0: x -= 1          # Move up
    elif action == 1 and x < grid_size - 1: x += 1  # Move down
    elif action == 2 and y > 0: y -= 1        # Move left
    elif action == 3 and y < grid_size - 1: y += 1  # Move right
    return (x, y)

# Training function with Q-learning
def train_agent(episodes, use_qlearning, display_area):
    agent_position = start_position
    alpha = 0.1  # Learning rate
    gamma = 0.9  # Discount factor
    epsilon = 1.0  # Exploration rate (starts high, decays over time)
    epsilon_decay = 0.995  # Decay factor for exploration rate

    for _ in range(episodes):
        # Decide on action using epsilon-greedy strategy (balance exploration and exploitation)
        if np.random.rand() < epsilon:
            action = np.random.randint(0, 4)  # Explore: Random action
        else:
            action = np.argmax(Q_table[agent_position])  # Exploit: Best known action

        # Perform the action and move the agent
        new_position = move_agent(agent_position, action)

        # Check if the new position is out of bounds (penalty)
        if new_position[0] < 0 or new_position[0] >= grid_size or new_position[1] < 0 or new_position[1] >= grid_size:
            reward = -1  # Penalize for hitting the border
            new_position = agent_position  # Don't let agent move out of bounds
        elif new_position == goal_position:
            reward = 1  # Positive reward for reaching the goal
        else:
            reward = -0.1  # Small penalty for regular moves

        # Update Q-value using the Q-learning formula
        old_q_value = Q_table[agent_position[0], agent_position[1], action]
        future_q_value = np.max(Q_table[new_position[0], new_position[1]])  # Max Q-value for the new position
        Q_table[agent_position[0], agent_position[1], action] = old_q_value + alpha * (reward + gamma * future_q_value - old_q_value)

        # Update agent's position
        agent_position = new_position

        # Decrease epsilon over time to reduce exploration
        epsilon *= epsilon_decay

        # Display the agent's progress
        display_grid(agent_position, display_area)
        time.sleep(0.2)  # Short pause to simulate video-like animation

        # Stop if the agent reaches the goal
        if agent_position == goal_position:
            st.write("Goal reached!")
            break

# Streamlit app structure
st.title("Interactive AI Agent in Grid World")

# Sidebar controls for customization
st.sidebar.header("Simulation Controls")
episodes = st.sidebar.slider("Number of Episodes", 1, 100, 20)
use_qlearning = st.sidebar.checkbox("Use Q-learning", True)

# Manual control for agent movement
st.subheader("Manual Agent Control")
if st.button("Up"): start_position = move_agent(start_position, 0)
if st.button("Down"): start_position = move_agent(start_position, 1)
if st.button("Left"): start_position = move_agent(start_position, 2)
if st.button("Right"): start_position = move_agent(start_position, 3)

# Create a display area for the grid visualization
display_area = st.empty()
display_grid(start_position, display_area)

# Run training when user clicks the button
if st.sidebar.button("Start Training"):
    train_agent(episodes, use_qlearning, display_area)
