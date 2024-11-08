# app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Set grid size and positions
grid_size = 5
goal_position = (4, 4)
start_position = (0, 0)

# Q-table for demonstration purposes
Q_table = np.zeros((grid_size, grid_size, 4))  # 4 actions (up, down, left, right)
action_mapping = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}

# Display function
def display_grid(agent_position):
    grid = np.zeros((grid_size, grid_size))
    grid[goal_position] = 0.5  # Goal marked in a different color
    grid[agent_position] = 1  # Agent marked in another color

    plt.imshow(grid, cmap="coolwarm", origin="upper")
    plt.grid(True)
    plt.xticks(range(grid_size))
    plt.yticks(range(grid_size))
    st.pyplot(plt)

# Move agent based on action
def move_agent(agent_position, action):
    x, y = agent_position
    if action == 0 and x > 0: x -= 1          # Move up
    elif action == 1 and x < grid_size - 1: x += 1  # Move down
    elif action == 2 and y > 0: y -= 1        # Move left
    elif action == 3 and y < grid_size - 1: y += 1  # Move right
    return (x, y)

# Training function with Q-learning
def train_agent(episodes, use_qlearning):
    agent_position = start_position
    for episode in range(episodes):
        st.write(f"Episode: {episode + 1}")
        display_grid(agent_position)

        if use_qlearning:
            action = np.argmax(Q_table[agent_position])  # Choose best action
        else:
            action = np.random.randint(0, 4)  # Random action for exploration

        agent_position = move_agent(agent_position, action)

        # Update Q-table (simple rule for demonstration)
        reward = 1 if agent_position == goal_position else -0.1
        Q_table[agent_position][action] += 0.1 * (reward + np.max(Q_table[agent_position]) - Q_table[agent_position][action])

        # Check if goal is reached
        if agent_position == goal_position:
            st.write("Goal reached!")
            break
        st.sleep(0.3)  # Pause for visualization

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
display_grid(start_position)

# Run training when user clicks the button
if st.sidebar.button("Start Training"):
    train_agent(episodes, use_qlearning)
