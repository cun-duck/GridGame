import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# Constants
GRID_SIZE = 5
CELL_SIZE = 100
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

# Initialize Q-table
Q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))

# Initialize Model for Dyna-Q
model = {}

# Hyperparameters for Q-learning and Dyna-Q
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1  # exploration vs exploitation
PLANNING_STEPS = 5  # Number of planning steps in Dyna-Q

# Agent starting and goal position
agent_position = (0, 0)
goal_position = (4, 4)
obstacles = [(1, 1), (3, 2)]

# Function to get possible actions given the state
def get_possible_actions(state):
    actions = []
    x, y = state
    if x > 0: actions.append("UP")  # Can move up
    if x < GRID_SIZE - 1: actions.append("DOWN")  # Can move down
    if y > 0: actions.append("LEFT")  # Can move left
    if y < GRID_SIZE - 1: actions.append("RIGHT")  # Can move right
    return actions

# Function to select the best action based on Q-table (epsilon-greedy policy)
def get_max_action(state):
    x, y = state
    if random.uniform(0, 1) < EPSILON:
        return random.choice(ACTIONS)  # Exploration
    else:
        action_index = np.argmax(Q_table[x, y])
        return ACTIONS[action_index]  # Exploitation

# Function to update Q-values using Q-learning formula
def update_q(state, action, reward, next_state):
    x, y = state
    action_index = ACTIONS.index(action)
    next_x, next_y = next_state
    best_next_action = np.argmax(Q_table[next_x, next_y])
    Q_table[x, y, action_index] += ALPHA * (reward + GAMMA * Q_table[next_x, next_y, best_next_action] - Q_table[x, y, action_index])

# Function to perform Dyna-Q planning (simulate experiences)
def dyna_q_planning():
    for _ in range(PLANNING_STEPS):
        state = random.choice(list(model.keys()))
        action = random.choice(model[state].keys())
        next_state = model[state][action]["next_state"]
        reward = model[state][action]["reward"]
        
        # Update Q-table based on simulated experience
        update_q(state, action, reward, next_state)

# Function to move the agent
def move_agent(state, action):
    x, y = state
    if action == "UP":
        return (x-1, y)
    elif action == "DOWN":
        return (x+1, y)
    elif action == "LEFT":
        return (x, y-1)
    elif action == "RIGHT":
        return (x, y+1)

# Function to run an episode
def run_episode():
    global agent_position
    total_reward = 0
    agent_position = (0, 0)  # Reset agent position at start of each episode

    trajectory = [agent_position]  # Store agent's path for video visualization

    while agent_position != goal_position:
        # Choose an action
        action = get_max_action(agent_position)

        # Simulate agent's next state
        next_state = move_agent(agent_position, action)

        # Check if the next state is an obstacle or out of bounds
        if next_state in obstacles or next_state[0] < 0 or next_state[0] >= GRID_SIZE or next_state[1] < 0 or next_state[1] >= GRID_SIZE:
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

# Function to create the video animation from the agent's trajectory
def create_video(trajectory):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    agent_dot, = ax.plot([], [], 'bo', markersize=12)  # Agent position

    def update_frame(frame):
        ax.clear()
        ax.set_xlim(0, GRID_SIZE)
        ax.set_ylim(0, GRID_SIZE)
        agent_position = frame
        ax.plot(agent_position[1], agent_position[0], 'bo', markersize=12)  # Plot agent
        ax.plot(goal_position[1], goal_position[0], 'go', markersize=12)  # Plot goal
        for obs in obstacles:
            ax.plot(obs[1], obs[0], 'ro', markersize=12)  # Plot obstacles
        return agent_dot,

    # Create animation from the agent's trajectory
    ani = animation.FuncAnimation(fig, update_frame, frames=trajectory, interval=500, blit=True)

    # Save the animation as a video
    video_path = "agent_movement.mp4"
    ani.save(video_path, writer='ffmpeg', fps=1)

    return video_path

# Streamlit interface
st.title("Dyna-Q Agent Video Visualization")

# Input for number of episodes
episodes = st.slider("Number of Episodes", min_value=1, max_value=100, value=10)

# Button to start training
if st.button("Start Dyna-Q Training"):
    total_rewards = []
    all_trajectories = []

    # Run specified number of episodes
    for episode in range(episodes):
        st.write(f"Running Episode {episode + 1}...")
        total_reward, trajectory = run_episode()
        total_rewards.append(total_reward)
        all_trajectories.append(trajectory)
        st.write(f"Episode {episode + 1} completed. Total Reward: {total_reward}")

    # Display the average reward
    avg_reward = np.mean(total_rewards)
    st.write(f"Average Reward: {avg_reward}")

    # Create video for the last episode
    video_path = create_video(all_trajectories[-1])

    # Display the video
    st.write("Agent's Movement Video:")
    st.video(video_path)
