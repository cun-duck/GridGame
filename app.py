# app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# Sidebar untuk mengatur ukuran grid dan jumlah episode
st.sidebar.header("Grid Settings")
grid_size = st.sidebar.slider("Grid Size", min_value=5, max_value=10, value=5)
episodes = st.sidebar.slider("Number of Episodes", min_value=1, max_value=100, value=20)
use_dyna_q = st.sidebar.checkbox("Use Dyna-Q", True)

# Tempat menyimpan posisi halangan
obstacles = set()

# Fungsi untuk menampilkan grid interaktif dengan tombol
def display_interactive_grid(grid_size, obstacles):
    st.write("Klik pada sel untuk menambahkan atau menghapus halangan.")
    for i in range(grid_size):
        cols = st.columns(grid_size)  # Buat kolom sesuai grid_size
        for j in range(grid_size):
            # Tandai sel sebagai "Halangan" jika berada di dalam set obstacles
            label = "X" if (i, j) in obstacles else ""
            if cols[j].button(label, key=f"btn-{i}-{j}"):
                # Tambah atau hapus halangan tergantung status saat ini
                if (i, j) in obstacles:
                    obstacles.remove((i, j))  # Hapus halangan jika sudah ada
                else:
                    obstacles.add((i, j))  # Tambahkan halangan baru
    return obstacles

# Fungsi untuk memvisualisasikan grid dengan halangan dan posisi agen
def display_grid(agent_position, obstacles, goal_position):
    grid = np.zeros((grid_size, grid_size))
    grid[goal_position] = 0.5  # Mark the goal
    grid[agent_position] = 1   # Mark the agent

    for obs in obstacles:
        if 0 <= obs[0] < grid_size and 0 <= obs[1] < grid_size:
            grid[obs] = -1  # Mark obstacles

    fig, ax = plt.subplots()
    ax.imshow(grid, cmap="coolwarm", origin="upper")
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.grid(True)
    
    st.pyplot(fig)

# Fungsi untuk menggerakkan agen dengan mempertimbangkan halangan
def move_agent(agent_position, action):
    x, y = agent_position
    if action == 0 and x > 0: x -= 1  # Move up
    elif action == 1 and x < grid_size - 1: x += 1  # Move down
    elif action == 2 and y > 0: y -= 1  # Move left
    elif action == 3 and y < grid_size - 1: y += 1  # Move right

    # Check if the new position is an obstacle
    if (x, y) in obstacles:
        return agent_position  # Prevent moving into obstacle
    return (x, y)

# Fungsi untuk melatih agen menggunakan Dyna-Q
def train_agent(episodes, use_dyna_q):
    Q_table = np.zeros((grid_size, grid_size, 4))  # 4 actions (up, down, left, right)
    model = {}  # Model untuk menyimpan pengalaman dan transisi
    agent_position = (0, 0)
    goal_position = (grid_size - 1, grid_size - 1)

    alpha = 0.1  # Learning rate
    gamma = 0.9  # Discount factor
    epsilon = 1.0  # Exploration rate (starts high, decays over time)
    epsilon_decay = 0.995  # Decay factor for exploration rate
    n_planning_steps = 10  # Jumlah langkah perencanaan dalam Dyna-Q

    display_area = st.empty()  # Area untuk visualisasi pergerakan agen

    for ep in range(episodes):
        agent_position = (0, 0)  # Mulai dari posisi awal di setiap episode
        for step in range(100):  # Maksimal 100 langkah per episode
            # Tentukan aksi dengan strategi epsilon-greedy
            if np.random.rand() < epsilon:
                action = np.random.randint(0, 4)  # Jelajahi: Aksi acak
            else:
                action = np.argmax(Q_table[agent_position])  # Eksploitasi: Aksi terbaik

            # Lakukan aksi dan pindahkan agen
            new_position = move_agent(agent_position, action)

            # Perhitungan reward
            if new_position == agent_position:
                reward = -1  # Penalti untuk tabrakan dengan halangan
            elif new_position == goal_position:
                reward = 1  # Reward untuk mencapai tujuan
            else:
                reward = -0.1  # Penalti kecil untuk setiap langkah

            # Update Q-value berdasarkan pengalaman nyata
            old_q_value = Q_table[agent_position[0], agent_position[1], action]
            future_q_value = np.max(Q_table[new_position[0], new_position[1]])
            Q_table[agent_position[0], agent_position[1], action] = old_q_value + alpha * (reward + gamma * future_q_value - old_q_value)

            # Simpan transisi dalam model untuk pembelajaran imajiner
            if (agent_position, action) not in model:
                model[(agent_position, action)] = []
            model[(agent_position, action)].append((reward, new_position))

            # Pembelajaran imajiner (Dyna-Q)
            for _ in range(n_planning_steps):
                # Pilih transisi acak dari model dan lakukan pembaruan Q
                (s, a) = list(model.keys())[np.random.randint(0, len(model))]
                reward_sim, next_state_sim = model[(s, a)][np.random.randint(0, len(model[(s, a)]))]
                old_q_value_sim = Q_table[s[0], s[1], a]
                future_q_value_sim = np.max(Q_table[next_state_sim[0], next_state_sim[1]])
                Q_table[s[0], s[1], a] = old_q_value_sim + alpha * (reward_sim + gamma * future_q_value_sim - old_q_value_sim)

            # Update posisi agen
            agent_position = new_position

            # Visualisasikan grid dengan agen
            display_area.empty()  # Clear the previous frame
            display_grid(agent_position, obstacles, goal_position)
            time.sleep(0.2)  # Waktu jeda untuk animasi (frame per frame)

            # Hentikan jika mencapai tujuan
            if agent_position == goal_position:
                st.write(f"Goal reached in episode {ep + 1}!")
                break

        # Kurangi epsilon untuk mengurangi eksplorasi seiring waktu
        epsilon *= epsilon_decay

# Struktur aplikasi utama Streamlit
st.title("Interactive Grid World with Dyna-Q Agent")

# Sidebar controls
st.sidebar.header("Simulation Controls")

# Tampilkan grid interaktif dan ambil posisi halangan dari pengguna
obstacles = display_interactive_grid(grid_size, obstacles)

# Tombol untuk memulai pelatihan agen
if st.sidebar.button("Start Training"):
    train_agent(episodes, use_dyna_q)
