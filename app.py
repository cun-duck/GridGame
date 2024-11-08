import streamlit as st
import numpy as np
import random
import time
import cv2

# Parameter untuk GridWorld
GRID_SIZE = 5
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
GOAL_STATE = (GRID_SIZE - 1, GRID_SIZE - 1)
OBSTACLE_STATE = (2, 2)  # Contoh posisi halangan
Q_TABLE = {}
MODEL = {}

# Inisialisasi Q-table
def init_q_table():
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            Q_TABLE[(x, y)] = {action: 0 for action in ACTIONS}

# Fungsi untuk memilih aksi berdasarkan epsilon-greedy
def choose_action(state, epsilon=0.1):
    if random.uniform(0, 1) < epsilon:
        return random.choice(ACTIONS)
    else:
        q_values = Q_TABLE[state]
        return max(q_values, key=q_values.get)

# Fungsi untuk melakukan aksi dan mendapatkan reward
def take_action(state, action):
    x, y = state
    if action == 'UP' and x > 0:
        return (x - 1, y), -1  # Reward negatif untuk setiap langkah
    elif action == 'DOWN' and x < GRID_SIZE - 1:
        return (x + 1, y), -1
    elif action == 'LEFT' and y > 0:
        return (x, y - 1), -1
    elif action == 'RIGHT' and y < GRID_SIZE - 1:
        return (x, y + 1), -1
    return state, -1  # Tetap di tempat jika aksi tidak valid

# Fungsi untuk memperbarui model Dyna-Q
def update_model(state, action, next_state, reward):
    if state not in MODEL:
        MODEL[state] = {}
    if action not in MODEL[state]:
        MODEL[state][action] = {"next_state": next_state, "reward": reward}

# Dyna-Q Planning
def dyna_q_planning():
    for state in list(MODEL.keys()):
        for action in list(MODEL[state].keys()):
            next_state = MODEL[state][action]["next_state"]
            reward = MODEL[state][action]["reward"]
            q_values = Q_TABLE[state]
            q_values[action] += 0.1 * (reward + 0.9 * max(Q_TABLE[next_state].values()) - q_values[action])

# Fungsi untuk menjalankan satu episode pelatihan
def run_episode():
    state = (0, 0)  # Mulai dari pojok kiri atas
    total_reward = 0
    trajectory = []
    
    for step in range(100):  # Mengatur panjang langkah di satu episode
        action = choose_action(state)
        next_state, reward = take_action(state, action)
        total_reward += reward
        trajectory.append((state, action, reward))
        
        # Pembelajaran Dyna-Q
        update_model(state, action, next_state, reward)
        dyna_q_planning()
        
        state = next_state
        
        # Visualisasi (Real-time canvas)
        img = np.ones((400, 400, 3), dtype=np.uint8) * 255  # Latar belakang putih
        cv2.rectangle(img, (state[1]*80, state[0]*80), (state[1]*80 + 80, state[0]*80 + 80), (0, 0, 255), -1)  # Gambar agen
        cv2.rectangle(img, (GOAL_STATE[1]*80, GOAL_STATE[0]*80), (GOAL_STATE[1]*80 + 80, GOAL_STATE[0]*80 + 80), (0, 255, 0), -1)  # Goal
        cv2.putText(img, f'Step {step + 1}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Halangan
        if state == OBSTACLE_STATE:
            cv2.rectangle(img, (OBSTACLE_STATE[1]*80, OBSTACLE_STATE[0]*80), (OBSTACLE_STATE[1]*80 + 80, OBSTACLE_STATE[0]*80 + 80), (255, 0, 0), -1)
        
        # Display image as real-time video
        st.image(img, channels="BGR", use_column_width=True)
        time.sleep(0.1)  # Atur kecepatan visualisasi (lebih lambat untuk efek video)
    
    return total_reward, trajectory

# Fungsi utama Streamlit
def main():
    st.title('Pelatihan Agen AI dengan Dyna-Q')

    # Sidebar untuk pengaturan
    st.sidebar.header('Pengaturan')
    episodes = st.sidebar.slider('Jumlah Episode', min_value=1, max_value=100, value=50, step=1)
    start_button = st.sidebar.button('Mulai Pelatihan')

    # Inisialisasi Q-table
    init_q_table()

    # Menjalankan pelatihan jika tombol ditekan
    if start_button:
        st.sidebar.text("Pelatihan dimulai! Visualisasi akan ditampilkan secara langsung.")
        
        # Menjalankan pelatihan tanpa pemisahan per episode
        for episode in range(episodes):
            total_reward, trajectory = run_episode()
            st.sidebar.text(f"Total reward episode {episode + 1}: {total_reward}")
            time.sleep(0.5)  # Memberikan sedikit waktu antara episode

if __name__ == "__main__":
    main()
