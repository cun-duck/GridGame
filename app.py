import streamlit as st
from agent import DynaQAgent
from environment import GridEnvironment
from video_visualizer import VideoVisualizer
from utils import wait

# Pengaturan Streamlit
st.sidebar.header('Pengaturan')
num_episodes = st.sidebar.slider('Jumlah Episode', min_value=1, max_value=100, value=10)
start_button = st.sidebar.button('Mulai Pelatihan')

# Inisialisasi agen dan lingkungan
grid_size = 5
actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
goal_state = (4, 4)
obstacle_state = (2, 2)

# Membuat objek-objek
agent = DynaQAgent(grid_size, actions)
environment = GridEnvironment(grid_size, goal_state, obstacle_state)
video_visualizer = VideoVisualizer()

def run_training():
    agent_position = (0, 0)  # Mulai dari pojok kiri atas
    trajectory = []
    
    # Memulai video visualizer
    video_visualizer.initialize_video_writer()

    for episode in range(num_episodes):
        state = agent_position
        for step in range(100):  # Batas langkah dalam satu episode
            action = agent.choose_action(state)
            next_state, reward = agent.take_action(state, action)
            
            # Pembelajaran Dyna-Q
            agent.update_model(state, action, next_state, reward)
            agent.dyna_q_planning()
            
            # Mengupdate posisi agen
            state = next_state
            agent_position = state

            # Render lingkungan
            frame = environment.render(agent_position)
            
            # Update frame ke video visualizer
            video_visualizer.update_frame(frame)
            
            # Tunggu sedikit untuk efek visual
            wait(0.1)

    # Menyelesaikan video setelah episode selesai
    video_visualizer.release()

def main():
    if start_button:
        st.sidebar.text("Pelatihan dimulai! Visualisasi akan ditampilkan secara langsung.")
        run_training()

if __name__ == "__main__":
    main()
