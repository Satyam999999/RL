import streamlit as st
import time
import yaml
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys
import os

# Fix import path to allow importing 'src'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.agents.dqn_agent import DQNAgent

# 1. Page Config (Browser Title & Icon)
st.set_page_config(page_title="DeepOptions AI Trader", page_icon="🤖", layout="wide")

# 2. Load Model & Config (Cached so it doesn't reload every loop)
@st.cache_resource
def load_system():
    try:
        with open("experiments/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        agent = DQNAgent(state_dim=2, action_dim=2, config=config)
        model_path = "checkpoints/dqn_option.pth"
        
        if os.path.exists(model_path):
            agent.policy_net.load_state_dict(torch.load(model_path, map_location=agent.device))
            agent.policy_net.eval()
        else:
            st.error(f"❌ Model not found at {model_path}")
            return None, None
            
        return agent, config
    except Exception as e:
        st.error(f"Error loading system: {e}")
        return None, None

agent, config = load_system()

# 3. Sidebar Controls
st.sidebar.title("🎮 Control Panel")
simulation_speed = st.sidebar.slider("Simulation Speed (sec)", 0.1, 2.0, 0.5)
start_btn = st.sidebar.button("🚀 Start Live Trading", type="primary")
stop_btn = st.sidebar.button("🛑 Stop")

# 4. Main Dashboard Layout
st.title("🤖 DeepOptions AI: Real-Time Inference")
st.markdown("Watching the AI Agent price American Options in real-time.")

# Create placeholders for live updates
# Metrics Row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    price_placeholder = st.empty()
with kpi2:
    strike_placeholder = st.empty()
with kpi3:
    time_placeholder = st.empty()
with kpi4:
    signal_placeholder = st.empty()

# Chart Area
chart_placeholder = st.empty()

# 5. The Trading Loop
if start_btn:
    # Initialize session state for history if needed
    if 'history' not in st.session_state:
        st.session_state.history = []

    stop_signal = False
    
    while not stop_signal:
        # A. Simulate Data
        live_price = np.random.uniform(85, 105)
        live_time = np.random.uniform(0.01, 1.0)
        
        # B. AI Inference
        norm_price = live_price / config['simulation']['k']
        state = np.array([norm_price, live_time], dtype=np.float32)
        
        with torch.no_grad():
            st_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            q_values = agent.policy_net(st_tensor)
            action = q_values.argmax().item()
            confidence = q_values.max().item()

        # C. Update History for Chart
        st.session_state.history.append({
            "Time": pd.Timestamp.now(),
            "Price": live_price,
            "Action": "Exercise" if action == 1 else "Hold"
        })
        # Keep last 50 points
        df = pd.DataFrame(st.session_state.history[-50:])

        # D. Visual Updates
        
        # 1. Update Metrics
        price_placeholder.metric(label="Live Price", value=f"${live_price:.2f}", delta=f"{live_price - 100:.2f}")
        strike_placeholder.metric(label="Strike Price", value=f"${config['simulation']['k']:.2f}")
        time_placeholder.metric(label="Time to Maturity", value=f"{live_time:.2f} yr")
        
        # 2. Update Signal (The "Matrix" Effect)
        if action == 1:
            signal_placeholder.markdown(
                """<div style="background-color: #ff4b4b; padding: 10px; border-radius: 5px; text-align: center;">
                <h2 style="color: white; margin:0;">🔴 EXERCISE</h2>
                </div>""", unsafe_allow_html=True
            )
        else:
            signal_placeholder.markdown(
                """<div style="background-color: #2e86c1; padding: 10px; border-radius: 5px; text-align: center;">
                <h2 style="color: white; margin:0;">🔵 HOLD</h2>
                </div>""", unsafe_allow_html=True
            )

        # 3. Update Chart (Live Plotly)
        fig = go.Figure()
        
        # Price Line
        fig.add_trace(go.Scatter(x=df.index, y=df['Price'], mode='lines', name='Price', line=dict(color='gray')))
        
        # Exercise Points (Red Dots)
        exercise_points = df[df['Action'] == "Exercise"]
        if not exercise_points.empty:
            fig.add_trace(go.Scatter(
                x=exercise_points.index, 
                y=exercise_points['Price'], 
                mode='markers', 
                name='AI Exercise',
                marker=dict(color='red', size=12, symbol='x')
            ))
            
        fig.update_layout(
            title="Live Price Stream & AI Decisions",
            xaxis_title="Tick",
            yaxis_title="Price",
            height=400,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)

        # Loop Control
        time.sleep(simulation_speed)
        
        # Check if user clicked another button (Streamlit trick: restarts script on interaction)
        # Note: In a real loop, handling 'Stop' gracefully needs st.session_state logic, 
        # but for a demo, simply clicking "Stop" will reload the page and kill the loop.