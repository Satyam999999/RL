import time
import yaml
import torch
import numpy as np
import sys
import os

# Fix import path so we can see 'src' from the deployment folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agents.dqn_agent import DQNAgent

def run_trading_bot():
    print("🤖 Booting up DeepOptions Trading Signal...")
    
    # 1. Load Config
    try:
        with open("experiments/config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("❌ Error: config.yaml not found. Run from root directory!")
        return

    # 2. Load Brain
    agent = DQNAgent(state_dim=2, action_dim=2, config=config)
    try:
        model_path = "checkpoints/dqn_option.pth"
        agent.policy_net.load_state_dict(torch.load(model_path, map_location=agent.device))
        agent.policy_net.eval()
        print("✅ AI Model Loaded.")
    except FileNotFoundError:
        print(f"❌ Error: Model not found at {model_path}")
        return

    # 3. Interactive Loop (Simulating a Live Feed)
    print("\n📡 Listening for Market Data... (Press Ctrl+C to stop)")
    print("-" * 65)
    print(f"{'PRICE':<10} | {'STRIKE':<10} | {'TIME LEFT':<10} | {'AI DECISION':<15}")
    print("-" * 65)

    try:
        while True:
            # SIMULATE LIVE DATA:
            # Prices fluctuate randomly around Strike ($100)
            live_price = np.random.uniform(85, 105) 
            live_time = np.random.uniform(0.01, 1.0) # Years
            
            # Prepare Data for AI (Normalize it!)
            norm_price = live_price / config['simulation']['k']
            norm_time = live_time 
            
            state = np.array([norm_price, norm_time], dtype=np.float32)
            
            # Ask AI
            with torch.no_grad():
                st_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                q_values = agent.policy_net(st_tensor)
                action = q_values.argmax().item()

            # Format Output
            # If Action is 1 (Exercise), make it RED and BOLD
            if action == 1:
                decision = "\033[91m🔴 EXERCISE\033[0m" # ANSI Color Code for Red
            else:
                decision = "\033[94m🔵 HOLD\033[0m"     # ANSI Color Code for Blue
            
            # Print Signal
            print(f"${live_price:<9.2f} | ${config['simulation']['k']:<9.2f} | {live_time:<4.2f} yr   | {decision}")
            
            # Wait 0.5 seconds before next "tick"
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n🛑 Bot shut down.")

if __name__ == "__main__":
    run_trading_bot()