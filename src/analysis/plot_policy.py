import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from src.agents.dqn_agent import DQNAgent

def plot_exercise_boundary():
    # 1. Load Config & Model
    with open("experiments/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    agent = DQNAgent(state_dim=2, action_dim=2, config=config)
    
    try:
        agent.policy_net.load_state_dict(torch.load("checkpoints/dqn_option.pth", map_location=agent.device))
        print("✅ Model loaded successfully.")
    except FileNotFoundError:
        print("❌ Model not found. Train the agent first!")
        return

    agent.policy_net.eval()

    # 2. Create Grid (Normalized!)
    # We scan Moneyness (S/K) from 0.8 to 1.2
    moneyness = np.linspace(0.8, 1.2, 100)
    times = np.linspace(0, 1, 100)

    decision_map = np.zeros((len(moneyness), len(times)))

    print("🧠 Scanning agent policy...")
    for i, m in enumerate(moneyness):
        for j, t in enumerate(times):
            # State = [Moneyness, Time_Remaining]
            # This matches exactly what the Agent sees in training
            state = np.array([m, t], dtype=np.float32)
            
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                q_values = agent.policy_net(state_t)
                action = q_values.argmax().item()
            
            decision_map[i, j] = action

    # 3. Plot
    plt.figure(figsize=(10, 6))
    
    # X-Axis: Time (1.0 = Start, 0.0 = End)
    # Y-Axis: Moneyness (0.8 = OTM, 1.2 = ITM)
    sns.heatmap(
        decision_map, 
        cmap="coolwarm", 
        cbar=False,
        xticklabels=10, 
        yticklabels=10
    )
    
    plt.gca().invert_yaxis() # Put high prices (1.2) at the top
    plt.gca().invert_xaxis() # Put Time 1.0 (Start) at the left
    
    plt.title("AI Exercise Boundary (Red = Exercise)")
    plt.xlabel("Time Remaining")
    plt.ylabel("Moneyness (Price / Strike)")
    
    plt.savefig("exercise_boundary_normalized.png")
    print("✅ Plot saved to 'exercise_boundary_normalized.png'")

if __name__ == "__main__":
    plot_exercise_boundary()