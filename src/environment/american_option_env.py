import gymnasium as gym
import numpy as np
from gymnasium import spaces

class AmericanOptionEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    The agent chooses to Exercise (1) or Hold (0) an American Put Option.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        super(AmericanOptionEnv, self).__init__()
        
        self.config = config
        self.K = config['simulation']['k'] # Strike Price
        
        # SIMULATION SETUP
        from src.environment.gbm import GBMGenerator
        self.simulator = GBMGenerator(config)
        self.price_path = None
        self.current_step = 0

        # ACTION SPACE: 0 (Hold) or 1 (Exercise)
        self.action_space = spaces.Discrete(2)

        # OBSERVATION SPACE: [Normalized Price, Normalized Time Remaining]
        # Price is normalized by Strike K (so 1.0 is ATM)
        # Time is normalized (0.0 to 1.0)
        high_bound = np.array([np.inf, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=high_bound, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Resets the environment to start a new episode."""
        super().reset(seed=seed)
        
        # Generate a new market scenario
        self.price_path = self.simulator.generate_path()
        self.current_step = 0
        
        # Return initial state
        return self._get_obs(), {}

    def step(self, action):
        """
        Executes the action.
        Reward = Discounted Payoff if exercised, else 0.
        """
        current_price = self.price_path[self.current_step]
        terminated = False
        truncated = False
        reward = 0.0

        # OPTION 1: AGENT EXERCISES
        if action == 1:
            reward = max(self.K - current_price, 0.0) # Put Payoff
            terminated = True # Episode ends immediately
        
        # OPTION 2: AGENT HOLDS
        else:
            self.current_step += 1
            
            # Check if we reached maturity (end of time)
            if self.current_step >= len(self.price_path) - 1:
                terminated = True
                # At maturity, you MUST exercise if profitable (rational behavior)
                final_price = self.price_path[self.current_step]
                reward = max(self.K - final_price, 0.0)
        
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        """Returns [Normalized Price, Normalized Time Remaining]"""
        price = self.price_path[self.current_step]
        
        # NORMALIZE PRICE: Divide by Strike (K)
        # S/K = 1.0 (At the money), 1.1 (ITM), 0.9 (OTM)
        norm_price = price / self.K 
        
        T_max = self.config['simulation']['t']
        dt = T_max / self.config['simulation']['n_steps']
        
        time_left = T_max - (self.current_step * dt)
        norm_time_left = time_left / T_max 
        
        return np.array([norm_price, norm_time_left], dtype=np.float32)