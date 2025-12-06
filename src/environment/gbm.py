import numpy as np

class GBMGenerator:
    """
    Simulates Geometric Brownian Motion (GBM) price paths.
    Formula: S_t = S_{t-1} * exp((r - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)
    """
    def __init__(self, config):
        self.s0 = config['simulation']['s0']       # Initial Price
        self.mu = config['simulation']['r']        # Drift (Risk-free rate)
        self.sigma = config['simulation']['sigma'] # Volatility
        self.T = config['simulation']['t']         # Time to maturity (years)
        self.steps = config['simulation']['n_steps'] # Number of steps (days)
        self.dt = self.T / self.steps

    def generate_path(self):
        """
        Generates a single price path.
        Returns: numpy array of shape (steps + 1,)
        """
        # Brownian Motion (random noise)
        Z = np.random.normal(0, 1, self.steps)
        
        # Calculate log returns
        drift = (self.mu - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt) * Z
        
        # Accumulate returns to get Price Path
        log_returns = np.concatenate([[0], drift + diffusion])
        cumulative_returns = np.cumsum(log_returns)
        
        # Convert back to Price
        price_path = self.s0 * np.exp(cumulative_returns)
        
        return price_path