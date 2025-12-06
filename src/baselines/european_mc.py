import numpy as np
from scipy.stats import norm

class EuropeanOptionPricing:
    def __init__(self, config):
        self.config = config
        self.r = config['simulation']['r']
        self.T = config['simulation']['t']
        self.K = config['simulation']['k']
        self.S0 = config['simulation']['s0']
        self.sigma = config['simulation']['sigma']
        
    def black_scholes_price(self):
        """
        Calculates the theoretical price using the exact Black-Scholes formula.
        Used as the 'Ground Truth'.
        """
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        # Formula for Put Option
        put_price = (self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)) - (self.S0 * norm.cdf(-d1))
        return put_price

    def monte_carlo_price(self, n_sims=None):
        """
        Estimates price by simulating N paths and averaging the payoff.
        """
        if n_sims is None:
            n_sims = self.config['simulation']['n_sims']
            
        print(f"📊 Simulating {n_sims} paths for validation...")
        
        # 1. Generate random Z for all paths at once (Vectorized = Fast)
        # We only need the FINAL price at T, not the whole path
        Z = np.random.normal(0, 1, n_sims)
        
        # ST = S0 * exp((r - 0.5*sigma^2)T + sigma*sqrt(T)*Z)
        drift = (self.r - 0.5 * self.sigma**2) * self.T
        diffusion = self.sigma * np.sqrt(self.T) * Z
        ST = self.S0 * np.exp(drift + diffusion)
        
        # 2. Calculate Payoff: Max(K - ST, 0)
        payoffs = np.maximum(self.K - ST, 0)
        
        # 3. Discount back to today: Payoff * exp(-rT)
        discounted_price = np.exp(-self.r * self.T) * np.mean(payoffs)
        
        return discounted_price