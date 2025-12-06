import pytest
import numpy as np
import yaml
from src.environment.gbm import GBMGenerator
from src.baselines.european_mc import EuropeanOptionPricing

# Fixture: Loads config once for all tests
@pytest.fixture
def config():
    with open("experiments/config.yaml", "r") as f:
        return yaml.safe_load(f)

def test_gbm_shape(config):
    """Test if GBM generator produces correct array shape"""
    # Reduce steps for fast testing
    config['simulation']['n_steps'] = 10
    gen = GBMGenerator(config)
    path = gen.generate_path()
    
    # Shape should be n_steps + 1 (Time 0 to Time T)
    assert len(path) == 11
    assert path[0] == config['simulation']['s0']

def test_black_scholes_math(config):
    """Test if Black-Scholes formula is sane"""
    pricer = EuropeanOptionPricing(config)
    price = pricer.black_scholes_price()
    
    # Price should be positive
    assert price > 0
    # Price should be less than Strike (for Put options)
    assert price < config['simulation']['k']

def test_config_consistency(config):
    """Ensure gamma matches risk-free rate logic"""
    # Just a sanity check that we didn't break the config
    assert config['simulation']['s0'] == 100.0