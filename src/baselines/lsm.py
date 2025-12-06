import numpy as np

def longstaff_schwartz(price_paths, K, r, T):
    """
    Vectorized implementation of Longstaff-Schwartz (LSM).
    price_paths: (n_paths, n_steps) numpy array
    K: Strike Price
    r: Risk-free rate
    T: Maturity
    """
    n_paths, n_steps = price_paths.shape
    dt = T / (n_steps - 1)
    df = np.exp(-r * dt) # Discount factor per step

    # 1. Initialize Cashflow Matrix at Maturity (Last column)
    # Payoff = Max(K - S, 0)
    cashflows = np.maximum(K - price_paths[:, -1], 0)

    # 2. Backward Induction
    # Walk backwards from T-1 to 1
    for t in range(n_steps - 2, 0, -1):
        # Prices at current time step t
        S_t = price_paths[:, t]
        
        # Select paths where option is currently in-the-money (ITM)
        # If out-of-money, we never exercise, so ignore them for regression
        itm_mask = (K - S_t) > 0
        
        if np.sum(itm_mask) > 0:
            # X = Current Price (ITM paths only)
            X = S_t[itm_mask]
            # Y = Discounted Cashflow from next step (ITM paths only)
            Y = cashflows[itm_mask] * df
            
            # REGRESSION: Estimate "Continuation Value" (Expected Future Payoff)
            # We use a polynomial of degree 2 (Laguerre-like)
            # Regress Y (future money) against X (current price)
            coeffs = np.polyfit(X, Y, 2)
            continuation_value = np.polyval(coeffs, X)
            
            # EXERCISE DECISION
            exercise_value = K - X
            
            # If Exercise > Wait, we exercise
            exercise_decisions = exercise_value > continuation_value
            
            # Update cashflows
            # For paths where we exercise, cashflow = exercise_value
            # For paths where we wait, cashflow = discounted previous cashflow
            
            # First, discount EVERYONE
            cashflows = cashflows * df
            
            # Then overwrite the exercisers
            # We need to map back the ITM mask to the full array
            full_exercise_decision = np.zeros(n_paths, dtype=bool)
            full_exercise_decision[itm_mask] = exercise_decisions
            
            cashflows[full_exercise_decision] = exercise_value[exercise_decisions]
        else:
            cashflows *= df

    # Discount back to time 0
    return np.mean(cashflows * df)