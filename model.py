import numpy as np

# ==============================================================================
# Bilinear interpolation function
# ==============================================================================
def interpolate_value(w_coords, h_coords, V):
    """
    Performs vectorized bilinear interpolation to get values from the grid V.
    This allows us to estimate the value of a continuous state (w, h) that
    falls between the discrete points on the grid.

    Args:
        w_coords (np.ndarray): Array of continuous wealth coordinates (1-based).
        h_coords (np.ndarray): Array of continuous health coordinates (1-based).
        V (np.ndarray): The 2D value function grid.

    Returns:
        np.ndarray: An array of interpolated values.
    """
    N = V.shape[0]

    # Convert from 1-based model coordinates to 0-based array indices
    w_idx = w_coords - 1
    h_idx = h_coords - 1

    # Find the integer indices of the four corners surrounding each point
    w_floor = np.floor(w_idx).astype(int)
    h_floor = np.floor(h_idx).astype(int)
    w_ceil = np.ceil(w_idx).astype(int)
    h_ceil = np.ceil(h_idx).astype(int)

    # Clip indices to be within the bounds of the V array [0, N-1]
    w_floor = np.clip(w_floor, 0, N - 1)
    h_floor = np.clip(h_floor, 0, N - 1)
    w_ceil = np.clip(w_ceil, 0, N - 1)
    h_ceil = np.clip(h_ceil, 0, N - 1)

    # Calculate fractional distances from the floor corner
    w_frac = w_idx - w_floor
    h_frac = h_idx - h_floor

    # Get the values at the four corners from the V grid
    V00 = V[w_floor, h_floor]  # Value at (w_floor, h_floor)
    V10 = V[w_ceil, h_floor]   # Value at (w_ceil, h_floor)
    V01 = V[w_floor, h_ceil]   # Value at (w_floor, h_ceil)
    V11 = V[w_ceil, h_ceil]    # Value at (w_ceil, h_ceil)

    # Interpolate along the wealth (w) axis
    V_h_floor = (1 - w_frac) * V00 + w_frac * V10
    V_h_ceil  = (1 - w_frac) * V01 + w_frac * V11

    # Interpolate along the health (h) axis
    interpolated_val = (1 - h_frac) * V_h_floor + h_frac * V_h_ceil

    return interpolated_val

# ==============================================================================
# Original helper functions
# ==============================================================================
def probability_weighting(p, gamma):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return (p**gamma) / ((p**gamma + (1-p)**gamma)**(1/gamma))

def cpt_value(x, theta, lambduh, eta):
    v = np.empty_like(x, dtype=float)
    pos_mask = x >= 0
    neg_mask = x < 0
    v[pos_mask] = x[pos_mask] ** theta
    v[neg_mask] = -lambduh * (-x[neg_mask]) ** eta
    return v

def compute_health_cost(h, N, rate, Cmin=1, Cmax=10):
    h = (h - 1) / (N - 1)
    y = 1 - (1 - h) ** rate
    return Cmin + y * (Cmax - Cmin)

def compute_health_delta(h, N, rate, Cmin=1, Cmax=10):
    h = (h - 1) / (N - 1)
    y = (1 - h) ** rate
    return Cmin + y * (Cmax - Cmin)

def utility(w, h, alpha):
    return w**alpha * h**(1 - alpha)

# ==============================================================================
# Core model functions
# ==============================================================================
def wealth_growth_rate(u, N, scale=0.05):
    u_norm = (u - 1) / (N - 1)
    return scale * (2 * u_norm - 1)

def compute_new_wealth(w, utility_value, N):
    out = w * (1 + wealth_growth_rate(utility_value, N))
    out = np.clip(out, 1, N)
    return out

def value_iteration_vectorized(
    N,
    alpha,
    gamma,
    lambduh,
    eta,
    P_H_increase,
    P_H_decrease,
    rate,
    theta,
    beta
):
    """
    Perform value iteration with interpolation to compute optimal policy.
    """
    V = np.zeros((N, N))
    policy = np.zeros((N, N), dtype=np.int16)
    parameters = {
        "N": N,
        "alpha": alpha,
        "gamma": gamma,
        "lambda": lambduh,
        "eta": eta,
        "P_H_increase": P_H_increase,
        "P_H_decrease": P_H_decrease,
        "rate": rate,
        "theta": theta,
        "beta": beta,
    }

    cpt_P_increase = probability_weighting(P_H_increase, gamma)
    cpt_P_increase_complement = probability_weighting(1 - P_H_increase, gamma)
    cpt_P_decrease = probability_weighting(P_H_decrease, gamma)
    cpt_P_decrease_complement = probability_weighting(1 - P_H_decrease, gamma)

    w_vals, h_vals = np.arange(1, N + 1), np.arange(1, N + 1)
    W, H = np.meshgrid(w_vals, h_vals, indexing='ij')

    reference_utility = utility(W, H, alpha)
    health_delta = compute_health_delta(H, N, rate)
    invest_cost = compute_health_cost(H, N, rate)
    invest_possible_mask = W > invest_cost

    norm = np.inf
    while norm > 1e-3:
        # --- Save Action ---
        # Calculate the potential next states as continuous floats
        new_wealth_save_float = compute_new_wealth(W, reference_utility, N)
        H_decrease_float = np.maximum(H - health_delta, 1).astype(float)
        # Note: H_steady is just H

        delta_util_decrease = utility(new_wealth_save_float, H_decrease_float, alpha) - reference_utility
        delta_util_steady = utility(new_wealth_save_float, H, alpha) - reference_utility
        immediate_cpt_save = (cpt_P_decrease * cpt_value(delta_util_decrease, theta, lambduh, eta) +
                              cpt_P_decrease_complement * cpt_value(delta_util_steady, theta, lambduh, eta))

        # Use interpolation to get the expected future value
        val_decrease = interpolate_value(new_wealth_save_float.ravel(), H_decrease_float.ravel(), V).reshape(N, N)
        val_steady = interpolate_value(new_wealth_save_float.ravel(), H.ravel().astype(float), V).reshape(N, N)
        expected_future_val_save = cpt_P_decrease * val_decrease + cpt_P_decrease_complement * val_steady
        save_value = immediate_cpt_save + beta * expected_future_val_save

        # --- Invest Action ---
        invest_value = np.full((N, N), -np.inf)
        if np.any(invest_possible_mask):
            W_invest = W[invest_possible_mask]
            H_invest = H[invest_possible_mask]

            W_after_cost = W_invest - invest_cost[invest_possible_mask]
            utility_after_cost = utility(W_after_cost, H_invest, alpha)
            new_wealth_invest_float = compute_new_wealth(W_after_cost, utility_after_cost, N)
            H_success_float = np.minimum(H_invest + health_delta[invest_possible_mask], N).astype(float)

            delta_util_success = utility(new_wealth_invest_float, H_success_float, alpha) - reference_utility[invest_possible_mask]
            delta_util_fail = utility(new_wealth_invest_float, H_invest, alpha) - reference_utility[invest_possible_mask]
            immediate_cpt_invest = (cpt_P_increase * cpt_value(delta_util_success, theta, lambduh, eta) +
                                    cpt_P_increase_complement * cpt_value(delta_util_fail, theta, lambduh, eta))

            # Use interpolation for the masked subset of states
            val_success = interpolate_value(new_wealth_invest_float, H_success_float, V)
            val_fail = interpolate_value(new_wealth_invest_float, H_invest.astype(float), V)
            expected_future_val_invest = cpt_P_increase * val_success + cpt_P_increase_complement * val_fail
            invest_value[invest_possible_mask] = immediate_cpt_invest + beta * expected_future_val_invest

        # --- Policy and Value Update ---
        new_V = np.maximum(invest_value, save_value)
        policy = (invest_value > save_value).astype(np.int16)
        norm = np.linalg.norm(new_V - V)
        V = new_V

    return policy, parameters

def simulate(params, policy, num_steps, initial_states):
    """
    MODIFIED: Agent wealth and health are now continuous floats.
    The simulation evolves the agent's state in continuous space. To decide
    on an action, the agent finds the nearest grid point and uses the
    pre-computed optimal policy from that point.
    """
    N, P_H_increase, P_H_decrease, alpha, rate = params["N"], params["P_H_increase"], params["P_H_decrease"], params["alpha"], params["rate"]
    num_agents = initial_states.shape[0]

    # Agent state is now stored as float
    wealth = np.zeros((num_agents, num_steps), dtype=np.float32)
    health = np.zeros((num_agents, num_steps), dtype=np.float32)
    rng = np.random.uniform(0, 1, size=(num_agents, num_steps - 1))

    wealth[:, 0] = initial_states[:, 0]
    health[:, 0] = initial_states[:, 1]

    for step in range(1, num_steps):
        w = wealth[:, step - 1].copy()
        h = health[:, step - 1].copy()

        # Round the continuous state to the nearest grid index to look up the discrete policy
        w_idx = np.clip(np.round(w - 1), 0, N - 1).astype(int)
        h_idx = np.clip(np.round(h - 1), 0, N - 1).astype(int)
        action = policy[w_idx, h_idx]

        # State update calculations are done using the precise float state
        invest_cost = compute_health_cost(h, N, rate)
        health_delta = compute_health_delta(h, N, rate)
        invest_mask = (action == 1) & (w > invest_cost)
        save_mask = ~invest_mask

        # Invest action state changes
        if np.any(invest_mask):
            w_after_cost = w[invest_mask] - invest_cost[invest_mask]
            w[invest_mask] = compute_new_wealth(w_after_cost, utility(w_after_cost, h[invest_mask], alpha), N)
            h[invest_mask] = np.where(rng[invest_mask, step - 1] < P_H_increase, h[invest_mask] + health_delta[invest_mask], h[invest_mask])

        # Save action state changes
        if np.any(save_mask):
            w[save_mask] = compute_new_wealth(w[save_mask], utility(w[save_mask], h[save_mask], alpha), N)
            h[save_mask] = np.where(rng[save_mask, step - 1] < P_H_decrease, h[save_mask] - health_delta[save_mask], h[save_mask])

        wealth[:, step] = np.clip(w, 1, N)
        health[:, step] = np.clip(h, 1, N)

    assert np.all((wealth >= 1) & (wealth <= N)), f"Wealth out of bounds: {wealth.flatten().min()}, {wealth.flatten().max()}"
    assert np.all((health >= 1) & (health <= N)), "Health out of bounds"
    return wealth, health