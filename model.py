import numpy as np

def probability_weighting(p, gamma):
    p = np.clip(p, 1e-10, 1 - 1e-10) # Avoid division by zero
    return (p**gamma) / ((p**gamma + (1-p)**gamma)**(1/gamma))

def cpt_value(x, theta, omega, eta):
    v = np.empty_like(x, dtype=float)
    pos_mask = x >= 0
    neg_mask = x < 0
    v[pos_mask] = x[pos_mask] ** theta
    v[neg_mask] = -omega * (-x[neg_mask]) ** eta
    return v

def compute_new_wealth(w, w_delta_scale, utility_value, N):
    delta = utility_value - w
    out = np.round(w + w_delta_scale * delta).astype(int)
    out = np.clip(out, 1, N)
    return out

def compute_health_delta(h):
    k = np.log(10) / 150
    return np.round(10 * np.exp(-k * h) + 1).astype(int)

def compute_health_cost(h):
    return - compute_health_delta(h) + 11

def utility(w, h, alpha, rate=1):
    return w**alpha * h**(rate - alpha)

def value_iteration_vectorized(
    N,
    alpha,
    gamma,
    theta,
    omega,
    eta,
    beta,
    P_H_increase,
    w_delta_scale,
    P_H_decrease
):
    V = np.zeros((N, N))
    policy = np.zeros((N, N), dtype=np.int16)
    norms = []

    parameters = {
        "N": N,
        "alpha": alpha,
        "gamma": gamma,
        "theta": theta,
        "omega": omega,
        "eta": eta,
        "beta": beta,
        "P_H_increase": P_H_increase,
        "w_delta_scale": w_delta_scale,
        "P_H_decrease": P_H_decrease
    }

    cpt_P_increase = probability_weighting(P_H_increase, gamma)
    cpt_P_increase_complement = probability_weighting(1 - P_H_increase, gamma)
    cpt_P_decrease = probability_weighting(P_H_decrease, gamma)
    cpt_P_decrease_complement = probability_weighting(1 - P_H_decrease, gamma)

    w_vals, h_vals = np.arange(1, N + 1), np.arange(1, N + 1)
    W, H = np.meshgrid(w_vals, h_vals, indexing='ij')

    reference_utility = utility(W, H, alpha)
    health_delta = compute_health_delta(H)
    invest_cost = compute_health_cost(H)

    invest_possible_mask = W > invest_cost

    norm = np.inf
    while norm > 1e-3:
        # --- Save Action Value Calculation ---
        new_wealth_save = compute_new_wealth(W, w_delta_scale, reference_utility, N)
        H_decrease = np.maximum(H - health_delta, 1)

        delta_util_decrease = utility(new_wealth_save, H_decrease, alpha) - reference_utility
        delta_util_steady = utility(new_wealth_save, H, alpha) - reference_utility
        cpt_delta_decrease = cpt_value(delta_util_decrease, theta, omega, eta)
        cpt_delta_steady = cpt_value(delta_util_steady, theta, omega, eta)
        immediate_cpt_save = cpt_P_decrease * cpt_delta_decrease + cpt_P_decrease_complement * cpt_delta_steady

        idx_w_save = np.clip(new_wealth_save - 1, 0, N - 1).astype(int)
        idx_h_decrease = np.clip(H_decrease - 1, 0, N - 1).astype(int)
        idx_h_steady = np.clip(H - 1, 0, N - 1).astype(int)
        val_decrease = V[idx_w_save, idx_h_decrease]
        val_steady = V[idx_w_save, idx_h_steady]
        expected_future_val_save = cpt_P_decrease * val_decrease + cpt_P_decrease_complement * val_steady
        save_value = immediate_cpt_save + beta * expected_future_val_save

        # --- Invest Action Value Calculation ---
        # Initialize invest_value to a very low number. For states where investment
        # is not possible, this ensures 'save' will be chosen.
        invest_value = np.full((N, N), -np.inf)

        # Perform investment calculations ONLY on the states where it's possible
        if np.any(invest_possible_mask):
            W_invest = W[invest_possible_mask]
            H_invest = H[invest_possible_mask]
            invest_cost_invest = invest_cost[invest_possible_mask]
            reference_utility_invest = reference_utility[invest_possible_mask]
            health_delta_invest = health_delta[invest_possible_mask]

            W_after_cost = W_invest - invest_cost_invest
            utility_after_cost = utility(W_after_cost, H_invest, alpha)
            new_wealth_invest = compute_new_wealth(W_after_cost, w_delta_scale, utility_after_cost, N)

            H_success = np.minimum(H_invest + health_delta_invest, N)
            delta_util_success = utility(new_wealth_invest, H_success, alpha) - reference_utility_invest
            delta_util_fail = utility(new_wealth_invest, H_invest, alpha) - reference_utility_invest
            cpt_delta_success = cpt_value(delta_util_success, theta, omega, eta)
            cpt_delta_fail = cpt_value(delta_util_fail, theta, omega, eta)
            immediate_cpt_invest = cpt_P_increase * cpt_delta_success + cpt_P_increase_complement * cpt_delta_fail

            idx_w_invest = np.clip(new_wealth_invest - 1, 0, N - 1).astype(int)
            idx_h_success = np.clip(H_success - 1, 0, N - 1).astype(int)
            idx_h_fail = np.clip(H_invest - 1, 0, N - 1).astype(int)
            val_success = V[idx_w_invest, idx_h_success]
            val_fail = V[idx_w_invest, idx_h_fail]
            expected_future_val_invest = cpt_P_increase * val_success + cpt_P_increase_complement * val_fail
            
            # Place the calculated values back into the main invest_value grid
            invest_value[invest_possible_mask] = immediate_cpt_invest + beta * expected_future_val_invest

        # --- Policy and Value Update (Vectorized) ---
        new_V = np.maximum(invest_value, save_value)
        policy = (invest_value > save_value).astype(np.int16)

        norm = np.linalg.norm(new_V - V)
        norms.append(norm)
        V = new_V

    return policy, parameters, V

def simulate(params, policy, num_steps, num_agents, initial_states):
    N = params["N"]
    w_delta_scale = params["w_delta_scale"]
    P_H_increase = params["P_H_increase"]
    P_H_decrease = params["P_H_decrease"]
    alpha = params["alpha"]

    wealth = np.zeros((num_agents, num_steps), dtype=np.int16)
    health = np.zeros((num_agents, num_steps), dtype=np.int16)
    rng = np.random.uniform(0, 1, size=(num_agents, num_steps-1))

    wealth[:,0] = initial_states[:,0]
    health[:,0] = initial_states[:,1]

    for step in range(1, num_steps):
        w = wealth[:, step-1].copy()
        h = health[:, step-1].copy()
        action = policy[w-1, h-1]
        invest_cost = compute_health_cost(h)
        health_delta = compute_health_delta(h)

        # Apply actions
        invest_mask = (action == 1) & (w > invest_cost)
        save_mask = ~invest_mask

        # Invest action
        if np.any(invest_mask):
            w_after_cost = w[invest_mask] - invest_cost[invest_mask]
            w[invest_mask] = compute_new_wealth(
                w_after_cost,
                w_delta_scale,
                utility(w_after_cost, h[invest_mask], alpha),
                N
            )
            h[invest_mask] = np.where(
                rng[invest_mask, step-1] < P_H_increase,
                h[invest_mask] + health_delta[invest_mask],
                h[invest_mask]
            )

        # Save action
        if np.any(save_mask):
            w[save_mask] = compute_new_wealth(
                w[save_mask],
                w_delta_scale,
                utility(w[save_mask], h[save_mask], alpha),
                N
            )
            h[save_mask] = np.where(
                rng[save_mask, step - 1] < P_H_decrease,
                h[save_mask] - health_delta[save_mask],
                h[save_mask]
            )

        wealth[:, step] = np.clip(w, 1, N)
        health[:, step] = np.clip(h, 1, N)

    assert np.all((wealth >= 1) & (wealth <= N)), f"Wealth out of bounds: min={wealth.min()}, max={wealth.max()}"
    assert np.all((health >= 1) & (health <= N)), f"Health out of bounds: min={health.min()}, max={health.max()}"
    return wealth, health
