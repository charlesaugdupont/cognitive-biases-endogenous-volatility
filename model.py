import numpy as np
from scipy.stats import qmc

def probability_weighting(p, gamma):
    p = np.clip(p, 1e-10, 1 - 1e-10)  # Avoid division by zero
    return (p**gamma) / ((p**gamma + (1-p)**gamma)**(1/gamma))

# def cpt_value(x, theta, omega, eta):
#     if x >= 0:
#         return x**theta
#     return -omega * (-x)**eta

def cpt_value(x, theta, omega, eta):
    return np.where(x >= 0, x**theta, -omega * (-x)**eta)

def compute_new_wealth(w, w_delta_scale, utility):
    delta = utility - w
    return np.round(w + w_delta_scale * delta).astype(int)

def compute_health_delta(h):
    k = np.log(10) / 150
    return np.round(10 * np.exp(-k * h) + 1).astype(int)

def compute_health_cost(h):
    return - compute_health_delta(h) + 11

def utility(w, h, alpha, rate=1):
    return w**alpha * h**(rate - alpha)

def value_iteration(
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
    V = np.zeros((N,N))
    policy = np.zeros((N,N))
    norms = []

    parameters = {
        "N":N,
        "alpha":alpha,
        "gamma":gamma,
        "theta":theta,
        "omega":omega,
        "eta":eta,
        "beta":beta,
        "P_H_increase":P_H_increase,
        "w_delta_scale":w_delta_scale,
        "P_H_decrease":P_H_decrease,
    }

    cpt_P_increase = probability_weighting(P_H_increase, gamma)
    cpt_P_increase_complement = probability_weighting(1 - P_H_increase, gamma)
    cpt_P_decrease = probability_weighting(P_H_decrease, gamma)
    cpt_P_decrease_complement = probability_weighting(1 - P_H_decrease, gamma)

    norm = np.inf
    while norm > 1e-3:
        new_V = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                w, h = i+1, j+1 # w and h range from 1, 2, ..., to N inclusive
                reference_utility = utility(w, h, alpha)
                invest_cost = compute_health_cost(h)
                health_delta = compute_health_delta(h)

                # compute value of investing
                weighted_success = 0
                weighted_fail = 0
                expected_future_val_invest = 0
                if w > invest_cost:
                    w_ = w - invest_cost
                    new_wealth = min(compute_new_wealth(w_, w_delta_scale, utility(w_, h, alpha)), N)
                    # calculate utility changes for investment case
                    util_invest_success = utility(new_wealth, min(h+health_delta, N), alpha)
                    util_invest_fail = utility(new_wealth, h, alpha)

                    # calculate change in utility
                    delta_util_success = util_invest_success - reference_utility
                    delta_util_fail = util_invest_fail - reference_utility

                    # apply CPT transformation
                    cpt_delta_success = cpt_value(delta_util_success, theta, omega, eta)
                    cpt_delta_fail = cpt_value(delta_util_fail, theta, omega, eta)
                    weighted_success = cpt_P_increase * cpt_delta_success
                    weighted_fail = cpt_P_increase_complement * cpt_delta_fail

                    # calculate expected future value of investing
                    val_success = V[new_wealth-1][min(j+health_delta, N-1)]
                    val_fail = V[new_wealth-1][j]
                    expected_future_val_invest = cpt_P_increase * val_success + cpt_P_increase_complement * val_fail

                # compute value of NOT investing
                new_wealth = min(compute_new_wealth(w, w_delta_scale, reference_utility), N)
                utility_save_decrease = utility(new_wealth, max(h-health_delta, 1), alpha)
                utility_save_steady = utility(new_wealth, h, alpha)

                delta_util_decrease = utility_save_decrease - reference_utility
                delta_util_steady = utility_save_steady - reference_utility

                cpt_delta_decrease = cpt_value(delta_util_decrease, theta, omega, eta)
                cpt_delta_steady = cpt_value(delta_util_steady, theta, omega, eta)
                weighted_decrease = cpt_P_decrease * cpt_delta_decrease
                weighted_steady = cpt_P_decrease_complement * cpt_delta_steady

                val_decrease = V[new_wealth-1][max(1, h-health_delta) - 1]
                val_steady = V[new_wealth-1][h-1]
                expected_future_val_save = cpt_P_decrease * val_decrease + cpt_P_decrease_complement * val_steady

                # compare invest vs. saving
                invest = weighted_success + weighted_fail + beta * expected_future_val_invest
                save = weighted_decrease + weighted_steady + beta * expected_future_val_save

                if w <= invest_cost:
                    new_V[i][j] = save
                    policy[i][j] = 0
                else:
                    new_V[i][j] = max(invest, save)
                    policy[i][j] = 1 if invest > save else 0

        norm = np.linalg.norm(new_V-V)
        norms.append(norm)
        V = new_V

    return policy, parameters, V

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
    policy = np.zeros((N, N), dtype=int)
    norms = []

    parameters = {
        "N": N, "alpha": alpha, "gamma": gamma, "theta": theta,
        "omega": omega, "eta": eta, "beta": beta, "P_H_increase": P_H_increase,
        "w_delta_scale": w_delta_scale, "P_H_decrease": P_H_decrease,
    }

    cpt_P_increase = probability_weighting(P_H_increase, gamma)
    cpt_P_increase_complement = probability_weighting(1 - P_H_increase, gamma)
    cpt_P_decrease = probability_weighting(P_H_decrease, gamma)
    cpt_P_decrease_complement = probability_weighting(1 - P_H_decrease, gamma)

    w_vals, h_vals = np.arange(1, N + 1), np.arange(1, N + 1)
    W, H = np.meshgrid(w_vals, h_vals, indexing='ij')

    Reference_utility = utility(W, H, alpha)
    Health_delta = compute_health_delta(H)
    Invest_cost = compute_health_cost(H)

    # This mask is key to preventing calculations on invalid states
    invest_possible_mask = W > Invest_cost

    norm = np.inf
    while norm > 1e-3:
        # --- Save Action Value Calculation (Vectorized) ---
        # This is safe to compute for all states
        new_wealth_save = np.minimum(compute_new_wealth(W, w_delta_scale, Reference_utility), N)
        H_decrease = np.maximum(H - Health_delta, 1)

        delta_util_decrease = utility(new_wealth_save, H_decrease, alpha) - Reference_utility
        delta_util_steady = utility(new_wealth_save, H, alpha) - Reference_utility
        cpt_delta_decrease = cpt_value(delta_util_decrease, theta, omega, eta)
        cpt_delta_steady = cpt_value(delta_util_steady, theta, omega, eta)
        immediate_cpt_save = (cpt_P_decrease * cpt_delta_decrease +
                              cpt_P_decrease_complement * cpt_delta_steady)

        idx_w_save = np.clip(new_wealth_save - 1, 0, N - 1).astype(int)
        idx_h_decrease = np.clip(H_decrease - 1, 0, N - 1).astype(int)
        idx_h_steady = np.clip(H - 1, 0, N - 1).astype(int)
        val_decrease = V[idx_w_save, idx_h_decrease]
        val_steady = V[idx_w_save, idx_h_steady]
        expected_future_val_save = (cpt_P_decrease * val_decrease +
                                    cpt_P_decrease_complement * val_steady)
        save_value = immediate_cpt_save + beta * expected_future_val_save

        # --- Invest Action Value Calculation (Masked and Vectorized) ---
        # Initialize invest_value to a very low number. For states where investment
        # is not possible, this ensures 'save' will be chosen.
        invest_value = np.full((N, N), -np.inf)

        # Now, perform investment calculations ONLY on the states where it's possible
        if np.any(invest_possible_mask):
            W_invest = W[invest_possible_mask]
            H_invest = H[invest_possible_mask]
            Invest_cost_invest = Invest_cost[invest_possible_mask]
            Reference_utility_invest = Reference_utility[invest_possible_mask]
            Health_delta_invest = Health_delta[invest_possible_mask]

            W_after_cost = W_invest - Invest_cost_invest
            New_wealth_invest = np.minimum(compute_new_wealth(
                W_after_cost, w_delta_scale, utility(W_after_cost, H_invest, alpha)
            ), N)

            H_success = np.minimum(H_invest + Health_delta_invest, N)
            delta_util_success = utility(New_wealth_invest, H_success, alpha) - Reference_utility_invest
            delta_util_fail = utility(New_wealth_invest, H_invest, alpha) - Reference_utility_invest
            cpt_delta_success = cpt_value(delta_util_success, theta, omega, eta)
            cpt_delta_fail = cpt_value(delta_util_fail, theta, omega, eta)
            immediate_cpt_invest = (cpt_P_increase * cpt_delta_success +
                                    cpt_P_increase_complement * cpt_delta_fail)

            idx_w_invest = np.clip(New_wealth_invest - 1, 0, N - 1).astype(int)
            idx_h_success = np.clip(H_success - 1, 0, N - 1).astype(int)
            idx_h_fail = np.clip(H_invest - 1, 0, N - 1).astype(int)
            val_success = V[idx_w_invest, idx_h_success]
            val_fail = V[idx_w_invest, idx_h_fail]
            expected_future_val_invest = (cpt_P_increase * val_success +
                                          cpt_P_increase_complement * val_fail)
            
            # Place the calculated values back into the main `invest_value` grid
            invest_value[invest_possible_mask] = immediate_cpt_invest + beta * expected_future_val_invest

        # --- Policy and Value Update (Vectorized) ---
        new_V = np.maximum(invest_value, save_value)
        policy = (invest_value > save_value).astype(int)

        norm = np.linalg.norm(new_V - V)
        norms.append(norm)
        V = new_V

    return policy, parameters, V

def simulate(params, policy, num_steps, num_agents):
    N = params["N"]
    w_delta_scale = params["w_delta_scale"]
    P_H_increase = params["P_H_increase"]
    P_H_decrease = params["P_H_decrease"]

    wealth = np.zeros((num_agents, num_steps), dtype=np.uint8)
    health = np.zeros((num_agents, num_steps), dtype=np.uint8)
    rng = np.random.uniform(0, 1, size=(num_agents, num_steps-1))

    # Initialize wealth and health for all agents using LHS
    l_bounds = [1,1]
    u_bounds = [N,N]
    sampler = qmc.LatinHypercube(d=2)
    sample = sampler.random(n=num_agents)

    # Scale the sample from [0, 1) to our desired integer range [1, N].
    scaled_sample = qmc.scale(sample, l_bounds, u_bounds)

    # Convert the scaled floating-point values to integers.
    initial_states = np.round(scaled_sample)
    wealth[:, 0] = initial_states[:, 0]
    health[:, 0] = initial_states[:, 1]

    for step in range(1, num_steps):
        w = wealth[:, step-1]
        h = health[:, step-1]
        action = policy[w-1, h-1]
        invest_cost = compute_health_cost(h)
        health_delta = compute_health_delta(h)

        # Apply actions
        invest_mask = (action == 1) & (w > invest_cost)
        no_invest_mask = (action == 0) | (w <= invest_cost)

        # Invest action
        w[invest_mask] = compute_new_wealth(
            w[invest_mask] - invest_cost[invest_mask],
            w_delta_scale,
            utility(w[invest_mask] - invest_cost[invest_mask], h[invest_mask], params["alpha"])
        )
        h[invest_mask] = np.where(
            rng[invest_mask, step-1] < P_H_increase,
            np.minimum(h[invest_mask] + health_delta[invest_mask], N),
            h[invest_mask]
        )

        # No invest action
        w[no_invest_mask] = compute_new_wealth(
            w[no_invest_mask],
            w_delta_scale,
            utility(w[no_invest_mask], h[no_invest_mask], params["alpha"])
        )
        h[no_invest_mask] = np.where(
            rng[no_invest_mask, step-1] < P_H_decrease,
            np.maximum(h[no_invest_mask] - health_delta[no_invest_mask], 1),
            h[no_invest_mask]
        )

        # Update utility, wealth, and health
        wealth[:, step] = np.minimum(w, N)
        health[:, step] = np.minimum(h, N)

    return wealth, health
