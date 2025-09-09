import numpy as np

def probability_weighting(p, gamma):
    return (p**gamma) / ((p**gamma + (1-p)**gamma)**(1/gamma))

def cpt_value(x, theta, omega, eta):
    if x >= 0:
        return x**theta
    return -omega * (-x)**eta

def compute_new_wealth(w, w_delta_scale, utility):
    delta = utility - w
    return w + w_delta_scale * delta

def compute_health_delta(h):
    k = np.log(10) / 150
    return (10 * np.exp(-k * h) + 1).astype(int)

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
    w_delta_scale_grid,
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
        "w_delta_scale_grid":w_delta_scale_grid,
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
                current_w_delta_scale = w_delta_scale_grid[j, i]
                reference_utility = utility(w, h, alpha)
                invest_cost = compute_health_cost(h)
                health_delta = compute_health_delta(h)

                # compute value of investing
                weighted_success = 0
                weighted_fail = 0
                expected_future_val_invest = 0
                if w > invest_cost:
                    w_ = w - invest_cost
                    new_wealth = int(compute_new_wealth(w_, current_w_delta_scale, utility(w_, h, alpha)))
                    if new_wealth > N:
                        print(w_, h, alpha, current_w_delta_scale)
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
                new_wealth = int(compute_new_wealth(w, current_w_delta_scale, reference_utility))
                if new_wealth > N:
                    print(w, h, alpha, current_w_delta_scale)
                utility_save_decrease = utility(new_wealth, max(h-health_delta, 1), alpha)
                utility_save_steady = utility(new_wealth, h, alpha)

                delta_util_decrease = utility_save_decrease - reference_utility
                delta_util_steady = utility_save_steady - reference_utility

                cpt_delta_decrease = cpt_value(delta_util_decrease, theta, omega, eta)
                cpt_delta_steady = cpt_value(delta_util_steady, theta, omega, eta)
                weighted_decrease = cpt_P_decrease * cpt_delta_decrease
                weighted_steady = cpt_P_decrease_complement * cpt_delta_steady

                val_decrease = V[new_wealth-1][max(1, j-health_delta)]
                val_steady = V[new_wealth-1][j]
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

def simulate(params, policy, num_steps, num_agents):
    N = params["N"]
    w_delta_scale_grid = params["w_delta_scale_grid"]
    P_H_increase = params["P_H_increase"]
    P_H_decrease = params["P_H_decrease"]

    wealth = np.zeros((num_agents, num_steps), dtype=np.uint8)
    health = np.zeros((num_agents, num_steps), dtype=np.uint8)
    rng = np.random.uniform(0, 1, size=(num_agents, num_steps-1))

    # Initialize wealth and health for all agents
    w_init = np.random.randint(1, N+1, size=num_agents)
    h_init = np.random.randint(1, N+1, size=num_agents)
    wealth[:, 0] = w_init
    health[:, 0] = h_init

    for step in range(1, num_steps):
        w = wealth[:, step-1].astype(int)
        h = health[:, step-1].astype(int)
        action = policy[w-1, h-1]
        invest_cost = compute_health_cost(h)
        health_delta = compute_health_delta(h)

        current_w_delta_scales = w_delta_scale_grid[h-1, w-1]

        # Apply actions
        invest_mask = (action == 1) & (w > invest_cost)
        no_invest_mask = (action == 0) | (w <= invest_cost)

        # Invest action
        w_after_cost = w[invest_mask] - invest_cost[invest_mask]
        updated_w = compute_new_wealth(
            w_after_cost,
            current_w_delta_scales[invest_mask], # Pass the specific scales
            utility(w_after_cost, h[invest_mask], params["alpha"])
        )
        w[invest_mask] = updated_w
        h[invest_mask] = np.where(
            rng[invest_mask, step-1] < P_H_increase,
            np.minimum(h[invest_mask] + health_delta[invest_mask], N),
            h[invest_mask]
        )

        # No invest action
        updated_w_no_invest = compute_new_wealth(
            w[no_invest_mask],
            current_w_delta_scales[no_invest_mask], # Pass the specific scales
            utility(w[no_invest_mask], h[no_invest_mask], params["alpha"])
        )
        w[no_invest_mask] = updated_w_no_invest
        h[no_invest_mask] = np.where(
            rng[no_invest_mask, step-1] < P_H_decrease,
            np.maximum(h[no_invest_mask] - health_delta[no_invest_mask], 1),
            h[no_invest_mask]
        )

        # w = np.minimum(w, 200)
        h = np.minimum(h, 200)

        # Update utility, wealth, and health
        wealth[:, step] = w
        health[:, step] = h

    return wealth, health
