import numpy as np

def probability_weighting(p, gamma):
    return (p**gamma) / ((p**gamma + (1-p)**gamma)**(1/gamma))

def cpt_value(x, theta, omega, eta):
    if x >= 0:
        return x**theta
    return -omega * (-x)**eta

def wealth_change(h, k=0.05, x0=100.5, bound=3):
    return (2*bound / (1 + np.exp(-k * (h - x0))) - bound).round().astype(int)

def utility(w, h, alpha):
    return w**alpha * h**(1 - alpha)

def value_iteration(
    N,
    alpha,
    gamma,
    theta,
    omega,
    eta,
    beta,
    P_H_increase, 
    invest_cost,
    health_delta,
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
        "invest_cost":invest_cost,
        "health_delta":health_delta,
        "P_H_decrease":P_H_decrease,
    }

    cpt_P_increase = probability_weighting(P_H_increase, gamma)
    cpt_P_increase_complement = probability_weighting(1 - P_H_increase, gamma)
    cpt_P_decrease = probability_weighting(P_H_decrease, gamma)
    cpt_P_decrease_complement = probability_weighting(1 - P_H_decrease, gamma)
    
    while True:
        new_V = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                w, h = i+1, j+1 # w and h range from 1, 2, ..., to N inclusive
                reference_utility = utility(w, h, alpha)

                # compute value of investing
                weighted_success = 0
                weighted_fail = 0
                expected_future_val_invest = 0
                if w > invest_cost:
                    # calculate utility changes for investment case
                    util_invest_success = utility(w-invest_cost, min(h+health_delta, N), alpha)
                    util_invest_fail = utility(w-invest_cost, h, alpha)

                    # calculate change in utility
                    delta_util_success = util_invest_success - reference_utility
                    delta_util_fail = util_invest_fail - reference_utility

                    # apply CPT transformation
                    cpt_delta_success = cpt_value(delta_util_success, theta, omega, eta)
                    cpt_delta_fail = cpt_value(delta_util_fail, theta, omega, eta)
                    weighted_success = cpt_P_increase * cpt_delta_success
                    weighted_fail = cpt_P_increase_complement * cpt_delta_fail

                    # calculate expected future value of investing
                    val_success = V[i-invest_cost][min(j+health_delta, N-1)]
                    val_fail = V[i-invest_cost][j]
                    expected_future_val_invest = cpt_P_increase * val_success + cpt_P_increase_complement * val_fail

                # compute value of NOT investing
                wealth_delta = wealth_change(h)
                utility_save_decrease = utility(max(1, min(N, w+wealth_delta)), max(h-health_delta, 1), alpha)
                utility_save_steady = utility(max(1, min(N, w+wealth_delta)), h, alpha)

                delta_util_decrease = utility_save_decrease - reference_utility
                delta_util_steady = utility_save_steady - reference_utility

                cpt_delta_decrease = cpt_value(delta_util_decrease, theta, omega, eta)
                cpt_delta_steady = cpt_value(delta_util_steady, theta, omega, eta)
                weighted_decrease = cpt_P_decrease * cpt_delta_decrease
                weighted_steady = cpt_P_decrease_complement * cpt_delta_steady

                val_decrease = V[max(1, min(N-1, i+wealth_delta))][max(1, j-health_delta)]
                val_steady = V[max(1, min(N-1, i+wealth_delta))][j]
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
        if (len(norms) and norm > norms[-1]) or norm < 1e-3:
            break
        norms.append(norm)
        V = new_V

    return policy, parameters, V

def simulate(params, policy, num_steps, num_agents):
    N = params["N"]
    health_delta = params["health_delta"]
    invest_cost = params["invest_cost"]
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

        # Apply actions
        invest_mask = (action == 1) & (w > invest_cost)
        no_invest_mask = (action == 0) | (w <= invest_cost)

        # Invest action
        h[invest_mask] = np.where(
            rng[invest_mask, step-1] < P_H_increase,
            np.minimum(h[invest_mask] + health_delta, N),
            h[invest_mask]
        )
        w[invest_mask] = np.maximum(w[invest_mask] - invest_cost, 1)

        # No invest action
        w_delta = wealth_change(h)
        h[no_invest_mask] = np.where(
            rng[no_invest_mask, step-1] < P_H_decrease,
            np.maximum(h[no_invest_mask] - health_delta, 1),
            h[no_invest_mask]
        )
        w[no_invest_mask] = np.maximum(1, np.minimum(w[no_invest_mask] + w_delta[no_invest_mask], N))

        # Update utility, wealth, and health
        wealth[:, step] = w
        health[:, step] = h

    return wealth, health
