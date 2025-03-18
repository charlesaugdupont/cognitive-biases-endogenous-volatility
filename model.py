import numpy as np

def probability_weighting(p, gamma):
    return (p**gamma) / ((p**gamma + (1-p)**gamma)**(1/gamma))

def cpt_value(x, theta, omega, eta):
    if x >= 0:
        return x**theta
    return -omega * (-x)**eta

def utility(w, h, alpha):
    return w**alpha * h**(1 - alpha)

def prob_health_decrease(w, h, N, scale):
    return  1 - scale*(w/N)*(h/N)

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
    health_decrease_scale
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
        "health_decrease_scale":health_decrease_scale,
    }
    
    while True:
        new_V = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                w, h = i+1, j+1 # w and h range from 1, 2, ..., to N inclusive
                reference_utility = utility(w, h, alpha)
                decrease_prob = prob_health_decrease(w, h, N, health_decrease_scale)

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
                    weighted_success = probability_weighting(P_H_increase, gamma) * cpt_delta_success
                    weighted_fail = probability_weighting(1-P_H_increase, gamma) * cpt_delta_fail

                    # calculate expected future value of investing
                    val_success = V[i-invest_cost][min(j+health_delta, N-1)]
                    val_fail = V[i-invest_cost][j]
                    expected_future_val_invest = probability_weighting(P_H_increase, gamma) * val_success + probability_weighting(1-P_H_increase, gamma) * val_fail

                # compute value of NOT investing
                utility_save_decrease = utility(w+1, max(h-health_delta, 1), alpha)
                utility_save_steady = utility(w+1, h, alpha)

                delta_util_decrease = utility_save_decrease - reference_utility
                delta_util_steady = utility_save_steady - reference_utility

                cpt_delta_decrease = cpt_value(delta_util_decrease, theta, omega, eta)
                cpt_delta_steady = cpt_value(delta_util_steady, theta, omega, eta)
                weighted_decrease = probability_weighting(decrease_prob, gamma) * cpt_delta_decrease
                weighted_steady = probability_weighting(1-decrease_prob, gamma) * cpt_delta_steady

                val_decrease = V[min(N-1, i+1)][max(1, j-health_delta)]
                val_steady = V[min(N-1, i+1)][j]
                expected_future_val_save = probability_weighting(decrease_prob, gamma) * val_decrease + probability_weighting(1-decrease_prob, gamma) * val_steady

                # compare invest vs. saving
                invest = weighted_success + weighted_fail + beta * expected_future_val_invest
                save = weighted_decrease + weighted_steady + beta * expected_future_val_save

                new_V[i][j] = max(invest, save)
                policy[i][j] = 1 if invest > save else 0

        norm = np.linalg.norm(new_V-V)
        if (len(norms) and norm > norms[-1]) or norm < 1e-6:
            break
        norms.append(norm)
        V = new_V

    return policy, parameters, V

def simulate(params, policy, num_steps, num_agents):
    N = params["N"]
    alpha = params["alpha"]
    health_delta = params["health_delta"]
    invest_cost = params["invest_cost"]
    P_H_increase = params["P_H_increase"]
    health_decrease_scale = params["health_decrease_scale"]

    util = np.zeros((num_agents, num_steps))
    wealth = np.zeros((num_agents, num_steps), dtype=np.uint8)
    health = np.zeros((num_agents, num_steps), dtype=np.uint8)
    rng = np.random.uniform(0, 1, size=(num_agents, num_steps-1))

    # Initialize wealth and health for all agents
    w_init = np.random.randint(1, N+1, size=num_agents)
    h_init = np.random.randint(1, N+1, size=num_agents)
    util[:, 0] = utility(w_init, h_init, alpha) / N
    wealth[:, 0] = w_init
    health[:, 0] = h_init

    for step in range(1, num_steps):
        w = wealth[:, step-1].astype(int)
        h = health[:, step-1].astype(int)
        action = policy[w-1, h-1]

        # Apply actions
        invest_mask = action == 1
        no_invest_mask = action == 0

        # Invest action
        h[invest_mask] = np.where(
            rng[invest_mask, step-1] < P_H_increase,
            np.minimum(h[invest_mask] + health_delta, N),
            h[invest_mask]
        )
        w[invest_mask] = np.maximum(w[invest_mask] - invest_cost, 1)

        # No invest action
        decrease_prob = prob_health_decrease(w[no_invest_mask], h[no_invest_mask], N, health_decrease_scale)
        h[no_invest_mask] = np.where(
            rng[no_invest_mask, step-1] < decrease_prob,
            np.maximum(h[no_invest_mask] - health_delta, 1),
            h[no_invest_mask]
        )
        w[no_invest_mask] = np.minimum(w[no_invest_mask] + 1, N)

        # Update utility, wealth, and health
        util[:, step] = utility(w, h, alpha) / N
        wealth[:, step] = w
        health[:, step] = h

    return util, wealth, health
