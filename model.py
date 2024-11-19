import numpy as np
import time

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
    time_start = time.time()
    V = np.zeros((N,N))
    policy = np.zeros((N,N))
    norms = []
    cpt_P_H_increase = probability_weighting(P_H_increase, gamma)

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
                w, h = i+1, j+1
                u = cpt_value(utility(w, h, alpha), theta, omega, eta)

                # option 1: invest in health
                invest_increase = 0
                invest_no_change = 0
                if w > invest_cost:
                    if h <= N - health_delta:
                        invest_increase = cpt_P_H_increase * cpt_value(V[i-invest_cost][j+health_delta], theta, omega, eta)
                        invest_no_change = probability_weighting(1-P_H_increase, gamma) * cpt_value(V[i-invest_cost][j], theta, omega, eta)
                    else:
                        invest_increase = cpt_P_H_increase * cpt_value(V[i-invest_cost][N-1], theta, omega, eta)
                        invest_no_change = probability_weighting(1-P_H_increase, gamma) * cpt_value(V[i-invest_cost][j], theta, omega, eta)
                invest_value = u + beta * (invest_increase + invest_no_change)

                # option 2: no investment
                decrease_prob = prob_health_decrease(w, h, N, health_decrease_scale)
                cpt_decrease_prob = probability_weighting(decrease_prob, gamma)
                if w < N:
                    if h > health_delta:
                        no_invest_decrease = cpt_decrease_prob * cpt_value(V[i+1][j-health_delta], theta, omega, eta)
                        no_invest_no_change = probability_weighting(1-decrease_prob, gamma) * cpt_value(V[i+1][j], theta, omega, eta)
                    else:
                        no_invest_decrease = cpt_decrease_prob * cpt_value(V[i+1][0], theta, omega, eta)
                        no_invest_no_change = probability_weighting(1-decrease_prob, gamma) * cpt_value(V[i+1][j], theta, omega, eta)
                else:
                    if h > health_delta:
                        no_invest_decrease = cpt_decrease_prob * cpt_value(V[i][j-health_delta], theta, omega, eta)
                        no_invest_no_change = probability_weighting(1-decrease_prob, gamma) * cpt_value(V[i][j], theta, omega, eta)
                    else:
                        no_invest_decrease = cpt_decrease_prob * cpt_value(V[i][0], theta, omega, eta)
                        no_invest_no_change = probability_weighting(1-decrease_prob, gamma) * cpt_value(V[i][j], theta, omega, eta)                   
                no_invest_value = u + beta * (no_invest_decrease + no_invest_no_change)

                # update V and policy
                new_V[i][j] = max(invest_value, no_invest_value)
                policy[i][j] = 1 if invest_value > no_invest_value else 0

        norm = np.linalg.norm(new_V-V)
        if (len(norms) and norm > norms[-1]) or norm < 1e-9:
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
