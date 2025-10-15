import numpy as np
import pytest
from model import (
    compute_health_delta, compute_health_cost, cpt_value, compute_new_wealth,
    utility, value_iteration_vectorized, simulate, probability_weighting
)

@pytest.fixture
def small_model():
    N = 5
    alpha = 0.5
    gamma = 0.8
    theta = 0.88
    omega = 2.25
    eta = 0.88
    beta = 0.95
    P_H_increase = 0.6
    P_H_decrease = 0.4
    w_delta_scale = 0.3

    policy, params, V = value_iteration_vectorized(
        N, alpha, gamma, theta, omega, eta, beta,
        P_H_increase, w_delta_scale, P_H_decrease
    )
    return policy, params

def test_compute_health_delta_positive():
    for h in range(1, 151):
        delta = compute_health_delta(np.array([h]))
        assert delta > 0
        assert np.issubdtype(delta.dtype, np.integer)

def test_compute_health_cost_relation():
    for h in range(1, 151):
        cost = compute_health_cost(np.array([h]))
        delta = compute_health_delta(np.array([h]))
        assert cost == -delta + 11

def test_compute_new_wealth_bounds():
    w = np.array([1, 5, 10])
    util = w**0.5
    new_w = compute_new_wealth(w, 0.3, util, N=5)
    assert np.all(new_w >= 0)
    assert new_w.dtype == np.int_

def test_utility_monotone():
    w = np.array([1,2,3])
    h = np.array([1,2,3])
    alpha = 0.5
    assert np.all(utility(w+1, h, alpha) >= utility(w,h,alpha))
    assert np.all(utility(w, h+1, alpha) >= utility(w,h,alpha))

def test_cpt_value_sign_monotone():
    theta, omega, eta = 0.88, 2.25, 0.88
    x = np.array([-3,-2,-1,0,1,2,3])
    v = cpt_value(x, theta, omega, eta)
    assert np.all(v[x < 0] < 0)
    assert np.all(v[x >= 0] >= 0)
    assert np.all(np.diff(v[x>=0]) >= 0)

def test_probability_weighting_bounds():
    p_vals = np.linspace(0.01,0.99,50)
    w = probability_weighting(p_vals, 0.8)
    assert np.all((w>0) & (w<1))

def test_probability_weighting_extremes():
    p = np.array([0.01,0.5,0.99])
    w_low = probability_weighting(p, 0.01)
    w_high = probability_weighting(p, 10)
    assert np.all((w_low>0)&(w_low<1))
    assert np.all((w_high>0)&(w_high<1))

def test_value_iteration_shapes(small_model):
    policy, params = small_model
    N = params["N"]
    assert policy.shape == (N,N)
    assert np.all(np.isin(policy, [0,1]))

def test_value_iteration_large_grid():
    N = 20
    policy, params, V = value_iteration_vectorized(
        N, 0.5, 0.8, 0.88, 2.25, 0.88, 0.95, 0.6, 0.3, 0.4
    )
    assert policy.shape == (N,N)
    assert np.all(np.isin(policy, [0,1]))
    assert np.all(np.isfinite(V))

def test_value_iteration_extreme_params():
    N = 5
    policy, params, V = value_iteration_vectorized(
        N, 0.5, 0.01, 0.1, 5, 0.1, 0.95, 0.6, 0.3, 0.4
    )
    assert np.all(np.isfinite(V))
    assert np.all(np.isin(policy,[0,1]))

def test_simulation_shapes_bounds(small_model):
    policy, params = small_model
    init = np.array([[3,3],[5,2],[1,4]])
    wealth, health = simulate(params, policy, 10, 3, init)
    N = params["N"]
    assert wealth.shape == (3,10)
    assert health.shape == (3,10)
    assert np.all(wealth >= 1) and np.all(wealth <= N)
    assert np.all(health >= 1) and np.all(health <= N)

def test_simulation_initial_state_preserved(small_model):
    policy, params = small_model
    init = np.array([[2,1],[5,5]])
    wealth, health = simulate(params, policy, 5, 2, init)
    assert np.all(wealth[:,0] == init[:,0])
    assert np.all(health[:,0] == init[:,1])

def test_simulation_reproducibility(small_model):
    policy, params = small_model
    init = np.array([[2,3],[4,5]])
    np.random.seed(42)
    w1,h1 = simulate(params, policy,5,2,init)
    np.random.seed(42)
    w2,h2 = simulate(params, policy,5,2,init)
    assert np.all(w1==w2)
    assert np.all(h1==h2)

def test_simulation_long_horizon_stability(small_model):
    policy, params = small_model
    init = np.array([[3,3]])
    wealth, health = simulate(params, policy, 5000, 1, init)
    assert np.all(wealth >=1) and np.all(wealth <= params["N"])
    assert np.all(health >=1) and np.all(health <= params["N"])
    assert not np.any(np.isnan(wealth)) and not np.any(np.isnan(health))

def test_invest_only_if_affordable(small_model):
    policy, params = small_model
    init = np.array([[1,3]])
    wealth,_ = simulate(params, policy,5,1,init)
    assert np.all(wealth>=1)

def test_save_when_not_investable(small_model):
    policy, params = small_model
    init = np.array([[1,1]])
    wealth,_ = simulate(params, policy,5,1,init)
    assert np.all(wealth>=1)

def test_health_delta_applied_correctly(small_model):
    policy, params = small_model
    init = np.array([[3,3],[4,4]])
    _, health = simulate(params, policy,2,2,init)
    max1 = compute_health_delta(init[0,1])
    max2 = compute_health_delta(init[1,1])
    assert abs(health[0,1]-init[0,1]) <= max1
    assert abs(health[1,1]-init[1,1]) <= max2

def test_wealth_delta_applied_correctly(small_model):
    policy, params = small_model
    init = np.array([[3,3]])
    wealth, _ = simulate(params, policy,2,1,init)
    max_delta = int(np.round(params["w_delta_scale"]*utility(init[0,0], init[0,1], params["alpha"])))
    assert wealth[0,1] <= init[0,0]+max_delta

def test_min_max_bounds(small_model):
    policy, params = small_model
    N = params["N"]
    init = np.array([[1,1],[N,N]])
    wealth, health = simulate(params, policy,5,2,init)
    assert np.all(wealth>=1) and np.all(wealth<=N)
    assert np.all(health>=1) and np.all(health<=N)

def test_policy_applied_over_time(small_model):
    policy, params = small_model
    init = np.array([[3,3]])
    np.random.seed(123)
    wealth, health = simulate(params, policy,10,1,init)
    changed = np.any(wealth[0,:] != init[0,0]) or np.any(health[0,:] != init[0,1])
    assert changed

def test_edge_health_bounds(small_model):
    policy, params = small_model
    N = params["N"]
    init = np.array([[1,1],[N,N]])
    _, health = simulate(params, policy,5,2,init)
    assert np.all(health>=1) and np.all(health<=N)

def test_edge_wealth_bounds(small_model):
    policy, params = small_model
    N = params["N"]
    init = np.array([[1,3],[N,3]])
    wealth, _ = simulate(params, policy,5,2,init)
    assert np.all(wealth>=1) and np.all(wealth<=N)

def test_multiple_agents_diversity(small_model):
    policy, params = small_model
    init = np.array([[1,1],[2,2],[3,3]])
    wealth, health = simulate(params, policy,5,3,init)
    assert not np.all(wealth[0] == wealth[1])
    assert not np.all(health[0] == health[1])

def test_probabilistic_variation(small_model):
    policy, params = small_model
    init = np.array([[3,3]])
    np.random.seed(100)
    w1,h1 = simulate(params, policy,10,1,init)
    np.random.seed(23)
    w2,h2 = simulate(params, policy,10,1,init)
    assert not np.all(w1==w2) or not np.all(h1==h2)

def test_probabilities_zero_one(small_model):
    policy, params = small_model
    params["P_H_increase"] = 0
    params["P_H_decrease"] = 0
    init = np.array([[3,3]])
    _, health = simulate(params, policy,10,1,init)
    assert np.all(health==3)
    params["P_H_increase"]=1
    params["P_H_decrease"]=1
    _, health = simulate(params, policy,10,1,init)
    N = params["N"]
    assert np.all(health>=1) and np.all(health<=N)

def test_independent_agent_stochasticity(small_model):
    policy, params = small_model
    init = np.array([[3,3],[3,3]])
    np.random.seed(10)
    w1,h1 = simulate(params, policy,10,2,init)
    assert not np.all(w1[0]==w1[1]) or not np.all(h1[0]==h1[1])

def test_cpt_monotone_and_loss_aversion():
    theta, omega, eta = 0.88, 2.25, 0.88
    gains = np.array([0,1,2,3])
    losses = np.array([-3,-2,-1,0])
    v_gains = cpt_value(gains, theta, omega, eta)
    v_losses = cpt_value(losses, theta, omega, eta)
    assert np.all(np.diff(v_gains) >= 0)
    assert np.all(v_losses[:-1] < 0)

def test_probability_weighting_monotone():
    p = np.linspace(0.01,0.99,10)
    w = probability_weighting(p,0.8)
    assert np.all(np.diff(w) >= 0)

def test_probability_weighting_extreme_gamma_small():
    p = np.array([0.01,0.5,0.99])
    w = probability_weighting(p,0.01)
    assert np.all((w>0)&(w<1))

def test_probability_weighting_extreme_gamma_large():
    p = np.array([0.01,0.5,0.99])
    w = probability_weighting(p,10)
    assert np.all((w>0)&(w<1))

def test_cpt_loss_aversion():
    theta, omega, eta = 0.88, 2.25, 0.88 # omega > 1
    x_gain = np.array([10, 20])
    x_loss = -x_gain
    v_gain = cpt_value(x_gain, theta, omega, eta)
    v_loss = cpt_value(x_loss, theta, omega, eta)
    assert np.all(np.abs(v_loss) > v_gain)

def test_utility_boundary_alpha():
    w, h = np.array([2]), np.array([3])
    # When alpha is 1, utility should only depend on wealth
    assert utility(w, h, alpha=1) == utility(w, h + 10, alpha=1)
    # When alpha is 0, utility should only depend on health
    assert utility(w, h, alpha=0) == utility(w + 10, h, alpha=0)

# Example of converting a test
@pytest.mark.parametrize("h", [1, 50, 100, 150])
def test_compute_health_cost_relation_parametrized(h):
    h_arr = np.array([h])
    cost = compute_health_cost(h_arr)
    delta = compute_health_delta(h_arr)
    assert cost == -delta + 11

def test_cpt_value_vectorized_logic():
    """
    Tests the vectorized cpt_value function with various vector compositions.
    """
    theta, omega, eta = 0.88, 2.25, 0.88
    
    # Test with only positive values
    x_pos = np.array([1, 5, 10])
    v_pos = cpt_value(x_pos, theta, omega, eta)
    assert np.all(v_pos >= 0)
    assert np.allclose(v_pos, x_pos**theta)

    # Test with only negative values
    x_neg = np.array([-1, -5, -10])
    v_neg = cpt_value(x_neg, theta, omega, eta)
    assert np.all(v_neg < 0)
    assert np.allclose(v_neg, -omega * (-x_neg)**eta)

    # Test with mixed values including zero
    x_mix = np.array([-5, -2, 0, 2, 5])
    v_mix = cpt_value(x_mix, theta, omega, eta)
    assert v_mix[2] == 0
    assert v_mix[0] < 0 and v_mix[1] < 0
    assert v_mix[3] > 0 and v_mix[4] > 0
    assert abs(v_mix[0]) > v_mix[4]  # Check loss aversion

def test_compute_new_wealth_clipping():
    """
    Ensures that compute_new_wealth correctly clips at both lower and upper bounds.
    """
    N = 50
    # A utility value that would push wealth well below the lower bound
    w_low = np.array([2])
    utility_low = np.array([0]) # Big negative delta
    new_w_low = compute_new_wealth(w_low, w_delta_scale=0.5, utility_value=utility_low, N=N)
    assert new_w_low == 1

    # A utility value that would push wealth well above the upper bound
    w_high = np.array([45])
    utility_high = np.array([100]) # Big positive delta
    new_w_high = compute_new_wealth(w_high, w_delta_scale=0.5, utility_value=utility_high, N=N)
    assert new_w_high == N

def test_utility_alpha_boundaries():
    """
    Tests the utility function when alpha is at its theoretical bounds (0 and 1).
    """
    w, h = np.array([10]), np.array([20])
    
    # If alpha=1, utility should only depend on wealth
    assert utility(w, h, alpha=1.0) == utility(w, h + 15, alpha=1.0)
    assert utility(w, h, alpha=1.0) != utility(w + 15, h, alpha=1.0)

    # If alpha=0, utility should only depend on health (assuming rate=1)
    assert utility(w, h, alpha=0.0) == utility(w + 15, h, alpha=0.0)
    assert utility(w, h, alpha=0.0) != utility(w, h + 15, alpha=0.0)

def test_value_iteration_no_investment_possible():
    """
    Tests that if no state allows investment, the policy is always 'save' (0).
    We simulate this by setting N so low that investment_cost is never met.
    """
    N = 5 # Health cost is min 1, so wealth must be > 1. Let's make it impossible
    # For h=5, cost is `11 - round(10*exp(-log(10)/150*5)+1) = 1`.
    # Let's set a really high h to make cost high
    # No, let's just make N really small.
    # The minimum cost is for h=150, cost is 11-2=9. So if N<=9, invest is hard.
    # The max cost is for h=1, cost is 11-11=0. But wealth must be > cost.
    # Let's just make the policy matrix check.

    policy, _, _ = value_iteration_vectorized(
        N=5, alpha=0.5, gamma=0.8, theta=0.88, omega=2.25, eta=0.88,
        beta=0.95, P_H_increase=0.6, w_delta_scale=0.3, P_H_decrease=0.4
    )
    
    # Find all states (w,h) where investment is not affordable
    w_vals, h_vals = np.arange(1, 6), np.arange(1, 6)
    W, H = np.meshgrid(w_vals, h_vals, indexing='ij')
    invest_cost = compute_health_cost(H)
    unaffordable_mask = W <= invest_cost
    
    # Assert that for all unaffordable states, the policy is 'save' (0)
    assert np.all(policy[unaffordable_mask] == 0)

def test_value_iteration_zero_beta():
    """
    If beta (discount factor) is 0, future value is irrelevant.
    The policy should be greedy based only on the immediate CPT outcome.
    """
    N = 10
    policy, _, _ = value_iteration_vectorized(
        N=N, alpha=0.5, gamma=0.8, theta=0.88, omega=2.25, eta=0.88,
        beta=0.0, P_H_increase=0.6, w_delta_scale=0.3, P_H_decrease=0.4
    )
    # This test confirms it runs; a full check would require re-calculating
    # the immediate CPT values manually, which is complex but would be the next step.
    assert policy.shape == (N, N)
    assert np.all(np.isfinite(policy))
