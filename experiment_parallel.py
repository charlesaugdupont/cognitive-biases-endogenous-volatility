from SALib.sample import sobol
from tqdm.auto import tqdm
from model import *
import pickle
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

np.random.seed(23)

MAX_WORKERS = 8
N_SOBOL_SAMPLES = 128
N_AGENTS = 10000
N_STEPS = 4000

def process_row(row):
    # unpack model parameters
    alpha, invest_cost, health_delta = row
    invest_cost, health_delta = int(invest_cost), int(health_delta)

    # compute optimal policy
    policy, params, _ = value_iteration(
        N=200,
        alpha=alpha,
        gamma=0.6,
        theta=0.88,
        omega=2.25,
        eta=0.88,
        beta=0.95,
        P_H_increase=0.9,
        invest_cost=invest_cost,
        health_delta=health_delta,
        health_decrease_scale=1
    )

    # run agent simulation
    _, wealth, health = simulate(
        params,
        policy,
        N_STEPS,
        N_AGENTS
    )

    result = {
        "params": params,
        "wealth": wealth,
        "health": health,
        "policy": policy
    }

    output_file_name = os.path.join("results", f"{alpha}_{invest_cost}_{health_delta}.pickle")
    with open(output_file_name, 'wb') as f:
        pickle.dump(result, f)

    return output_file_name

if __name__ == "__main__":
    problem = {
        'num_vars': 3,
        'names': ['alpha', 'invest_cost', 'health_delta'],
        'bounds': [[0, 1], [0, 1], [0, 1]]
    }

    param_values = sobol.sample(problem, N_SOBOL_SAMPLES)
    param_values[:, 1] = np.ceil(param_values[:,1]*10).astype(int)
    param_values[:, 2] = np.ceil(param_values[:,2]*10).astype(int)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_row, row) for row in param_values]
        for future in tqdm(as_completed(futures), total=len(futures)):
            output_file_name = future.result()
