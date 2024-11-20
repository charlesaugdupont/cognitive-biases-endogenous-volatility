from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import generate_samples
from tqdm.auto import tqdm
from model import *
import argparse
import pickle
import os

np.random.seed(23)

def process_row(row, n_steps, n_agents):
    # unpack model parameters
    alpha, invest_cost, health_delta, health_decrease_scale = row
    invest_cost, health_delta = int(invest_cost), int(health_delta)

    # compute optimal policy
    policy, params, _ = value_iteration(
        N=200,
        gamma=0.6,
        theta=0.88,
        omega=2.25,
        eta=0.88,
        beta=0.95,
        P_H_increase=0.9,
        alpha=alpha,
        invest_cost=invest_cost,
        health_delta=health_delta,
        health_decrease_scale=health_decrease_scale
    )

    # run agent simulation
    _, wealth, health = simulate(
        params,
        policy,
        n_steps,
        n_agents
    )

    result = {
        "params": params,
        "wealth": wealth,
        "health": health,
        "policy": policy
    }

    output_file_name = os.path.join("results", f"{alpha}_{invest_cost}_{health_delta}_{round(health_decrease_scale, 2)}.pickle")
    with open(output_file_name, 'wb') as f:
        pickle.dump(result, f)

    return output_file_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--n-agents", type=int, default=10000)
    parser.add_argument("--n-steps", type=int, default=4000)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    N_SAMPLES = args.n_samples
    N_AGENTS = args.n_agents
    N_STEPS = args.n_steps
    MAX_WORKERS = args.max_workers

    samples = generate_samples(N_SAMPLES)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_row, row, N_STEPS, N_AGENTS) for row in samples]
        for future in tqdm(as_completed(futures), total=len(futures)):
            output_file_name = future.result()
