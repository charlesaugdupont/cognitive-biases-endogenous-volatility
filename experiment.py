from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import generate_samples
from tqdm.auto import tqdm
from model import *
import argparse
import pickle
import os

np.random.seed(23)

def process_row(row, n_steps, n_agents, output_dir):
    # unpack model parameters
    alpha, invest_cost, health_delta, prob_health_decrease, prob_health_increase, gamma = row
    invest_cost, health_delta = int(invest_cost), int(health_delta)

    # compute optimal policy
    policy, params, _ = value_iteration(
        N=200,
        theta=0.88,
        omega=2.25,
        eta=0.88,
        beta=0.95,
        gamma=gamma,
        P_H_increase=prob_health_increase,
        alpha=alpha,
        invest_cost=invest_cost,
        health_delta=health_delta,
        P_H_decrease=prob_health_decrease
    )

    # run agent simulation
    wealth, health = simulate(
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

    output_file_name = os.path.join(output_dir, f"{alpha}_{invest_cost}_{health_delta}_{prob_health_decrease}_{prob_health_increase}_{gamma}.pickle")
    with open(output_file_name, 'wb') as f:
        pickle.dump(result, f)

    return output_file_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--n-agents", type=int, default=10000)
    parser.add_argument("--n-steps", type=int, default=4000)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    N_SAMPLES = args.n_samples
    N_AGENTS = args.n_agents
    N_STEPS = args.n_steps
    MAX_WORKERS = args.max_workers
    OUTPUT_DIR = args.output_dir

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    samples = generate_samples(N_SAMPLES, 6)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_row, row, N_STEPS, N_AGENTS, OUTPUT_DIR) for row in samples]
        for future in tqdm(as_completed(futures), total=len(futures)):
            output_file_name = future.result()
