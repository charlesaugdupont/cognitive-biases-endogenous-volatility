from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import generate_initial_agent_states
from tqdm.auto import tqdm
from model import *
import argparse
import pickle
import random
import os

# constants
GRID_SIZE = 200
THETA = 0.88
BETA = 0.95
SEED = 23
np.random.seed(SEED)
random.seed(SEED)

def process_row(row, n_steps, n_agents, output_dir):
    # unpack model parameters
    alpha, prob_health_decrease, prob_health_increase, gamma, w_delta_scale, omega, eta, initial_states = row

    # compute optimal policy
    policy, params, _ = value_iteration_vectorized(
        N=GRID_SIZE,
        theta=THETA,
        omega=omega,
        eta=eta,
        beta=BETA,
        alpha=alpha,
        P_H_decrease=prob_health_decrease,
        P_H_increase=prob_health_increase,
        gamma=gamma,
        w_delta_scale=w_delta_scale
    )

    # run agent simulation
    wealth, health = simulate(
        params,
        policy,
        n_steps,
        n_agents,
        initial_states
    )

    result = {
        "params": params,
        "wealth": wealth.astype(np.uint8),
        "health": health.astype(np.uint8),
        "policy": policy.astype(np.uint8)
    }

    output_file_name = os.path.join(output_dir, f"{alpha}_{prob_health_decrease}_{prob_health_increase}_{gamma}_{w_delta_scale}_{omega}_{eta}.pickle")
    with open(output_file_name, 'wb') as f:
        pickle.dump(result, f)

    return output_file_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=1024)
    parser.add_argument("--n-agents", type=int, default=10000)
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--model", type=str, default="cpt")
    args = parser.parse_args()

    N_SAMPLES = args.n_samples
    N_AGENTS = args.n_agents
    N_STEPS = args.n_steps
    MAX_WORKERS = args.max_workers
    OUTPUT_DIR = args.model

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    initial_states = generate_initial_agent_states(num_agents=N_AGENTS, N=GRID_SIZE, seed=SEED)

    with open("parameter_combinations.pickle", "rb") as f:
        parmameter_combinations = pickle.load(f)

    if OUTPUT_DIR == "cpt":
        samples = [(p["alpha"], p["P_H_decrease"], p["P_H_increase"], p["gamma"], p["w_delta_scale"], p["omega"], p["eta"], initial_states) for p in parmameter_combinations]
    elif OUTPUT_DIR == "nocpt":
        samples = [(p["alpha"], p["P_H_decrease"], p["P_H_increase"], 1, p["w_delta_scale"], 1, 1, initial_states) for p in parmameter_combinations]
    else:
        raise Exception("Invalid model.")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_row, row, N_STEPS, N_AGENTS, OUTPUT_DIR) for row in samples]
        for future in tqdm(as_completed(futures), total=len(futures)):
            output_file_name = future.result()
