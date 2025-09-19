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
    alpha, prob_health_decrease, prob_health_increase, w_delta_scale = row

    # compute optimal policy
    policy, params, _ = value_iteration(
        N=200,
        theta=1,
        omega=1,
        eta=1,
        beta=0.95,
        alpha=alpha,
        P_H_decrease=prob_health_decrease,
        P_H_increase=prob_health_increase,
        gamma=1,
        w_delta_scale=w_delta_scale
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

    output_file_name = os.path.join(output_dir, f"{alpha}_{prob_health_decrease}_{prob_health_increase}_{w_delta_scale}.pickle")
    with open(output_file_name, 'wb') as f:
        pickle.dump(result, f)

    return output_file_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=1024)
    parser.add_argument("--n-agents", type=int, default=10000)
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--output-dir", type=str, default="cpt_no_effect")
    args = parser.parse_args()

    N_SAMPLES = args.n_samples
    N_AGENTS = args.n_agents
    N_STEPS = args.n_steps
    MAX_WORKERS = args.max_workers
    OUTPUT_DIR = args.output_dir

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    with open("parameter_combinations.pickle", "rb") as f:
        parmameter_combinations = pickle.load(f)

    samples = [(p["alpha"], p["P_H_decrease"], p["P_H_increase"], p["w_delta_scale"]) for p in parmameter_combinations]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_row, row, N_STEPS, N_AGENTS, OUTPUT_DIR) for row in samples]
        for future in tqdm(as_completed(futures), total=len(futures)):
            output_file_name = future.result()
