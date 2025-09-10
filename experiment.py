from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import generate_samples
from tqdm.auto import tqdm
from model import *
import argparse
import pickle
import random
import os

SEED = 23
np.random.seed(SEED)
random.seed(SEED)

def process_row(row, n_steps, n_agents, output_dir):
    # unpack model parameters
    alpha, prob_health_decrease, prob_health_increase, gamma, omega, eta = row

    N = 200
    x = np.linspace(1, N, N)
    y = np.linspace(1, N, N)
    X, Y = np.meshgrid(x, y)
    mu = 100               # center of diagonal
    sigma_perp = 30        # controls decay perpendicular to diagonal
    sigma_diag = 30        # controls decay along diagonal
    dist_perp = (Y - X)
    dist_diag = (X + Y)/2 - mu
    decay_perp = np.exp(-(dist_perp**2) / (2*sigma_perp**2))
    decay_diag = np.exp(-(dist_diag**2) / (2*sigma_diag**2))
    magnitude = decay_perp * decay_diag
    w_delta_scale_grid = np.where(Y >= X, magnitude, -magnitude)


    # compute optimal policy
    policy, params, _ = value_iteration(
        N=N,
        theta=0.88,
        omega=omega,
        eta=eta,
        beta=0.95,
        alpha=alpha,
        P_H_decrease=prob_health_decrease,
        P_H_increase=prob_health_increase,
        gamma=gamma,
        w_delta_scale_grid=w_delta_scale_grid # Pass the grid here
    )

    output_file_name = os.path.join(output_dir, f"{alpha}_{prob_health_decrease}_{prob_health_increase}_{gamma}_{omega}_{eta}.pickle")

    if not os.path.exists(output_file_name):
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

        with open(output_file_name, 'wb') as f:
            pickle.dump(result, f)

        return output_file_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=1024)
    parser.add_argument("--n-agents", type=int, default=10000)
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--output-dir", type=str, default="state_dependent_growth_rate_full")
    args = parser.parse_args()

    N_SAMPLES = args.n_samples
    N_AGENTS = args.n_agents
    N_STEPS = args.n_steps
    MAX_WORKERS = args.max_workers
    OUTPUT_DIR = args.output_dir

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    samples = generate_samples(N_SAMPLES, 6, seed=SEED)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_row, row, N_STEPS, N_AGENTS, OUTPUT_DIR) for row in samples]
        for future in tqdm(as_completed(futures), total=len(futures)):
            output_file_name = future.result()
