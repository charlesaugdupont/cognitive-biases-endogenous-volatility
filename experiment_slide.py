from experiment import quantize_and_pack, THETA, ETA, BETA, P_H_CATASTROPHE, P_H_DECREASE, P_H_INCREASE, SEED
from generate_parameter_sample import PARAMETER_RANGES
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
from model import *
import argparse
import pickle
import random
import os

np.random.seed(SEED)
random.seed(SEED)

def process_row(row, n_steps, model, grid_size, initial_states):
    alpha, gamma, lambduh, rate, A, shock_size = row

    # compute optimal policy
    policy, params = value_iteration_pt_cpt(
        N=grid_size,
        alpha=alpha,
        gamma=gamma,
        lambduh=lambduh,
        eta=ETA,
        P_H_increase=P_H_INCREASE,
        P_H_decrease=P_H_DECREASE,
        rate=rate,
        A=A,
        theta=THETA,
        beta=BETA,
        P_health_catastrophe=P_H_CATASTROPHE,
        shock_size=shock_size
    )

    # run agent simulation
    wealth, health = simulate(params, policy, n_steps, initial_states)

    storage_dtype = np.uint16
    result = {
        "params": params,
        "wealth": quantize_and_pack(wealth, grid_size, storage_dtype),
        "health": quantize_and_pack(health, grid_size, storage_dtype),
        "policy": policy.astype(np.uint8),
        "storage_dtype_info": str(storage_dtype)
    }

    output_file_name = os.path.join(model, f"{alpha}_{gamma}_{lambduh}_{rate}_{A}_{shock_size}.pickle")
    with open(output_file_name, 'wb') as f:
        pickle.dump(result, f)

    return output_file_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--grid-size", type=int, default=200)
    args = parser.parse_args()

    N_STEPS = args.n_steps
    MAX_WORKERS = args.max_workers
    MODEL = args.model
    GRID_SIZE = args.grid_size

    if MODEL not in ["slide_gamma", "slide_lambda", "slide_rate"]:
        raise Exception("Model must be one of {slide_gamma, slide_lambda, slide_rate}")

    if not os.path.exists(MODEL):
        os.makedirs(MODEL)

    with open("initial_states.pickle", "rb") as f:
        initial_states = pickle.load(f)

    # constants
    ALPHA = 0.5840430741439839          # (0.0, 1.0)
    A = 0.6694087990448646              # (0.0, 1.0)
    GAMMA = 0.7487308119303844          # (0.4, 0.8)
    LAMBDUH = 2.529457110841415         # (1.5, 3.0)
    RATE = 3.2762230765095177           # (1.0, 5,0)
    SHOCK_SIZE = 0.9474815806644064     # (0.8, 1.0)

    # construct samples
    num_samples = 25
    if "gamma" in MODEL:
        vals = np.linspace(PARAMETER_RANGES["gamma"][0], PARAMETER_RANGES["gamma"][1], num_samples)
        samples = [(ALPHA, v, LAMBDUH, RATE, A, SHOCK_SIZE) for v in vals]
    elif "lambda" in MODEL:
        vals = np.linspace(PARAMETER_RANGES["lambda"][0], PARAMETER_RANGES["lambda"][1], num_samples)
        samples = [(ALPHA, GAMMA, v, RATE, A, SHOCK_SIZE) for v in vals]
    elif "rate" in MODEL:
        vals = np.linspace(PARAMETER_RANGES["rate"][0], PARAMETER_RANGES["rate"][1], num_samples)
        samples = [(ALPHA, GAMMA, LAMBDUH, v, A, SHOCK_SIZE) for v in vals]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_row, row, N_STEPS, MODEL, GRID_SIZE, initial_states) for row in samples]
        for future in tqdm(as_completed(futures), total=len(futures)):
            output_file_name = future.result()