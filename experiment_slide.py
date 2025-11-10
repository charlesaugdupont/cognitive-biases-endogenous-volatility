from concurrent.futures import ProcessPoolExecutor, as_completed
from experiment import quantize_and_pack
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
P_H_CATASTROPHE = 0.00
HEALTH_SHOCK_SIZE = 1.0
SEED = 23
np.random.seed(SEED)
random.seed(SEED)

def process_row(row, n_steps, model, grid_size, initial_states):
    alpha, gamma, lambduh, eta, P_H_increase, P_H_decrease, rate, w_delta_scale = row

    # compute optimal policy
    policy, params = value_iteration_vectorized(
        N=grid_size,
        alpha=alpha,
        gamma=gamma,
        lambduh=lambduh,
        eta=eta,
        P_H_increase=P_H_increase,
        P_H_decrease=P_H_decrease,
        rate=rate,
        w_delta_scale=w_delta_scale,
        theta=THETA,
        beta=BETA,
        P_health_catastrophe=P_H_CATASTROPHE,
        health_shock_size=HEALTH_SHOCK_SIZE
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

    output_file_name = os.path.join(model, f"{alpha}_{gamma}_{lambduh}_{eta}_{P_H_increase}_{P_H_decrease}_{rate}.pickle")
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

    if MODEL not in ["slide_eta", "slide_gamma", "slide_lambda"]:
        raise Exception("Model must be one of {slide_eta, slide_gamma, slide_lambda}")

    if not os.path.exists(MODEL):
        os.makedirs(MODEL)

    with open("initial_states.pickle", "rb") as f:
        initial_states = pickle.load(f)

    # constant parameters
    ALPHA = 0.6609983472679867          # (0,1)
    P_H_DECREASE = 0.7385942711087851   # (0,1)
    P_H_INCREASE = 0.599589963875504    # (0,1)
    A = 0.8332704421368914              # (0,1)
    OMEGA = 1.2779085024959067          # (1,4)
    ETA = 0.6558972460820358            # (0.5,1)
    GAMMA = 0.7613187062319042          # (0.4,0.8)
    LAMBDUH = 2.6273535616937997        # (1.5,3.0)
    RATE = 4.0805161561520205           # (1.0,5,0)

    # construct samples
    if "gamma" in MODEL:
        vals = np.linspace(0.4, 0.8, 25)
        samples = [(ALPHA, v, LAMBDUH, ETA, P_H_INCREASE, P_H_DECREASE, RATE, A) for v in vals]
    elif "lambda" in MODEL:
        vals = np.linspace(1.5, 3.0, 25)
        samples = [(ALPHA, GAMMA, v, ETA, P_H_INCREASE, P_H_DECREASE, RATE, A) for v in vals]
    elif "eta" in MODEL:
        vals = np.linspace(0.5, 1.0, 25)
        samples = [(ALPHA, GAMMA, LAMBDUH, v, P_H_INCREASE, P_H_DECREASE, RATE, A) for v in vals]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_row, row, N_STEPS, MODEL, GRID_SIZE, initial_states) for row in samples]
        for future in tqdm(as_completed(futures), total=len(futures)):
            output_file_name = future.result()