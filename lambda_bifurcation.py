from experiment import quantize_and_pack, THETA, ETA, BETA, P_H_DECREASE, P_H_INCREASE, SEED
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
    alpha, gamma, lambduh, rate, A = row

    # compute optimal policy
    policy, params = value_iteration_pt_cpt(
        N=grid_size,
        alpha=alpha,
        gamma=1,
        lambduh=lambduh,
        eta=ETA,
        P_H_increase=P_H_INCREASE,
        P_H_decrease=P_H_DECREASE,
        rate=rate,
        A=A,
        theta=THETA,
        beta=BETA
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

    output_file_name = os.path.join(model, f"{alpha}_{gamma}_{lambduh}_{rate}_{A}.pickle")
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

    if MODEL != "lambda_bifurcation":
        raise Exception("Model must be 'lambda_bifurcation'")
    if not os.path.exists(MODEL):
        os.makedirs(MODEL)

    with open("initial_states.pickle", "rb") as f:
        initial_states = pickle.load(f)

    # identify PT simulations with average amplitude > 10
    with open("pt_dominant_frequencies_amplitudes.pickle", "rb") as f:
        pt_data = pickle.load(f)

    THRESHOLD = 10
    sims = [np.mean(x["amplitudes"]) for x in pt_data]
    params = []
    for idx, f_name in enumerate(os.listdir("pt")):
        if sims[idx] > THRESHOLD:
            with open(os.path.join("pt", f_name), "rb") as f:
                res = pickle.load(f)
            P = res["params"]
            params.append((
                P["alpha"], P["A"], P["rate"], P["lambda"]
            ))

    # run a sweep of lambda values
    lambda_values = np.linspace(1, 2.5, 9)
    samples = []
    for p in params:
        for L in lambda_values:
            samples.append(
                (p[0], 1, L, p[2], p[1])
            )

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_row, row, N_STEPS, MODEL, GRID_SIZE, initial_states) for row in samples]
        for future in tqdm(as_completed(futures), total=len(futures)):
            output_file_name = future.result()