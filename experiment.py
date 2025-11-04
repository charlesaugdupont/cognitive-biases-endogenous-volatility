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

def quantize_and_pack(data: np.ndarray, grid_size: int, dtype=np.uint16):
    """
    Quantizes continuous data from [1, grid_size] to the integer range of dtype.

    Args:
        data (np.ndarray): The floating-point array to convert.
        grid_size (int): The maximum value of the original range (e.g., 200).
        dtype: The target integer type (e.g., np.uint8 or np.uint16).

    Returns:
        np.ndarray: The quantized data as the specified integer type.
    """
    if np.issubdtype(dtype, np.integer):
        max_val = np.iinfo(dtype).max
        # Scale the data from [1, grid_size] to [0, max_val]
        # 1. Shift range to [0, grid_size-1]
        shifted_data = data - 1
        # 2. Scale to [0, max_val]
        scale_factor = max_val / (grid_size - 1)
        scaled_data = np.round(shifted_data * scale_factor)
        return scaled_data.astype(dtype)
    else:
        raise ValueError("dtype must be an integer type.")

def unpack_and_dequantize(data: np.ndarray, grid_size: int, dtype=np.uint16):
    """
    De-quantizes integer data back to its approximate float value in [1, grid_size].
    This is for use in your analysis code, not the simulation script.
    """
    if np.issubdtype(data.dtype, np.integer):
        max_val = np.iinfo(dtype).max
        scale_factor = max_val / (grid_size - 1)
        # Convert back to float and reverse the scaling
        unscaled_data = data.astype(np.float32) / scale_factor
        # Shift back to the original [1, grid_size] range
        return unscaled_data + 1
    else:
        raise ValueError("data must be an integer array.")

def process_row(row, n_steps, n_agents, output_dir):
    # unpack model parameters
    alpha, prob_health_decrease, prob_health_increase, gamma, w_delta_scale, omega, eta, initial_states = row

    # compute optimal policy (this now uses the interpolation method)
    policy, params, _ = value_iteration_vectorized(
        N=GRID_SIZE, theta=THETA, omega=omega, eta=eta, beta=BETA, alpha=alpha,
        P_H_decrease=prob_health_decrease, P_H_increase=prob_health_increase,
        gamma=gamma, w_delta_scale=w_delta_scale
    )

    # run agent simulation (this now simulates in continuous space)
    wealth, health = simulate(
        params, policy, n_steps, n_agents, initial_states
    )

    storage_dtype = np.uint16
    result = {
        "params": params,
        # Quantize the float data before saving
        "wealth": quantize_and_pack(wealth, GRID_SIZE, storage_dtype),
        "health": quantize_and_pack(health, GRID_SIZE, storage_dtype),
        "policy": policy.astype(np.uint8),
        # It's helpful to save the dtype you used for unpacking later
        "storage_dtype_info": str(storage_dtype)
    }

    output_file_name = os.path.join(output_dir, f"{alpha}_{prob_health_decrease}_{prob_health_increase}_{gamma}_{w_delta_scale}_{omega}_{eta}.pickle")
    with open(output_file_name, 'wb') as f:
        pickle.dump(result, f)

    return output_file_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=1024)
    parser.add_argument("--n-agents", type=int, default=5000)
    parser.add_argument("--n-steps", type=int, default=4000)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--model", type=str, default="nocpt_continuous_linear")
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

    parmameter_combinations = [p for p in parmameter_combinations if p["alpha"]==0.6808952922484166]

    if OUTPUT_DIR == "cpt_continuous_linear":
        samples = [(p["alpha"], p["P_H_decrease"], p["P_H_increase"], p["gamma"], p["w_delta_scale"], p["omega"], p["eta"], initial_states) for p in parmameter_combinations]
    elif OUTPUT_DIR == "nocpt_continuous_linear":
        samples = [(p["alpha"], p["P_H_decrease"], p["P_H_increase"], 1, p["w_delta_scale"], 1, 1, initial_states) for p in parmameter_combinations]
    else:
        raise Exception("Invalid model.")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_row, row, N_STEPS, N_AGENTS, OUTPUT_DIR) for row in samples]
        for future in tqdm(as_completed(futures), total=len(futures)):
            output_file_name = future.result()