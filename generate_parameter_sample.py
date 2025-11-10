from utils import generate_samples
import numpy as np
import argparse
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=1024)
    parser.add_argument("--model", type=str, default="cpt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    N_SAMPLES = args.n_samples
    MODEL = args.model
    SEED = args.seed

    if MODEL not in ["cpt", "nocpt"]:
        raise Exception(f"Invalid model name: {MODEL}")

    parameter_ranges = {
        "alpha":        [0.05, 0.95],
        "gamma":        [0.50, 0.80],
        "lambda":       [1.50, 3.00],
        "eta":          [0.65, 0.95],
        "P_H_increase": [0.05, 0.95],
        "P_H_decrease": [0.05, 0.95],
        "rate":         [1.00, 5.00],
        "w_delta_scale":[0.05, 0.95]
    }

    samples = generate_samples(N_SAMPLES, len(parameter_ranges), SEED)

    scaled_samples = np.zeros_like(samples)
    for i, (param, (low, high)) in enumerate(parameter_ranges.items()):
        if MODEL == "nocpt" and param in ["gamma", "lambda", "eta"]:
            scaled_samples[:, i] = 1
        else:
            scaled_samples[:, i] = samples[:, i] * (high - low) + low

    save_path = f"{MODEL}_samples.pickle"
    with open(save_path, "wb") as f:
        pickle.dump(scaled_samples, f)

    print(f"\nSuccesfully generated parameter sample with shape {scaled_samples.shape} for '{MODEL}' model")
    print(f"Results saved to: {save_path}")