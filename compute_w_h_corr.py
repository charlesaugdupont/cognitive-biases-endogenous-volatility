import os
import pickle
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

directory = "results_5"
file_list = os.listdir(directory)

def process_file(f_name):
    """Load a file, compute utility, and return dominant frequencies for all agents."""
    with open(os.path.join(directory, f_name), "rb") as f:
        res = pickle.load(f)
    
    w = res["wealth"]
    h = res["health"]

    # Step 1: Compute row-wise means
    w_mean = w.mean(axis=1, keepdims=True)  # shape (10000, 1)
    h_mean = h.mean(axis=1, keepdims=True)  # shape (10000, 1)

    # Step 2: Compute row-wise deviations from mean
    w_dev = w - w_mean
    h_dev = h - h_mean

    # Step 3: Compute row-wise covariance and standard deviations
    cov = np.sum(w_dev * h_dev, axis=1)  # shape (10000,)
    w_std = np.sqrt(np.sum(w_dev**2, axis=1))
    h_std = np.sqrt(np.sum(h_dev**2, axis=1))

    # Step 4: Compute row-wise correlation
    denom = w_std * h_std
    # Set denom=0 to 1 temporarily to avoid division by zero
    denom_safe = np.where(denom == 0, 1, denom)
    corr = cov / denom_safe
    
    return corr

if __name__ == "__main__":

    with Pool(6) as pool:
        results = list(tqdm(pool.imap(process_file, file_list), total=len(file_list)))

    with open(directory + "_w_h_correlation.pickle", "wb") as f:
        pickle.dump(results, f)
