import os
import pickle
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from model import utility

directory = "cpt_no_effect"
file_list = os.listdir(directory)

def process_file(f_name):
    """Load a file, compute utility, and return dominant frequencies for all agents."""
    with open(os.path.join(directory, f_name), "rb") as f:
        res = pickle.load(f)
    
    w = res["wealth"]
    h = res["health"]
    u = utility(w, h, res["params"]["alpha"])
    
    # Compute FFT along the time axis (axis=1)
    fft_vals = np.fft.fft(u, axis=1)
    
    # Corresponding frequencies
    fft_freqs = np.fft.fftfreq(5000, 1)
    
    # Only positive frequencies
    pos_mask = fft_freqs > 0
    fft_vals = np.abs(fft_vals[:, pos_mask])
    fft_freqs = fft_freqs[pos_mask]
    
    # Find the index of the dominant frequency for each agent
    dominant_idx = np.argmax(fft_vals, axis=1)
    
    # Map indices to actual frequencies
    dominant_frequencies = fft_freqs[dominant_idx]
    
    return dominant_frequencies

if __name__ == "__main__":

    with Pool(6) as pool:
        results = list(tqdm(pool.imap(process_file, file_list), total=len(file_list)))

    with open(directory + "_dominant_frequencies.pickle", "wb") as f:
        pickle.dump(results, f)
