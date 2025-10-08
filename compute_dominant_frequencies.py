import os
import pickle
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from model import utility
from scipy.signal import detrend

directory = "results_5"
file_list = os.listdir(directory)

def process_file_robust(f_name, power_threshold_ratio=0.1, discard_steps=1000):
    """
    Load a file, compute utility, and return dominant frequencies
    using a robust method with windowing and power thresholding.
    """
    with open(os.path.join(directory, f_name), "rb") as f:
        res = pickle.load(f)

    w = res["wealth"]
    h = res["health"]
    u = detrend(utility(w, h, res["params"]["alpha"])[:, discard_steps:], axis=1)

    # 1. Apply a window function (e.g., Hann window) to each agent's signal
    window = np.hanning(u.shape[1])
    u_windowed = u * window

    # 2. Compute FFT
    fft_vals_complex = np.fft.fft(u_windowed, axis=1)
    fft_freqs = np.fft.fftfreq(u.shape[1], 1)

    # Consider only positive frequencies
    pos_mask = fft_freqs > 0
    fft_freqs_pos = fft_freqs[pos_mask]
    
    # 3. Calculate Power Spectral Density (PSD)
    psd = np.abs(fft_vals_complex[:, pos_mask])**2

    # 4. Denoise by thresholding the PSD for each agent
    # Set a threshold relative to the peak power of each agent
    peak_power_per_agent = np.max(psd, axis=1, keepdims=True)
    # Avoid division by zero for agents with no signal
    peak_power_per_agent[peak_power_per_agent == 0] = 1 
    
    power_threshold = power_threshold_ratio * peak_power_per_agent
    psd[psd < power_threshold] = 0

    # 5. Find the dominant frequency from the cleaned PSD
    # If an agent's max power is now 0, it means no dominant frequency was found
    dominant_idx = np.argmax(psd, axis=1)
    dominant_frequencies = fft_freqs_pos[dominant_idx]
    
    # Set frequency to NaN for agents with no significant signal
    dominant_frequencies[np.max(psd, axis=1) == 0] = np.nan

    return dominant_frequencies

if __name__ == "__main__":
    from functools import partial

    with Pool(6) as pool:
        # Use partial to pass the threshold ratio to the processing function
        process_func = partial(process_file_robust, power_threshold_ratio=0.1)
        results = list(tqdm(pool.imap(process_func, file_list), total=len(file_list)))

    with open(directory + "_dominant_frequencies_robust.pickle", "wb") as f:
        pickle.dump(results, f)