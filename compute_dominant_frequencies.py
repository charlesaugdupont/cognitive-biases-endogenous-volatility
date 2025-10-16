import os
import pickle
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from model import utility
from scipy.signal import detrend

directory = "cpt"
file_list = os.listdir(directory)

def process_file_robust(f_name, power_threshold_ratio=0.20, discard_steps=3000, trivial_range_threshold=30):
    """
    Load a file, compute utility, and return dominant frequencies
    using a robust method with windowing and power thresholding.
    """
    with open(os.path.join(directory, f_name), "rb") as f:
        res = pickle.load(f)

    w = res["wealth"][:,discard_steps:].astype(np.int16)
    h = res["health"][:,discard_steps:].astype(np.int16)
    u = utility(w, h, res["params"]["alpha"])
    
    # threshold absed on utility range during steady state
    u_last = u[:,-500:]
    u_range = np.max(u_last, axis=1) - np.min(u_last, axis=1)
    is_trivial = u_range < trivial_range_threshold

    dominant_frequencies = np.full(w.shape[0], np.nan)
    is_significant = ~is_trivial
    if not np.any(is_significant):
        return dominant_frequencies

    u_to_process = u[is_significant]
    u = detrend(u_to_process, axis=1)

    # Apply a window function (e.g., Hann window) to each agent's signal
    window = np.hanning(u.shape[1])
    u_windowed = u * window

    # Compute FFT
    fft_vals_complex = np.fft.fft(u_windowed, axis=1)
    fft_freqs = np.fft.fftfreq(u.shape[1], 1)

    # Consider only positive frequencies
    pos_mask = fft_freqs > 0
    fft_freqs_pos = fft_freqs[pos_mask]
    
    # Calculate Power Spectral Density (PSD)
    psd = np.abs(fft_vals_complex[:, pos_mask])**2

    # Denoise by thresholding the PSD for each agent
    peak_power_per_agent = np.max(psd, axis=1, keepdims=True)
    peak_power_per_agent[peak_power_per_agent == 0] = 1 
    
    power_threshold = power_threshold_ratio * peak_power_per_agent
    psd[psd < power_threshold] = 0

    # Find the dominant frequency from the cleaned PSD
    dominant_idx = np.argmax(psd, axis=1)
    calculated_freqs = fft_freqs_pos[dominant_idx]
    calculated_freqs[np.max(psd, axis=1) == 0] = np.nan
    dominant_frequencies[is_significant] = calculated_freqs

    return dominant_frequencies

if __name__ == "__main__":
    from functools import partial

    with Pool(6) as pool:
        # Use partial to pass the threshold ratio to the processing function
        process_func = partial(process_file_robust, power_threshold_ratio=0.1)
        results = list(tqdm(pool.imap(process_func, file_list), total=len(file_list)))

    with open(directory + "_dominant_frequencies.pickle", "wb") as f:
        pickle.dump(results, f)