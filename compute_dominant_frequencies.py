import os
import pickle
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from model import utility
from scipy.signal import detrend
from functools import partial

directory = "cpt"
file_list = os.listdir(directory)

def process_file_robust(f_name, power_threshold_ratio=0.20, discard_steps=3000, trivial_range_threshold=0):
    """
    Load a file, compute utility, and return dominant frequencies
    using a robust method with windowing and power thresholding.
    """
    with open(os.path.join(directory, f_name), "rb") as f:
        res = pickle.load(f)

    w = res["wealth"][:, discard_steps:].astype(np.int16)
    h = res["health"][:, discard_steps:].astype(np.int16)
    u = utility(w, h, res["params"]["alpha"])

    # Threshold based on utility range during steady state
    u_last = u[:, -500:]
    u_range = np.max(u_last, axis=1) - np.min(u_last, axis=1)
    is_trivial = u_range < trivial_range_threshold

    dominant_frequencies = np.full(w.shape[0], np.nan)
    dominant_amplitudes = np.full(w.shape[0], np.nan)
    is_significant = ~is_trivial
    if not np.any(is_significant):
        return {
            "frequencies":dominant_frequencies, 
            "amplitudes":dominant_amplitudes
        }   

    # Prepare and detrend signals
    u_to_process = u[is_significant]
    u = detrend(u_to_process, axis=1)

    # Apply Hann window
    N = u.shape[1]
    window = np.hanning(N)
    u_windowed = u * window

    # FFT with proper normalization (divide by N and window RMS correction)
    window_rms = np.sqrt(np.mean(window ** 2))
    fft_vals_complex = np.fft.fft(u_windowed, axis=1) / (N * window_rms)
    fft_freqs = np.fft.fftfreq(N, 1)

    # Positive frequencies only
    pos_mask = fft_freqs > 0
    fft_freqs_pos = fft_freqs[pos_mask]

    # Power spectral density
    psd = np.abs(fft_vals_complex[:, pos_mask])**2

    # Denoise by thresholding
    peak_power_per_agent = np.max(psd, axis=1, keepdims=True)
    peak_power_per_agent[peak_power_per_agent == 0] = 1
    power_threshold = power_threshold_ratio * peak_power_per_agent
    psd[psd < power_threshold] = 0

    # Find dominant frequency and corresponding amplitude
    dominant_idx = np.argmax(psd, axis=1)
    dominant_freq_vals = fft_freqs_pos[dominant_idx]
    dominant_power_vals = psd[np.arange(psd.shape[0]), dominant_idx]

    # Handle zero-power cases
    zero_power_mask = np.max(psd, axis=1) == 0
    dominant_freq_vals[zero_power_mask] = np.nan
    dominant_power_vals[zero_power_mask] = np.nan

    # Store results
    dominant_frequencies[is_significant] = dominant_freq_vals
    dominant_amplitudes[is_significant] = 2 * np.sqrt(dominant_power_vals)

    return {
        "frequencies":dominant_frequencies, 
        "amplitudes":dominant_amplitudes
    }

if __name__ == "__main__":

    with Pool(6) as pool:
        # Use partial to pass the threshold ratio to the processing function
        process_func = partial(process_file_robust)
        results = list(tqdm(pool.imap(process_func, file_list), total=len(file_list)))

    with open(directory + "_dominant_frequencies_amplitudes.pickle", "wb") as f:
        pickle.dump(results, f)
