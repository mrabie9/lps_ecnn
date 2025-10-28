import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt

# Load the IQ data from the .npz file
data = np.load('radar/tasks-sw-nonorm/task0/radar_dataset.npz')
iq_matrix = data['xtr'][data['ytr'] != 0]  # Shape: (n, 1024)

# Flatten into a 1D array for analysis
iq_samples = iq_matrix.flatten()

# Convert interleaved real and imag to complex numbers if necessary
# (Only if it's still interleaved. If already complex dtype, skip this.)
if np.isrealobj(iq_samples):
    iq_samples = iq_samples[::2] + 1j * iq_samples[1::2]

# Compute the Power Spectral Density using Welch's method
fs = 10e6  # Sampling rate is 20 MHz
frequencies, psd = welch(iq_samples, fs=fs, nperseg=1024)
psd_db = 10 * np.log10(psd)

# Estimate noise and signal power
noise_floor = np.mean(psd_db[psd_db < np.percentile(psd_db, 20)])
signal_power = np.mean(psd_db[psd_db > np.percentile(psd_db, 95)])

snr_db = signal_power - noise_floor

print(f"Estimated SNR: {snr_db:.2f} dB")

# Optional: Plot PSD
plt.figure()
plt.semilogy(frequencies / 1e6, psd)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Power Spectral Density')
plt.title('PSD of IQ Samples')
plt.grid(True)
plt.show()
