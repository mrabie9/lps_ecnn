import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from skimage.transform import resize
from skimage.color import gray2rgb

def iq_to_spectrogram(iq_sample, nperseg=64, noverlap=32):
    f, t, Sxx = spectrogram(iq_sample, fs=50e7, nperseg=nperseg, noverlap=noverlap)
    Sxx_log = 10 * np.log10(Sxx + 1e-10)
    Sxx_log = (Sxx_log - np.min(Sxx_log)) / (np.max(Sxx_log) - np.min(Sxx_log))
    return Sxx_log

def process_dataset(data, labels):
    spectrograms = []
    for idx, iq in enumerate(data):
        if labels[idx] != 1:
            print(f"Label: {labels[idx]}")
            complex_iq = iq[::2] + 1j * iq[1::2]
            spec = iq_to_spectrogram(complex_iq)
            plt.imshow(spec)
            plt.show()
            spec_resized = resize(spec, (50, 50), anti_aliasing=True)
            spec_rgb = gray2rgb(spec_resized)
            spectrograms.append(spec_rgb)
    return np.array(spectrograms)

def visualize_spectrograms(spectrograms, labels=None, num_samples=6):
    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        plt.subplot(2, 3, i + 1)
        plt.imshow(spectrograms[i])
        if labels is not None:
            plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def main(input_path, output_path):
    data = np.load(input_path)
    yte_subset = data['yte']
    mask = yte_subset == 3
    print(np.unique_counts(data['yte']))
    xtr_spec = process_dataset(data['xte'], yte_subset)
    # xte_spec = process_dataset(data['xte'])

    # np.savez(output_path, xtr=xtr_spec, xte=xte_spec, ytr=data['ytr'], yte=data['yte'])
    print(f"Saved spectrogram data to {output_path}")

    # Optional visualization
    visualize_spectrograms(xtr_spec, labels=data['ytr'])

if __name__ == "__main__":
    input_file = "radar/tasks-highfs-4096sl/task0/radar_dataset.npz"   # Replace with your input file
    output_file = "output_spectrograms.npz"
    main(input_file, output_file)
