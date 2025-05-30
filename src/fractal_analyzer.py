# src/fractal_analyzer.py
import numpy as np
import matplotlib.pyplot as plt

# Parameters
sampling_rate = 44100
duration = 2
epsilon = 1e-9
coherence_threshold = 0.5  # Adjusted to prevent overflow
dissipation_threshold = 0.05

# Generate sample signals
time = np.linspace(0, duration, int(sampling_rate * duration))
baseline_freq = 3000
baseline_signal = np.sin(2 * np.pi * baseline_freq * time)
disturbance_freq = 3500
current_signal = baseline_signal + 0.5 * np.sin(2 * np.pi * disturbance_freq * time)

# FFT
frequencies = np.fft.rfftfreq(len(time), 1/sampling_rate)
baseline_fft = np.fft.rfft(baseline_signal)
current_fft = np.fft.rfft(current_signal)

# SASF²: Constructive fractal transform (amplify coherent structures)
def sASF2_transform(fft_vals, freqs, coherence_threshold):
    mag = np.abs(fft_vals)
    with np.errstate(divide='ignore', invalid='ignore'):
        fractal = np.log(mag + epsilon) / np.log(freqs + epsilon)
        fractal[np.isnan(fractal) | np.isinf(fractal)] = 0
        fractal = np.clip(fractal, -10, 10)  # Prevent extreme values
        coherence = np.exp(-fractal / coherence_threshold)
        return fractal * coherence

# DASF²: Dissipative fractal transform (suppress divergent components)
def dASF2_transform(fft_vals, freqs, dissipation_threshold):
    mag = np.abs(fft_vals)
    with np.errstate(divide='ignore', invalid='ignore'):
        fractal = np.log(mag + epsilon) / np.log(freqs + epsilon)
        fractal[np.isnan(fractal) | np.isinf(fractal)] = 0
        divergence = np.abs(fractal - np.mean(fractal))
        return fractal * np.where(divergence > dissipation_threshold, 0.1, 1)

# Apply transforms
baseline_sasf2 = sASF2_transform(baseline_fft, frequencies, coherence_threshold)
current_sasf2 = sASF2_transform(current_fft, frequencies, coherence_threshold)
baseline_dasf2 = dASF2_transform(baseline_fft, frequencies, dissipation_threshold)
current_dasf2 = dASF2_transform(current_fft, frequencies, dissipation_threshold)

# Fractal divergence (based on SASF² for anomaly detection)
fractal_divergence = np.abs(baseline_sasf2 - current_sasf2)
fractal_divergence[np.isnan(fractal_divergence) | np.isinf(fractal_divergence)] = 0

# Compute numerical metrics
spectral_div = np.abs(current_fft - baseline_fft)
SDI = np.mean(spectral_div)  # Spectral Divergence Index
RMSE = np.sqrt(np.mean((current_signal - baseline_signal) ** 2))  # Root Mean Square Error

# DFS computation with debugging
b_peak_idx = np.argmax(np.abs(baseline_fft))
c_peak_idx = np.argmax(np.abs(current_fft))
b_peak = frequencies[b_peak_idx]  # Baseline peak frequency
c_peak = frequencies[c_peak_idx]  # Current peak frequency
DFS = abs(c_peak - b_peak)  # Dominant Frequency Shift
print(f"Debug: Baseline peak frequency = {b_peak:.2f} Hz, Current peak frequency = {c_peak:.2f} Hz")  # Debug

sig_pow = np.mean(np.abs(current_signal) ** 2)  # Signal power
noise_pow = np.mean(np.abs(current_signal - baseline_signal) ** 2)  # Noise power
SNR = 10 * np.log10(sig_pow / (noise_pow + 1e-12))  # Signal-to-Noise Ratio
CI = 1.96 * np.std(spectral_div) / np.sqrt(len(spectral_div))  # Confidence Interval
TCE = max(0, (1000 - SDI) / (SDI + 0.001))  # Time-to-Collapse (in minutes)

# Plotting
plt.figure(figsize=(14, 10))

# 1) FFT Magnitude
plt.subplot(3, 1, 1)
plt.plot(frequencies, np.abs(baseline_fft), label='Baseline')
plt.plot(frequencies, np.abs(current_fft), label='Disturbed', alpha=0.7)
plt.title('1) FFT Magnitude - Baseline vs. Disturbed')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# 2) SASF² Fractal Analysis
plt.subplot(3, 1, 2)
plt.plot(frequencies, baseline_sasf2, label='Fractal Baseline (SASF²)')
plt.plot(frequencies, current_sasf2, label='Fractal Disturbed (SASF²)', alpha=0.7)
plt.title('2) SASF² Fractal Spectral Analysis')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Fractal Dimension (approx)')
plt.grid(True)
plt.legend()

# 3) Fractal Divergence
plt.subplot(3, 1, 3)
plt.plot(frequencies, fractal_divergence, color='red', label='Fractal Divergence')
# Adjust threshold to be more meaningful: 2 * standard deviation above mean
threshold_line = np.mean(fractal_divergence) + 2 * np.std(fractal_divergence)
plt.axhline(y=threshold_line, color='purple', linestyle='--', label='Alert Threshold')
plt.title('3) Fractal Divergence Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Divergence')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Display numerical results
print("\nNumerical Metrics:")
print(f"Spectral Divergence Index (SDI): {SDI:.2f}")
print(f"Root Mean Square Error (RMSE): {RMSE:.2f}")
print(f"Dominant Frequency Shift (DFS): {DFS:.2f} Hz")
print(f"Signal-to-Noise Ratio (SNR): {SNR:.2f} dB")
print(f"Confidence Interval (CI) for SDI: ±{CI:.4f}")
print(f"Estimated Time-to-Collapse (TCE): {TCE:.2f} minutes")

# Alert based on SDI threshold
if SDI > 500:
    print("ALERT: System approaching collapse!")
else:
    print("System vibrations are within safe limits.")
