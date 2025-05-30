# V2 - Enhanced Fractal Signal Analyzer
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime

# --- Basic Logging Configuration ---
log_filename = f"signal_analysis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])

# --- Configuration Parameters ---
# Signal Generation
SAMPLING_RATE = 44100  # Hz
DURATION = 2  # seconds
BASELINE_FREQ_HZ = 3000
DISTURBANCE_FREQ_HZ = 3500
DISTURBANCE_AMPLITUDE = 0.5

# Fractal Transform Parameters
EPSILON = 1e-9  # To prevent log(0) or division by zero
SASF2_COHERENCE_THRESHOLD = 0.5
DASF2_DISSIPATION_THRESHOLD = 0.05
FRACTAL_CLIP_RANGE = (-10, 10) # To prevent extreme fractal values

# Analysis & Alerting
SDI_ALERT_THRESHOLD = 500

# Output Configuration
OUTPUT_DIR = "analysis_results"
SAVE_PLOTS = True
SAVE_METRICS = True
PLOT_FILENAME_PREFIX = "V2_analysis_plot"
METRICS_FILENAME = "V2_numerical_metrics.txt"

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    logging.info(f"Created output directory: {OUTPUT_DIR}")

# --- Helper Functions ---

def generate_signals(sampling_rate, duration, baseline_freq, disturbance_freq, disturbance_amplitude):
    """Generates baseline and current (disturbed) signals."""
    logging.info("Generating signals...")
    time = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    baseline_signal = np.sin(2 * np.pi * baseline_freq * time)
    disturbance = disturbance_amplitude * np.sin(2 * np.pi * disturbance_freq * time)
    current_signal = baseline_signal + disturbance
    logging.info("Signals generated successfully.")
    return time, baseline_signal, current_signal

def calculate_fft(signal, sampling_rate):
    """Calculates FFT and corresponding frequencies."""
    logging.info("Calculating FFT...")
    n = len(signal)
    fft_values = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(n, 1/sampling_rate)
    logging.info("FFT calculated successfully.")
    return frequencies, fft_values

def sASF2_transform(fft_vals, freqs, coherence_threshold, epsilon, clip_range):
    """SASF²: Constructive fractal transform (amplifies coherent structures)."""
    mag = np.abs(fft_vals)
    with np.errstate(divide='ignore', invalid='ignore'):
        # Add epsilon to freqs as well to avoid log(0) if a frequency is 0
        fractal_denominator = np.log(np.abs(freqs) + epsilon) # abs for safety with freqs
        fractal_denominator[fractal_denominator == 0] = epsilon # ensure no zero in denominator for log
        
        fractal = np.log(mag + epsilon) / fractal_denominator
        fractal[np.isnan(fractal) | np.isinf(fractal)] = 0
        fractal = np.clip(fractal, clip_range[0], clip_range[1])
        
        # Avoid division by zero or very small numbers in coherence_threshold if it's made variable
        safe_coherence_threshold = coherence_threshold if coherence_threshold != 0 else epsilon
        coherence_metric = np.exp(-fractal / safe_coherence_threshold) # Renamed for clarity
    return fractal * coherence_metric

def DASF2_transform(fft_vals, freqs, dissipation_threshold, epsilon, clip_range):
    """DASF²: Dissipative fractal transform (suppresses divergent components)."""
    mag = np.abs(fft_vals)
    with np.errstate(divide='ignore', invalid='ignore'):
        fractal_denominator = np.log(np.abs(freqs) + epsilon)
        fractal_denominator[fractal_denominator == 0] = epsilon

        fractal = np.log(mag + epsilon) / fractal_denominator
        fractal[np.isnan(fractal) | np.isinf(fractal)] = 0
        fractal = np.clip(fractal, clip_range[0], clip_range[1])
        
        divergence = np.abs(fractal - np.mean(fractal))
    return fractal * np.where(divergence > dissipation_threshold, 0.1, 1)

def compute_numerical_metrics(baseline_signal, current_signal,
                              baseline_fft_mag, current_fft_mag,
                              frequencies, baseline_sasf2, current_sasf2, sdi_alert_threshold):
    """Computes various numerical metrics for signal comparison."""
    logging.info("Computing numerical metrics...")
    metrics = {}

    # Time-domain metrics
    metrics['RMSE'] = np.sqrt(np.mean((current_signal - baseline_signal) ** 2))

    # Frequency-domain metrics
    spectral_diff_mag = np.abs(current_fft_mag - baseline_fft_mag) # Using magnitudes
    metrics['SDI'] = np.mean(spectral_diff_mag)

    b_peak_idx = np.argmax(baseline_fft_mag)
    c_peak_idx = np.argmax(current_fft_mag)
    metrics['DFS'] = abs(frequencies[c_peak_idx] - frequencies[b_peak_idx])

    sig_pow = np.mean(np.abs(current_signal) ** 2)
    noise_pow = np.mean(np.abs(current_signal - baseline_signal) ** 2)
    metrics['SNR'] = 10 * np.log10(sig_pow / (noise_pow + EPSILON)) # Added epsilon for noise_pow

    # Fractal metrics
    fractal_divergence_sasf2 = np.abs(baseline_sasf2 - current_sasf2)
    fractal_divergence_sasf2[np.isnan(fractal_divergence_sasf2) | np.isinf(fractal_divergence_sasf2)] = 0
    metrics['Mean_SASF2_Divergence'] = np.mean(fractal_divergence_sasf2)
    metrics['Max_SASF2_Divergence'] = np.max(fractal_divergence_sasf2)

    # Other derived metrics
    metrics['CI_SDI'] = 1.96 * np.std(spectral_diff_mag) / np.sqrt(len(spectral_diff_mag))
    metrics['TCE'] = max(0, (1000 - metrics['SDI']) / (metrics['SDI'] + EPSILON)) # Added epsilon

    metrics['SDI_Alert_Threshold'] = sdi_alert_threshold
    metrics['Alert_Status'] = "ALERT: System approaching collapse!" if metrics['SDI'] > sdi_alert_threshold else "System vibrations are within safe limits."
    
    logging.info("Numerical metrics computed.")
    return metrics, fractal_divergence_sasf2


def plot_analysis_results(frequencies, baseline_fft_mag, current_fft_mag,
                          baseline_sasf2, current_sasf2,
                          baseline_dasf2, current_dasf2,
                          fractal_divergence_sasf2, output_dir, file_prefix, save_plots=True):
    """Generates and optionally saves plots of the analysis results."""
    logging.info("Generating plots...")
    plt.figure(figsize=(15, 18)) # Increased figure size for more plots

    # 1) FFT Magnitude
    plt.subplot(4, 1, 1)
    plt.plot(frequencies, baseline_fft_mag, label='Baseline FFT Mag')
    plt.plot(frequencies, current_fft_mag, label='Disturbed FFT Mag', alpha=0.7)
    plt.title('1) FFT Magnitude - Baseline vs. Disturbed')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    # 2) SASF² Fractal Analysis
    plt.subplot(4, 1, 2)
    plt.plot(frequencies, baseline_sasf2, label='Baseline SASF²')
    plt.plot(frequencies, current_sasf2, label='Disturbed SASF²', alpha=0.7)
    plt.title('2) SASF² Fractal Spectral Analysis')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('SASF² Value')
    plt.grid(True)
    plt.legend()

    # 3) DASF² Fractal Analysis
    plt.subplot(4, 1, 3)
    plt.plot(frequencies, baseline_dasf2, label='Baseline DASF²')
    plt.plot(frequencies, current_dasf2, label='Disturbed DASF²', alpha=0.7)
    plt.title('3) DASF² Fractal Spectral Analysis')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('DASF² Value')
    plt.grid(True)
    plt.legend()

    # 4) Fractal Divergence (SASF²)
    plt.subplot(4, 1, 4)
    plt.plot(frequencies, fractal_divergence_sasf2, color='red', label='SASF² Fractal Divergence')
    threshold_line = np.mean(fractal_divergence_sasf2) + 2 * np.std(fractal_divergence_sasf2)
    plt.axhline(y=threshold_line, color='purple', linestyle='--', label=f'Divergence Alert Threshold ({threshold_line:.2f})')
    plt.title('4) SASF² Fractal Divergence Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Divergence')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    
    if save_plots:
        plot_filename = os.path.join(output_dir, f"{file_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_filename)
        logging.info(f"Plots saved to {plot_filename}")
    else:
        plt.show()
    plt.close() # Close plot to free memory


def save_metrics_to_file(metrics, output_dir, filename):
    """Saves the computed numerical metrics to a text file."""
    filepath = os.path.join(output_dir, filename)
    logging.info(f"Saving metrics to {filepath}...")
    with open(filepath, 'w') as f:
        f.write(f"Signal Analysis Numerical Metrics (V2) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
        f.write("="*50 + "\n")
        f.write(f"Log file for this run: {log_filename}\n")
    logging.info("Metrics saved successfully.")

# --- Main Analysis Workflow ---
def main_analysis_workflow():
    """Orchestrates the entire signal analysis process."""
    logging.info("Starting V2 Signal Analysis Workflow...")

    # 1. Generate Signals
    time, baseline_signal, current_signal = generate_signals(
        SAMPLING_RATE, DURATION, BASELINE_FREQ_HZ, DISTURBANCE_FREQ_HZ, DISTURBANCE_AMPLITUDE
    )

    # 2. Calculate FFT
    frequencies, baseline_fft_vals = calculate_fft(baseline_signal, SAMPLING_RATE)
    _, current_fft_vals = calculate_fft(current_signal, SAMPLING_RATE)
    
    baseline_fft_mag = np.abs(baseline_fft_vals)
    current_fft_mag = np.abs(current_fft_vals)


    # 3. Apply Fractal Transforms
    logging.info("Applying fractal transforms...")
    baseline_sasf2 = sASF2_transform(baseline_fft_vals, frequencies, SASF2_COHERENCE_THRESHOLD, EPSILON, FRACTAL_CLIP_RANGE)
    current_sasf2 = sASF2_transform(current_fft_vals, frequencies, SASF2_COHERENCE_THRESHOLD, EPSILON, FRACTAL_CLIP_RANGE)
    
    baseline_dasf2 = DASF2_transform(baseline_fft_vals, frequencies, DASF2_DISSIPATION_THRESHOLD, EPSILON, FRACTAL_CLIP_RANGE)
    current_dasf2 = DASF2_transform(current_fft_vals, frequencies, DASF2_DISSIPATION_THRESHOLD, EPSILON, FRACTAL_CLIP_RANGE)
    logging.info("Fractal transforms applied.")

    # 4. Compute Numerical Metrics
    metrics, fractal_divergence_sasf2 = compute_numerical_metrics(
        baseline_signal, current_signal,
        baseline_fft_mag, current_fft_mag, frequencies,
        baseline_sasf2, current_sasf2, SDI_ALERT_THRESHOLD
    )
    
    # Print metrics to console (logging also handles this for some parts)
    print("\n--- Numerical Metrics (V2) ---")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print(f"Alert Status: {metrics['Alert_Status']}")
    print("--- End of Metrics ---")


    # 5. Plot Analysis Results
    plot_analysis_results(
        frequencies, baseline_fft_mag, current_fft_mag,
        baseline_sasf2, current_sasf2,
        baseline_dasf2, current_dasf2,
        fractal_divergence_sasf2,
        OUTPUT_DIR, PLOT_FILENAME_PREFIX, SAVE_PLOTS
    )

    # 6. Save Metrics to File
    if SAVE_METRICS:
        save_metrics_to_file(metrics, OUTPUT_DIR, METRICS_FILENAME)

    logging.info("V2 Signal Analysis Workflow Completed.")

# --- Main Execution ---
if __name__ == "__main__":
    main_analysis_workflow()
