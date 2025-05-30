# -Vibrational/Fregventional Signal-Analyzer-with-Fractal-Transforms-SASF²/DASF²/FFT-
Signal  Fractal Transforms (SASF²/DASF²/FFT)  This project implements an innovative method for signal analysis, particularly for anomaly detection and vibration characterization, using fractal spectral transforms
# Signal Analyzer with Fractal Transforms (SASF²/DASF²)
# By: V.Lucian Borbeleac
This project implements an innovative method for signal analysis, particularly for anomaly detection and vibration characterization, using fractal spectral transforms named SASF² (Constructive Fractal Transform) and DASF² (Dissipative Fractal Transform). The program simulates a baseline signal and a disturbed signal, applies these transforms, and calculates various metrics to quantify divergence and assess system status.

## Features

* **Synthetic Signal Generation**: Creates a baseline signal and a disturbed signal for demonstration.
* **Standard FFT Analysis**: Calculates the Fast Fourier Transform to switch to the frequency domain.
* **Innovative Fractal Transforms**:
    * **SASF² (Constructive Fractal Transform)**: Amplifies coherent structures in the frequency spectrum.
    * **dASF² (Dissipative Fractal Transform)**: Suppresses divergent or irregular components.
* **Anomaly Detection**: Uses fractal divergence (based on SASF²) to identify significant differences between the baseline and current signals.
* **Comprehensive Numerical Metrics**: Calculates indicators such as:
    * Spectral Divergence Index (SDI)
    * Root Mean Square Error (RMSE)
    * Dominant Frequency Shift (DFS)
    * Signal-to-Noise Ratio (SNR)
    * Confidence Interval (CI) for SDI
    * Estimated Time-to-Collapse (TCE)
* **Detailed Visualization**: Generates plots for FFT magnitude, SASF² fractal analysis, and the fractal divergence spectrum.
* **Simple Alert System**: Issues an alert based on the SDI value.

## Requirements

* Python 3.x
* NumPy
* Matplotlib

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/fractal_signal_analyzer.git](https://github.com/YourUsername/fractal_signal_analyzer.git)
    cd fractal_signal_analyzer
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the main script from the `src` directory:

```bash
python src/fractal_analyzer.py
