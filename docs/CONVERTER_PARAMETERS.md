# Buck Converter Parameters for Anomaly Detection

This document summarizes additional parameters and signals that can be extracted from Buck converters for enhanced anomaly detection, based on recent research.

---

## 1. Current Data: Small-Signal Transfer Functions ($G_{vd}(s)$)

### Primary Frequency-Response Representation
- **Frequency Grid**: 101 log-spaced points from **100 Hz to 50 kHz** (switching frequency $f_{\mathrm{sw}} = 100\text{ kHz}$, Nyquist $50\text{ kHz}$).
- **Raw PSIM AC Sweep**: Linear gain $|H(j\omega)|$ (V) and unwrapped phase $\phi(\omega)$ (degrees).
- **Model Input Tensor (`[101, 2]`)**: Smooth Cartesian projection
  $$\mathrm{Re}\{H(j\omega_k)\} = |H(j\omega_k)|\cos\phi(\omega_k), \qquad \mathrm{Im}\{H(j\omega_k)\} = |H(j\omega_k)|\sin\phi(\omega_k),$$
  standardized channel-wise via `DataPreprocessor`.

### Exact State-Space Averaged (SSA) Buck Model (`BuckConverter`)
For a synchronous/diode-clamped CCM Buck converter ($V_{\mathrm{in}}=50\text{ V}$, $D=0.40$, nominal $L=200\ \mu\text{H}$, $C=100\ \mu\text{F}$, $R_{\mathrm{out}}=4\ \Omega$, $R_{ds,1}=10\text{ m}\Omega$, $r_L=100\text{ m}\Omega$, $r_C=20\text{ m}\Omega$), accounting for series conduction resistances $r_{\mathrm{eq}} = r_L + D R_{ds,1} + (1-D) R_{ds,2}^{\mathrm{eff}}$ and steady-state inductor current $I_L = D V_{\mathrm{in}} / (R_{\mathrm{out}} + r_{\mathrm{eq}})$:
$$G_{vd}(s) = \big(V_{\mathrm{in}} - I_L (R_{ds,1} - R_{ds,2}^{\mathrm{eff}})\big) \cdot \frac{Z_{RC}(s)}{Z_L(s) + Z_{RC}(s)},$$
where $Z_L(s) = r_{\mathrm{eq}} + sL$ and $Z_{RC}(s) = \frac{R_{\mathrm{out}}(1 + s r_C C)}{1 + s(R_{\mathrm{out}} + r_C)C}$.
- Note: In `buck.psimsch`, the low-side MOSFET is paralleled by an ideal $0\text{-V}$ freewheeling diode (`Rds_2` shunted during $(1-D)T_s$, so $R_{ds,2}^{\mathrm{eff}} = 0$). With this exact SSA formulation (`include_parasitic_drop=True`), analytic responses match all 4,723 PSIM AC-sweep curves to **0.0917% median relative $L_2$ error**.

### Component Multiplier Bands (`data/buck/component_ranges.json`)
| Component | Nominal | Healthy Band $\mathcal{I}_j^{\mathrm{norm}}$ | Degraded Band $\mathcal{I}_j^{\mathrm{anom}}$ | Physical Role |
|-----------|---------|---------------------------------------------|----------------------------------------------|---------------|
| `Cout` | $100\ \mu\text{F}$ | $[0.80, 1.20]\times$ | $[0.30, 0.70]\times$ | Filter capacitance loss / derating |
| `Esr_C` | $20\text{ m}\Omega$ | $[0.50, 2.00]\times$ | $[3.00, 8.00]\times$ | Electrolyte dry-out (ESR zero shift) |
| `Lout` | $200\ \mu\text{H}$ | $[0.80, 1.20]\times$ | $[0.40, 0.70]\times$ | Core saturation / turn degradation |
| `Esr_L` | $100\text{ m}\Omega$ | $[0.85, 1.40]\times$ | $[2.00, 6.00]\times$ | Winding resistance / thermal drift |
| `Rds_1` | $10\text{ m}\Omega$ | $[1.00, 2.20]\times$ | $[3.00, 15.00]\times$ | High-side MOSFET bond-wire / die-attach aging |
| `Rout` | $4.0\ \Omega$ | $[0.50, 2.00]\times$ | `null` (operating point) | Healthy load envelope ($2\ \Omega$ to $8\ \Omega$, $Q$ variation) |

### Derived Features (can extract)
- DC Gain ($|H(0)|$)
- Bandwidth ($-3\text{ dB}$ point)
- Phase margin & Gain margin
- Resonance frequency $f_0 = \frac{1}{2\pi\sqrt{LC}}$ and Quality factor $Q$
- ESR zero $f_z = \frac{1}{2\pi r_C C}$ and high-frequency roll-off rate

---

## 2. Time-Domain Waveforms

### Primary Signals (from PSIM simulation or hardware)

| Signal | Description | Fault Sensitivity |
|--------|-------------|-------------------|
| **Output Voltage v_out(t)** | DC output with ripple | Capacitor degradation, load changes |
| **Inductor Current i_L(t)** | Triangular waveform | Inductor faults, DCM operation |
| **Input Current i_in(t)** | Pulsed waveform | Switch faults, input issues |
| **Switch Node Voltage v_sw(t)** | PWM waveform | Switch degradation, dead-time |

### Extractable Features

```python
time_domain_features = {
    # Statistical Features
    'mean': np.mean(signal),
    'std': np.std(signal),
    'rms': np.sqrt(np.mean(signal**2)),
    'peak_to_peak': np.max(signal) - np.min(signal),
    'crest_factor': np.max(np.abs(signal)) / np.sqrt(np.mean(signal**2)),
    'form_factor': np.sqrt(np.mean(signal**2)) / np.mean(np.abs(signal)),
    'skewness': scipy.stats.skew(signal),
    'kurtosis': scipy.stats.kurtosis(signal),
    
    # Shape Features
    'ripple_amplitude': extract_ripple(signal),
    'duty_cycle': estimate_duty_cycle(signal),
    'rise_time': measure_rise_time(signal),
    'fall_time': measure_fall_time(signal),
}
```

---

## 3. Ripple Analysis

### Voltage Ripple
The output voltage ripple is a critical health indicator:

$$\Delta V_{out} = \frac{I_L \cdot (1-D)}{f_s \cdot C_{out}}$$

Where:
- $I_L$ = Inductor current
- $D$ = Duty cycle
- $f_s$ = Switching frequency
- $C_{out}$ = Output capacitance

**Anomaly indicators:**
- Increased ripple → Capacitor degradation (↓ Cout or ↑ ESR)
- Asymmetric ripple → Switch imbalance
- Subharmonic ripple → Instability

### Current Ripple
Inductor current ripple:

$$\Delta I_L = \frac{V_{in} \cdot D}{f_s \cdot L}$$

**Anomaly indicators:**
- Increased ripple → Inductance degradation
- Distorted waveform → Core saturation
- DCM operation → Load issues

---

## 4. Component Health Indicators

### Capacitor Health
| Parameter | Normal Range | Degraded | Failed |
|-----------|-------------|----------|--------|
| **Capacitance (C)** | ±10% | -20% to -30% | <-30% |
| **ESR** | Datasheet | 2-3x increase | >5x |
| **Ripple Current Rating** | Within spec | Approaching limit | Exceeded |

### Inductor Health
| Parameter | Normal Range | Degraded | Failed |
|-----------|-------------|----------|--------|
| **Inductance (L)** | ±10% | -20% to -30% | <-30% |
| **ESR (Esr_L)** | Datasheet | 2x increase | >3x |
| **Saturation Current** | Within spec | Reduced | Severely reduced |

### Switch Health (MOSFETs)
| Parameter | Normal Range | Degraded | Failed |
|-----------|-------------|----------|--------|
| **Rds_on** | Datasheet | 1.5-2x increase | >3x or open |
| **Switching time** | Datasheet | 2x slower | Very slow |
| **Gate threshold** | Datasheet | Shifted | Out of range |

---

## 5. Frequency Domain Features (from Time-Domain Signals)

### FFT-based Features

```python
def extract_frequency_features(signal, fs):
    """Extract frequency domain features from time-domain signal."""
    # FFT
    fft_result = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1/fs)
    magnitude = np.abs(fft_result)
    
    features = {
        'fundamental_magnitude': magnitude[fundamental_idx],
        'dc_component': magnitude[0],
        'thd': calculate_thd(magnitude),
        'harmonic_2': magnitude[2 * fundamental_idx],
        'harmonic_3': magnitude[3 * fundamental_idx],
        'spectral_centroid': np.sum(freqs * magnitude) / np.sum(magnitude),
        'spectral_spread': calculate_spread(freqs, magnitude),
        'spectral_entropy': calculate_entropy(magnitude),
    }
    return features
```

### Key Harmonics
- **Fundamental (f_sw)**: Switching frequency component
- **2nd Harmonic**: Often indicates asymmetry
- **Subharmonics (f_sw/2)**: Instability indicator
- **Sideband components**: Load modulation

---

## 6. Dynamic Response Features

### Step Response Analysis
| Feature | Description | Calculation |
|---------|-------------|-------------|
| **Settling Time** | Time to reach ±2% of final value | From step response |
| **Overshoot** | Peak deviation / final value | % |
| **Rise Time** | 10% to 90% transition time | seconds |
| **Delay Time** | Time to reach 50% of final value | seconds |

### Load Transient Response
- Voltage dip/spike magnitude
- Recovery time
- Ringing frequency
- Damping ratio

---

## 7. Efficiency and Loss Features

### Power Efficiency
$$\eta = \frac{P_{out}}{P_{in}} = \frac{V_{out} \cdot I_{out}}{V_{in} \cdot I_{in}}$$

### Loss Breakdown
| Loss Type | Depends On | Anomaly Indicator |
|-----------|-----------|-------------------|
| **Conduction (Switch)** | Rds_on | Increased Rds |
| **Switching Loss** | Rise/fall time | Switch degradation |
| **Inductor Core Loss** | Frequency, flux | Core saturation |
| **Inductor Copper Loss** | ESR_L | Winding damage |
| **Capacitor ESR Loss** | ESR, ripple current | Cap degradation |

---

## 8. Multi-Modal Data Architecture

### Proposed Data Structure

```python
class ConverterSample:
    """Multi-modal data sample from Buck converter."""
    
    # Transfer Function (existing)
    transfer_function: np.ndarray  # [101, 3] - freq, amp, phase
    
    # Time-Domain Waveforms (new)
    output_voltage: np.ndarray     # [N_samples]
    inductor_current: np.ndarray   # [N_samples]
    input_current: np.ndarray      # [N_samples]
    switch_voltage: np.ndarray     # [N_samples]
    
    # Component Parameters (labels/ground truth)
    parameters: dict  # {Cout, Rds_1, Rds_2, Esr_L, Esr_C}
    
    # Derived Features (computed)
    time_features: dict    # Statistical features
    freq_features: dict    # FFT-based features
    ripple_features: dict  # Ripple analysis
    efficiency: float      # Power efficiency
    
    # Metadata
    operating_point: dict  # {Vin, Vout, Iout, fs, D}
    timestamp: datetime
```

---

## 9. Anomaly Detection Model Options

### Model 1: Transfer Function Only (Current)
- **Input**: [101, 2] - Amplitude & Phase
- **Models**: Conv1D-AE, LSTM-AE, VAE, Transformer-AE
- **Detects**: Frequency response anomalies

### Model 2: Multi-Modal Fusion
- **Input**: Transfer function + Time-domain features
- **Architecture**: Dual-encoder with fusion layer
- **Detects**: Broader range of anomalies

### Model 3: Multivariate Time Series
- **Input**: Multiple synchronized waveforms
- **Architecture**: 2D Conv-AE or Multi-channel LSTM
- **Detects**: Waveform shape anomalies

### Model 4: Feature-Level Detection
- **Input**: Extracted features (statistical + frequency)
- **Architecture**: Dense AE or VAE
- **Detects**: Parametric anomalies

---

## 10. Implementation Priority

### Phase 1: Feature Extraction (Immediate)
1. ✅ Add transfer function derived features (DC gain, bandwidth, margins)
2. 🔲 Create feature extraction module for time-domain signals
3. 🔲 Implement ripple analysis functions

### Phase 2: Enhanced Models (Short-term)
1. 🔲 Multi-input autoencoder (transfer function + features)
2. 🔲 Contrastive learning with synthetic anomalies
3. 🔲 Physics-informed loss functions

### Phase 3: Advanced Architectures (Medium-term)
1. 🔲 Mamba-based autoencoder
2. 🔲 KAN layers for interpretability
3. 🔲 Multi-modal fusion network

---

## References

1. **PINN for Buck Converter Parameter Identification** (arXiv:2504.20528)
   - Uses loss landscape to identify reliable vs unreliable parameters
   - Inductance more reliable than inductor resistance

2. **Data-Driven Condition Monitoring for MMC Capacitors** (arXiv:2404.13399)
   - PSO-based estimation of capacitance AND ESR
   - Uses voltage prediction error for health assessment

3. **VSI Switch Fault Classification using STFT** (arXiv:2111.06566)
   - Time-frequency analysis for fault classification
   - 98.3% accuracy on open/short circuit faults

4. **Unsupervised Clustering for Power System Faults** (arXiv:2505.17763)
   - FFT features + K-Means clustering
   - Voltage and current waveform analysis

5. **LLM for DC-Link Capacitor Ripple Prediction** (arXiv:2407.01724)
   - Fine-tuned GPT-3.5 for ripple prediction
   - Minimal invasive measurements

6. **Multimodal Hypergraph Contrastive Network for Fault Diagnosis** (arXiv:2510.15547)
   - 99.82% accuracy with multimodal sensor fusion
   - Hypergraph models inter-modal dependencies

---

*Last updated: January 2026*
