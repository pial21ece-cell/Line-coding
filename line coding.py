import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd

# Set global plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (12, 10)

# ==========================================
# 1. LINE ENCODING FUNCTIONS (NO TOOLBOX)
# ==========================================

def encode_unipolar_nrz(bits, spb):
    """Unipolar NRZ: 1 -> +1, 0 -> 0"""
    return np.repeat(np.where(bits == 1, 1.0, 0.0), spb)

def encode_polar_nrz(bits, spb):
    """Polar NRZ: 1 -> +1, 0 -> -1"""
    return np.repeat(np.where(bits == 1, 1.0, -1.0), spb)

def encode_polar_rz(bits, spb):
    """Polar RZ: 1 -> [+1, 0], 0 -> [-1, 0]"""
    wave = []
    half = spb // 2
    for b in bits:
        val = 1.0 if b == 1 else -1.0
        wave.extend([val] * half + [0.0] * (spb - half))
    return np.array(wave)

def encode_manchester(bits, spb):
    """Manchester: 1 -> [+1, -1], 0 -> [-1, +1]"""
    wave = []
    half = spb // 2
    for b in bits:
        val = 1.0 if b == 1 else -1.0
        wave.extend([val] * half + [-val] * (spb - half))
    return np.array(wave)

def encode_diff_manchester(bits, spb):
    """Differential Manchester: Transition in mid-bit; 0 has start transition, 1 does not"""
    wave = []
    half = spb // 2
    curr = 1.0
    for b in bits:
        if b == 0:
            curr = -curr  # Transition at start for bit 0
        wave.extend([curr] * half)
        curr = -curr      # Mid-bit transition
        wave.extend([curr] * (spb - half))
    return np.array(wave)

def encode_ami(bits, spb):
    """Bipolar AMI: 0 -> 0, 1 -> alternating +1 and -1"""
    wave = []
    last_polarity = -1.0
    for b in bits:
        if b == 1:
            last_polarity = -last_polarity
            wave.extend([last_polarity] * spb)
        else:
            wave.extend([0.0] * spb)
    return np.array(wave)

# Map encoders to a dictionary
ENCODERS = {
    'Unipolar NRZ': encode_unipolar_nrz,
    'Polar NRZ': encode_polar_nrz,
    'Polar RZ': encode_polar_rz,
    'Manchester': encode_manchester,
    'Diff Manchester': encode_diff_manchester,
    'Bipolar AMI': encode_ami
}

# ==========================================
# 2. TASK 19 & MANDATORY VALIDATION (8-BIT WORD)
# ==========================================
test_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])  # Specified 8-bit test word
spb = 100                                        # Samples per bit
fs = 1000                                       # Sampling rate (Hz)
t = np.arange(len(test_bits) * spb) / fs

fig, axes = plt.subplots(len(ENCODERS), 1, figsize=(12, 10), sharex=True)
fig.suptitle('Aligned Line-Code Waveforms (Mandatory 8-bit Validation)', fontsize=14, fontweight='bold')

waveforms_8bit = {}
for i, (name, func) in enumerate(ENCODERS.items()):
    wave = func(test_bits, spb)
    waveforms_8bit[name] = wave
    axes[i].plot(t, wave, linewidth=2)
    axes[i].set_ylabel(name, fontsize=9)
    axes[i].set_ylim(-1.5, 1.5)
    axes[i].grid(True, linestyle='--')

    # Draw bit boundary guides
    for b_idx in range(len(test_bits) + 1):
        axes[i].axvline(b_idx * spb / fs, color='gray', linestyle=':', alpha=0.5)

axes[-1].set_xlabel('Time (s)', fontsize=11)
plt.tight_layout()
plt.show()

# ==========================================
# 3. TASK 21: DC CONTENT & RUNNING DIGITAL SUM
# ==========================================
fig, ax = plt.subplots(figsize=(10, 5))
print("--- AVERAGE LEVEL (DC CONTENT) FOR 8-BIT WORD ---")

for name, wave in waveforms_8bit.items():
    avg_level = np.mean(wave)
    print(f"{name:20s}: Average Level = {avg_level:.4f} V")

    # Cumulative discrete sum (RDS) per sample normalized by samples per bit
    rds = np.cumsum(wave) / spb
    ax.plot(t, rds, label=name, linewidth=1.8)

ax.set_title('Running Digital Sum (RDS) Comparison', fontsize=12, fontweight='bold')
ax.set_xlabel('Time (s)')
ax.set_ylabel('RDS Accumulation')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ==========================================
# 4. TASK 20: ESTIMATE PSD USING WELCH'S METHOD
# ==========================================
np.random.seed(42)
long_bits = np.random.randint(0, 2, 2000)  # Random sequence for PSD evaluation

plt.figure(figsize=(10, 6))
for name, func in ENCODERS.items():
    wave_long = func(long_bits, spb)

    # Welch's Method for PSD estimation
    freqs, psd = signal.welch(wave_long, fs=fs, nperseg=1024)

    # Normalize PSD to peak (0 dB max)
    psd_norm = psd / np.max(psd)
    psd_db = 10 * np.log10(psd_norm + 1e-12)  # Avoid log(0)

    plt.plot(freqs[:200], psd_db[:200], label=name, linewidth=1.5)

plt.title("Normalized Power Spectral Density (PSD) using Welch's Method", fontsize=12, fontweight='bold')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized Power (dB)")
plt.ylim(-40, 5)
plt.legend()
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.show()

# ==========================================
# 5. TASK 22: LONG-RUN BEHAVIOR ANALYSIS
# ==========================================
# Sequence with a long run of identical bits (30 ones followed by 30 zeros)
long_run_bits = np.array([1]*30 + [0]*30)
t_long_run = np.arange(len(long_run_bits) * spb) / fs

fig, axes = plt.subplots(2, 1, figsize=(11, 6))
fig.suptitle('Long-Run Behavior Test (30 Ones + 30 Zeros)', fontsize=12, fontweight='bold')

for name in ['Polar NRZ', 'Manchester', 'Bipolar AMI']:
    wave = ENCODERS[name](long_run_bits, spb)
    rds = np.cumsum(wave) / spb

    axes[0].plot(t_long_run, wave, label=name)
    axes[1].plot(t_long_run, rds, label=f"{name} RDS")

axes[0].set_title('Waveform Behavior')
axes[0].set_ylabel('Amplitude')
axes[0].legend(loc='upper right')

axes[1].set_title('Running Digital Sum (Baseline Wander Hazard)')
axes[1].set_ylabel('Accumulated Sum')
axes[1].set_xlabel('Time (s)')
axes[1].legend(loc='upper left')

plt.tight_layout()
plt.show()

# ==========================================
# 6. SUMMARY COMPARISON TABLE
# ==========================================
summary_data = {
    "Line Code": ["Unipolar NRZ", "Polar NRZ", "Polar RZ", "Manchester", "Diff Manchester", "Bipolar AMI"],
    "DC Component": ["High (Non-zero)", "Depends on data", "Depends on data", "Zero (DC-free)", "Zero (DC-free)", "Zero (DC-free)"],
    "Bandwidth Requirement": ["Low ($R_b$)", "Low ($R_b$)", "High ($2R_b$)", "High ($2R_b$)", "High ($2R_b$)", "Low ($R_b$)"],
    "Polarity": ["Unipolar", "Polar", "Polar", "Bipolar", "Bipolar", "Pseudoternary"],
    "Self-Clocking Capability": ["Poor", "Poor", "Moderate", "Excellent", "Excellent", "Poor (on long 0s)"]
}

df = pd.DataFrame(summary_data)
print("\n" + "="*80)
print("LINE CODING COMPARISON SUMMARY")
print("="*80)
display(df)
