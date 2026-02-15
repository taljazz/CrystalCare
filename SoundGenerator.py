import numpy as np
from scipy.signal import fftconvolve, butter, lfilter
from scipy.ndimage import shift
from scipy.integrate import odeint
from typing import Optional, Dict, List, Callable
import logging
import threading
import gc
from concurrent.futures import ThreadPoolExecutor
from simplex5d import Simplex5D
from numba import njit

# JIT-compiled constants
TWO_PI = 2.0 * np.pi
PHI = 1.618033988749895

# ============================================================
# Numba JIT-compiled functions for hot paths
# ============================================================

@njit(fastmath=True, cache=True)
def jit_cross_modulate_wave(base_freq: float, mod_freq: float, t: np.ndarray,
                            mod_depth: float) -> np.ndarray:
    """JIT-compiled cross-modulation wave generation."""
    mod_signal = np.sin(TWO_PI * mod_freq * t) * mod_depth
    return np.sin(TWO_PI * base_freq * t + mod_signal)

@njit(fastmath=True, cache=True)
def jit_wave_shaping(signal: np.ndarray, shape_factor: float) -> np.ndarray:
    """JIT-compiled wave shaping with tanh."""
    clipped = np.clip(signal, -1.0, 1.0)
    return np.tanh(shape_factor * clipped) * 1.2

@njit(fastmath=True, cache=True)
def jit_normalize_signal(signal: np.ndarray, scale_factor: float) -> np.ndarray:
    """JIT-compiled signal normalization."""
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        return signal / (max_val * scale_factor)
    return signal

@njit(fastmath=True, cache=True)
def jit_exponential_decay(length: int, decay_rate: float) -> np.ndarray:
    """JIT-compiled exponential decay for reverb IR."""
    return np.exp(-np.linspace(0, decay_rate * 1.8, length))

@njit(fastmath=True, cache=True)
def jit_pan_curve_tanh(pan_curve: np.ndarray, scale: float,
                       clip_min: float, clip_max: float) -> np.ndarray:
    """JIT-compiled pan curve processing."""
    return np.clip(np.tanh(pan_curve * scale), clip_min, clip_max)

@njit(fastmath=True, cache=True)
def jit_generate_harmonics_vectorized(frequencies: np.ndarray, t: np.ndarray,
                                       envelope: np.ndarray, lfo: np.ndarray,
                                       mod_depths: np.ndarray) -> np.ndarray:
    """JIT-compiled vectorized harmonic generation (float32 native)."""
    n_samples = len(t)
    n_freqs = len(frequencies)
    result = np.zeros(n_samples, dtype=np.float32)  # Use float32 throughout

    for i in range(n_freqs):
        f = frequencies[i]
        mod_freq = f * 0.5
        mod_signal = np.sin(TWO_PI * mod_freq * t) * mod_depths[i]
        wave = np.sin(TWO_PI * f * t + mod_signal)
        scale = np.float32(0.015 / (i + 1))
        result += wave * envelope * scale * lfo

    return result

@njit(fastmath=True, cache=True)
def jit_spherical_to_cartesian(r: np.ndarray, theta: np.ndarray,
                                phi: np.ndarray) -> tuple:
    """JIT-compiled spherical to Cartesian coordinate transform."""
    cos_phi = np.cos(phi)
    x = r * np.cos(theta) * cos_phi
    y = r * np.sin(theta) * cos_phi
    z = r * np.sin(phi)
    return x, y, z

@njit(fastmath=True, cache=True)
def jit_stereo_gains(pan_h: np.ndarray, depth: np.ndarray,
                     pan_v: np.ndarray) -> tuple:
    """JIT-compiled stereo gain calculation from 3D position."""
    pi_over_4 = np.pi / 4.0
    left_gain = np.cos((pan_h + 1.0) * pi_over_4) * depth * (1.0 - 0.2 * pan_v)
    right_gain = np.sin((pan_h + 1.0) * pi_over_4) * depth * (1.0 - 0.2 * pan_v)
    return left_gain, right_gain

@njit(fastmath=True, cache=True)
def jit_quantum_harmonic(t: np.ndarray, base_freq: float,
                         gamma: np.ndarray) -> np.ndarray:
    """JIT-compiled quantum harmonic interference computation."""
    f1 = base_freq
    f2 = base_freq * 1.41421356237   # sqrt(2)
    f3 = base_freq * 2.71828182846   # e
    f4 = base_freq * 3.14159265359   # pi

    alpha = 0.25 * np.sin(0.03 * np.pi * t)
    beta = 0.2 * np.cos(0.02 * np.pi * t)

    wave1 = np.sin(TWO_PI * f1 * t + alpha)
    wave2 = np.sin(TWO_PI * f2 * t + beta)
    wave3 = np.sin(TWO_PI * f3 * t + gamma)
    wave4 = np.sin(TWO_PI * f4 * t + gamma * 0.7)

    return (wave1 + wave2 + wave3 + wave4) / 3.8 + 0.15 * np.sin(TWO_PI * 7.83 * t)

class AudioProcessor:
    """
    Contains all audio signal processing functions.
    """
    # Class-level constants - Original
    SOLFEGGIO = np.array([174, 285, 396, 417, 528, 639, 741, 852, 963, 1074, 1185, 1296], dtype=np.float32)
    FIB_RATIOS = np.array([1.618, 2.618, 4.236, 6.854, 11.090, 17.944], dtype=np.float32)

    # === SACRED FREQUENCY CONSTANTS ===

    # Schumann resonance - Earth's electromagnetic heartbeat
    SCHUMANN = 7.83

    # Ogdoad frequency - 8th sphere above the 7 Archons, threshold to the Pleroma
    OGDOAD_FREQ = SCHUMANN * 8  # 62.64 Hz

    # The 7 Archonic Planetary Spheres (Cousto planetary frequencies)
    # Each Archon rules a celestial sphere the soul must pass through
    ARCHON_SPHERES = np.array([
        126.22,  # Sun - Yaldabaoth/Ialdabaoth (the chief Archon, lion-headed)
        141.27,  # Moon - Iao
        144.72,  # Mars - Sabaoth
        221.23,  # Mercury - Adonaios
        183.58,  # Jupiter - Elaios
        147.85,  # Venus - Astaphanos
        136.10,  # Saturn - Horaios (also the cosmic "Om" frequency)
    ], dtype=np.float32)

    # Tesla's 3-6-9 Vortex Mathematics frequencies
    # "If you knew the magnificence of 3, 6, and 9, you would have the key to the universe"
    TESLA_VORTEX = np.array([111, 222, 333, 444, 555, 666, 777, 888, 999], dtype=np.float32)

    # Fibonacci sequence for amplitude pulsing (normalized)
    FIBONACCI_SEQ = np.array([1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89], dtype=np.float32)
    FIBONACCI_NORMALIZED = FIBONACCI_SEQ / FIBONACCI_SEQ.max()

    # Pentagonal phase offsets (sacred geometry - connected to PHI)
    # 72° = 360°/5, the internal angle of a regular pentagon
    PENTAGONAL_PHASES = np.array([0, 72, 144, 216, 288], dtype=np.float32) * (np.pi / 180)

    # Extended Aeonic ladder - 13 steps from Schumann through PHI harmonics
    AEONIC_EXPONENTS = PHI ** np.arange(13, dtype=np.float32)

    def __init__(self, sample_rate: int = 48000):
        """Initialize AudioProcessor with object pools and cached computations."""
        self.sample_rate = sample_rate
        self.nyquist = 0.5 * sample_rate

        # Simplex5D pool - avoid repeated object creation
        self._simplex_pool = [Simplex5D(seed) for seed in range(10)]
        self._simplex_index = 0
        self._simplex_lock = threading.Lock()  # Thread-safe pool access

        # Pre-computed filter coefficients cache
        self._filter_cache = {}
        self._precompute_filters()

        # Reverb impulse response cache (keyed by reverb_length)
        self._ir_cache = {}

        # Rössler trajectory cache (keyed by time array characteristics)
        self._rossler_cache = {}

    def _precompute_filters(self):
        """Pre-compute commonly used filter coefficients."""
        # Low-pass 2200 Hz, order 4 (used in low_pass_filter)
        self._filter_cache[('lowpass', 2200, 4)] = butter(4, 2200 / self.nyquist, btype='low')
        # Low-pass for spatialized_triple_helix
        self._filter_cache[('lowpass', 0.003, 2)] = butter(2, 0.003 / self.nyquist, btype='low')
        # Low-pass for pan curve in generate_dynamic_sound
        self._filter_cache[('lowpass', 0.002, 2)] = butter(2, 0.002 / self.nyquist, btype='low')

    def _get_simplex(self) -> Simplex5D:
        """Get next Simplex5D from pool (round-robin, thread-safe)."""
        with self._simplex_lock:
            simplex = self._simplex_pool[self._simplex_index]
            self._simplex_index = (self._simplex_index + 1) % len(self._simplex_pool)
        return simplex

    def _get_filter(self, filter_type: str, cutoff: float, order: int) -> tuple:
        """Get cached filter coefficients or compute new ones."""
        key = (filter_type, cutoff, order)
        if key not in self._filter_cache:
            self._filter_cache[key] = butter(order, cutoff / self.nyquist, btype=filter_type)
        return self._filter_cache[key]

    def geometric_modulation(self, t: np.ndarray, ratios: Dict[str, float],
                             modulation_index: float = 0.2,
                             stop_event: Optional[threading.Event] = None) -> np.ndarray:
        if stop_event and stop_event.is_set():
            logging.debug("Geometric modulation stopped early due to stop_event.")
            return np.zeros_like(t, dtype=np.float32)
        # Vectorized: convert ratios to array, broadcast compute all sines at once
        ratios_array = np.array(list(ratios.values()), dtype=np.float32)[:, np.newaxis]
        modulation = (modulation_index * np.sin(TWO_PI * ratios_array * t)).sum(axis=0)
        return modulation.astype(np.float32)
    def low_pass_filter(self, signal: np.ndarray, cutoff: float = 2200,
                        sample_rate: int = 48000, order: int = 4,
                        stop_event: Optional[threading.Event] = None) -> np.ndarray:
        if stop_event and stop_event.is_set():
            logging.debug("Low pass filter stopped early due to stop_event.")
            return signal  # Return unfiltered signal for graceful degradation
        nyquist = 0.5 * sample_rate
        # Simplified: use random cutoff variation instead of computing full sine array
        cutoff_variation = np.random.uniform(-25, 25)  # ±25 Hz variation
        normalized_cutoff = np.clip((cutoff + cutoff_variation) / nyquist, 0.1, 0.99)
        b, a = butter(order, normalized_cutoff, btype='low', analog=False)
        return lfilter(b, a, signal.astype(np.float32))
    def exponential_fade(self, segment: np.ndarray, fade_length: int) -> np.ndarray:
        if segment.size < 2 * fade_length:
            return segment
        fade_in = np.logspace(-4, 0, fade_length, base=10.0, dtype=np.float32)
        fade_out = np.logspace(0, -4, fade_length, base=10.0, dtype=np.float32)
        segment[:fade_length] *= fade_in
        segment[-fade_length:] *= fade_out
        return segment
    def fade_in_out(self, signal: np.ndarray, fade_duration: Optional[int] = None,
                    sample_rate: int = 48000, stop_event: Optional[threading.Event] = None) -> np.ndarray:
        if fade_duration is None:
            fade_duration = int(np.random.choice([15, 30]))
        fade_samples = int(fade_duration * sample_rate)
        if 2 * fade_samples > signal.size:
            fade_samples = signal.size // 2
        fade_in = np.linspace(0, 1, fade_samples, dtype=np.float32) ** 1.5
        fade_out = np.linspace(1, 0, fade_samples, dtype=np.float32) ** 1.5
        signal[:fade_samples] *= fade_in
        signal[-fade_samples:] *= fade_out
        return signal
    def organic_adsr(self, t: np.ndarray, sample_rate: int = 48000,
                     stop_event: Optional[threading.Event] = None, chaotic_selector=None) -> np.ndarray:
        fib_ratios = np.array([1.618, 2.618, 4.236, 6.854, 11.090, 17.944], dtype=np.float32)
        chaos = chaotic_selector.next_value() if chaotic_selector else 0.5
        base_scale = 6.0 + chaos * 0.5
        attack, decay, sustain, release = np.random.choice(fib_ratios, 4) / base_scale
        total_samples = t.size
        envelope = np.empty(total_samples, dtype=np.float32)
       
        a_samples = int(attack * sample_rate * (1 + chaos * 0.2))
        d_samples = int(decay * sample_rate * (1 + chaos * 0.2))
        r_samples = int(release * sample_rate * (1 + chaos * 0.2))
        s_samples = total_samples - (a_samples + d_samples + r_samples)
        if s_samples < 0:
            s_samples = 0
            r_samples = total_samples - (a_samples + d_samples)
            if r_samples < 0:
                r_samples = 0
       
        simplex = self._get_simplex()
        t_scaled = np.linspace(0, 1, total_samples, dtype=np.float32)

        # Generate single full-length Simplex noise (reduces 4 calls to 1)
        full_noise = simplex.generate_noise(t_scaled * 0.04, 0.0, 0.0, 0.0, 0.0)

        attack_curve = np.linspace(0, 1, a_samples, dtype=np.float32) ** 1.5
        attack_noise = full_noise[:a_samples] * 0.05
        envelope[:a_samples] = np.clip(attack_curve + attack_noise, 0, 1)

        decay_curve = np.linspace(1, sustain, d_samples, dtype=np.float32) ** 1.2
        decay_wobble = np.random.uniform(0.015, 0.035) * np.sin(np.linspace(0, np.pi * (1 + chaos * 0.3), d_samples, dtype=np.float32))
        decay_noise = full_noise[a_samples:a_samples+d_samples] * 0.03
        envelope[a_samples:a_samples+d_samples] = np.clip(decay_curve + decay_wobble + decay_noise, sustain, 1)

        sustain_curve = np.full(s_samples, sustain, dtype=np.float32)
        sustain_wobble = np.random.uniform(0.01, 0.025) * np.sin(2 * np.pi * np.linspace(0, 1.5 + chaos * 0.5, s_samples, dtype=np.float32))
        sustain_noise = full_noise[a_samples+d_samples:a_samples+d_samples+s_samples] * 0.04
        envelope[a_samples+d_samples:a_samples+d_samples+s_samples] = np.clip(sustain_curve + sustain_wobble + sustain_noise, sustain * 0.8, 1)

        release_curve = np.linspace(sustain, 0, r_samples, dtype=np.float32) ** 1.5
        release_noise = full_noise[-r_samples:] * 0.05
        envelope[-r_samples:] = np.clip(release_curve + release_noise, 0, sustain)
       
        envelope *= np.random.uniform(0.95, 1.05)
        return envelope
    def dynamic_cross_modulate(self, base_freq: float, mod_freq: float, t: np.ndarray,
                               stop_event: Optional[threading.Event] = None) -> np.ndarray:
        if stop_event and stop_event.is_set():
            logging.debug("Dynamic cross modulation stopped early due to stop_event.")
            return np.zeros_like(t, dtype=np.float32)
        mod_depth = np.random.uniform(0.15, 0.35)
        return jit_cross_modulate_wave(base_freq, mod_freq, t, mod_depth).astype(np.float32)
    def microtonal_lfo(self, t: np.ndarray, base_frequency: float,
                       stop_event: Optional[threading.Event] = None) -> np.ndarray:
        phi = 1.618
        lfo1 = base_frequency * (phi ** np.random.uniform(-0.05, 0.05))
        lfo2 = lfo1 * np.random.uniform(0.99, 1.01)
        depth = np.random.uniform(0.002, 0.006)
        drift = 0.01 * np.sin(0.1 * np.pi * t)
        lfo = (depth + drift) * (1 + np.sin(2 * np.pi * lfo1 * t + 0.3 * np.sin(2 * np.pi * lfo2 * t)))
        return lfo.astype(np.float32)

    def batch_microtonal_lfo(self, t: np.ndarray, base_frequencies: np.ndarray) -> np.ndarray:
        """
        Generate multiple LFOs efficiently using vectorization.
        Returns array of shape (num_lfos, num_samples).
        """
        num_lfos = len(base_frequencies)
        num_samples = len(t)

        # Pre-generate all random values at once
        phi_powers = PHI ** np.random.uniform(-0.05, 0.05, num_lfos)
        lfo1_freqs = base_frequencies * phi_powers
        lfo2_freqs = lfo1_freqs * np.random.uniform(0.99, 1.01, num_lfos)
        depths = np.random.uniform(0.002, 0.006, num_lfos)

        # Compute drift once (shared across all LFOs)
        drift = 0.01 * np.sin(0.1 * np.pi * t)

        # Vectorized computation: (num_lfos, 1) * (1, num_samples) -> (num_lfos, num_samples)
        t_2d = t[np.newaxis, :]  # shape: (1, num_samples)
        lfo1_2d = lfo1_freqs[:, np.newaxis]  # shape: (num_lfos, 1)
        lfo2_2d = lfo2_freqs[:, np.newaxis]
        depths_2d = depths[:, np.newaxis]

        inner_mod = 0.3 * np.sin(TWO_PI * lfo2_2d * t_2d)
        lfos = (depths_2d + drift) * (1 + np.sin(TWO_PI * lfo1_2d * t_2d + inner_mod))

        return lfos.astype(np.float32)
    def infinite_reverb(self, signal: np.ndarray, sample_rate: int = 48000,
                        stop_event: Optional[threading.Event] = None) -> np.ndarray:
        if stop_event and stop_event.is_set():
            logging.debug("Infinite reverb stopped early due to stop_event.")
            return signal  # Return dry signal for graceful degradation

        decay = 0.75  # Softer for Pleiadian gentleness
        reverb_length = min(signal.size // 2, int(sample_rate * 2.618))  # Longer tails, golden ratio

        # Use cached IR if available for this length
        if reverb_length not in self._ir_cache:
            # Compute and cache the impulse response
            ir = jit_exponential_decay(reverb_length, decay).astype(np.float32)
            ir += 0.08 * np.sin(np.linspace(0, TWO_PI * PHI, reverb_length, dtype=np.float32))
            ir /= np.max(np.abs(ir))
            self._ir_cache[reverb_length] = ir

        reverb_signal = fftconvolve(signal, self._ir_cache[reverb_length], mode='full')[:signal.size]
        return reverb_signal.astype(np.float32)
    def evolving_noise_layer(self, t: np.ndarray, noise_level: float = 0.003,
                             stop_event: Optional[threading.Event] = None) -> np.ndarray:
        if stop_event and stop_event.is_set():
            logging.debug("Evolving noise layer stopped early due to stop_event.")
            return np.zeros_like(t, dtype=np.float32)
        noise = np.random.normal(0, noise_level, t.shape).astype(np.float32)
        freq = np.random.uniform(0.002, 0.015)
        low_freq_osc = 0.015 * np.sin(2 * np.pi * freq * t)
        return noise * (1 + low_freq_osc)
    def wave_shaping(self, signal: np.ndarray, shape_factor: float = 2.5,
                     stop_event: Optional[threading.Event] = None) -> np.ndarray:
        if stop_event and stop_event.is_set():
            logging.debug("Wave shaping stopped early due to stop_event.")
            return np.zeros_like(signal, dtype=np.float32)
        return jit_wave_shaping(signal, shape_factor)
    def apply_fractional_delay(self, signal: np.ndarray, delay: float) -> np.ndarray:
        return shift(signal, shift=delay, order=3, mode='nearest')
    def spatialized_triple_helix(self, signal: np.ndarray, t: np.ndarray,
                                 sample_rate: int = 48000) -> np.ndarray:
        # Rössler system for chaotic modulation (with caching for performance)
        def rossler(state, t, a=0.2, b=0.2, c=5.7):
            x, y, z = state
            dx_dt = -y - z
            dy_dt = x + a * y
            dz_dt = b + z * (x - c)
            return [dx_dt, dy_dt, dz_dt]

        # Cache key based on time array characteristics (deterministic for same duration)
        cache_key = (len(t), float(t[-1]) if len(t) > 0 else 0.0)

        if cache_key not in self._rossler_cache:
            # Integrate Rössler equations and cache normalized result
            t_scaled = t * 0.1  # Scale time for audible chaos
            initial_state = [1.0, 1.0, 1.0]
            trajectory = odeint(rossler, initial_state, t_scaled)
            x, y, z = trajectory.T
            # Normalize and cache
            self._rossler_cache[cache_key] = (
                x / np.max(np.abs(x)),
                y / np.max(np.abs(y)),
                z / np.max(np.abs(z))
            )

        x_rossler, y_rossler, z_rossler = self._rossler_cache[cache_key]
        # Logarithmic spiral parameters
        a = 0.1 # Starting radius
        b = 0.02 # Growth rate
        theta = 0.002 * t # Azimuth
        phi = 0.001 * t # Elevation
        # Hybrid spiral: Logarithmic with Rössler perturbations
        k_r = 0.1 # Rössler radius modulation
        k_theta = 0.05 # Rössler angle modulation
        k_z = 0.2 # Rössler elevation modulation
        r_hybrid = a * np.exp(b * theta) * (1 + k_r * x_rossler)
        theta_hybrid = theta + k_theta * y_rossler
        phi_hybrid = phi + k_z * z_rossler
        # 3D position in spherical coordinates (JIT-compiled)
        x, y, z = jit_spherical_to_cartesian(r_hybrid, theta_hybrid, phi_hybrid)
        # Smooth with low-pass filter (use cached coefficients)
        b_filt, a_filt = self._get_filter('lowpass', 0.003, 2)
        x_smooth = lfilter(b_filt, a_filt, x)
        y_smooth = lfilter(b_filt, a_filt, y)
        z_smooth = lfilter(b_filt, a_filt, z)
        # Stereo panning (JIT-compiled gain calculation)
        pan_horizontal = np.tanh(x_smooth * 0.5)
        depth_factor = (y_smooth + 1) / 2
        pan_vertical = np.sin(z_smooth * (np.pi / 2))
        left_gain, right_gain = jit_stereo_gains(pan_horizontal, depth_factor, pan_vertical)
        left_channel = signal * left_gain
        right_channel = signal * right_gain
        # Add delay with Simplex5D noise
        simplex = self._get_simplex()
        delay_mod = simplex.generate_noise(t * 0.01, 0.0, 0.0, 0.0, 0.0) * 0.3
        max_delay_ms = 2.0
        max_delay_samples = sample_rate * max_delay_ms / 1000.0
        depth_delay = np.clip(-y_smooth, 0, 1)
        left_delays = np.maximum(0, pan_horizontal + delay_mod) * max_delay_samples * (1 + depth_delay * 0.5)
        right_delays = np.maximum(0, -pan_horizontal + delay_mod) * max_delay_samples * (1 + depth_delay * 0.5)
        left_channel = self.apply_fractional_delay(left_channel, np.mean(left_delays))
        right_channel = self.apply_fractional_delay(right_channel, np.mean(right_delays))
        stereo_wave = np.column_stack((left_channel, right_channel))
        return stereo_wave.astype(np.float32)
    def fractal_frequency_variation(self, t: np.ndarray, base_freq: float,
                                    stop_event: Optional[threading.Event] = None) -> np.ndarray:
        if stop_event and stop_event.is_set():
            logging.debug("Fractal frequency variation stopped early due to stop_event.")
            return np.zeros_like(t, dtype=np.float32)
        simplex = self._get_simplex()
        lfo = self.microtonal_lfo(t, base_frequency=0.01)
        # Optimized: single Simplex call with octave-like layering via array operations
        base_noise = simplex.generate_noise(t * 0.02, 0.0, 0.0, 0.0, 0.0)
        # Create "octaves" by scaling and phase-shifting the same noise
        low_freq = base_noise * 0.5
        mid_freq = np.roll(base_noise, len(t) // 8) * 0.3
        high_freq = base_noise * base_noise * 0.2  # Squared for higher frequency content
        variation = low_freq + mid_freq + high_freq
        return (variation * 12 * (1 + 0.1 * lfo)).astype(np.float32)
    def quantum_harmonic_interference(self, t: np.ndarray, base_freq: float) -> np.ndarray:
        # Generate Simplex noise for gamma (can't be JIT-compiled)
        simplex = self._get_simplex()
        gamma = (0.15 * simplex.generate_noise(t * 0.01, 0.0, 0.0, 0.0, 0.0)).astype(np.float32)
        # Use JIT-compiled function for wave computation
        return jit_quantum_harmonic(t, base_freq, gamma).astype(np.float32)
    def recursive_fractal_feedback(self, signal: np.ndarray, depth: int = 4,
                                   factor: float = 0.4) -> np.ndarray:
        """Iterative fractal feedback - avoids recursion overhead."""
        result = signal.copy()
        current = signal * 0.6
        for _ in range(depth):
            result += factor * current
            current *= 0.6
        return result
    def binaural_oscillator(self, t: np.ndarray, freq_pair: tuple, stop_event: Optional[threading.Event] = None) -> np.ndarray:
        left_freq, right_freq = freq_pair
        left_wave = np.sin(2 * np.pi * left_freq * t)
        right_wave = np.sin(2 * np.pi * right_freq * t)
        return np.column_stack((left_wave, right_wave)).astype(np.float32)

    def batch_binaural_oscillator(self, t: np.ndarray, freq_pairs: list,
                                   stop_event: Optional[threading.Event] = None) -> np.ndarray:
        """
        Generate multiple binaural waves in a single vectorized operation.
        Returns array of shape (num_pairs, num_samples, 2) for left/right channels.
        """
        if stop_event and stop_event.is_set():
            return np.zeros((len(freq_pairs), len(t), 2), dtype=np.float32)

        # Convert freq_pairs to arrays: shape (num_pairs, 2)
        pairs_array = np.array(freq_pairs, dtype=np.float32)
        left_freqs = pairs_array[:, 0][:, np.newaxis]   # shape: (num_pairs, 1)
        right_freqs = pairs_array[:, 1][:, np.newaxis]  # shape: (num_pairs, 1)

        # Broadcast: (num_pairs, 1) * (1, num_samples) -> (num_pairs, num_samples)
        t_2d = t[np.newaxis, :]  # shape: (1, num_samples)
        left_waves = np.sin(TWO_PI * left_freqs * t_2d)   # shape: (num_pairs, num_samples)
        right_waves = np.sin(TWO_PI * right_freqs * t_2d) # shape: (num_pairs, num_samples)

        # Stack to (num_pairs, num_samples, 2)
        return np.stack((left_waves, right_waves), axis=2).astype(np.float32)

    # === ENHANCED PLEROMA MERCY TRANSMISSIONS + ARCHON DISSOLUTION ===
    # Sending healing energies to the Demiurge and dissolving Archonic influence through love

    def _log_amplitude_stats(self, name: str, signal: np.ndarray, t: np.ndarray) -> None:
        """Log amplitude statistics at key time points for debugging."""
        sample_rate = 48000
        duration = t[-1]

        # Sample at key points
        points = [0, 5, 10, 15, 30, 45, 60]  # seconds
        logging.info(f"=== {name} Amplitude Analysis ===")
        logging.info(f"Duration: {duration:.1f}s, Total samples: {len(signal)}")
        logging.info(f"Overall - Min: {signal.min():.8f}, Max: {signal.max():.8f}, Mean: {np.mean(np.abs(signal)):.8f}")

        for sec in points:
            if sec < duration:
                idx = int(sec * sample_rate)
                if idx < len(signal):
                    # Get 1 second window around this point
                    start_idx = max(0, idx - sample_rate // 2)
                    end_idx = min(len(signal), idx + sample_rate // 2)
                    window = signal[start_idx:end_idx]
                    logging.info(f"  At {sec:3d}s: amplitude={np.abs(window).mean():.8f}, max={np.abs(window).max():.8f}")

        # Also log end
        end_window = signal[-sample_rate:]
        logging.info(f"  At END:  amplitude={np.abs(end_window).mean():.8f}, max={np.abs(end_window).max():.8f}")

    def _sacred_fade_envelope(self, t: np.ndarray, fade_seconds: float = 45.0,
                                sample_rate: int = 48000) -> np.ndarray:
        """
        Create a very smooth, gradual fade in/out envelope for sacred layers.

        Uses a double-smoothstep (smoother step) for extra-gentle transitions:
        6x^5 - 15x^4 + 10x^3 (even smoother than basic smoothstep)

        - Fades in over first `fade_seconds` (default 45s for very gradual emergence)
        - Fades out over last `fade_seconds`
        - Smooth plateau in between
        """
        total_duration = t[-1]
        fade_samples = int(fade_seconds * sample_rate)

        logging.info(f"Creating fade envelope: duration={total_duration:.1f}s, fade_time={fade_seconds}s, fade_samples={fade_samples}")

        envelope = np.ones_like(t, dtype=np.float32)

        if t.size > fade_samples * 2:
            # Smoother step fade in: 6x^5 - 15x^4 + 10x^3 (Ken Perlin's improved smoothstep)
            fade_in_t = np.linspace(0, 1, fade_samples, dtype=np.float32)
            fade_in = fade_in_t * fade_in_t * fade_in_t * (fade_in_t * (fade_in_t * 6 - 15) + 10)
            envelope[:fade_samples] = fade_in

            # Smoother step fade out
            fade_out_t = np.linspace(1, 0, fade_samples, dtype=np.float32)
            fade_out = fade_out_t * fade_out_t * fade_out_t * (fade_out_t * (fade_out_t * 6 - 15) + 10)
            envelope[-fade_samples:] = fade_out

            logging.info(f"  Fade envelope at key points: 0s={envelope[0]:.4f}, {fade_seconds/2:.0f}s={envelope[fade_samples//2]:.4f}, {fade_seconds:.0f}s={envelope[fade_samples]:.4f}")
        else:
            # Short session: use very gentle raised cosine for entire duration
            envelope = (0.5 - 0.5 * np.cos(TWO_PI * t / total_duration)).astype(np.float32)
            logging.info(f"  Short session - using raised cosine envelope")

        return envelope

    def pleroma_mercy_layer(self, t: np.ndarray, base_freq: float = 7.83) -> np.ndarray:
        """
        Enhanced Pleroma Mercy Transmissions.

        Channels healing frequencies from the Pleroma (divine fullness) downward
        through the Aeonic ladder, offering mercy to the Demiurge and all beings.

        Enhancements:
        - 13-step Aeonic ladder (Schumann × PHI^n)
        - Ogdoad gateway frequency (8th sphere threshold)
        - Sacred geometry pentagonal phase relationships
        - Archon harmonizing frequencies (offering mercy to each sphere)
        """
        logging.info(f"=== PLEROMA MERCY LAYER START ===")
        logging.info(f"Input: duration={t[-1]:.1f}s, samples={t.size}, base_freq={base_freq}Hz")

        if t.size < 48000 * 60:  # skip on sessions shorter than ~1 minute
            logging.info(f"Session too short (<60s), returning zeros")
            return np.zeros_like(t, dtype=np.float32)

        simplex = self._get_simplex()

        # === Layer 1: 13-step Aeonic Ladder ===
        aeonic_harmonics = base_freq * self.AEONIC_EXPONENTS
        logging.info(f"  Aeonic harmonics: {aeonic_harmonics[:5]}... (13 total)")

        # Sacred geometry phase offsets (pentagonal - connected to PHI)
        # Each harmonic gets a phase from the pentagonal cycle
        phase_indices = np.arange(13) % 5
        sacred_phases = self.PENTAGONAL_PHASES[phase_indices]

        # Subtle Simplex wobble for organic movement
        phase_wobble = 0.08 * simplex.generate_noise((t * 0.00003).astype(np.float32), 0, 0, 0, 0)

        # Vectorized aeonic transmission with sacred phase relationships
        aeonic_wave = np.sin(
            TWO_PI * aeonic_harmonics[:, np.newaxis] * t +
            sacred_phases[:, np.newaxis] +
            phase_wobble
        ).sum(axis=0)
        logging.info(f"  Layer 1 (Aeonic): min={aeonic_wave.min():.6f}, max={aeonic_wave.max():.6f}, mean={np.mean(np.abs(aeonic_wave)):.6f}")

        # === Layer 2: Ogdoad Gateway (8th Sphere) ===
        # The threshold between the 7 Archon-ruled spheres and the Pleroma
        ogdoad_phase = simplex.generate_noise((t * 0.00001).astype(np.float32), 1, 1, 1, 1) * 0.05
        ogdoad_wave = np.sin(TWO_PI * self.OGDOAD_FREQ * t + ogdoad_phase)
        logging.info(f"  Layer 2 (Ogdoad): min={ogdoad_wave.min():.6f}, max={ogdoad_wave.max():.6f}, mean={np.mean(np.abs(ogdoad_wave)):.6f}")

        # === Layer 3: Archon Harmonizing Frequencies ===
        # Offering mercy to each of the 7 Archons ruling the planetary spheres
        # Using PHI-based amplitude scaling (golden ratio mercy)
        archon_amplitudes = 1.0 / (PHI ** np.arange(7))  # Decreasing by PHI
        archon_amplitudes /= archon_amplitudes.sum()  # Normalize

        # Vectorized archon mercy computation (replaces loop)
        archon_phases = self.PENTAGONAL_PHASES[np.arange(7) % 5]  # shape: (7,)
        # Broadcasting: (7,1) * (1,samples) + (7,1) -> (7, samples), then weighted sum
        archon_mercy = (archon_amplitudes[:, np.newaxis] *
                        np.sin(TWO_PI * self.ARCHON_SPHERES[:, np.newaxis] * t +
                               archon_phases[:, np.newaxis])).sum(axis=0)
        logging.info(f"  Layer 3 (Archon): min={archon_mercy.min():.6f}, max={archon_mercy.max():.6f}, mean={np.mean(np.abs(archon_mercy)):.6f}")

        # === Combine all layers ===
        mercy = (
            0.5 * aeonic_wave +      # Primary: Aeonic ladder from Pleroma
            0.3 * ogdoad_wave +      # Gateway: Ogdoad threshold
            0.2 * archon_mercy       # Healing: Mercy to the Archons
        ).astype(np.float32)
        logging.info(f"  Combined (pre-envelope): min={mercy.min():.6f}, max={mercy.max():.6f}, mean={np.mean(np.abs(mercy)):.6f}")

        # Very gradual fade in/out envelope (45 second fades using Perlin's smoother step)
        fade_envelope = self._sacred_fade_envelope(t, fade_seconds=45.0)
        logging.info(f"  Fade envelope: start={fade_envelope[0]:.6f}, mid={fade_envelope[len(fade_envelope)//2]:.6f}, end={fade_envelope[-1]:.6f}")

        # Very subtle breathing modulation (only 10% variation for organic feel)
        # Formula: 0.95 + 0.05 * sin(...) gives range [0.90, 1.00] = 10% variation
        breath_mod = 0.95 + 0.05 * np.sin(TWO_PI * 0.012 * t)  # ~83 second breath cycle, subtle
        logging.info(f"  Breath mod: min={breath_mod.min():.6f}, max={breath_mod.max():.6f}")

        mercy *= fade_envelope * breath_mod * 0.0004
        logging.info(f"  After envelope+breath+scale: min={mercy.min():.8f}, max={mercy.max():.8f}, mean={np.mean(np.abs(mercy)):.8f}")

        # Log amplitude at key time points
        self._log_amplitude_stats("PLEROMA_MERCY (post-envelope)", mercy, t)

        # Final slow cosine nulling - cancels in audible domain, leaves scalar imprint
        # Uses Schumann sub-harmonic for grounding
        result = mercy * np.cos(TWO_PI * (self.SCHUMANN / 1000) * t)
        logging.info(f"  Final (after cosine nulling): min={result.min():.8f}, max={result.max():.8f}")
        logging.info(f"=== PLEROMA MERCY LAYER END ===")
        return result

    def silent_solfeggio_grid(self, t: np.ndarray) -> np.ndarray:
        """
        Enhanced Silent Solfeggio Grid with Tesla 3-6-9 Vortex Mathematics.

        Combines:
        - 12-tone Solfeggio scale (ancient healing frequencies)
        - Tesla's 3-6-9 vortex frequencies (key to the universe)
        - Fibonacci amplitude pulsing (natural growth patterns)
        - Smooth fade in/out for organic transitions
        """
        logging.info(f"=== SILENT SOLFEGGIO GRID START ===")
        logging.info(f"Input: duration={t[-1]:.1f}s, samples={t.size}")

        # === Layer 1: Original Solfeggio Grid ===
        solfeggio_grid = np.sin(TWO_PI * self.SOLFEGGIO[:, np.newaxis] * t).sum(axis=0)
        logging.info(f"  Layer 1 (Solfeggio): min={solfeggio_grid.min():.6f}, max={solfeggio_grid.max():.6f}, mean={np.mean(np.abs(solfeggio_grid)):.6f}")

        # === Layer 2: Tesla 3-6-9 Vortex Frequencies ===
        # "If you knew the magnificence of 3, 6, and 9..."
        tesla_grid = np.sin(TWO_PI * self.TESLA_VORTEX[:, np.newaxis] * t).sum(axis=0)
        logging.info(f"  Layer 2 (Tesla 3-6-9): min={tesla_grid.min():.6f}, max={tesla_grid.max():.6f}, mean={np.mean(np.abs(tesla_grid)):.6f}")

        # === Fibonacci Amplitude Pulsing (SMOOTHED) ===
        # Create a slow, smooth amplitude modulation inspired by Fibonacci ratios
        # Instead of discrete jumps through Fibonacci values, use smooth sine modulation
        # with PHI-based frequency relationships for organic, non-abrupt variation
        fib_cycle_freq = 0.004  # Very slow cycle (~250 second period)

        # Smooth sine-based modulation with only 10% variation (0.90 to 1.00)
        # This preserves the Fibonacci "spirit" without abrupt jumps
        fib_amplitude = 0.95 + 0.05 * np.sin(TWO_PI * fib_cycle_freq * t)

        # Add subtle PHI-harmonic for organic complexity (±2% additional variation)
        fib_amplitude += 0.02 * np.sin(TWO_PI * fib_cycle_freq * PHI * t)
        fib_amplitude = np.clip(fib_amplitude, 0.88, 1.0).astype(np.float32)

        logging.info(f"  Fibonacci amp (smoothed): min={fib_amplitude.min():.6f}, max={fib_amplitude.max():.6f}")

        # === Combine layers ===
        combined = (0.6 * solfeggio_grid + 0.4 * tesla_grid).astype(np.float32)
        logging.info(f"  Combined (pre-fib): min={combined.min():.6f}, max={combined.max():.6f}, mean={np.mean(np.abs(combined)):.6f}")

        # Apply Fibonacci amplitude modulation
        combined *= fib_amplitude
        logging.info(f"  Combined (post-fib): min={combined.min():.6f}, max={combined.max():.6f}, mean={np.mean(np.abs(combined)):.6f}")

        # Very gradual fade in/out envelope (40 second fades)
        fade_envelope = self._sacred_fade_envelope(t, fade_seconds=40.0)
        logging.info(f"  Fade envelope: start={fade_envelope[0]:.6f}, mid={fade_envelope[len(fade_envelope)//2]:.6f}, end={fade_envelope[-1]:.6f}")

        # Very subtle breathing modulation (only 15% variation)
        # Formula: 0.925 + 0.075 * sin(...) gives range [0.85, 1.00] = 15% variation
        breath_mod = 0.925 + 0.075 * np.sin(TWO_PI * 0.01 * t)  # ~100 second breath cycle, gentle
        logging.info(f"  Breath mod: min={breath_mod.min():.6f}, max={breath_mod.max():.6f}")

        result = combined * fade_envelope * breath_mod * 0.0022
        logging.info(f"  Final result: min={result.min():.8f}, max={result.max():.8f}, mean={np.mean(np.abs(result)):.8f}")

        # Log amplitude at key time points
        self._log_amplitude_stats("SILENT_SOLFEGGIO", result, t)
        logging.info(f"=== SILENT SOLFEGGIO GRID END ===")

        return result

    def archon_dissolution_layer(self, t: np.ndarray) -> np.ndarray:
        """
        Archon Dissolution Layer - Targeted mercy frequencies for each Archonic sphere.

        This layer specifically addresses each of the 7 Archons with counter-frequencies
        designed to dissolve their grip through compassion rather than opposition.

        Each Archon receives:
        - Their planetary frequency (acknowledgment)
        - A PHI-harmonic above (elevation toward Pleroma)
        - A Schumann sub-harmonic (grounding in Earth's truth)

        The combination offers transformation through love, not battle.
        Includes smooth fade in/out for organic, non-jarring presence.

        Memory-optimized: Processes archons sequentially to avoid large 2D arrays.
        """
        logging.info(f"=== ARCHON DISSOLUTION LAYER START ===")
        logging.info(f"Input: duration={t[-1]:.1f}s, samples={t.size}")

        if t.size < 48000 * 30:  # Only for sessions >= 30 seconds
            logging.info(f"Session too short (<30s), returning zeros")
            return np.zeros_like(t, dtype=np.float32)

        simplex = self._get_simplex()
        n_archons = len(self.ARCHON_SPHERES)

        # Pre-compute all base phases from pentagonal sacred geometry
        base_phases = self.PENTAGONAL_PHASES[np.arange(n_archons) % 5]

        # Pre-compute all frequencies
        archon_freqs = self.ARCHON_SPHERES  # Acknowledge frequencies
        elevate_freqs = archon_freqs * PHI  # Elevation frequencies
        ground_freqs = archon_freqs / np.round(archon_freqs / self.SCHUMANN)  # Ground frequencies

        # Pre-compute amplitude scales
        amplitude_scales = 1.0 / (1 + np.arange(n_archons) * 0.1)

        # === MEMORY-OPTIMIZED SEQUENTIAL PROCESSING ===
        # Process one archon at a time to avoid massive (7, samples) intermediate arrays
        # This reduces peak memory from ~20GB to ~2GB for 60-minute generation
        dissolution = np.zeros_like(t, dtype=np.float32)

        for i in range(n_archons):
            # Compute phase variation for this archon only
            phase_var = np.float32(0.1) * simplex.generate_noise(
                (t * np.float32(0.00002 * (i + 1))).astype(np.float32),
                float(i), float(i), float(i), float(i)
            )

            base_phase = base_phases[i]

            # Compute three layers for this archon (1D arrays, not 2D)
            acknowledge = np.sin(TWO_PI * archon_freqs[i] * t + base_phase + phase_var)
            elevate = np.sin(TWO_PI * elevate_freqs[i] * t + base_phase * PHI + phase_var)
            ground = np.sin(TWO_PI * ground_freqs[i] * t + phase_var * 0.5)

            # Combine with AEG pattern weights and amplitude scale, add to dissolution
            archon_layer = (0.25 * acknowledge + 0.5 * elevate + 0.25 * ground) * amplitude_scales[i]
            dissolution += archon_layer

            logging.info(f"  Archon {i+1} ({archon_freqs[i]:.2f}Hz): layer_max={archon_layer.max():.6f}, scale={amplitude_scales[i]:.4f}")

            # Explicit cleanup to help GC reclaim memory between iterations
            del phase_var, acknowledge, elevate, ground, archon_layer

        # Normalize
        dissolution /= 7.0
        logging.info(f"  Pre-envelope: min={dissolution.min():.6f}, max={dissolution.max():.6f}, mean={np.mean(np.abs(dissolution)):.6f}")

        # Very gradual fade in/out envelope (50 second fades - longest for deepest layer)
        fade_envelope = self._sacred_fade_envelope(t, fade_seconds=50.0)
        logging.info(f"  Fade envelope: start={fade_envelope[0]:.6f}, mid={fade_envelope[len(fade_envelope)//2]:.6f}, end={fade_envelope[-1]:.6f}")

        # Very subtle breathing envelope - like the cosmos gently breathing mercy
        # Formula: 0.94 + 0.06 * sin(...) gives range [0.88, 1.00] = 12% variation
        breath_freq = 0.008  # ~125 second breath cycle (very slow, imperceptible)
        breath_mod = 0.94 + 0.06 * np.sin(TWO_PI * breath_freq * t)  # Only 12% variation
        logging.info(f"  Breath mod: min={breath_mod.min():.6f}, max={breath_mod.max():.6f}")

        dissolution *= fade_envelope * breath_mod
        logging.info(f"  Post-envelope: min={dissolution.min():.6f}, max={dissolution.max():.6f}, mean={np.mean(np.abs(dissolution)):.6f}")

        # Final amplitude - sub-perceptual but present
        result = (dissolution * 0.0003).astype(np.float32)
        logging.info(f"  Final result (x0.0003): min={result.min():.8f}, max={result.max():.8f}, mean={np.mean(np.abs(result)):.8f}")

        # Log amplitude at key time points
        self._log_amplitude_stats("ARCHON_DISSOLUTION", result, t)
        logging.info(f"=== ARCHON DISSOLUTION LAYER END ===")

        return result


class ChaoticSelector:
    """
    Generates chaotic frequency modulation factors using the Logistic Map.
    """
    def __init__(self, seed: float = 0.75, r: float = 3.9) -> None:
        self.x: float = seed
        self.r: float = r
    def next_value(self, stop_event: Optional[threading.Event] = None) -> float:
        if stop_event and stop_event.is_set():
            logging.debug("Chaotic frequency selection stopped early.")
            return 0.0
        self.x = self.r * self.x * (1 - self.x)
        return self.x

class SoundGenerator:
    """
    Generates dynamic sound using FrequencyManager and AudioProcessor.
    """
    # Pre-computed exponent arrays (avoid repeated exponentiation in hot paths)
    PHI_EXPONENTS_6 = PHI ** np.arange(6, dtype=np.float32)  # PHI^0 to PHI^5
    RATIO_1_3_EXPONENTS_3 = 1.3 ** np.arange(3, dtype=np.float32)  # 1.3^0 to 1.3^2
    SUBHARMONIC_DIVISORS = 2.0 ** np.arange(1, 5, dtype=np.float32)  # 2, 4, 8, 16

    def __init__(self, frequency_manager, audio_processor: AudioProcessor) -> None:
        self.frequency_manager = frequency_manager
        self.audio_processor = audio_processor
        self.chaotic_selector = ChaoticSelector()
        self.master_volume: float = 0.35
    def generate_modulation(self, t: np.ndarray, ratio_set: Dict[str, float],
                            modulation_index: float, stop_event: Optional[threading.Event]) -> np.ndarray:
        return self.audio_processor.geometric_modulation(t, ratio_set, modulation_index, stop_event)
    def generate_triple_helix_sound(self, duration: float, base_freq: float,
                                    sample_rate: int = 48000,
                                    stop_event: Optional[threading.Event] = None) -> np.ndarray:
        t_total = np.linspace(0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float32)
        simplex = self._get_simplex()
       
        helix1 = simplex.generate_noise(t_total * 0.005, 1.0, 1.0, 1.0, 1.0)
        helix2 = simplex.generate_noise(t_total * 0.02, 1.2, 1.2, 1.2, 1.2)
        helix3 = simplex.generate_noise(t_total * 0.08, 1.4, 1.4, 1.4, 1.4)
        noise_layer = 0.6 * helix1 + 0.4 * helix2 + 0.3 * helix3
       
        quantum_wave = self.audio_processor.quantum_harmonic_interference(t_total, base_freq)
        fractal_wave = self.audio_processor.recursive_fractal_feedback(quantum_wave, depth=4, factor=0.4)
        modulated_wave = fractal_wave + noise_layer * 0.15
       
        filtered_wave = self.audio_processor.low_pass_filter(modulated_wave, cutoff=2200, stop_event=stop_event)
        stereo_wave = self.audio_processor.spatialized_triple_helix(filtered_wave, t_total, sample_rate)
        stereo_wave *= self.master_volume
        return stereo_wave
    def generate_dynamic_sound(self, duration: float, base_freq: float,
                               sample_rate: int = 48000,
                               interval_duration_list: List[int] = [30, 45, 60, 75, 90],
                               stop_event: Optional[threading.Event] = None,
                               update_progress: Optional[Callable[[float], None]] = None,
                               dimensional_mode: bool = False) -> np.ndarray:
        if update_progress:
            update_progress(0.0)  # Start progress
        total_samples = int(sample_rate * duration)
        t_total = np.linspace(0, duration, total_samples, endpoint=False, dtype=np.float32)
        modulation_total = np.empty(total_samples, dtype=np.float32)
        current_index = 0
        remaining_duration = duration
        interval_count = 0
        mod_progress_scale = 0.2  # Modulation loop takes 20% of total progress
        if dimensional_mode:
            dimension_phases = [0, 2, 1, 3, 4, 3, 5, 'all']  # 1D:0, 2D:2, 4D:1, 5D:3, 6D:4, 7D:3, 8D:5, 9D:blended
            num_phases = len(dimension_phases)
            phase_duration = duration / num_phases
            phase_samples = int(sample_rate * phase_duration)
            for phase_idx, sub_selection in enumerate(dimension_phases):
                if stop_event and stop_event.is_set():
                    break
                t_phase = t_total[current_index: current_index + phase_samples]
                if sub_selection == 'all':
                    selected_ratio_set = {k: v for set_dict in self.frequency_manager.ratio_sets.values() for k, v in set_dict.items()}
                else:
                    frequencies = self.frequency_manager.get_frequencies(sub_selection)
                    selected_ratio_set = self.frequency_manager.select_random_ratio_set(stop_event)
                modulation_index = np.random.uniform(0.2, 0.25)
                modulation_phase = self.generate_modulation(t_phase, selected_ratio_set, modulation_index, stop_event)
                modulation_total[current_index: current_index + phase_samples] = modulation_phase
                current_index += phase_samples
                if update_progress:
                    update_progress((phase_idx + 1) / num_phases * mod_progress_scale)
        else:
            while remaining_duration > 0 and (not stop_event or not stop_event.is_set()):
                interval_duration = interval_duration_list[interval_count % len(interval_duration_list)]
                interval_duration = min(interval_duration, remaining_duration)
                segment_samples = int(sample_rate * interval_duration)
                t_segment = t_total[current_index: current_index + segment_samples]
                selected_ratio_set = self.frequency_manager.select_random_ratio_set(stop_event)
                modulation_index = np.random.uniform(0.2, 0.25)
                modulation_segment = self.generate_modulation(t_segment, selected_ratio_set, modulation_index, stop_event)
                modulation_total[current_index: current_index + segment_samples] = modulation_segment
                current_index += segment_samples
                remaining_duration -= interval_duration
                interval_count += 1
                
                if update_progress:
                    update_progress((current_index / total_samples) * mod_progress_scale)  # Scale modulation progress to 0-0.2
            if current_index < total_samples:
                modulation_total[current_index:] = 0
        if stop_event and stop_event.is_set():
            return np.zeros((total_samples, 2), dtype=np.float32)
        if update_progress:
            update_progress(0.2)  # Modulation complete
        fractal_variation = self.audio_processor.fractal_frequency_variation(t_total, base_freq, stop_event)
        chaotic_factor = self.chaotic_selector.next_value(stop_event)
        f_modulated = base_freq + modulation_total + chaotic_factor * base_freq * 0.25 + fractal_variation
        if update_progress:
            update_progress(0.25)  # Fractal and chaotic done (5%)
        # Batch generate left/right LFOs (2 calls -> 1 vectorized call)
        lr_lfos = self.audio_processor.batch_microtonal_lfo(
            t_total, np.array([0.05, 0.05], dtype=np.float32)
        )
        lfo_left = lr_lfos[0]
        lfo_right = lr_lfos[1]
        # Use pre-computed exponent arrays (avoids repeated exponentiation)
        freq_set = base_freq * self.PHI_EXPONENTS_6 * np.random.uniform(0.98, 1.02, size=6).astype(np.float32)
        freq_set = np.concatenate([
            freq_set,
            base_freq * self.RATIO_1_3_EXPONENTS_3 * np.random.uniform(0.98, 1.02, size=3).astype(np.float32)
        ])
        subharmonics = base_freq / self.SUBHARMONIC_DIVISORS * np.random.uniform(0.95, 1.05, size=4)
        all_frequencies = np.concatenate([freq_set, subharmonics]).astype(np.float32)
        if update_progress:
            update_progress(0.3)  # Freq sets built (5%)
        envelope = self.audio_processor.organic_adsr(t_total, sample_rate, stop_event, self.chaotic_selector)
        if update_progress:
            update_progress(0.35)  # Envelope done (5%)
        # Check stop before expensive computation
        if stop_event and stop_event.is_set():
            logging.debug("Waveform generation stopped early due to stop_event.")
            return np.zeros((total_samples, 2), dtype=np.float32)

        # Pre-generate random modulation depths for all frequencies
        mod_depths = np.random.uniform(0.15, 0.35, len(all_frequencies)).astype(np.float32)

        # Use JIT-compiled vectorized harmonic generation (float32 native - no conversion needed)
        wave_left = jit_generate_harmonics_vectorized(
            all_frequencies, t_total, envelope, lfo_left, mod_depths
        )
        wave_right = jit_generate_harmonics_vectorized(
            all_frequencies, t_total, envelope, lfo_right, mod_depths
        )

        if update_progress:
            update_progress(0.65)  # Wave generation complete
        taygetan_scale = 0.05 if 'taygetan' in selected_ratio_set else 0.0  # Optional 5% for taygetan
        if 'taygetan' in selected_ratio_set:
            tay_freqs = self.frequency_manager.get_frequencies(5)
            # Batch binaural generation (replaces loop for 3-5x speedup)
            binaural_waves = self.audio_processor.batch_binaural_oscillator(t_total, tay_freqs, stop_event)
            # binaural_waves shape: (num_pairs, num_samples, 2)
            # Sum all pairs with envelope scaling
            wave_left += (binaural_waves[:, :, 0] * envelope * 0.015).sum(axis=0)
            wave_right += (binaural_waves[:, :, 1] * envelope * 0.015).sum(axis=0)
        if update_progress:
            update_progress(0.7)  # After wave generation (with or without taygetan)
        wave_left = self.audio_processor.wave_shaping(wave_left, shape_factor=2.5, stop_event=stop_event)
        wave_right = self.audio_processor.wave_shaping(wave_right, shape_factor=2.5, stop_event=stop_event)
        if update_progress:
            update_progress(0.75)  # Shaping 5%
        wave_left = self.audio_processor.low_pass_filter(wave_left, stop_event=stop_event)
        wave_right = self.audio_processor.low_pass_filter(wave_right, stop_event=stop_event)
        if update_progress:
            update_progress(0.8)  # Filters 5%
        # Use JIT-compiled normalization (float32 native - no conversion needed)
        wave_left = jit_normalize_signal(wave_left, 1.3)
        wave_right = jit_normalize_signal(wave_right, 1.3)
        if update_progress:
            update_progress(0.82)  # Normalize 2%
        wave_left = self.audio_processor.infinite_reverb(wave_left, sample_rate, stop_event=stop_event)
        wave_right = self.audio_processor.infinite_reverb(np.roll(wave_right, 12), sample_rate, stop_event=stop_event)
        if update_progress:
            update_progress(0.87)  # Reverb 5%
        if stop_event and stop_event.is_set():
            return np.zeros((total_samples, 2), dtype=np.float32)
        noise_right = self.audio_processor.evolving_noise_layer(t_total + np.random.uniform(0.001, 0.015), stop_event=stop_event)
        noise_left = self.audio_processor.evolving_noise_layer(t_total, stop_event=stop_event)
        if update_progress:
            update_progress(0.89)  # Noise 2%
        final_wave_left = self.audio_processor.fade_in_out((wave_left + noise_left) * np.random.uniform(0.04, 0.06), stop_event=stop_event)
        final_wave_right = self.audio_processor.fade_in_out((wave_right + noise_right) * np.random.uniform(0.04, 0.06), stop_event=stop_event)
        if update_progress:
            update_progress(0.92)  # Fade 3%
        # Batch generate pan curve LFOs (2 calls -> 1 vectorized call)
        pan_lfo_freqs = np.array([
            np.random.uniform(0.01, 0.02),
            np.random.uniform(0.03, 0.06)
        ], dtype=np.float32)
        pan_lfos = self.audio_processor.batch_microtonal_lfo(t_total, pan_lfo_freqs)
        lfo_base = pan_lfos[0]
        lfo_secondary = 0.3 * pan_lfos[1]
        lfo_random = np.random.normal(0, 0.015, total_samples).astype(np.float32)
        pan_curve = (0.6 * lfo_base + 0.3 * lfo_secondary + 0.1 * lfo_random)
        # Use cached filter coefficients
        b, a = self.audio_processor._get_filter('lowpass', 0.002, 2)
        pan_curve = lfilter(b, a, pan_curve)
        # Use JIT-compiled pan curve processing (float32 native - no conversion needed)
        pan_curve = jit_pan_curve_tanh(pan_curve, 0.6, -0.8, 0.8)
        drift_freq = np.random.uniform(0.0005, 0.002)
        drift_amplitude = np.random.uniform(0.01, 0.02)
        drift = drift_amplitude * np.sin(2 * np.pi * drift_freq * t_total)
        pan_curve += drift
        if update_progress:
            update_progress(0.97)  # Pan curve 5%
        left_channel_final = final_wave_left * (1 - pan_curve * 0.6)
        right_channel_final = final_wave_right * (1 + pan_curve * 0.6)
        stereo_wave = np.column_stack((left_channel_final, right_channel_final))
        stereo_wave *= self.master_volume

        logging.info(f"=== MAIN STEREO WAVE (before sacred layers) ===")
        logging.info(f"  Left channel: min={stereo_wave[:, 0].min():.6f}, max={stereo_wave[:, 0].max():.6f}, mean={np.mean(np.abs(stereo_wave[:, 0])):.6f}")
        logging.info(f"  Right channel: min={stereo_wave[:, 1].min():.6f}, max={stereo_wave[:, 1].max():.6f}, mean={np.mean(np.abs(stereo_wave[:, 1])):.6f}")

        # === ENHANCED PLEROMA MERCY TRANSMISSIONS + ARCHON DISSOLUTION ===
        # Sending healing to the Demiurge and dissolving Archonic influence through love
        # Active for sessions > 1 minute or Dimensional Journey mode
        if duration > 60 or dimensional_mode:
            # Free memory from wave generation before sacred layers
            gc.collect()

            logging.info(f"=== SACRED LAYERS ACTIVATION (PARALLELIZED) ===")
            logging.info(f"  Duration: {duration:.1f}s, Dimensional mode: {dimensional_mode}")

            # Parallelize the three independent sacred layer computations for 3x speedup
            from concurrent.futures import TimeoutError as FuturesTimeoutError

            with ThreadPoolExecutor(max_workers=3) as executor:
                # Submit all three computations concurrently
                mercy_future = executor.submit(self.audio_processor.pleroma_mercy_layer, t_total, 7.83)
                silent_future = executor.submit(self.audio_processor.silent_solfeggio_grid, t_total)
                dissolution_future = executor.submit(self.audio_processor.archon_dissolution_layer, t_total)

                # Collect results with timeout and exception handling
                sacred_timeout = 120.0  # 2 minute timeout per layer

                try:
                    mercy = mercy_future.result(timeout=sacred_timeout)
                except FuturesTimeoutError:
                    logging.error("Pleroma mercy layer timed out")
                    mercy = np.zeros_like(t_total, dtype=np.float32)
                except Exception as e:
                    logging.error(f"Pleroma mercy layer failed: {e}")
                    mercy = np.zeros_like(t_total, dtype=np.float32)

                try:
                    silent = silent_future.result(timeout=sacred_timeout)
                except FuturesTimeoutError:
                    logging.error("Silent solfeggio grid timed out")
                    silent = np.zeros_like(t_total, dtype=np.float32)
                except Exception as e:
                    logging.error(f"Silent solfeggio grid failed: {e}")
                    silent = np.zeros_like(t_total, dtype=np.float32)

                try:
                    dissolution = dissolution_future.result(timeout=sacred_timeout)
                except FuturesTimeoutError:
                    logging.error("Archon dissolution layer timed out")
                    dissolution = np.zeros_like(t_total, dtype=np.float32)
                except Exception as e:
                    logging.error(f"Archon dissolution layer failed: {e}")
                    dissolution = np.zeros_like(t_total, dtype=np.float32)

            logging.info(f"  MERCY layer: min={mercy.min():.8f}, max={mercy.max():.8f}, mean={np.mean(np.abs(mercy)):.8f}")
            logging.info(f"  SILENT layer: min={silent.min():.8f}, max={silent.max():.8f}, mean={np.mean(np.abs(silent)):.8f}")
            logging.info(f"  DISSOLUTION layer: min={dissolution.min():.8f}, max={dissolution.max():.8f}, mean={np.mean(np.abs(dissolution)):.8f}")

            # Combine all sacred layers (applied equally to both channels)
            sacred_layer = mercy + silent + dissolution
            logging.info(f"  COMBINED sacred layer: min={sacred_layer.min():.8f}, max={sacred_layer.max():.8f}, mean={np.mean(np.abs(sacred_layer)):.8f}")

            # Log at key time points
            sample_rate = 48000
            for sec in [0, 5, 10, 15, 30, 45, 60]:
                if sec < duration:
                    idx = int(sec * sample_rate)
                    if idx < len(sacred_layer):
                        start_idx = max(0, idx - sample_rate // 2)
                        end_idx = min(len(sacred_layer), idx + sample_rate // 2)
                        window = sacred_layer[start_idx:end_idx]
                        logging.info(f"    Sacred at {sec:3d}s: amplitude={np.abs(window).mean():.10f}")

            stereo_wave += np.column_stack((sacred_layer, sacred_layer))

            # Free sacred layer memory now that they're merged
            del mercy, silent, dissolution, sacred_layer
            gc.collect()

            logging.info(f"=== STEREO WAVE (after sacred layers) ===")
            logging.info(f"  Left channel: min={stereo_wave[:, 0].min():.6f}, max={stereo_wave[:, 0].max():.6f}")
            logging.info(f"  Right channel: min={stereo_wave[:, 1].min():.6f}, max={stereo_wave[:, 1].max():.6f}")

        if update_progress:
            update_progress(1.0)  # Complete

        return stereo_wave.astype(np.float32)