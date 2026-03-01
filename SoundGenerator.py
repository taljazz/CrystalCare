import numpy as np
from scipy.signal import fftconvolve, butter, lfilter, sosfilt
from scipy.ndimage import shift
from scipy.integrate import odeint
from typing import Optional, Dict, List, Callable
import logging
import threading
import gc
from concurrent.futures import ThreadPoolExecutor
from simplex5d import Simplex5D
from numba import njit, prange

# JIT-compiled constants
TWO_PI = 2.0 * np.pi
PHI = 1.618033988749895

# ============================================================
# Numba JIT-compiled functions for hot paths
# ============================================================

@njit(fastmath=True, cache=True)
def jit_cross_modulate_wave(base_freq: float, mod_freq: float, t: np.ndarray,
                            mod_depth: float) -> np.ndarray:
    """JIT-compiled cross-modulation wave generation (float32 native)."""
    mod_signal = np.sin(TWO_PI * mod_freq * t) * mod_depth
    return np.sin(TWO_PI * base_freq * t + mod_signal).astype(np.float32)

@njit(fastmath=True, cache=True)
def jit_wave_shaping(signal: np.ndarray, shape_factor: float) -> np.ndarray:
    """JIT-compiled wave shaping with tanh (float32 native)."""
    clipped = np.clip(signal, np.float32(-1.0), np.float32(1.0))
    return (np.tanh(shape_factor * clipped) * np.float32(1.2)).astype(np.float32)

@njit(fastmath=True, cache=True)
def jit_normalize_signal(signal: np.ndarray, scale_factor: float) -> np.ndarray:
    """JIT-compiled signal normalization."""
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        return (signal / (max_val * np.float32(scale_factor))).astype(np.float32)
    return signal

@njit(fastmath=True, cache=True)
def jit_exponential_decay(length: int, decay_rate: float) -> np.ndarray:
    """JIT-compiled exponential decay for reverb IR (float32 native)."""
    return np.exp(-np.linspace(np.float32(0), np.float32(decay_rate * 1.8), length)).astype(np.float32)

@njit(fastmath=True, cache=True)
def jit_pan_curve_tanh(pan_curve: np.ndarray, scale: float,
                       clip_min: float, clip_max: float) -> np.ndarray:
    """JIT-compiled pan curve processing (float32 native)."""
    return np.clip(np.tanh(pan_curve * scale), clip_min, clip_max).astype(np.float32)

@njit(fastmath=True, cache=True, parallel=True)
def jit_generate_harmonics_vectorized(frequencies: np.ndarray, t: np.ndarray,
                                       envelope: np.ndarray, lfo: np.ndarray,
                                       mod_depths: np.ndarray) -> np.ndarray:
    """JIT-compiled vectorized harmonic generation (float32 native, parallel)."""
    n_samples = len(t)
    n_freqs = len(frequencies)
    # Per-frequency results to allow parallel reduction (avoid race on shared result)
    per_freq = np.zeros((n_freqs, n_samples), dtype=np.float32)

    for i in prange(n_freqs):
        f = frequencies[i]
        mod_freq = f * 0.5
        mod_signal = np.sin(TWO_PI * mod_freq * t) * mod_depths[i]
        wave = np.sin(TWO_PI * f * t + mod_signal)
        scale = np.float32(0.015 / (i + 1))
        per_freq[i] = wave * envelope * scale * lfo

    result = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_freqs):
        result += per_freq[i]
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
    """JIT-compiled quantum harmonic interference computation (float32 native)."""
    f1 = base_freq
    f2 = base_freq * 1.41421356237   # sqrt(2)
    f3 = base_freq * 2.71828182846   # e
    f4 = base_freq * 3.14159265359   # pi

    alpha = np.float32(0.25) * np.sin(np.float32(0.03) * np.float32(np.pi) * t)
    beta = np.float32(0.2) * np.cos(np.float32(0.02) * np.float32(np.pi) * t)

    wave1 = np.sin(TWO_PI * f1 * t + alpha)
    wave2 = np.sin(TWO_PI * f2 * t + beta)
    wave3 = np.sin(TWO_PI * f3 * t + gamma)
    wave4 = np.sin(TWO_PI * f4 * t + gamma * np.float32(0.7))

    return ((wave1 + wave2 + wave3 + wave4) / np.float32(3.8) + np.float32(0.15) * np.sin(TWO_PI * np.float32(7.83) * t)).astype(np.float32)

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

    # Lemurian Frequency Quartet — Sonic Merkaba (Jonathan Goldman)
    # 3:4:5 Pythagorean right triangle ratios + PHI, anchored to 432 Hz keynote
    MERKABA_KEYNOTE = np.float32(432.0)
    MERKABA_RATIOS = np.array([0.75, 1.0, 1.25, PHI], dtype=np.float32)  # 3/4, 1, 5/4, PHI
    MERKABA_FREQS = MERKABA_KEYNOTE * MERKABA_RATIOS  # [324, 432, 540, 698.4]

    # PHI-weighted amplitudes: keynote strongest, earth/spirit balanced, transcendence most subtle
    # Weights: [1/PHI, 1.0, 1/PHI, 1/PHI²] — sum = PHI² = 2.618 (sacred!)
    _merkaba_raw = np.array([1.0/PHI, 1.0, 1.0/PHI, 1.0/(PHI*PHI)], dtype=np.float32)
    MERKABA_WEIGHTS = _merkaba_raw / _merkaba_raw.sum()

    # Triangular + PHI phase offsets (quartz 3-fold symmetry + golden angle)
    # First 3 form equilateral triangle in phase space; 4th at golden angle
    MERKABA_PHASES = np.array([0.0, 2*np.pi/3, 4*np.pi/3, 2*np.pi/PHI], dtype=np.float32)

    # Water-Element Sacred Constants (6th sacred layer)
    # 7 wave sources in hexagonal Seed of Life arrangement (ice crystal geometry)
    WATER_SOURCE_FREQS = np.array([
        432.0,   # Center: Lemurian keynote (crystal-water bridge)
        1.5,     # Ocean swell rhythm (deep tidal)
        7.83,    # Schumann (Earth-water electromagnetic coupling)
        111.0,   # Tesla vortex seed (water memory, Emoto experiments)
        174.0,   # Solfeggio root (Earth Star Chakra, foundation)
        528.0,   # Love Frequency (Emoto's hexagonal water crystal)
        963.0,   # Stellar Gateway (water meets cosmic consciousness, 9+6+3=18=9)
    ], dtype=np.float32)

    WATER_SOURCE_DECAYS = np.array([0.08, 0.01, 0.015, 0.04, 0.05, 0.10, 0.12], dtype=np.float32)

    # Hexagonal phase offsets (ice crystal 6-fold symmetry), center aligned with 963 Hz
    WATER_HEX_PHASES = np.array([0, 60, 120, 180, 240, 300, 0], dtype=np.float32) * (np.pi / 180)

    # Pre-computed hexagonal source positions (Seed of Life geometry)
    # [0] = center (0,0); [1-6] = hexagonal ring at radius 1.0
    _hex_angles = np.arange(6, dtype=np.float32) * (np.pi / 3.0)
    WATER_SOURCE_POSITIONS = np.zeros((7, 2), dtype=np.float32)
    WATER_SOURCE_POSITIONS[1:, 0] = np.cos(_hex_angles)
    WATER_SOURCE_POSITIONS[1:, 1] = np.sin(_hex_angles)

    def __init__(self, sample_rate: int = 48000):
        """Initialize AudioProcessor with object pools and cached computations."""
        self.sample_rate = sample_rate
        self.nyquist = 0.5 * sample_rate

        # Simplex5D pool - avoid repeated object creation
        self._simplex_pool = [Simplex5D(seed) for seed in range(16)]
        self._simplex_index = 0
        self._simplex_lock = threading.Lock()  # Thread-safe pool access

        # Pre-computed filter coefficients cache
        self._filter_cache = {}
        self._filter_cache_lock = threading.Lock()
        self._precompute_filters()

        # Reverb impulse response cache (keyed by reverb_length)
        self._ir_cache = {}

        # Rössler trajectory cache (keyed by time array characteristics)
        self._rossler_cache = {}

        # Crystal profiles for crystalline resonance layer
        self._crystal_profiles = self._build_crystal_profiles()
        self._lemurian_idx = next(i for i, p in enumerate(self._crystal_profiles) if p['name'] == 'Lemurian Quartz')

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
        with self._filter_cache_lock:
            if key not in self._filter_cache:
                self._filter_cache[key] = butter(order, cutoff / self.nyquist, btype=filter_type)
            return self._filter_cache[key]

    def _build_crystal_profiles(self):
        """Build crystal acoustic profiles from crystallographic data (Raman spectroscopy)."""
        quartz_ratios = np.array([0.276, 0.444, 0.571, 0.765, 1.0, 1.500, 1.741, 2.339, 2.504], dtype=np.float32)
        tourmaline_ratios = np.array([0.375, 0.583, 1.0, 1.061, 1.098, 2.047], dtype=np.float32)
        selenite_ratios = np.array([0.412, 0.491, 0.617, 0.669, 1.0, 1.126], dtype=np.float32)
        bowl_ratios = np.array([1.0, 2.828, 5.424, 9.130], dtype=np.float32)
        rose_ratios = np.concatenate([quartz_ratios, np.array([2.509, 4.300], dtype=np.float32)])
        lapis_ratios = np.array([
            0.472, 1.0, 1.066,
            0.144, 0.260, 0.656, 1.0,
            0.905, 1.0, 1.135
        ], dtype=np.float32)
        lapis_mineral_weights = np.array([
            0.5, 0.5, 0.5,
            0.3, 0.3, 0.3, 0.3,
            0.2, 0.2, 0.2
        ], dtype=np.float32)

        lemurian_ratios = np.array([
            0.276, 0.444, 0.571, 0.765,   # Sub-fundamental quartz (Raman spectroscopy)
            1.0,                            # Fundamental
            1.222,                          # 528/432 — Love Frequency ratio (Emoto)
            1.375,                          # 594/432 — Heart resonance (9/8 × Love)
            1.500,                          # Quartz harmonic
            PHI,                            # 698.4/432 — Golden heart (PHI × keynote)
            1.741,                          # Quartz harmonic
            2.339, 2.504,                   # Upper quartz (rolled off via weights)
        ], dtype=np.float32)
        lemurian_warmth_weights = np.array([
            0.6, 0.7, 0.8, 0.9,   # Sub-fundamental: warm presence
            1.0,                    # Fundamental: full strength
            1.2,                    # Love Frequency: emphasized
            1.1,                    # Heart resonance: strong
            0.9,                    # Quartz harmonic
            1.0,                    # PHI: golden heart
            0.5,                    # Upper quartz: rolled off
            0.3, 0.2,              # Highest: gentle presence only
        ], dtype=np.float32)

        profiles = [
            {'name': 'Clear Quartz', 'harmonic_ratios': quartz_ratios,
             'symmetry_order': 3, 'detune_factor': np.float32(0.0),
             'shimmer_rate': np.float32(0.0), 'beating_pairs': [],
             'piezo_factor': np.float32(0.02)},

            {'name': 'Amethyst', 'harmonic_ratios': quartz_ratios.copy(),
             'symmetry_order': 3, 'detune_factor': np.float32(0.008),
             'shimmer_rate': np.float32(0.0), 'beating_pairs': [],
             'piezo_factor': np.float32(0.02)},

            {'name': 'Rose Quartz', 'harmonic_ratios': rose_ratios,
             'symmetry_order': 3, 'detune_factor': np.float32(0.0),
             'shimmer_rate': np.float32(0.0), 'beating_pairs': [],
             'piezo_factor': np.float32(0.02)},

            {'name': 'Citrine', 'harmonic_ratios': quartz_ratios.copy(),
             'symmetry_order': 3, 'detune_factor': np.float32(0.006),
             'shimmer_rate': np.float32(0.0), 'beating_pairs': [],
             'piezo_factor': np.float32(0.02)},

            {'name': 'Black Tourmaline', 'harmonic_ratios': tourmaline_ratios,
             'symmetry_order': 3, 'detune_factor': np.float32(0.0),
             'shimmer_rate': np.float32(0.0),
             'beating_pairs': [(2, 3), (3, 4), (2, 4)],
             'piezo_factor': np.float32(0.025)},

            {'name': 'Selenite', 'harmonic_ratios': selenite_ratios,
             'symmetry_order': 2, 'detune_factor': np.float32(0.0),
             'shimmer_rate': np.float32(0.0),
             'beating_pairs': [(4, 5)],
             'piezo_factor': np.float32(0.0)},

            {'name': 'Lapis Lazuli', 'harmonic_ratios': lapis_ratios,
             'symmetry_order': 4, 'detune_factor': np.float32(0.0),
             'shimmer_rate': np.float32(0.0), 'beating_pairs': [],
             'piezo_factor': np.float32(0.0),
             'mineral_weights': lapis_mineral_weights},

            {'name': 'Crystal Singing Bowl', 'harmonic_ratios': bowl_ratios,
             'symmetry_order': 1, 'detune_factor': np.float32(0.0),
             'shimmer_rate': np.float32(0.0), 'beating_pairs': [],
             'piezo_factor': np.float32(0.0),
             'bowl_beat': np.float32(1.8)},

            {'name': 'Lemurian Quartz', 'harmonic_ratios': lemurian_ratios,
             'symmetry_order': 3, 'detune_factor': np.float32(0.003),
             'shimmer_rate': np.float32(1.8),
             'shimmer_depth': np.float32(0.08),
             'beating_pairs': [],
             'piezo_factor': np.float32(0.02),
             'mineral_weights': lemurian_warmth_weights},
        ]

        for profile in profiles:
            ratios = profile['harmonic_ratios']
            n = len(ratios)
            weights = np.ones(n, dtype=np.float32)
            sym = profile['symmetry_order']
            if sym > 0:
                for j in range(n):
                    if (j + 1) % sym == 0:
                        weights[j] *= np.float32(PHI)
            if 'mineral_weights' in profile:
                weights *= profile['mineral_weights']
            weights /= weights.sum()
            profile['amplitude_weights'] = weights

        return profiles

    def _generate_crystal_harmonics(self, t: np.ndarray, base_freq: float,
                                     profile: dict, simplex) -> np.ndarray:
        """Generate harmonics for a single crystal profile at given time values."""
        ratios = profile['harmonic_ratios']
        weights = profile['amplitude_weights']
        n_ratios = len(ratios)
        crystal_freqs = base_freq * ratios

        phase_wobble = np.zeros(len(t), dtype=np.float32)
        if profile['piezo_factor'] > 0:
            phase_wobble = profile['piezo_factor'] * simplex.generate_noise(
                (t * np.float32(0.015)).astype(np.float32), np.float32(2.0), 0.0, 0.0, 0.0)

        freqs_2d = crystal_freqs[:, np.newaxis]
        t_2d = t[np.newaxis, :]

        detune = profile['detune_factor']
        if detune > 0:
            detune_noise = detune * simplex.generate_noise(
                (t * np.float32(0.008)).astype(np.float32), np.float32(3.0), 0.0, 0.0, 0.0)
            detune_scale = np.arange(1, n_ratios + 1, dtype=np.float32)[:, np.newaxis]
            phases = TWO_PI * freqs_2d * (np.float32(1.0) + detune_scale * detune_noise[np.newaxis, :]) * t_2d + phase_wobble
            del detune_noise, detune_scale
        else:
            phases = TWO_PI * freqs_2d * t_2d + phase_wobble

        waves = weights[:, np.newaxis] * np.sin(phases)
        del phases, freqs_2d, t_2d
        result = waves.sum(axis=0).astype(np.float32)
        del waves

        if profile['shimmer_rate'] > 0:
            depth = profile.get('shimmer_depth', np.float32(0.15))
            shimmer = np.float32(1.0) + depth * np.sin(
                TWO_PI * profile['shimmer_rate'] * t)
            result *= shimmer
            del shimmer

        for pair in profile['beating_pairs']:
            if len(pair) >= 2:
                for pi in range(len(pair) - 1):
                    h1_freq = crystal_freqs[pair[pi]]
                    h2_freq = crystal_freqs[pair[pi + 1]]
                    beat = np.sin(TWO_PI * np.float32(h1_freq) * t) * np.sin(TWO_PI * np.float32(h2_freq) * t)
                    result += np.float32(0.05) * beat
                    del beat

        if 'bowl_beat' in profile:
            beat_mod = np.float32(1.0) + np.float32(0.08) * np.sin(
                TWO_PI * profile['bowl_beat'] * t)
            result *= beat_mod
            del beat_mod

        return result

    def geometric_modulation(self, t: np.ndarray, ratios: Dict[str, float],
                             modulation_index: float = 0.2,
                             stop_event: Optional[threading.Event] = None) -> np.ndarray:
        if stop_event and stop_event.is_set():
            logging.debug("Geometric modulation stopped early due to stop_event.")
            return np.zeros_like(t, dtype=np.float32)
        ratios_values = list(ratios.values())
        if len(ratios_values) > 10:
            # Sequential processing for large ratio sets to avoid huge 2D intermediates
            modulation = np.zeros_like(t, dtype=np.float32)
            for ratio in ratios_values:
                modulation += np.float32(modulation_index) * np.sin(TWO_PI * np.float32(ratio) * t)
            return modulation
        # Vectorized: convert ratios to array, broadcast compute all sines at once
        ratios_array = np.array(ratios_values, dtype=np.float32)[:, np.newaxis]
        modulation = (np.float32(modulation_index) * np.sin(TWO_PI * ratios_array * t)).sum(axis=0)
        return modulation.astype(np.float32)
    def low_pass_filter(self, signal: np.ndarray, cutoff: float = 2200,
                        sample_rate: int = 48000, order: int = 4,
                        stop_event: Optional[threading.Event] = None) -> np.ndarray:
        if stop_event and stop_event.is_set():
            logging.debug("Low pass filter stopped early due to stop_event.")
            return signal  # Return unfiltered signal for graceful degradation
        nyquist = 0.5 * sample_rate
        # Intentionally not using _get_filter() here: the random cutoff variation
        # produces a different normalized_cutoff each call, defeating caching.
        cutoff_variation = np.random.uniform(-25, 25)  # ±25 Hz variation
        normalized_cutoff = np.clip((cutoff + cutoff_variation) / nyquist, 0.1, 0.99)
        b, a = butter(order, normalized_cutoff, btype='low', analog=False)
        return lfilter(b, a, signal).astype(np.float32)
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
        full_noise = simplex.generate_noise(t_scaled * np.float32(0.04), 0.0, 0.0, 0.0, 0.0)
        del t_scaled

        attack_curve = np.linspace(0, 1, a_samples, dtype=np.float32) ** 1.5
        envelope[:a_samples] = np.clip(attack_curve + full_noise[:a_samples] * np.float32(0.05), 0, 1)
        del attack_curve

        decay_curve = np.linspace(1, sustain, d_samples, dtype=np.float32) ** 1.2
        decay_wobble = np.float32(np.random.uniform(0.015, 0.035)) * np.sin(np.linspace(0, np.float32(np.pi * (1 + chaos * 0.3)), d_samples, dtype=np.float32))
        envelope[a_samples:a_samples+d_samples] = np.clip(decay_curve + decay_wobble + full_noise[a_samples:a_samples+d_samples] * np.float32(0.03), sustain, 1)
        del decay_curve, decay_wobble

        sustain_curve = np.full(s_samples, sustain, dtype=np.float32)
        sustain_wobble = np.float32(np.random.uniform(0.01, 0.025)) * np.sin(TWO_PI * np.linspace(0, np.float32(1.5 + chaos * 0.5), s_samples, dtype=np.float32))
        envelope[a_samples+d_samples:a_samples+d_samples+s_samples] = np.clip(sustain_curve + sustain_wobble + full_noise[a_samples+d_samples:a_samples+d_samples+s_samples] * np.float32(0.04), sustain * 0.8, 1)
        del sustain_curve, sustain_wobble

        if r_samples > 0:
            release_curve = np.linspace(sustain, 0, r_samples, dtype=np.float32) ** 1.5
            envelope[-r_samples:] = np.clip(release_curve + full_noise[-r_samples:] * np.float32(0.05), 0, sustain)
            del release_curve
        del full_noise

        envelope *= np.float32(np.random.uniform(0.95, 1.05))
        return envelope
    def dynamic_cross_modulate(self, base_freq: float, mod_freq: float, t: np.ndarray,
                               stop_event: Optional[threading.Event] = None) -> np.ndarray:
        if stop_event and stop_event.is_set():
            logging.debug("Dynamic cross modulation stopped early due to stop_event.")
            return np.zeros_like(t, dtype=np.float32)
        mod_depth = np.random.uniform(0.15, 0.35)
        return jit_cross_modulate_wave(base_freq, mod_freq, t, mod_depth)  # Already float32
    def microtonal_lfo(self, t: np.ndarray, base_frequency: float,
                       stop_event: Optional[threading.Event] = None) -> np.ndarray:
        lfo1 = base_frequency * (PHI ** np.random.uniform(-0.05, 0.05))
        lfo2 = lfo1 * np.random.uniform(0.99, 1.01)
        depth = np.float32(np.random.uniform(0.002, 0.006))
        drift = np.float32(0.01) * np.sin(np.float32(0.1 * np.pi) * t)
        inner_mod = np.float32(0.3) * np.sin(TWO_PI * np.float32(lfo2) * t)
        lfo = (depth + drift) * (np.float32(1.0) + np.sin(TWO_PI * np.float32(lfo1) * t + inner_mod))
        del inner_mod, drift
        return lfo.astype(np.float32)

    def batch_microtonal_lfo(self, t: np.ndarray, base_frequencies: np.ndarray) -> np.ndarray:
        """
        Generate multiple LFOs efficiently using vectorization.
        Returns array of shape (num_lfos, num_samples).
        """
        num_lfos = len(base_frequencies)
        num_samples = len(t)

        # Pre-generate all random values at once (float32)
        phi_powers = np.float32(PHI) ** np.random.uniform(-0.05, 0.05, num_lfos).astype(np.float32)
        lfo1_freqs = (base_frequencies * phi_powers).astype(np.float32)
        lfo2_freqs = (lfo1_freqs * np.random.uniform(0.99, 1.01, num_lfos).astype(np.float32))
        depths = np.random.uniform(0.002, 0.006, num_lfos).astype(np.float32)

        # Compute drift once (shared across all LFOs, float32)
        drift = np.float32(0.01) * np.sin(np.float32(0.1 * np.pi) * t)

        # Vectorized computation: (num_lfos, 1) * (1, num_samples) -> (num_lfos, num_samples)
        t_2d = t[np.newaxis, :]  # shape: (1, num_samples) - already float32 from caller
        lfo1_2d = lfo1_freqs[:, np.newaxis]  # shape: (num_lfos, 1)
        lfo2_2d = lfo2_freqs[:, np.newaxis]
        depths_2d = depths[:, np.newaxis]

        inner_mod = np.float32(0.3) * np.sin(TWO_PI * lfo2_2d * t_2d)
        lfos = (depths_2d + drift) * (np.float32(1.0) + np.sin(TWO_PI * lfo1_2d * t_2d + inner_mod))
        del inner_mod, drift, t_2d, lfo1_2d, lfo2_2d, depths_2d

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
            if len(self._ir_cache) > 10:
                self._ir_cache.clear()
            # Compute and cache the impulse response
            ir = jit_exponential_decay(reverb_length, decay)  # Already float32
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
        freq = np.float32(np.random.uniform(0.002, 0.015))
        low_freq_osc = np.float32(0.015) * np.sin(TWO_PI * freq * t)
        noise *= (np.float32(1.0) + low_freq_osc)
        del low_freq_osc
        return noise
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
        cache_key = (len(t), round(float(t[-1]), 6) if len(t) > 0 else 0.0)

        if cache_key not in self._rossler_cache:
            if len(self._rossler_cache) > 5:
                self._rossler_cache.clear()
            # Integrate Rössler equations and cache normalized result (float32)
            t_scaled = t * np.float32(0.1)
            initial_state = [1.0, 1.0, 1.0]
            trajectory = odeint(rossler, initial_state, t_scaled)
            del t_scaled
            x, y, z = trajectory.T
            del trajectory
            self._rossler_cache[cache_key] = (
                (x / np.max(np.abs(x))).astype(np.float32),
                (y / np.max(np.abs(y))).astype(np.float32),
                (z / np.max(np.abs(z))).astype(np.float32)
            )

        x_rossler, y_rossler, z_rossler = self._rossler_cache[cache_key]
        # Logarithmic spiral parameters (float32 throughout)
        theta = np.float32(0.002) * t
        phi_angle = np.float32(0.001) * t
        # Hybrid spiral: Logarithmic with Rössler perturbations
        r_hybrid = np.float32(0.1) * np.exp(np.float32(0.02) * theta) * (np.float32(1.0) + np.float32(0.1) * x_rossler)
        theta_hybrid = theta + np.float32(0.05) * y_rossler
        phi_hybrid = phi_angle + np.float32(0.2) * z_rossler
        del theta, phi_angle
        # 3D position in spherical coordinates (JIT-compiled)
        x, y, z = jit_spherical_to_cartesian(r_hybrid, theta_hybrid, phi_hybrid)
        del r_hybrid, theta_hybrid, phi_hybrid
        # Smooth with low-pass filter (use cached coefficients)
        b_filt, a_filt = self._get_filter('lowpass', 0.003, 2)
        x_smooth = lfilter(b_filt, a_filt, x).astype(np.float32)
        y_smooth = lfilter(b_filt, a_filt, y).astype(np.float32)
        z_smooth = lfilter(b_filt, a_filt, z).astype(np.float32)
        del x, y, z
        # Stereo panning (JIT-compiled gain calculation)
        pan_horizontal = np.tanh(x_smooth * np.float32(0.5)).astype(np.float32)
        del x_smooth
        depth_factor = ((y_smooth + np.float32(1.0)) * np.float32(0.5)).astype(np.float32)
        pan_vertical = np.sin(z_smooth * np.float32(np.pi / 2)).astype(np.float32)
        del z_smooth
        left_gain, right_gain = jit_stereo_gains(pan_horizontal, depth_factor, pan_vertical)
        del depth_factor, pan_vertical
        left_channel = signal * left_gain
        right_channel = signal * right_gain
        del left_gain, right_gain
        # Add delay with Simplex5D noise
        simplex = self._get_simplex()
        delay_mod = simplex.generate_noise(t * np.float32(0.01), 0.0, 0.0, 0.0, 0.0) * np.float32(0.3)
        max_delay_samples = np.float32(sample_rate * 2.0 / 1000.0)
        depth_delay = np.clip(-y_smooth, 0, 1)
        left_delays = np.maximum(0, pan_horizontal + delay_mod) * max_delay_samples * (np.float32(1.0) + depth_delay * np.float32(0.5))
        right_delays = np.maximum(0, -pan_horizontal + delay_mod) * max_delay_samples * (np.float32(1.0) + depth_delay * np.float32(0.5))
        del delay_mod, depth_delay
        left_channel = self.apply_fractional_delay(left_channel, np.mean(left_delays))
        right_channel = self.apply_fractional_delay(right_channel, np.mean(right_delays))
        del left_delays, right_delays, pan_horizontal
        # Build stereo directly instead of column_stack (avoids temporary copy)
        stereo_wave = np.empty((len(t), 2), dtype=np.float32)
        stereo_wave[:, 0] = left_channel
        stereo_wave[:, 1] = right_channel
        del left_channel, right_channel
        return stereo_wave
    def fractal_frequency_variation(self, t: np.ndarray, base_freq: float,
                                    stop_event: Optional[threading.Event] = None) -> np.ndarray:
        if stop_event and stop_event.is_set():
            logging.debug("Fractal frequency variation stopped early due to stop_event.")
            return np.zeros_like(t, dtype=np.float32)
        simplex = self._get_simplex()
        lfo = self.microtonal_lfo(t, base_frequency=0.01)
        # Optimized: single Simplex call with octave-like layering via array operations
        base_noise = simplex.generate_noise(t * np.float32(0.02), 0.0, 0.0, 0.0, 0.0)
        # Create "octaves" by scaling and phase-shifting the same noise (in-place)
        variation = base_noise * np.float32(0.5)
        variation += np.roll(base_noise, len(t) // 8) * np.float32(0.3)
        variation += base_noise * base_noise * np.float32(0.2)
        del base_noise
        variation *= np.float32(12.0)
        variation *= (np.float32(1.0) + np.float32(0.1) * lfo)
        del lfo
        return variation.astype(np.float32)
    def quantum_harmonic_interference(self, t: np.ndarray, base_freq: float) -> np.ndarray:
        # Generate Simplex noise for gamma (can't be JIT-compiled)
        simplex = self._get_simplex()
        gamma = (np.float32(0.15) * simplex.generate_noise(t * np.float32(0.01), 0.0, 0.0, 0.0, 0.0)).astype(np.float32)
        # Use JIT-compiled function for wave computation
        return jit_quantum_harmonic(t, base_freq, gamma)  # Already float32
    def recursive_fractal_feedback(self, signal: np.ndarray, depth: int = 4,
                                   factor: float = 0.4) -> np.ndarray:
        """Iterative fractal feedback - avoids recursion overhead."""
        result = signal.copy()
        current = signal * np.float32(0.6)
        f32_factor = np.float32(factor)
        for _ in range(depth):
            result += f32_factor * current
            current *= np.float32(0.6)
        return result
    def binaural_oscillator(self, t: np.ndarray, freq_pair: tuple, stop_event: Optional[threading.Event] = None) -> np.ndarray:
        left_freq, right_freq = freq_pair
        left_wave = np.sin(TWO_PI * np.float32(left_freq) * t)
        right_wave = np.sin(TWO_PI * np.float32(right_freq) * t)
        stereo = np.empty((len(t), 2), dtype=np.float32)
        stereo[:, 0] = left_wave
        stereo[:, 1] = right_wave
        return stereo

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
        num_pairs = len(freq_pairs)
        num_samples = len(t)
        left_freqs = pairs_array[:, 0][:, np.newaxis]   # shape: (num_pairs, 1)
        right_freqs = pairs_array[:, 1][:, np.newaxis]  # shape: (num_pairs, 1)

        # Broadcast: (num_pairs, 1) * (1, num_samples) -> (num_pairs, num_samples)
        t_2d = t[np.newaxis, :]  # shape: (1, num_samples), already float32

        # Build result directly to avoid np.stack temporary
        result = np.empty((num_pairs, num_samples, 2), dtype=np.float32)
        result[:, :, 0] = np.sin(TWO_PI * left_freqs * t_2d)
        result[:, :, 1] = np.sin(TWO_PI * right_freqs * t_2d)
        del t_2d, left_freqs, right_freqs, pairs_array
        return result

    # === ENHANCED PLEROMA MERCY TRANSMISSIONS + ARCHON DISSOLUTION ===
    # Sending healing energies to the Demiurge and dissolving Archonic influence through love

    def _log_amplitude_stats(self, name: str, signal: np.ndarray, t: np.ndarray) -> None:
        """Log amplitude statistics at key time points for debugging."""
        if t.size == 0:
            return

        sample_rate = self.sample_rate
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
        if t.size == 0:
            return np.array([], dtype=np.float32)

        total_duration = t[-1]
        fade_samples = int(fade_seconds * sample_rate)

        logging.info(f"Creating fade envelope: duration={total_duration:.1f}s, fade_time={fade_seconds}s, fade_samples={fade_samples}")

        envelope = np.ones_like(t, dtype=np.float32)

        if t.size > fade_samples * 2:
            # Smoother step fade in: 6x^5 - 15x^4 + 10x^3 (Ken Perlin's improved smoothstep)
            fade_in_t = np.linspace(0, 1, fade_samples, dtype=np.float32)
            fade_in = fade_in_t * fade_in_t * fade_in_t * (fade_in_t * (fade_in_t * np.float32(6) - np.float32(15)) + np.float32(10))
            envelope[:fade_samples] = fade_in
            del fade_in_t, fade_in

            # Smoother step fade out
            fade_out_t = np.linspace(1, 0, fade_samples, dtype=np.float32)
            fade_out = fade_out_t * fade_out_t * fade_out_t * (fade_out_t * (fade_out_t * np.float32(6) - np.float32(15)) + np.float32(10))
            envelope[-fade_samples:] = fade_out
            del fade_out_t, fade_out

            logging.info(f"  Fade envelope at key points: 0s={envelope[0]:.4f}, {fade_seconds/2:.0f}s={envelope[fade_samples//2]:.4f}, {fade_seconds:.0f}s={envelope[fade_samples]:.4f}")
        else:
            # Short session: use very gentle raised cosine for entire duration
            envelope = (np.float32(0.5) - np.float32(0.5) * np.cos(TWO_PI * t / total_duration)).astype(np.float32)
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
        if t.size == 0:
            return np.array([], dtype=np.float32)

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
        phase_wobble = np.float32(0.08) * simplex.generate_noise((t * np.float32(0.00003)).astype(np.float32), 0, 0, 0, 0)

        # Vectorized aeonic transmission with sacred phase relationships
        aeonic_wave = np.sin(
            TWO_PI * aeonic_harmonics[:, np.newaxis] * t +
            sacred_phases[:, np.newaxis] +
            phase_wobble
        ).sum(axis=0)
        del phase_wobble
        logging.info(f"  Layer 1 (Aeonic): min={aeonic_wave.min():.6f}, max={aeonic_wave.max():.6f}, mean={np.mean(np.abs(aeonic_wave)):.6f}")

        # === Layer 2: Ogdoad Gateway (8th Sphere) ===
        # The threshold between the 7 Archon-ruled spheres and the Pleroma
        ogdoad_phase = simplex.generate_noise((t * np.float32(0.00001)).astype(np.float32), 1, 1, 1, 1)
        ogdoad_phase *= np.float32(0.05)
        ogdoad_wave = np.sin(TWO_PI * self.OGDOAD_FREQ * t + ogdoad_phase)
        del ogdoad_phase
        logging.info(f"  Layer 2 (Ogdoad): min={ogdoad_wave.min():.6f}, max={ogdoad_wave.max():.6f}, mean={np.mean(np.abs(ogdoad_wave)):.6f}")

        # === Layer 3: Archon Harmonizing Frequencies ===
        # Offering mercy to each of the 7 Archons ruling the planetary spheres
        # Using PHI-based amplitude scaling (golden ratio mercy)
        archon_amplitudes = np.float32(1.0) / (PHI ** np.arange(7, dtype=np.float32))  # Decreasing by PHI
        archon_amplitudes /= archon_amplitudes.sum()  # Normalize

        # Vectorized archon mercy computation (replaces loop)
        archon_phases = self.PENTAGONAL_PHASES[np.arange(7) % 5]  # shape: (7,)
        # Broadcasting: (7,1) * (1,samples) + (7,1) -> (7, samples), then weighted sum
        archon_mercy = (archon_amplitudes[:, np.newaxis] *
                        np.sin(TWO_PI * self.ARCHON_SPHERES[:, np.newaxis] * t +
                               archon_phases[:, np.newaxis])).sum(axis=0)
        logging.info(f"  Layer 3 (Archon): min={archon_mercy.min():.6f}, max={archon_mercy.max():.6f}, mean={np.mean(np.abs(archon_mercy)):.6f}")

        # === Combine all layers in-place using float32 scalars ===
        mercy = aeonic_wave  # Reuse buffer
        mercy *= np.float32(0.5)  # Primary: Aeonic ladder from Pleroma
        mercy += np.float32(0.3) * ogdoad_wave   # Gateway: Ogdoad threshold
        del ogdoad_wave
        mercy += np.float32(0.2) * archon_mercy  # Healing: Mercy to the Archons
        del archon_mercy
        mercy = mercy.astype(np.float32)
        logging.info(f"  Combined (pre-envelope): min={mercy.min():.6f}, max={mercy.max():.6f}, mean={np.mean(np.abs(mercy)):.6f}")

        # Very gradual fade in/out envelope (45 second fades using Perlin's smoother step)
        fade_envelope = self._sacred_fade_envelope(t, fade_seconds=45.0)
        logging.info(f"  Fade envelope: start={fade_envelope[0]:.6f}, mid={fade_envelope[len(fade_envelope)//2]:.6f}, end={fade_envelope[-1]:.6f}")

        # Very subtle breathing modulation (only 10% variation for organic feel)
        # Formula: 0.95 + 0.05 * sin(...) gives range [0.90, 1.00] = 10% variation
        breath_mod = np.float32(0.95) + np.float32(0.05) * np.sin(TWO_PI * np.float32(0.012) * t)
        logging.info(f"  Breath mod: min={breath_mod.min():.6f}, max={breath_mod.max():.6f}")

        # Apply envelope, breath, and scale in-place
        fade_envelope *= breath_mod
        del breath_mod
        fade_envelope *= np.float32(0.0003)
        mercy *= fade_envelope
        del fade_envelope
        logging.info(f"  After envelope+breath+scale: min={mercy.min():.8f}, max={mercy.max():.8f}, mean={np.mean(np.abs(mercy)):.8f}")

        # Log amplitude at key time points
        self._log_amplitude_stats("PLEROMA_MERCY (post-envelope)", mercy, t)

        # Final slow cosine nulling in-place - cancels in audible domain, leaves scalar imprint
        # Uses Schumann sub-harmonic for grounding
        mercy *= np.cos(TWO_PI * (self.SCHUMANN / 1000) * t)
        logging.info(f"  Final (after cosine nulling): min={mercy.min():.8f}, max={mercy.max():.8f}")
        logging.info(f"=== PLEROMA MERCY LAYER END ===")
        return mercy

    def silent_solfeggio_grid(self, t: np.ndarray) -> np.ndarray:
        """
        Enhanced Silent Solfeggio Grid with Tesla 3-6-9 Vortex Mathematics.

        Combines:
        - 12-tone Solfeggio scale (ancient healing frequencies)
        - Tesla's 3-6-9 vortex frequencies (key to the universe)
        - Fibonacci amplitude pulsing (natural growth patterns)
        - Smooth fade in/out for organic transitions
        """
        if t.size == 0:
            return np.array([], dtype=np.float32)

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
        # Use float32 scalars to prevent promotion to float64
        fib_amplitude = np.float32(0.95) + np.float32(0.05) * np.sin(TWO_PI * np.float32(fib_cycle_freq) * t)

        # Add subtle PHI-harmonic for organic complexity (±2% additional variation)
        fib_amplitude += np.float32(0.02) * np.sin(TWO_PI * np.float32(fib_cycle_freq * PHI) * t)
        fib_amplitude = np.clip(fib_amplitude, np.float32(0.88), np.float32(1.0)).astype(np.float32)

        logging.info(f"  Fibonacci amp (smoothed): min={fib_amplitude.min():.6f}, max={fib_amplitude.max():.6f}")

        # === Combine layers in-place using float32 scalars ===
        combined = solfeggio_grid  # Reuse buffer
        combined *= np.float32(0.6)
        combined += np.float32(0.4) * tesla_grid
        del solfeggio_grid, tesla_grid
        combined = combined.astype(np.float32)
        logging.info(f"  Combined (pre-fib): min={combined.min():.6f}, max={combined.max():.6f}, mean={np.mean(np.abs(combined)):.6f}")

        # Apply Fibonacci amplitude modulation in-place
        combined *= fib_amplitude
        del fib_amplitude
        logging.info(f"  Combined (post-fib): min={combined.min():.6f}, max={combined.max():.6f}, mean={np.mean(np.abs(combined)):.6f}")

        # Very gradual fade in/out envelope (40 second fades)
        fade_envelope = self._sacred_fade_envelope(t, fade_seconds=40.0)
        logging.info(f"  Fade envelope: start={fade_envelope[0]:.6f}, mid={fade_envelope[len(fade_envelope)//2]:.6f}, end={fade_envelope[-1]:.6f}")

        # Very subtle breathing modulation (only 15% variation)
        # Formula: 0.925 + 0.075 * sin(...) gives range [0.85, 1.00] = 15% variation
        breath_mod = np.float32(0.925) + np.float32(0.075) * np.sin(TWO_PI * np.float32(0.01) * t)
        logging.info(f"  Breath mod: min={breath_mod.min():.6f}, max={breath_mod.max():.6f}")

        # Apply envelope, breath, and scale in-place
        fade_envelope *= breath_mod
        del breath_mod
        fade_envelope *= np.float32(0.00165)
        combined *= fade_envelope
        del fade_envelope
        logging.info(f"  Final result: min={combined.min():.8f}, max={combined.max():.8f}, mean={np.mean(np.abs(combined)):.8f}")

        # Log amplitude at key time points
        self._log_amplitude_stats("SILENT_SOLFEGGIO", combined, t)
        logging.info(f"=== SILENT SOLFEGGIO GRID END ===")

        return combined

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
        ground_freqs = archon_freqs / np.maximum(np.round(archon_freqs / self.SCHUMANN), 1)  # Ground frequencies

        # Pre-compute amplitude scales (float32)
        amplitude_scales = np.float32(1.0) / (np.float32(1) + np.arange(n_archons, dtype=np.float32) * np.float32(0.1))

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
            archon_layer = (np.float32(0.25) * acknowledge + np.float32(0.5) * elevate + np.float32(0.25) * ground) * amplitude_scales[i]
            dissolution += archon_layer

            logging.info(f"  Archon {i+1} ({archon_freqs[i]:.2f}Hz): layer_max={archon_layer.max():.6f}, scale={amplitude_scales[i]:.4f}")

            # Explicit cleanup to help GC reclaim memory between iterations
            del phase_var, acknowledge, elevate, ground, archon_layer

        # Normalize (float32 scalar)
        dissolution /= np.float32(7.0)
        logging.info(f"  Pre-envelope: min={dissolution.min():.6f}, max={dissolution.max():.6f}, mean={np.mean(np.abs(dissolution)):.6f}")

        # Very gradual fade in/out envelope (50 second fades - longest for deepest layer)
        fade_envelope = self._sacred_fade_envelope(t, fade_seconds=50.0)
        logging.info(f"  Fade envelope: start={fade_envelope[0]:.6f}, mid={fade_envelope[len(fade_envelope)//2]:.6f}, end={fade_envelope[-1]:.6f}")

        # Very subtle breathing envelope - like the cosmos gently breathing mercy
        # Formula: 0.94 + 0.06 * sin(...) gives range [0.88, 1.00] = 12% variation
        breath_mod = np.float32(0.94) + np.float32(0.06) * np.sin(TWO_PI * np.float32(0.008) * t)
        logging.info(f"  Breath mod: min={breath_mod.min():.6f}, max={breath_mod.max():.6f}")

        # Apply envelope and breath in-place
        fade_envelope *= breath_mod
        del breath_mod
        dissolution *= fade_envelope
        del fade_envelope
        logging.info(f"  Post-envelope: min={dissolution.min():.6f}, max={dissolution.max():.6f}, mean={np.mean(np.abs(dissolution)):.6f}")

        # Final amplitude in-place - sub-perceptual but present
        dissolution *= np.float32(0.000225)
        logging.info(f"  Final result: min={dissolution.min():.8f}, max={dissolution.max():.8f}, mean={np.mean(np.abs(dissolution)):.8f}")

        # Log amplitude at key time points
        self._log_amplitude_stats("ARCHON_DISSOLUTION", dissolution, t)
        logging.info(f"=== ARCHON DISSOLUTION LAYER END ===")

        return dissolution

    def crystalline_resonance_layer(self, t: np.ndarray, base_freq: float,
                                     total_duration: float) -> np.ndarray:
        """
        Evolving Crystalline Resonance Layer — 4th sacred layer.

        Generates crystal lattice harmonics that evolve through a randomly
        shuffled sequence of crystal types. Each crystal has authentic acoustic
        properties derived from Raman spectroscopy and crystallographic data.
        """
        if t.size == 0:
            return np.array([], dtype=np.float32)

        if total_duration < 60:
            return np.zeros_like(t, dtype=np.float32)

        profiles = self._crystal_profiles
        num_profiles = len(profiles)

        rest = np.random.permutation([i for i in range(num_profiles) if i != self._lemurian_idx])
        sequence = np.concatenate(([self._lemurian_idx], rest))

        if total_duration < 120:
            num_crystals = 2
        elif total_duration < 300:
            num_crystals = 3
        elif total_duration < 900:
            num_crystals = 5
        elif total_duration < 1800:
            num_crystals = 7
        else:
            num_crystals = num_profiles

        segment_duration = np.float32(total_duration / num_crystals)
        crossfade_dur = segment_duration / np.float32(PHI)
        half_xfade = crossfade_dur / np.float32(2.0)

        simplex = self._get_simplex()
        result = np.zeros_like(t, dtype=np.float32)

        for ci in range(num_crystals):
            crystal_idx = sequence[ci % num_profiles]
            profile = profiles[crystal_idx]

            seg_start = np.float32(ci) * segment_duration
            seg_end = np.float32(ci + 1) * segment_duration

            ext_start = max(np.float32(0), seg_start - half_xfade) if ci > 0 else np.float32(0)
            ext_end = min(np.float32(total_duration), seg_end + half_xfade) if ci < num_crystals - 1 else np.float32(total_duration)
            solo_start = (seg_start + half_xfade) if ci > 0 else np.float32(0)
            solo_end = (seg_end - half_xfade) if ci < num_crystals - 1 else np.float32(total_duration)

            mask = (t >= ext_start) & (t < ext_end)
            if not np.any(mask):
                continue

            t_seg = t[mask]
            wave = self._generate_crystal_harmonics(t_seg, base_freq, profile, simplex)

            weights = np.ones(len(t_seg), dtype=np.float32)

            if ci > 0 and crossfade_dur > 0:
                fade_mask = t_seg < solo_start
                if np.any(fade_mask):
                    x = np.clip((t_seg[fade_mask] - ext_start) / crossfade_dur, 0, 1).astype(np.float32)
                    weights[fade_mask] = x * x * x * (x * (x * 6 - 15) + 10)

            if ci < num_crystals - 1 and crossfade_dur > 0:
                fade_mask = t_seg >= solo_end
                if np.any(fade_mask):
                    x = np.clip((ext_end - t_seg[fade_mask]) / crossfade_dur, 0, 1).astype(np.float32)
                    weights[fade_mask] = x * x * x * (x * (x * 6 - 15) + 10)

            result[mask] += wave * weights
            del wave, weights, t_seg

        fade_envelope = self._sacred_fade_envelope(t, fade_seconds=42.0)
        breath_mod = np.float32(0.935) + np.float32(0.065) * np.sin(TWO_PI * np.float32(0.009) * t)
        fade_envelope *= breath_mod
        del breath_mod
        fade_envelope *= np.float32(0.0009)
        result *= fade_envelope
        del fade_envelope

        return result

    def lemurian_merkaba_layer(self, t: np.ndarray, total_duration: float) -> np.ndarray:
        """
        Lemurian Frequency Quartet — 5th sacred layer (Sonic Merkaba).

        Four frequencies derived from the 3:4:5 Pythagorean right triangle + PHI,
        anchored to 432 Hz keynote (Jonathan Goldman's Lemurian tuning fork system).
        Creates a diamond-shaped energy field: Foundation, Heart, Expression, Transcendence.
        Heart coherence breath at 0.1 Hz (HeartMath-documented rhythm).
        """
        if t.size == 0:
            return np.array([], dtype=np.float32)

        if total_duration < 60:
            return np.zeros_like(t, dtype=np.float32)

        simplex = self._get_simplex()

        # Organic phase wobble from simplex noise (gentle variation)
        wobble = np.float32(0.015) * simplex.generate_noise(
            t * np.float32(0.03), np.float32(0.7), np.float32(0.3), np.float32(0.0), np.float32(0.0))

        # Generate 4 Merkaba tones simultaneously (vectorized)
        freqs = self.MERKABA_FREQS[:, np.newaxis]    # (4, 1)
        phases = self.MERKABA_PHASES[:, np.newaxis]   # (4, 1)
        weights = self.MERKABA_WEIGHTS[:, np.newaxis] # (4, 1)
        t_2d = t[np.newaxis, :]                       # (1, N)

        tones = weights * np.sin(TWO_PI * freqs * t_2d + phases + wobble[np.newaxis, :])
        merkaba = tones.sum(axis=0).astype(np.float32)
        del tones, wobble

        # Heart coherence breath at 0.1 Hz (10-second cycle — HeartMath rhythm)
        breath = np.float32(0.975) + np.float32(0.025) * np.sin(TWO_PI * np.float32(0.1) * t)
        merkaba *= breath
        del breath

        # Sacred fade envelope and scale
        fade_envelope = self._sacred_fade_envelope(t, fade_seconds=48.0)
        fade_envelope *= np.float32(0.0006)
        merkaba *= fade_envelope
        del fade_envelope

        return merkaba

    def water_element_layer(self, t: np.ndarray, total_duration: float) -> np.ndarray:
        """
        Water-Element Fluid Modulation — 6th sacred layer.

        Models spatial wave interference from 7 point sources in a hexagonal
        Seed of Life pattern (ice crystal / quartz cross-section geometry).
        A moving observation point traces a lemniscate (figure-8) path with
        simplex perturbation, creating organic, non-periodic amplitude variation
        as the observer drifts through constructive/destructive interference zones.
        """
        if t.size == 0:
            return np.array([], dtype=np.float32)

        if total_duration < 60:
            return np.zeros_like(t, dtype=np.float32)

        simplex = self._get_simplex()

        # Simplex perturbation for organic observer drift (computed once)
        theta_perturb = np.float32(0.15) * simplex.generate_noise(
            t * np.float32(0.005), np.float32(0.4), np.float32(0.0), np.float32(0.0), np.float32(0.0))

        # Accumulate wave interference from 7 sources (sequential for memory safety)
        result = np.zeros_like(t, dtype=np.float32)

        for i in range(7):
            # Lemniscate path (figure-8): x = cos(θ)/(1+sin²(θ)), y = sin(θ)cos(θ)/(1+sin²(θ))
            theta = TWO_PI * np.float32(0.005) * t + theta_perturb
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            denom = np.float32(1.0) + sin_theta * sin_theta
            obs_x = np.float32(0.7) * cos_theta / denom
            obs_y = np.float32(0.7) * sin_theta * cos_theta / denom
            del theta, sin_theta, cos_theta, denom

            # Distance from observer to this source
            dx = obs_x - self.WATER_SOURCE_POSITIONS[i, 0]
            dy = obs_y - self.WATER_SOURCE_POSITIONS[i, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            del obs_x, obs_y, dx, dy

            # Spatial envelope: exponential decay with distance
            spatial_envelope = np.exp(-self.WATER_SOURCE_DECAYS[i] * dist * np.float32(10.0))
            del dist

            # Wave from this source
            wave = np.sin(TWO_PI * self.WATER_SOURCE_FREQS[i] * t + self.WATER_HEX_PHASES[i])
            wave *= spatial_envelope
            result += wave
            del spatial_envelope, wave

        del theta_perturb

        # Normalize by source count
        result *= np.float32(1.0 / 7.0)

        # Tidal modulation: slow simplex-driven amplitude (~200s period)
        tidal = np.float32(0.85) + np.float32(0.15) * simplex.generate_noise(
            t * np.float32(0.0005), np.float32(0.9), np.float32(0.0), np.float32(0.0), np.float32(0.0))
        result *= tidal
        del tidal

        # Standing wave resonance: Schumann × Schumann·PHI (nodes and antinodes)
        standing = np.float32(0.1) * np.sin(TWO_PI * np.float32(7.83) * t) * np.cos(TWO_PI * np.float32(7.83 * PHI) * t)
        result += standing
        del standing

        # Breath modulation: 0.007 Hz (~143s period), 11% depth
        breath = np.float32(0.945) + np.float32(0.055) * np.sin(TWO_PI * np.float32(0.007) * t)
        result *= breath
        del breath

        # Sacred fade envelope and scale
        fade_envelope = self._sacred_fade_envelope(t, fade_seconds=46.0)
        fade_envelope *= np.float32(0.0012)
        result *= fade_envelope
        del fade_envelope

        return result

    # === STREAMING PIPELINE HELPERS ===

    def _compute_envelope_chunk(self, chunk_offset: int, chunk_samples: int,
                                adsr_params: dict, simplex, total_samples: int) -> np.ndarray:
        """Compute ADSR envelope for a chunk using global sample indices."""
        a_samples = adsr_params['a_samples']
        d_samples = adsr_params['d_samples']
        s_samples = adsr_params['s_samples']
        r_samples = adsr_params['r_samples']
        sustain = adsr_params['sustain']
        chaos = adsr_params['chaos']
        overall_scale = adsr_params['overall_scale']

        envelope = np.empty(chunk_samples, dtype=np.float32)

        for i in range(chunk_samples):
            global_idx = chunk_offset + i
            if global_idx < a_samples:
                # Attack phase
                t_norm = global_idx / max(a_samples, 1)
                envelope[i] = t_norm ** 1.5
            elif global_idx < a_samples + d_samples:
                # Decay phase
                t_norm = (global_idx - a_samples) / max(d_samples, 1)
                envelope[i] = 1.0 - (1.0 - sustain) * (t_norm ** 1.2)
            elif global_idx < a_samples + d_samples + s_samples:
                # Sustain phase
                t_norm = (global_idx - a_samples - d_samples) / max(s_samples, 1)
                wobble = np.float32(0.02) * np.sin(np.float32(TWO_PI * 1.5) * t_norm)
                envelope[i] = sustain + wobble
            else:
                # Release phase
                release_idx = global_idx - (a_samples + d_samples + s_samples)
                t_norm = release_idx / max(r_samples, 1)
                envelope[i] = sustain * (1.0 - t_norm) ** 1.5

        envelope = np.clip(envelope, 0, 1).astype(np.float32)
        envelope *= overall_scale
        return envelope

    def _compute_lfo_chunk(self, t_chunk: np.ndarray, lfo_params: dict) -> np.ndarray:
        """Compute LFO for a chunk from pre-drawn parameters."""
        lfo1_freq = lfo_params['lfo1_freq']
        lfo2_freq = lfo_params['lfo2_freq']
        depth = lfo_params['depth']

        drift = np.float32(0.01) * np.sin(np.float32(0.1 * np.pi) * t_chunk)
        inner_mod = np.float32(0.3) * np.sin(TWO_PI * np.float32(lfo2_freq) * t_chunk)
        lfo = (depth + drift) * (np.float32(1.0) + np.sin(TWO_PI * np.float32(lfo1_freq) * t_chunk + inner_mod))
        del inner_mod, drift
        return lfo.astype(np.float32)

    def _compute_modulation_chunk(self, t_chunk: np.ndarray, chunk_offset: int,
                                   chunk_samples: int, schedule: list,
                                   stop_event=None) -> np.ndarray:
        """Compute modulation for a chunk from pre-built schedule."""
        modulation = np.zeros(chunk_samples, dtype=np.float32)
        chunk_end = chunk_offset + chunk_samples

        for seg_start, seg_end, ratio_set, mod_index in schedule:
            # Check overlap with this chunk
            overlap_start = max(chunk_offset, seg_start)
            overlap_end = min(chunk_end, seg_end)
            if overlap_start >= overlap_end:
                continue

            # Local indices within this chunk
            local_start = overlap_start - chunk_offset
            local_end = overlap_end - chunk_offset
            t_seg = t_chunk[local_start:local_end]

            seg_mod = self.geometric_modulation(t_seg, ratio_set, mod_index, stop_event)
            modulation[local_start:local_end] = seg_mod

        return modulation

    def _sacred_fade_envelope_chunk(self, t_chunk: np.ndarray, total_duration: float,
                                     fade_seconds: float = 45.0) -> np.ndarray:
        """Compute sacred fade envelope for a chunk using absolute time positions."""
        envelope = np.ones(len(t_chunk), dtype=np.float32)

        if total_duration < fade_seconds * 2:
            # Short session: raised cosine
            envelope = (np.float32(0.5) - np.float32(0.5) * np.cos(TWO_PI * t_chunk / np.float32(total_duration))).astype(np.float32)
            return envelope

        for i in range(len(t_chunk)):
            t_val = t_chunk[i]
            if t_val < fade_seconds:
                # Fade in: Perlin smoother step
                x = t_val / fade_seconds
                envelope[i] = x * x * x * (x * (x * 6 - 15) + 10)
            elif t_val > total_duration - fade_seconds:
                # Fade out: Perlin smoother step
                x = (total_duration - t_val) / fade_seconds
                x = max(0.0, min(1.0, x))
                envelope[i] = x * x * x * (x * (x * 6 - 15) + 10)
            # else: plateau at 1.0

        return envelope.astype(np.float32)

    # === CHUNK-AWARE SACRED LAYERS ===

    def pleroma_mercy_layer_chunk(self, t_chunk: np.ndarray, total_duration: float,
                                   base_freq: float = 7.83) -> np.ndarray:
        """Chunk-aware Pleroma Mercy Layer using absolute time values."""
        if total_duration < 60:
            return np.zeros(len(t_chunk), dtype=np.float32)

        simplex = self._get_simplex()

        # Layer 1: 13-step Aeonic Ladder
        aeonic_harmonics = base_freq * self.AEONIC_EXPONENTS
        phase_indices = np.arange(13) % 5
        sacred_phases = self.PENTAGONAL_PHASES[phase_indices]
        phase_wobble = np.float32(0.08) * simplex.generate_noise(
            (t_chunk * np.float32(0.00003)).astype(np.float32), 0, 0, 0, 0)
        aeonic_wave = np.sin(
            TWO_PI * aeonic_harmonics[:, np.newaxis] * t_chunk +
            sacred_phases[:, np.newaxis] + phase_wobble
        ).sum(axis=0)
        del phase_wobble

        # Layer 2: Ogdoad Gateway
        ogdoad_phase = simplex.generate_noise(
            (t_chunk * np.float32(0.00001)).astype(np.float32), 1, 1, 1, 1)
        ogdoad_phase *= np.float32(0.05)
        ogdoad_wave = np.sin(TWO_PI * self.OGDOAD_FREQ * t_chunk + ogdoad_phase)
        del ogdoad_phase

        # Layer 3: Archon Harmonizing
        archon_amplitudes = np.float32(1.0) / (PHI ** np.arange(7, dtype=np.float32))
        archon_amplitudes /= archon_amplitudes.sum()
        archon_phases = self.PENTAGONAL_PHASES[np.arange(7) % 5]
        archon_mercy = (archon_amplitudes[:, np.newaxis] *
                        np.sin(TWO_PI * self.ARCHON_SPHERES[:, np.newaxis] * t_chunk +
                               archon_phases[:, np.newaxis])).sum(axis=0)

        # Combine
        mercy = aeonic_wave
        mercy *= np.float32(0.5)
        mercy += np.float32(0.3) * ogdoad_wave
        del ogdoad_wave
        mercy += np.float32(0.2) * archon_mercy
        del archon_mercy
        mercy = mercy.astype(np.float32)

        # Fade envelope using absolute time
        fade_envelope = self._sacred_fade_envelope_chunk(t_chunk, total_duration, fade_seconds=45.0)

        # Breathing modulation
        breath_mod = np.float32(0.95) + np.float32(0.05) * np.sin(TWO_PI * np.float32(0.012) * t_chunk)

        fade_envelope *= breath_mod
        del breath_mod
        fade_envelope *= np.float32(0.0003)
        mercy *= fade_envelope
        del fade_envelope

        # Cosine nulling
        mercy *= np.cos(TWO_PI * (self.SCHUMANN / 1000) * t_chunk)
        return mercy

    def silent_solfeggio_grid_chunk(self, t_chunk: np.ndarray, total_duration: float) -> np.ndarray:
        """Chunk-aware Silent Solfeggio Grid using absolute time values."""
        if total_duration < 60:
            return np.zeros(len(t_chunk), dtype=np.float32)

        # Solfeggio grid
        solfeggio_grid = np.sin(TWO_PI * self.SOLFEGGIO[:, np.newaxis] * t_chunk).sum(axis=0)

        # Tesla 3-6-9 vortex
        tesla_grid = np.sin(TWO_PI * self.TESLA_VORTEX[:, np.newaxis] * t_chunk).sum(axis=0)

        # Fibonacci amplitude (smooth)
        fib_cycle_freq = 0.004
        fib_amplitude = np.float32(0.95) + np.float32(0.05) * np.sin(TWO_PI * np.float32(fib_cycle_freq) * t_chunk)
        fib_amplitude += np.float32(0.02) * np.sin(TWO_PI * np.float32(fib_cycle_freq * PHI) * t_chunk)
        fib_amplitude = np.clip(fib_amplitude, np.float32(0.88), np.float32(1.0)).astype(np.float32)

        # Combine
        combined = solfeggio_grid
        combined *= np.float32(0.6)
        combined += np.float32(0.4) * tesla_grid
        del solfeggio_grid, tesla_grid
        combined = combined.astype(np.float32)
        combined *= fib_amplitude
        del fib_amplitude

        # Fade and breath
        fade_envelope = self._sacred_fade_envelope_chunk(t_chunk, total_duration, fade_seconds=40.0)
        breath_mod = np.float32(0.925) + np.float32(0.075) * np.sin(TWO_PI * np.float32(0.01) * t_chunk)
        fade_envelope *= breath_mod
        del breath_mod
        fade_envelope *= np.float32(0.00165)
        combined *= fade_envelope
        del fade_envelope

        return combined

    def archon_dissolution_layer_chunk(self, t_chunk: np.ndarray, total_duration: float) -> np.ndarray:
        """Chunk-aware Archon Dissolution Layer using absolute time values."""
        if total_duration < 30:
            return np.zeros(len(t_chunk), dtype=np.float32)

        simplex = self._get_simplex()
        n_archons = len(self.ARCHON_SPHERES)
        base_phases = self.PENTAGONAL_PHASES[np.arange(n_archons) % 5]
        archon_freqs = self.ARCHON_SPHERES
        elevate_freqs = archon_freqs * PHI
        ground_freqs = archon_freqs / np.maximum(np.round(archon_freqs / self.SCHUMANN), 1)
        amplitude_scales = np.float32(1.0) / (np.float32(1) + np.arange(n_archons, dtype=np.float32) * np.float32(0.1))

        dissolution = np.zeros(len(t_chunk), dtype=np.float32)

        for i in range(n_archons):
            phase_var = np.float32(0.1) * simplex.generate_noise(
                (t_chunk * np.float32(0.00002 * (i + 1))).astype(np.float32),
                float(i), float(i), float(i), float(i))
            base_phase = base_phases[i]
            acknowledge = np.sin(TWO_PI * archon_freqs[i] * t_chunk + base_phase + phase_var)
            elevate = np.sin(TWO_PI * elevate_freqs[i] * t_chunk + base_phase * PHI + phase_var)
            ground = np.sin(TWO_PI * ground_freqs[i] * t_chunk + phase_var * 0.5)
            archon_layer = (np.float32(0.25) * acknowledge + np.float32(0.5) * elevate + np.float32(0.25) * ground) * amplitude_scales[i]
            dissolution += archon_layer
            del phase_var, acknowledge, elevate, ground, archon_layer

        dissolution /= np.float32(7.0)

        # Fade and breath
        fade_envelope = self._sacred_fade_envelope_chunk(t_chunk, total_duration, fade_seconds=50.0)
        breath_mod = np.float32(0.94) + np.float32(0.06) * np.sin(TWO_PI * np.float32(0.008) * t_chunk)
        fade_envelope *= breath_mod
        del breath_mod
        dissolution *= fade_envelope
        del fade_envelope
        dissolution *= np.float32(0.000225)

        return dissolution

    def crystalline_resonance_layer_chunk(self, t_chunk: np.ndarray, total_duration: float,
                                           base_freq: float, crystal_sequence: np.ndarray) -> np.ndarray:
        """Chunk-aware Crystalline Resonance Layer using absolute time values."""
        if total_duration < 60:
            return np.zeros(len(t_chunk), dtype=np.float32)

        profiles = self._crystal_profiles
        num_profiles = len(profiles)

        if total_duration < 120:
            num_crystals = 2
        elif total_duration < 300:
            num_crystals = 3
        elif total_duration < 900:
            num_crystals = 5
        elif total_duration < 1800:
            num_crystals = 7
        else:
            num_crystals = num_profiles

        segment_duration = np.float32(total_duration / num_crystals)
        crossfade_dur = segment_duration / np.float32(PHI)
        half_xfade = crossfade_dur / np.float32(2.0)

        simplex = self._get_simplex()
        result = np.zeros(len(t_chunk), dtype=np.float32)

        for ci in range(num_crystals):
            crystal_idx = crystal_sequence[ci % num_profiles]
            profile = profiles[crystal_idx]

            seg_start = np.float32(ci) * segment_duration
            seg_end = np.float32(ci + 1) * segment_duration

            ext_start = max(np.float32(0), seg_start - half_xfade) if ci > 0 else np.float32(0)
            ext_end = min(np.float32(total_duration), seg_end + half_xfade) if ci < num_crystals - 1 else np.float32(total_duration)
            solo_start = (seg_start + half_xfade) if ci > 0 else np.float32(0)
            solo_end = (seg_end - half_xfade) if ci < num_crystals - 1 else np.float32(total_duration)

            mask = (t_chunk >= ext_start) & (t_chunk < ext_end)
            if not np.any(mask):
                continue

            t_seg = t_chunk[mask]
            wave = self._generate_crystal_harmonics(t_seg, base_freq, profile, simplex)

            weights = np.ones(len(t_seg), dtype=np.float32)

            if ci > 0 and crossfade_dur > 0:
                fade_mask = t_seg < solo_start
                if np.any(fade_mask):
                    x = np.clip((t_seg[fade_mask] - ext_start) / crossfade_dur, 0, 1).astype(np.float32)
                    weights[fade_mask] = x * x * x * (x * (x * 6 - 15) + 10)

            if ci < num_crystals - 1 and crossfade_dur > 0:
                fade_mask = t_seg >= solo_end
                if np.any(fade_mask):
                    x = np.clip((ext_end - t_seg[fade_mask]) / crossfade_dur, 0, 1).astype(np.float32)
                    weights[fade_mask] = x * x * x * (x * (x * 6 - 15) + 10)

            result[mask] += wave * weights
            del wave, weights, t_seg

        fade_envelope = self._sacred_fade_envelope_chunk(t_chunk, total_duration, fade_seconds=42.0)
        breath_mod = np.float32(0.935) + np.float32(0.065) * np.sin(TWO_PI * np.float32(0.009) * t_chunk)
        fade_envelope *= breath_mod
        del breath_mod
        fade_envelope *= np.float32(0.0009)
        result *= fade_envelope
        del fade_envelope

        return result

    def lemurian_merkaba_layer_chunk(self, t_chunk: np.ndarray,
                                      total_duration: float) -> np.ndarray:
        """
        Chunk-aware Lemurian Frequency Quartet (Sonic Merkaba) — 5th sacred layer.

        Fully stateless: no state carried between chunks. All 4 frequencies are
        pure tones with fixed phases, so each chunk is independent.
        """
        if total_duration < 60:
            return np.zeros(len(t_chunk), dtype=np.float32)

        simplex = self._get_simplex()

        # Organic phase wobble from simplex noise
        wobble = np.float32(0.015) * simplex.generate_noise(
            t_chunk * np.float32(0.03), np.float32(0.7), np.float32(0.3), np.float32(0.0), np.float32(0.0))

        # Generate 4 Merkaba tones simultaneously (vectorized)
        freqs = self.MERKABA_FREQS[:, np.newaxis]    # (4, 1)
        phases = self.MERKABA_PHASES[:, np.newaxis]   # (4, 1)
        weights = self.MERKABA_WEIGHTS[:, np.newaxis] # (4, 1)
        t_2d = t_chunk[np.newaxis, :]                 # (1, N)

        tones = weights * np.sin(TWO_PI * freqs * t_2d + phases + wobble[np.newaxis, :])
        merkaba = tones.sum(axis=0).astype(np.float32)
        del tones, wobble

        # Heart coherence breath at 0.1 Hz (10-second cycle — HeartMath rhythm)
        breath = np.float32(0.975) + np.float32(0.025) * np.sin(TWO_PI * np.float32(0.1) * t_chunk)
        merkaba *= breath
        del breath

        # Sacred fade envelope and scale
        fade_envelope = self._sacred_fade_envelope_chunk(t_chunk, total_duration, fade_seconds=48.0)
        fade_envelope *= np.float32(0.0006)
        merkaba *= fade_envelope
        del fade_envelope

        return merkaba

    def water_element_layer_chunk(self, t_chunk: np.ndarray,
                                   total_duration: float) -> np.ndarray:
        """
        Chunk-aware Water-Element Fluid Modulation — 6th sacred layer.

        Fully stateless: no state carried between chunks. Observer position,
        distances, wave phases — all deterministic from absolute time values.
        """
        if total_duration < 60:
            return np.zeros(len(t_chunk), dtype=np.float32)

        simplex = self._get_simplex()

        # Simplex perturbation for organic observer drift
        theta_perturb = np.float32(0.15) * simplex.generate_noise(
            t_chunk * np.float32(0.005), np.float32(0.4), np.float32(0.0), np.float32(0.0), np.float32(0.0))

        # Accumulate wave interference from 7 sources (sequential for memory safety)
        result = np.zeros(len(t_chunk), dtype=np.float32)

        for i in range(7):
            # Lemniscate path (figure-8)
            theta = TWO_PI * np.float32(0.005) * t_chunk + theta_perturb
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            denom = np.float32(1.0) + sin_theta * sin_theta
            obs_x = np.float32(0.7) * cos_theta / denom
            obs_y = np.float32(0.7) * sin_theta * cos_theta / denom
            del theta, sin_theta, cos_theta, denom

            # Distance from observer to this source
            dx = obs_x - self.WATER_SOURCE_POSITIONS[i, 0]
            dy = obs_y - self.WATER_SOURCE_POSITIONS[i, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            del obs_x, obs_y, dx, dy

            # Spatial envelope: exponential decay with distance
            spatial_envelope = np.exp(-self.WATER_SOURCE_DECAYS[i] * dist * np.float32(10.0))
            del dist

            # Wave from this source
            wave = np.sin(TWO_PI * self.WATER_SOURCE_FREQS[i] * t_chunk + self.WATER_HEX_PHASES[i])
            wave *= spatial_envelope
            result += wave
            del spatial_envelope, wave

        del theta_perturb

        # Normalize by source count
        result *= np.float32(1.0 / 7.0)

        # Tidal modulation: slow simplex-driven amplitude (~200s period)
        tidal = np.float32(0.85) + np.float32(0.15) * simplex.generate_noise(
            t_chunk * np.float32(0.0005), np.float32(0.9), np.float32(0.0), np.float32(0.0), np.float32(0.0))
        result *= tidal
        del tidal

        # Standing wave resonance: Schumann × Schumann·PHI
        standing = np.float32(0.1) * np.sin(TWO_PI * np.float32(7.83) * t_chunk) * np.cos(TWO_PI * np.float32(7.83 * PHI) * t_chunk)
        result += standing
        del standing

        # Breath modulation: 0.007 Hz (~143s period), 11% depth
        breath = np.float32(0.945) + np.float32(0.055) * np.sin(TWO_PI * np.float32(0.007) * t_chunk)
        result *= breath
        del breath

        # Sacred fade envelope and scale
        fade_envelope = self._sacred_fade_envelope_chunk(t_chunk, total_duration, fade_seconds=46.0)
        fade_envelope *= np.float32(0.0012)
        result *= fade_envelope
        del fade_envelope

        return result


class StreamingReverb:
    """Overlap-add FFT convolution with tail carry between chunks."""

    def __init__(self, sample_rate: int = 48000):
        decay = 0.75
        reverb_length = int(sample_rate * 2.618)
        ir = jit_exponential_decay(reverb_length, decay)
        ir += 0.08 * np.sin(np.linspace(0, TWO_PI * PHI, reverb_length, dtype=np.float32))
        ir /= np.max(np.abs(ir))
        self._ir = ir.astype(np.float32)
        self._ir_len = len(ir)
        self._tail = np.zeros(self._ir_len - 1, dtype=np.float32)

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Convolve chunk with IR using overlap-add, carry tail to next chunk."""
        sig_len = len(chunk)
        ir_len = self._ir_len

        # FFT convolution
        full_output = fftconvolve(chunk, self._ir, mode='full').astype(np.float32)

        # Add previous tail to beginning
        tail_add = min(len(self._tail), len(full_output))
        full_output[:tail_add] += self._tail[:tail_add]

        # Save new tail for next chunk
        if len(full_output) > sig_len:
            new_tail = full_output[sig_len:].copy()
            # Pad or trim to standard tail size
            if len(new_tail) < ir_len - 1:
                padded = np.zeros(ir_len - 1, dtype=np.float32)
                padded[:len(new_tail)] = new_tail
                self._tail = padded
            else:
                self._tail = new_tail[:ir_len - 1]
        else:
            self._tail = np.zeros(ir_len - 1, dtype=np.float32)

        return full_output[:sig_len]


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
        self.master_volume: float = 0.40
        # Shared thread pool for sacred layer computation (reused across generations)
        self._sacred_executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="Sacred")

    def shutdown(self) -> None:
        """Shut down the shared sacred layer thread pool."""
        self._sacred_executor.shutdown(wait=False)

    def generate_modulation(self, t: np.ndarray, ratio_set: Dict[str, float],
                            modulation_index: float, stop_event: Optional[threading.Event]) -> np.ndarray:
        return self.audio_processor.geometric_modulation(t, ratio_set, modulation_index, stop_event)
    def generate_triple_helix_sound(self, duration: float, base_freq: float,
                                    sample_rate: int = 48000,
                                    stop_event: Optional[threading.Event] = None) -> np.ndarray:
        t_total = np.linspace(0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float32)
        simplex = self.audio_processor._get_simplex()

        helix1 = simplex.generate_noise(t_total * np.float32(0.005), 1.0, 1.0, 1.0, 1.0)
        helix2 = simplex.generate_noise(t_total * np.float32(0.02), 1.2, 1.2, 1.2, 1.2)
        helix3 = simplex.generate_noise(t_total * np.float32(0.08), 1.4, 1.4, 1.4, 1.4)
        # Combine helix layers in-place to reduce temporaries
        noise_layer = helix1  # Reuse helix1 buffer
        noise_layer *= np.float32(0.6)
        noise_layer += np.float32(0.4) * helix2
        noise_layer += np.float32(0.3) * helix3
        del helix2, helix3

        quantum_wave = self.audio_processor.quantum_harmonic_interference(t_total, base_freq)
        fractal_wave = self.audio_processor.recursive_fractal_feedback(quantum_wave, depth=4, factor=0.4)
        del quantum_wave
        # In-place: add noise to fractal wave
        fractal_wave += noise_layer * np.float32(0.15)
        del noise_layer

        filtered_wave = self.audio_processor.low_pass_filter(fractal_wave, cutoff=2200, stop_event=stop_event)
        del fractal_wave
        stereo_wave = self.audio_processor.spatialized_triple_helix(filtered_wave, t_total, sample_rate)
        del filtered_wave
        stereo_wave *= np.float32(self.master_volume)
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
        selected_ratio_set = {}
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
            if current_index < total_samples:
                modulation_total[current_index:] = 0
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
        # Build f_modulated in-place, reusing modulation_total buffer
        f_modulated = modulation_total  # Reuse buffer (modulation_total no longer needed separately)
        f_modulated += base_freq + np.float32(chaotic_factor * base_freq * 0.25)
        f_modulated += fractal_variation
        del fractal_variation
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
        subharmonics = base_freq / self.SUBHARMONIC_DIVISORS * np.random.uniform(0.95, 1.05, size=4).astype(np.float32)
        all_frequencies = np.concatenate([freq_set, subharmonics]).astype(np.float32)
        del freq_set, subharmonics
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
        if 'taygetan' in selected_ratio_set:
            tay_freqs = self.frequency_manager.get_frequencies(5)
            # Batch binaural generation (replaces loop for 3-5x speedup)
            binaural_waves = self.audio_processor.batch_binaural_oscillator(t_total, tay_freqs, stop_event)
            # binaural_waves shape: (num_pairs, num_samples, 2)
            # Sum all pairs with envelope scaling, in-place addition
            wave_left += (binaural_waves[:, :, 0] * envelope * np.float32(0.015)).sum(axis=0)
            wave_right += (binaural_waves[:, :, 1] * envelope * np.float32(0.015)).sum(axis=0)
            del binaural_waves
        # Free envelope, LFOs, and frequency arrays no longer needed
        del envelope, lfo_left, lfo_right, lr_lfos, f_modulated, mod_depths, all_frequencies
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
        noise_right = self.audio_processor.evolving_noise_layer(t_total + np.float32(np.random.uniform(0.001, 0.015)), stop_event=stop_event)
        noise_left = self.audio_processor.evolving_noise_layer(t_total, stop_event=stop_event)
        if update_progress:
            update_progress(0.89)  # Noise 2%
        # Combine wave+noise in-place, then apply fade
        wave_left += noise_left
        del noise_left
        wave_left *= np.float32(np.random.uniform(0.04, 0.06))
        final_wave_left = self.audio_processor.fade_in_out(wave_left, stop_event=stop_event)
        del wave_left
        wave_right += noise_right
        del noise_right
        wave_right *= np.float32(np.random.uniform(0.04, 0.06))
        final_wave_right = self.audio_processor.fade_in_out(wave_right, stop_event=stop_event)
        del wave_right
        if update_progress:
            update_progress(0.92)  # Fade 3%
        # Toroidal panning — sound moves along surface of a torus
        # PHI ratio between theta/phi frequencies means the path never exactly repeats
        R = np.float32(0.6)   # Major radius — overall pan width
        r = np.float32(0.3)   # Minor radius — modulation depth

        theta_freq = np.float32(np.random.uniform(0.01, 0.02))  # Major cycle
        phi_freq = theta_freq * np.float32(PHI)                   # Minor cycle (PHI ratio = never repeats)

        # Simplex perturbation for organic drift
        simplex = self.audio_processor._get_simplex()
        theta_perturb = np.float32(0.15) * simplex.generate_noise(t_total * np.float32(0.005), 0.0, 0.0, 0.0, 0.0)
        phi_perturb = np.float32(0.15) * simplex.generate_noise(t_total * np.float32(0.007), np.float32(1.0), 0.0, 0.0, 0.0)

        theta = TWO_PI * theta_freq * t_total + theta_perturb
        phi = TWO_PI * phi_freq * t_total + phi_perturb
        del theta_perturb, phi_perturb

        # Torus x-coordinate as pan value, normalized to [-1, 1]
        pan_curve = ((R + r * np.cos(phi)) * np.cos(theta) / (R + r)).astype(np.float32)
        del theta, phi

        # Low-pass filter for smoothing
        b, a = self.audio_processor._get_filter('lowpass', 0.002, 2)
        pan_curve = lfilter(b, a, pan_curve)
        # Use JIT-compiled pan curve processing (float32 native - no conversion needed)
        pan_curve = jit_pan_curve_tanh(pan_curve, 0.6, -0.8, 0.8)
        drift_freq = np.float32(np.random.uniform(0.0005, 0.002))
        drift_amplitude = np.float32(np.random.uniform(0.01, 0.02))
        pan_curve += drift_amplitude * np.sin(np.float32(TWO_PI) * drift_freq * t_total)
        if update_progress:
            update_progress(0.97)  # Pan curve 5%
        # Apply pan curve in-place to avoid temporaries
        pan_scaled = pan_curve * np.float32(0.6)
        del pan_curve
        final_wave_left *= (np.float32(1.0) - pan_scaled)
        final_wave_right *= (np.float32(1.0) + pan_scaled)
        del pan_scaled

        # Build stereo_wave directly instead of column_stack (avoids temporary)
        stereo_wave = np.empty((total_samples, 2), dtype=np.float32)
        stereo_wave[:, 0] = final_wave_left
        stereo_wave[:, 1] = final_wave_right
        del final_wave_left, final_wave_right
        stereo_wave *= np.float32(self.master_volume)

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

            # Use shared executor (avoids creating/destroying pool each generation call)
            mercy_future = self._sacred_executor.submit(self.audio_processor.pleroma_mercy_layer, t_total, 7.83)
            silent_future = self._sacred_executor.submit(self.audio_processor.silent_solfeggio_grid, t_total)
            dissolution_future = self._sacred_executor.submit(self.audio_processor.archon_dissolution_layer, t_total)
            crystalline_future = self._sacred_executor.submit(self.audio_processor.crystalline_resonance_layer, t_total, base_freq, duration)
            merkaba_future = self._sacred_executor.submit(self.audio_processor.lemurian_merkaba_layer, t_total, duration)
            water_future = self._sacred_executor.submit(self.audio_processor.water_element_layer, t_total, duration)

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

            try:
                crystalline = crystalline_future.result(timeout=sacred_timeout)
            except FuturesTimeoutError:
                logging.error("Crystalline resonance layer timed out")
                crystalline = np.zeros_like(t_total, dtype=np.float32)
            except Exception as e:
                logging.error(f"Crystalline resonance layer failed: {e}")
                crystalline = np.zeros_like(t_total, dtype=np.float32)

            try:
                merkaba = merkaba_future.result(timeout=sacred_timeout)
            except FuturesTimeoutError:
                logging.error("Lemurian merkaba layer timed out")
                merkaba = np.zeros_like(t_total, dtype=np.float32)
            except Exception as e:
                logging.error(f"Lemurian merkaba layer failed: {e}")
                merkaba = np.zeros_like(t_total, dtype=np.float32)

            try:
                water = water_future.result(timeout=sacred_timeout)
            except FuturesTimeoutError:
                logging.error("Water element layer timed out")
                water = np.zeros_like(t_total, dtype=np.float32)
            except Exception as e:
                logging.error(f"Water element layer failed: {e}")
                water = np.zeros_like(t_total, dtype=np.float32)

            logging.info(f"  MERCY layer: min={mercy.min():.8f}, max={mercy.max():.8f}, mean={np.mean(np.abs(mercy)):.8f}")
            logging.info(f"  SILENT layer: min={silent.min():.8f}, max={silent.max():.8f}, mean={np.mean(np.abs(silent)):.8f}")
            logging.info(f"  DISSOLUTION layer: min={dissolution.min():.8f}, max={dissolution.max():.8f}, mean={np.mean(np.abs(dissolution)):.8f}")

            # Log combined amplitude at key time points
            sacred_layer = mercy + silent + dissolution
            logging.info(f"  COMBINED sacred layer: min={sacred_layer.min():.8f}, max={sacred_layer.max():.8f}, mean={np.mean(np.abs(sacred_layer)):.8f}")
            for sec in [0, 5, 10, 15, 30, 45, 60]:
                if sec < duration:
                    idx = int(sec * sample_rate)
                    if idx < len(sacred_layer):
                        start_idx = max(0, idx - sample_rate // 2)
                        end_idx = min(len(sacred_layer), idx + sample_rate // 2)
                        window = sacred_layer[start_idx:end_idx]
                        logging.info(f"    Sacred at {sec:3d}s: amplitude={np.abs(window).mean():.10f}")
            del sacred_layer

            # Toroidal panning for each sacred layer — each drifts independently
            # Much slower than main panning, sacred layers are deep and gradual
            sacred_theta_freqs = [np.float32(0.003), np.float32(0.005), np.float32(0.004), np.float32(0.0035), np.float32(0.0028), np.float32(0.0045)]  # pleroma, solfeggio, archon, crystalline, merkaba, water
            sacred_R = np.float32(0.6)
            sacred_r = np.float32(0.3)
            for layer, st_freq in zip([mercy, silent, dissolution, crystalline, merkaba, water], sacred_theta_freqs):
                sp_freq = st_freq * np.float32(PHI)
                s_theta = TWO_PI * st_freq * t_total
                s_phi = TWO_PI * sp_freq * t_total
                sacred_pan = ((sacred_R + sacred_r * np.cos(s_phi)) * np.cos(s_theta) / (sacred_R + sacred_r)).astype(np.float32)
                sacred_pan_scaled = sacred_pan * np.float32(0.4)  # Gentler range than main
                stereo_wave[:, 0] += layer * (np.float32(1.0) - sacred_pan_scaled)
                stereo_wave[:, 1] += layer * (np.float32(1.0) + sacred_pan_scaled)
                del sacred_pan, sacred_pan_scaled

            # Free sacred layer memory now that it's merged
            del mercy, silent, dissolution, crystalline, merkaba, water
            gc.collect()

            logging.info(f"=== STEREO WAVE (after sacred layers) ===")
            logging.info(f"  Left channel: min={stereo_wave[:, 0].min():.6f}, max={stereo_wave[:, 0].max():.6f}")
            logging.info(f"  Right channel: min={stereo_wave[:, 1].min():.6f}, max={stereo_wave[:, 1].max():.6f}")

        if update_progress:
            update_progress(1.0)  # Complete

        return stereo_wave  # Already float32 from np.empty allocation

    def generate_dynamic_sound_stream(self, duration: float, base_freq: float,
                                       sample_rate: int = 48000,
                                       interval_duration_list: List[int] = [30, 45, 60, 75, 90],
                                       stop_event: Optional[threading.Event] = None,
                                       update_progress: Optional[Callable[[float], None]] = None,
                                       dimensional_mode: bool = False):
        """
        Streaming generator that yields (chunk_samples, 2) float32 stereo chunks.
        Produces identical audio to generate_dynamic_sound() but with ~50 MB peak memory.
        """
        total_samples = int(sample_rate * duration)
        chunk_seconds = 10
        chunk_size = chunk_seconds * sample_rate

        # ========== PRE-COMPUTATION PHASE ==========

        # --- Modulation schedule ---
        schedule = []
        current_index = 0
        remaining_duration = duration
        selected_ratio_set = {}

        if dimensional_mode:
            dimension_phases = [0, 2, 1, 3, 4, 3, 5, 'all']
            num_phases = len(dimension_phases)
            phase_duration = duration / num_phases
            phase_samples = int(sample_rate * phase_duration)
            for phase_idx, sub_selection in enumerate(dimension_phases):
                if stop_event and stop_event.is_set():
                    return
                if sub_selection == 'all':
                    selected_ratio_set = {k: v for set_dict in self.frequency_manager.ratio_sets.values() for k, v in set_dict.items()}
                else:
                    frequencies = self.frequency_manager.get_frequencies(sub_selection)
                    selected_ratio_set = self.frequency_manager.select_random_ratio_set(stop_event)
                modulation_index = np.random.uniform(0.2, 0.25)
                seg_end = min(current_index + phase_samples, total_samples)
                schedule.append((current_index, seg_end, dict(selected_ratio_set), modulation_index))
                current_index = seg_end
        else:
            interval_count = 0
            while remaining_duration > 0 and (not stop_event or not stop_event.is_set()):
                interval_duration = interval_duration_list[interval_count % len(interval_duration_list)]
                interval_duration = min(interval_duration, remaining_duration)
                segment_samples = int(sample_rate * interval_duration)
                selected_ratio_set = self.frequency_manager.select_random_ratio_set(stop_event)
                modulation_index = np.random.uniform(0.2, 0.25)
                seg_end = min(current_index + segment_samples, total_samples)
                schedule.append((current_index, seg_end, dict(selected_ratio_set), modulation_index))
                current_index = seg_end
                remaining_duration -= interval_duration
                interval_count += 1

        if stop_event and stop_event.is_set():
            return

        # --- Random parameters (drawn once) ---
        chaotic_factor = self.chaotic_selector.next_value(stop_event)
        chaos_val = self.chaotic_selector.next_value(stop_event) if not (stop_event and stop_event.is_set()) else 0.5

        # ADSR parameters
        fib_ratios = np.array([1.618, 2.618, 4.236, 6.854, 11.090, 17.944], dtype=np.float32)
        base_scale = 6.0 + chaos_val * 0.5
        attack, decay_r, sustain, release = np.random.choice(fib_ratios, 4) / base_scale
        a_samples = int(attack * sample_rate * (1 + chaos_val * 0.2))
        d_samples = int(decay_r * sample_rate * (1 + chaos_val * 0.2))
        r_samples = int(release * sample_rate * (1 + chaos_val * 0.2))
        s_samples = total_samples - (a_samples + d_samples + r_samples)
        if s_samples < 0:
            s_samples = 0
            r_samples = total_samples - (a_samples + d_samples)
            if r_samples < 0:
                r_samples = 0
        adsr_params = {
            'a_samples': a_samples, 'd_samples': d_samples,
            's_samples': s_samples, 'r_samples': r_samples,
            'sustain': float(sustain), 'chaos': chaos_val,
            'overall_scale': np.float32(np.random.uniform(0.95, 1.05))
        }

        # LFO parameters
        lfo_left_params = {
            'lfo1_freq': float(0.05 * (PHI ** np.random.uniform(-0.05, 0.05))),
            'lfo2_freq': float(0.05 * (PHI ** np.random.uniform(-0.05, 0.05)) * np.random.uniform(0.99, 1.01)),
            'depth': np.float32(np.random.uniform(0.002, 0.006))
        }
        lfo_right_params = {
            'lfo1_freq': float(0.05 * (PHI ** np.random.uniform(-0.05, 0.05))),
            'lfo2_freq': float(0.05 * (PHI ** np.random.uniform(-0.05, 0.05)) * np.random.uniform(0.99, 1.01)),
            'depth': np.float32(np.random.uniform(0.002, 0.006))
        }

        # Toroidal panning parameters (drawn once, applied per chunk)
        torus_theta_freq = np.float32(np.random.uniform(0.01, 0.02))
        torus_phi_freq = torus_theta_freq * np.float32(PHI)
        torus_R = np.float32(0.6)
        torus_r = np.float32(0.3)
        simplex_pan = Simplex5D(13)  # Dedicated instance for pan perturbation

        # Sacred layer toroidal panning frequencies (slow, independent per layer)
        sacred_theta_freqs = [np.float32(0.003), np.float32(0.005), np.float32(0.004), np.float32(0.0035), np.float32(0.0028), np.float32(0.0045)]  # pleroma, solfeggio, archon, crystalline, merkaba, water
        sacred_phi_freqs = [f * np.float32(PHI) for f in sacred_theta_freqs]
        sacred_R = np.float32(0.6)
        sacred_r = np.float32(0.3)

        # Crystal sequence for crystalline resonance layer (drawn once, passed to each chunk)
        # Lemurian Quartz always first — every session starts with divine feminine heart energy
        num_crystal_profiles = len(self.audio_processor._crystal_profiles)
        lemurian_idx = self.audio_processor._lemurian_idx
        rest = np.random.permutation([i for i in range(num_crystal_profiles) if i != lemurian_idx])
        crystal_sequence = np.concatenate(([lemurian_idx], rest))

        # Noise and fade parameters
        noise_scale_left = np.float32(np.random.uniform(0.04, 0.06))
        noise_scale_right = np.float32(np.random.uniform(0.04, 0.06))
        noise_offset_right = np.float32(np.random.uniform(0.001, 0.015))
        fade_duration = int(np.random.choice([15, 30]))
        fade_samples = int(fade_duration * sample_rate)
        if 2 * fade_samples > total_samples:
            fade_samples = total_samples // 2

        # Pan drift parameters
        drift_freq = np.float32(np.random.uniform(0.0005, 0.002))
        drift_amplitude = np.float32(np.random.uniform(0.01, 0.02))

        # Frequency sets
        freq_set = base_freq * self.PHI_EXPONENTS_6 * np.random.uniform(0.98, 1.02, size=6).astype(np.float32)
        freq_set = np.concatenate([
            freq_set,
            base_freq * self.RATIO_1_3_EXPONENTS_3 * np.random.uniform(0.98, 1.02, size=3).astype(np.float32)
        ])
        subharmonics = base_freq / self.SUBHARMONIC_DIVISORS * np.random.uniform(0.95, 1.05, size=4).astype(np.float32)
        all_frequencies = np.concatenate([freq_set, subharmonics]).astype(np.float32)
        del freq_set, subharmonics

        mod_depths = np.random.uniform(0.15, 0.35, len(all_frequencies)).astype(np.float32)

        # Dedicated Simplex instances
        simplex_envelope = Simplex5D(42)
        simplex_fractal = Simplex5D(7)

        # Check taygetan
        has_taygetan = 'taygetan' in selected_ratio_set
        tay_freqs = self.frequency_manager.get_frequencies(5) if has_taygetan else None

        # --- IIR filter states (SOS format for stateful sosfilt) ---
        # Compute low-pass coefficients ONCE — changing sos between chunks while
        # carrying zi state produces discontinuities (pops/clicks at boundaries).
        cutoff_variation = np.random.uniform(-25, 25)
        normalized_cutoff = np.clip((2200 + cutoff_variation) / (0.5 * sample_rate), 0.1, 0.99)
        sos_lowpass = butter(4, normalized_cutoff, btype='low', output='sos')
        zi_left = None
        zi_right = None

        # Pan curve filter
        sos_pan = butter(2, 0.002 / (0.5 * sample_rate), btype='low', output='sos')
        zi_pan = None

        # --- Reverb instances ---
        reverb_left = StreamingReverb(sample_rate)
        reverb_right = StreamingReverb(sample_rate)

        # --- 12-sample right-channel delay buffer ---
        delay_buffer = np.zeros(12, dtype=np.float32)

        # --- Normalization: estimate peak from 0.5s sample at full envelope + LFO peak ---
        # Use envelope=1.0 (peak of attack) to estimate the loudest the signal can get,
        # matching batch pipeline's global-max normalization behavior.
        est_samples = min(int(0.5 * sample_rate), total_samples)
        quarter_period = 1.0 / (4.0 * max(lfo_left_params['lfo1_freq'], 0.001))
        est_t = np.linspace(quarter_period, quarter_period + 0.5, est_samples, endpoint=False, dtype=np.float32)
        est_envelope = np.ones(est_samples, dtype=np.float32)  # Full envelope (ADSR peak)
        est_lfo = self.audio_processor._compute_lfo_chunk(est_t, lfo_left_params)
        est_wave = jit_generate_harmonics_vectorized(all_frequencies, est_t, est_envelope, est_lfo, mod_depths)
        est_wave = jit_wave_shaping(est_wave, 2.5)
        est_peak = max(np.max(np.abs(est_wave)), 0.001)
        norm_gain = np.float32(1.0 / (est_peak * 1.3))
        del est_t, est_envelope, est_lfo, est_wave

        # ========== PER-CHUNK LOOP ==========
        num_chunks = (total_samples + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            if stop_event and stop_event.is_set():
                return

            chunk_offset = chunk_idx * chunk_size
            chunk_samples = min(chunk_size, total_samples - chunk_offset)

            # Time array for this chunk (absolute time values)
            t_start = chunk_offset / sample_rate
            t_end = (chunk_offset + chunk_samples) / sample_rate
            t_chunk = np.linspace(t_start, t_end, chunk_samples, endpoint=False, dtype=np.float32)

            # --- Stage 1: Modulation ---
            modulation = self.audio_processor._compute_modulation_chunk(
                t_chunk, chunk_offset, chunk_samples, schedule, stop_event)

            # --- Stage 2: Fractal variation ---
            base_noise = simplex_fractal.generate_noise(
                t_chunk * np.float32(0.02), np.float32(base_freq * 0.01), 0.0, 0.0, 0.0)
            variation = base_noise * np.float32(0.5)
            # Use second simplex call instead of np.roll for streaming
            variation2 = simplex_fractal.generate_noise(
                t_chunk * np.float32(0.02), np.float32(base_freq * 0.01), np.float32(1.0), 0.0, 0.0)
            variation += variation2 * np.float32(0.3)
            del variation2
            variation += base_noise * base_noise * np.float32(0.2)
            del base_noise
            # LFO for fractal
            frac_lfo = self.audio_processor._compute_lfo_chunk(t_chunk, {'lfo1_freq': 0.01, 'lfo2_freq': 0.01, 'depth': np.float32(0.004)})
            variation *= np.float32(12.0)
            variation *= (np.float32(1.0) + np.float32(0.1) * frac_lfo)
            del frac_lfo
            fractal_variation = variation.astype(np.float32)
            del variation

            # --- Stage 3: f_modulated ---
            f_modulated = modulation
            f_modulated += base_freq + np.float32(chaotic_factor * base_freq * 0.25)
            f_modulated += fractal_variation
            del fractal_variation, modulation

            # --- Stage 4: Envelope ---
            envelope = self.audio_processor._compute_envelope_chunk(
                chunk_offset, chunk_samples, adsr_params, simplex_envelope, total_samples)

            # --- Stage 5: LFOs ---
            lfo_left = self.audio_processor._compute_lfo_chunk(t_chunk, lfo_left_params)
            lfo_right = self.audio_processor._compute_lfo_chunk(t_chunk, lfo_right_params)

            # --- Stage 6: Harmonics ---
            wave_left = jit_generate_harmonics_vectorized(
                all_frequencies, t_chunk, envelope, lfo_left, mod_depths)
            wave_right = jit_generate_harmonics_vectorized(
                all_frequencies, t_chunk, envelope, lfo_right, mod_depths)

            # --- Stage 7: Taygetan binaural ---
            if has_taygetan and tay_freqs:
                binaural_waves = self.audio_processor.batch_binaural_oscillator(t_chunk, tay_freqs, stop_event)
                wave_left += (binaural_waves[:, :, 0] * envelope * np.float32(0.015)).sum(axis=0)
                wave_right += (binaural_waves[:, :, 1] * envelope * np.float32(0.015)).sum(axis=0)
                del binaural_waves

            del envelope, lfo_left, lfo_right

            # --- Stage 8: Wave shaping ---
            wave_left = jit_wave_shaping(wave_left, 2.5)
            wave_right = jit_wave_shaping(wave_right, 2.5)

            # --- Stage 9: Low-pass filter with zi carry ---
            # Uses pre-computed sos_lowpass (fixed coefficients for consistent zi carry)
            if zi_left is None:
                wave_left, zi_left = sosfilt(sos_lowpass, wave_left, zi=np.zeros((sos_lowpass.shape[0], 2), dtype=np.float64))
            else:
                wave_left, zi_left = sosfilt(sos_lowpass, wave_left, zi=zi_left)
            if zi_right is None:
                wave_right, zi_right = sosfilt(sos_lowpass, wave_right, zi=np.zeros((sos_lowpass.shape[0], 2), dtype=np.float64))
            else:
                wave_right, zi_right = sosfilt(sos_lowpass, wave_right, zi=zi_right)

            wave_left = wave_left.astype(np.float32)
            wave_right = wave_right.astype(np.float32)

            # --- Stage 10: Normalization (pre-computed gain) ---
            wave_left *= norm_gain
            wave_right *= norm_gain

            # --- Stage 11: Reverb with tail carry ---
            wave_left = reverb_left.process_chunk(wave_left)
            wave_right_delayed = np.concatenate([delay_buffer, wave_right[:-12]])

            # --- Stage 12: 12-sample right delay with carry ---
            delay_buffer = wave_right[-12:].copy()
            wave_right = reverb_right.process_chunk(wave_right_delayed)
            del wave_right_delayed

            # --- Stage 13: Noise layers ---
            noise_left = self.audio_processor.evolving_noise_layer(t_chunk, stop_event=stop_event)
            noise_right = self.audio_processor.evolving_noise_layer(
                t_chunk + noise_offset_right, stop_event=stop_event)

            wave_left += noise_left
            del noise_left
            wave_left *= noise_scale_left

            wave_right += noise_right
            del noise_right
            wave_right *= noise_scale_right

            # --- Stage 14: Fade in/out ---
            if chunk_offset < fade_samples:
                # Chunk overlaps with fade-in region
                for i in range(chunk_samples):
                    global_idx = chunk_offset + i
                    if global_idx < fade_samples:
                        fade_val = (global_idx / fade_samples) ** 1.5
                        wave_left[i] *= fade_val
                        wave_right[i] *= fade_val

            if chunk_offset + chunk_samples > total_samples - fade_samples:
                # Chunk overlaps with fade-out region
                fade_out_start = total_samples - fade_samples
                for i in range(chunk_samples):
                    global_idx = chunk_offset + i
                    if global_idx >= fade_out_start:
                        fade_val = ((total_samples - global_idx) / fade_samples) ** 1.5
                        fade_val = max(0.0, fade_val)
                        wave_left[i] *= fade_val
                        wave_right[i] *= fade_val

            # --- Stage 15: Toroidal pan curve ---
            theta_perturb = np.float32(0.15) * simplex_pan.generate_noise(t_chunk * np.float32(0.005), 0.0, 0.0, 0.0, 0.0)
            phi_perturb = np.float32(0.15) * simplex_pan.generate_noise(t_chunk * np.float32(0.007), np.float32(1.0), 0.0, 0.0, 0.0)

            theta = TWO_PI * torus_theta_freq * t_chunk + theta_perturb
            phi = TWO_PI * torus_phi_freq * t_chunk + phi_perturb
            del theta_perturb, phi_perturb

            pan_curve = ((torus_R + torus_r * np.cos(phi)) * np.cos(theta) / (torus_R + torus_r)).astype(np.float32)
            del theta, phi

            # sosfilt with zi for pan curve
            if zi_pan is None:
                pan_curve, zi_pan = sosfilt(sos_pan, pan_curve, zi=np.zeros((sos_pan.shape[0], 2), dtype=np.float64))
            else:
                pan_curve, zi_pan = sosfilt(sos_pan, pan_curve, zi=zi_pan)

            pan_curve = jit_pan_curve_tanh(pan_curve.astype(np.float32), 0.6, -0.8, 0.8)
            pan_curve += drift_amplitude * np.sin(np.float32(TWO_PI) * drift_freq * t_chunk)

            pan_scaled = pan_curve * np.float32(0.6)
            del pan_curve
            wave_left *= (np.float32(1.0) - pan_scaled)
            wave_right *= (np.float32(1.0) + pan_scaled)
            del pan_scaled

            # Apply master volume
            wave_left *= np.float32(self.master_volume)
            wave_right *= np.float32(self.master_volume)

            # --- Stage 16: Sacred layers ---
            if duration > 60 or dimensional_mode:
                from concurrent.futures import TimeoutError as FuturesTimeoutError

                mercy_f = self._sacred_executor.submit(
                    self.audio_processor.pleroma_mercy_layer_chunk, t_chunk, duration, 7.83)
                silent_f = self._sacred_executor.submit(
                    self.audio_processor.silent_solfeggio_grid_chunk, t_chunk, duration)
                dissolution_f = self._sacred_executor.submit(
                    self.audio_processor.archon_dissolution_layer_chunk, t_chunk, duration)
                crystalline_f = self._sacred_executor.submit(
                    self.audio_processor.crystalline_resonance_layer_chunk, t_chunk, duration, base_freq, crystal_sequence)
                merkaba_f = self._sacred_executor.submit(
                    self.audio_processor.lemurian_merkaba_layer_chunk, t_chunk, duration)
                water_f = self._sacred_executor.submit(
                    self.audio_processor.water_element_layer_chunk, t_chunk, duration)

                sacred_timeout = 60.0
                try:
                    mercy = mercy_f.result(timeout=sacred_timeout)
                except Exception:
                    mercy = np.zeros(chunk_samples, dtype=np.float32)
                try:
                    silent = silent_f.result(timeout=sacred_timeout)
                except Exception:
                    silent = np.zeros(chunk_samples, dtype=np.float32)
                try:
                    dissolution = dissolution_f.result(timeout=sacred_timeout)
                except Exception:
                    dissolution = np.zeros(chunk_samples, dtype=np.float32)
                try:
                    crystalline = crystalline_f.result(timeout=sacred_timeout)
                except Exception:
                    crystalline = np.zeros(chunk_samples, dtype=np.float32)
                try:
                    merkaba = merkaba_f.result(timeout=sacred_timeout)
                except Exception:
                    merkaba = np.zeros(chunk_samples, dtype=np.float32)
                try:
                    water = water_f.result(timeout=sacred_timeout)
                except Exception:
                    water = np.zeros(chunk_samples, dtype=np.float32)

                # Toroidal panning for each sacred layer — independent spatial drift
                for layer, st_freq, sp_freq in zip(
                        [mercy, silent, dissolution, crystalline, merkaba, water], sacred_theta_freqs, sacred_phi_freqs):
                    s_theta = TWO_PI * st_freq * t_chunk
                    s_phi = TWO_PI * sp_freq * t_chunk
                    sacred_pan = ((sacred_R + sacred_r * np.cos(s_phi)) * np.cos(s_theta) / (sacred_R + sacred_r)).astype(np.float32)
                    sacred_pan_scaled = sacred_pan * np.float32(0.4)
                    wave_left += layer * (np.float32(1.0) - sacred_pan_scaled)
                    wave_right += layer * (np.float32(1.0) + sacred_pan_scaled)
                    del sacred_pan, sacred_pan_scaled
                del mercy, silent, dissolution, crystalline, merkaba, water

            # Build stereo chunk and yield
            stereo_chunk = np.empty((chunk_samples, 2), dtype=np.float32)
            stereo_chunk[:, 0] = wave_left
            stereo_chunk[:, 1] = wave_right
            del wave_left, wave_right, t_chunk

            if update_progress:
                update_progress((chunk_idx + 1) / num_chunks)

            yield stereo_chunk