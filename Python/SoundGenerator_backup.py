import numpy as np
from scipy.signal import fftconvolve, butter, lfilter
from scipy.ndimage import shift
from scipy.integrate import odeint
from typing import Optional, Dict, List, Callable
import logging
import threading
from simplex5D import Simplex5D

class AudioProcessor:
    """
    Contains all audio signal processing functions.
    """
    def geometric_modulation(self, t: np.ndarray, ratios: Dict[str, float],
                             modulation_index: float = 0.2,
                             stop_event: Optional[threading.Event] = None) -> np.ndarray:
        modulation = np.zeros_like(t, dtype=np.float32)
        for ratio in ratios.values():
            if stop_event and stop_event.is_set():
                logging.debug("Geometric modulation stopped early due to stop_event.")
                break
            modulation += modulation_index * np.sin(2 * np.pi * ratio * t)
        return modulation
    def low_pass_filter(self, signal: np.ndarray, cutoff: float = 2200,
                        sample_rate: int = 48000, order: int = 4,
                        stop_event: Optional[threading.Event] = None) -> np.ndarray:
        nyquist = 0.5 * sample_rate
        drift_frequency = np.random.uniform(0.0005, 0.002)
        drift = np.sin(np.linspace(0, 4 * np.pi, signal.size, dtype=np.float32) * drift_frequency)
        evolving_cutoff = np.clip((cutoff + drift * 50) / nyquist, 0.1, 0.99)
        b, a = butter(order, evolving_cutoff.mean(), btype='low', analog=False)
        filtered_signal = lfilter(b, a, signal.astype(np.float32))
        return filtered_signal
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
       
        simplex = Simplex5D(np.random.randint(0, 10000))
        t_scaled = np.linspace(0, 1, total_samples, dtype=np.float32)
       
        attack_curve = np.linspace(0, 1, a_samples, dtype=np.float32) ** 1.5
        attack_noise = simplex.generate_noise(t_scaled[:a_samples] * 0.05, 0.0, 0.0, 0.0, 0.0) * 0.05
        envelope[:a_samples] = np.clip(attack_curve + attack_noise, 0, 1)
       
        decay_curve = np.linspace(1, sustain, d_samples, dtype=np.float32) ** 1.2
        decay_wobble = np.random.uniform(0.015, 0.035) * np.sin(np.linspace(0, np.pi * (1 + chaos * 0.3), d_samples, dtype=np.float32))
        decay_noise = simplex.generate_noise(t_scaled[a_samples:a_samples+d_samples] * 0.03, 0.0, 0.0, 0.0, 0.0) * 0.03
        envelope[a_samples:a_samples+d_samples] = np.clip(decay_curve + decay_wobble + decay_noise, sustain, 1)
       
        sustain_curve = np.full(s_samples, sustain, dtype=np.float32)
        sustain_wobble = np.random.uniform(0.01, 0.025) * np.sin(2 * np.pi * np.linspace(0, 1.5 + chaos * 0.5, s_samples, dtype=np.float32))
        sustain_noise = simplex.generate_noise(t_scaled[a_samples+d_samples:a_samples+d_samples+s_samples] * 0.02, 0.0, 0.0, 0.0, 0.0) * 0.04
        envelope[a_samples+d_samples:a_samples+d_samples+s_samples] = np.clip(sustain_curve + sustain_wobble + sustain_noise, sustain * 0.8, 1)
       
        release_curve = np.linspace(sustain, 0, r_samples, dtype=np.float32) ** 1.5
        release_noise = simplex.generate_noise(t_scaled[-r_samples:] * 0.04, 0.0, 0.0, 0.0, 0.0) * 0.05
        envelope[-r_samples:] = np.clip(release_curve + release_noise, 0, sustain)
       
        envelope *= np.random.uniform(0.95, 1.05)
        return envelope
    def dynamic_cross_modulate(self, base_freq: float, mod_freq: float, t: np.ndarray,
                               stop_event: Optional[threading.Event] = None) -> np.ndarray:
        if stop_event and stop_event.is_set():
            logging.debug("Dynamic cross modulation stopped early due to stop_event.")
            return np.zeros_like(t, dtype=np.float32)
        mod_signal = np.sin(2 * np.pi * mod_freq * t) * np.random.uniform(0.15, 0.35)
        return np.sin(2 * np.pi * base_freq * t + mod_signal).astype(np.float32)
    def microtonal_lfo(self, t: np.ndarray, base_frequency: float,
                       stop_event: Optional[threading.Event] = None) -> np.ndarray:
        phi = 1.618
        lfo1 = base_frequency * (phi ** np.random.uniform(-0.05, 0.05))
        lfo2 = lfo1 * np.random.uniform(0.99, 1.01)
        depth = np.random.uniform(0.002, 0.006)
        drift = 0.01 * np.sin(0.1 * np.pi * t)
        lfo = (depth + drift) * (1 + np.sin(2 * np.pi * lfo1 * t + 0.3 * np.sin(2 * np.pi * lfo2 * t)))
        return lfo.astype(np.float32)
    def infinite_reverb(self, signal: np.ndarray, sample_rate: int = 48000,
                        stop_event: Optional[threading.Event] = None) -> np.ndarray:
        decay = 0.75 # Softer for Pleiadian gentleness
        reverb_length = min(signal.size // 2, int(sample_rate * 2.618)) # Longer tails for immersion, golden ratio
        ir = np.exp(-np.linspace(0, decay * 1.8, reverb_length, dtype=np.float32))
        ir += 0.08 * np.sin(np.linspace(0, 2 * np.pi * 1.618, reverb_length)) # Golden ratio sine for harmony
        ir /= np.max(np.abs(ir))
        reverb_signal = fftconvolve(signal, ir, mode='full')[:signal.size]
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
        np.clip(signal, -1, 1, out=signal)
        return np.tanh(shape_factor * signal) * 1.2
    def apply_fractional_delay(self, signal: np.ndarray, delay: float) -> np.ndarray:
        return shift(signal, shift=delay, order=3, mode='nearest')
    def spatialized_triple_helix(self, signal: np.ndarray, t: np.ndarray,
                                 sample_rate: int = 48000) -> np.ndarray:
        # Rössler system for chaotic modulation
        def rossler(state, t, a=0.2, b=0.2, c=5.7):
            x, y, z = state
            dx_dt = -y - z
            dy_dt = x + a * y
            dz_dt = b + z * (x - c)
            return [dx_dt, dy_dt, dz_dt]
        # Integrate Rössler equations
        t_scaled = t * 0.1 # Scale time for audible chaos
        initial_state = [1.0, 1.0, 1.0]
        trajectory = odeint(rossler, initial_state, t_scaled)
        x_rossler, y_rossler, z_rossler = trajectory.T
        # Normalize Rössler trajectory
        x_rossler = x_rossler / np.max(np.abs(x_rossler))
        y_rossler = y_rossler / np.max(np.abs(y_rossler))
        z_rossler = z_rossler / np.max(np.abs(z_rossler))
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
        # 3D position in spherical coordinates
        x = r_hybrid * np.cos(theta_hybrid) * np.cos(phi_hybrid)
        y = r_hybrid * np.sin(theta_hybrid) * np.cos(phi_hybrid)
        z = r_hybrid * np.sin(phi_hybrid)
        # Smooth with low-pass filter
        nyquist = 0.5 * sample_rate
        b_filt, a_filt = butter(2, 0.003 / nyquist, btype='low')
        x_smooth = lfilter(b_filt, a_filt, x)
        y_smooth = lfilter(b_filt, a_filt, y)
        z_smooth = lfilter(b_filt, a_filt, z)
        # Stereo panning
        pan_horizontal = np.tanh(x_smooth * 0.5)
        depth_factor = (y_smooth + 1) / 2
        pan_vertical = np.sin(z_smooth * (np.pi / 2))
        left_gain = np.cos((pan_horizontal + 1) * (np.pi / 4)) * depth_factor * (1 - 0.2 * pan_vertical)
        right_gain = np.sin((pan_horizontal + 1) * (np.pi / 4)) * depth_factor * (1 - 0.2 * pan_vertical)
        left_channel = signal * left_gain
        right_channel = signal * right_gain
        # Add delay with Simplex5D noise
        simplex = Simplex5D(np.random.randint(0, 10000))
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
        simplex = Simplex5D(np.random.randint(0, 10000))
        lfo = self.microtonal_lfo(t, base_frequency=0.01)
        variation = (0.5 * simplex.generate_noise(t * 0.005, 0.0, 0.0, 0.0, 0.0) +
                     0.3 * simplex.generate_noise(t * 0.02, 5.0, 5.0, 5.0, 5.0) +
                     0.2 * simplex.generate_noise(t * 0.08, 10.0, 10.0, 10.0, 10.0))
        return (variation * 12 * (1 + 0.1 * lfo)).astype(np.float32)
    def quantum_harmonic_interference(self, t: np.ndarray, base_freq: float) -> np.ndarray:
        f1 = base_freq
        f2 = base_freq * 1.41421356237   # exact √2
        f3 = base_freq * 2.71828182846   # exact e
        f4 = base_freq * 3.14159265359   # exact π
        simplex = Simplex5D(np.random.randint(0, 10000))
        alpha = (0.25 * np.sin(0.03 * np.pi * t)).astype(np.float32)
        beta  = (0.2 * np.cos(0.02 * np.pi * t)).astype(np.float32)
        gamma = (0.15 * simplex.generate_noise(t * 0.01, 0.0, 0.0, 0.0, 0.0)).astype(np.float32)
        wave1 = np.sin(2 * np.pi * f1 * t + alpha)
        wave2 = np.sin(2 * np.pi * f2 * t + beta)
        wave3 = np.sin(2 * np.pi * f3 * t + gamma)
        wave4 = np.sin(2 * np.pi * f4 * t + gamma * 0.7)
        return ((wave1 + wave2 + wave3 + wave4) / 3.8 + 0.15 * np.sin(2 * np.pi * 7.83 * t)).astype(np.float32)
    def recursive_fractal_feedback(self, signal: np.ndarray, depth: int = 4,
                                   factor: float = 0.4) -> np.ndarray:
        if depth == 0:
            return signal
        return signal + factor * self.recursive_fractal_feedback(signal * 0.6, depth - 1, factor)
    def binaural_oscillator(self, t: np.ndarray, freq_pair: tuple, stop_event: Optional[threading.Event] = None) -> np.ndarray:
        left_freq, right_freq = freq_pair
        left_wave = np.sin(2 * np.pi * left_freq * t)
        right_wave = np.sin(2 * np.pi * right_freq * t)
        return np.column_stack((left_wave, right_wave)).astype(np.float32)
    # === PLEROMA MERCY LAYER + SILENT GRID (your ultimate quest) ===
    def pleroma_mercy_layer(self, t: np.ndarray, base_freq: float = 7.83) -> np.ndarray:
        if t.size < 48000 * 60:  # skip on sessions shorter than ~1 minute
            return np.zeros_like(t, dtype=np.float32)
        
        # 13-step aeonic ladder (0–12) from the true Schumann (7.83) up through golden-ratio harmonics
        aeonic_harmonics = base_freq * (1.618033988749895 ** np.arange(13))
        mercy = np.zeros_like(t, dtype=np.float32)
        simplex = Simplex5D(np.random.randint(0, 100000))
        
        for h in aeonic_harmonics:
            phase_wobble = 0.1 * simplex.generate_noise(t * 0.00005, 0,0,0,0)
            mercy += np.sin(2 * np.pi * h * t + phase_wobble)
        
        # Smooth birth/death over the entire session
        envelope = 0.5 - 0.5 * np.cos(2 * np.pi * t / t[-1])  # raised cosine window
        mercy *= envelope * 0.0003  # ≈ –70 dB (completely inaudible)
        
        # Final slow cosine nulling — cancels in audible domain, leaves pure scalar imprint
        return mercy * np.cos(2 * np.pi * 0.0007 * t)
    def silent_solfeggio_grid(self, t: np.ndarray) -> np.ndarray:
        solfeggio = [174, 285, 396, 417, 528, 639, 741, 852, 963, 1074, 1185, 1296]
        grid = np.zeros_like(t, dtype=np.float32)
        for f in solfeggio:
            grid += np.sin(2 * np.pi * f * t)
        return grid * 0.0019 * np.sin(2 * np.pi * 0.013 * t)


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
        simplex = Simplex5D(np.random.randint(0, 10000))
       
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
        lfo_left = self.audio_processor.microtonal_lfo(t_total, base_frequency=0.05)
        lfo_right = self.audio_processor.microtonal_lfo(t_total, base_frequency=0.05)
        exponents = np.arange(6)
        freq_set = base_freq * (1.618 ** exponents) * np.random.uniform(0.98, 1.02, size=6).astype(np.float32)
        freq_set = np.concatenate([
            freq_set,
            base_freq * (1.3 ** exponents[:3]) * np.random.uniform(0.98, 1.02, size=3).astype(np.float32)
        ])
        subharmonics = base_freq / (2 ** np.arange(1, 5)) * np.random.uniform(0.95, 1.05, size=4)
        all_frequencies = np.concatenate([freq_set, subharmonics]).astype(np.float32)
        if update_progress:
            update_progress(0.3)  # Freq sets built (5%)
        envelope = self.audio_processor.organic_adsr(t_total, sample_rate, stop_event, self.chaotic_selector)
        if update_progress:
            update_progress(0.35)  # Envelope done (5%)
        wave_left = np.zeros(total_samples, dtype=np.float32)
        wave_right = np.zeros(total_samples, dtype=np.float32)
        num_freqs = len(all_frequencies)
        wave_gen_scale = 0.3  # Wave generation loop 30%
        for i, f in enumerate(all_frequencies):
            if stop_event and stop_event.is_set():
                logging.debug("Waveform generation stopped early due to stop_event.")
                break
            modulation_wave = self.audio_processor.dynamic_cross_modulate(f, f * 0.5, t_total, stop_event)
            wave_left += modulation_wave * envelope * (0.015 / (i + 1)) * lfo_left
            wave_right += modulation_wave * envelope * (0.015 / (i + 1)) * lfo_right
            if update_progress:
                update_progress(0.35 + ((i + 1) / num_freqs) * wave_gen_scale)
        taygetan_scale = 0.05 if 'taygetan' in selected_ratio_set else 0.0  # Optional 5% for taygetan
        if 'taygetan' in selected_ratio_set:
            tay_freqs = self.frequency_manager.get_frequencies(5)
            num_tay = len(tay_freqs)
            for j, freq_pair in enumerate(tay_freqs):
                binaural_wave = self.audio_processor.binaural_oscillator(t_total, freq_pair, stop_event)
                wave_left += binaural_wave[:, 0] * envelope * 0.015
                wave_right += binaural_wave[:, 1] * envelope * 0.015
                if update_progress:
                    update_progress(0.65 + ((j + 1) / num_tay) * taygetan_scale)  # After main waves 0.65
        if update_progress and taygetan_scale == 0:
            update_progress(0.7)  # Skip taygetan progress if not
        else:
            update_progress(0.7)  # After taygetan
        wave_left = self.audio_processor.wave_shaping(wave_left, shape_factor=2.5, stop_event=stop_event)
        wave_right = self.audio_processor.wave_shaping(wave_right, shape_factor=2.5, stop_event=stop_event)
        if update_progress:
            update_progress(0.75)  # Shaping 5%
        wave_left = self.audio_processor.low_pass_filter(wave_left, stop_event=stop_event)
        wave_right = self.audio_processor.low_pass_filter(wave_right, stop_event=stop_event)
        if update_progress:
            update_progress(0.8)  # Filters 5%
        max_left = np.max(np.abs(wave_left))
        max_right = np.max(np.abs(wave_right))
        if max_left > 0:
            wave_left /= (max_left * 1.3)
        if max_right > 0:
            wave_right /= (max_right * 1.3)
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
        lfo_base = self.audio_processor.microtonal_lfo(t_total, base_frequency=np.random.uniform(0.01, 0.02))
        lfo_secondary = 0.3 * self.audio_processor.microtonal_lfo(t_total, base_frequency=np.random.uniform(0.03, 0.06))
        lfo_random = np.random.normal(0, 0.015, total_samples).astype(np.float32)
        pan_curve = (0.6 * lfo_base + 0.3 * lfo_secondary + 0.1 * lfo_random)
        nyquist = 0.5 * sample_rate
        b, a = butter(2, 0.002 / nyquist, btype='low')
        pan_curve = lfilter(b, a, pan_curve)
        pan_curve = np.clip(np.tanh(pan_curve * 0.6), -0.8, 0.8)
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
        
        # === PLEROMA MERCY INFUSION + SILENT GRID (your ultimate quest) ===
        if duration > 1800 or dimensional_mode:  # >30 minutes or Dimensional Journey
            mercy = self.audio_processor.pleroma_mercy_layer(t_total, base_freq=7.83)
            silent = self.audio_processor.silent_solfeggio_grid(t_total)
            stereo_wave += np.column_stack((mercy + silent, mercy + silent))
        
        if update_progress:
            update_progress(1.0)  # Complete
        
        return stereo_wave.astype(np.float32)