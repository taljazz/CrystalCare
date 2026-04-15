import numpy as np
from typing import Dict, List, Optional
import logging
import threading

class FrequencyManager:
    """
    Manages frequency sets and handles random selection based on predefined weights.
    """
    def __init__(self) -> None:
        self.sacred_geometry_ratios: Dict[str, float] = {
            "metatron_ratio": 1.414,
            "vesica_piscis_ratio": 1.732,
            "hexagon_ratio": 2.0,
            "flower_of_life_ratio": 1.618,
            "circle_ratio": 3.0,
        }
        self.flower_of_life_ratios: Dict[str, float] = {
            "petal_ratio": 1.3,
            "intersect_ratio": 1.5,
            "symmetry_ratio": 2.5,
        }
        self.triple_helix_ratios: Dict[str, float] = {
            "strand_1": 1.0,
            "strand_2": 1.2,
            "strand_3": 1.4,
        }
        self.ratio_sets: Dict[str, Dict[str, float]] = {
            "sacred_geometry": self.sacred_geometry_ratios,
            "flower_of_life": self.flower_of_life_ratios,
            "triple_helix": self.triple_helix_ratios,
            "combined": {**self.sacred_geometry_ratios, **self.flower_of_life_ratios},
            "minimal": {"metatron_ratio": 1.414},
            "enhanced_geometry": {"octagon_ratio": 2.828, "spiral_ratio": 2.236},
            "fibonacci_set": {"fibonacci_ratio_1": 1.618, "fibonacci_ratio_2": 2.618, "fibonacci_ratio_3": 4.236},
            "fractal_set": {"mandelbrot_ratio": 3.1415, "julia_ratio": 2.718},
            "taygetan": {
                "root": 1.0,
                "etheric_body": 1.41421356237,    # exact √2
                "astral_bridge": 1.73205080757,   # exact √3
                "natural_log": 2.71828182846,     # exact e
                "crown_portal": 3.14159265359,    # exact π
                "zero_point": 4.2360679775,       # phi^3 (golden ratio cubed)
                "remembrance": 1.61803398875,     # exact golden ratio φ
            },
        }
        self.ratio_set_names: List[str] = list(self.ratio_sets.keys())
        self.ratio_set_weights: List[float] = [0.25, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.12]
        if len(self.ratio_set_weights) != len(self.ratio_set_names):
            logging.error("Number of weights must match number of ratio sets.")
            raise ValueError("Mismatch between weights and ratio sets.")
        self.ratio_set_weights = np.array(self.ratio_set_weights, dtype=np.float32)
        self.ratio_set_weights /= self.ratio_set_weights.sum()

    def get_geometric_frequency_set(self, base_freq: float, ratios: Dict[str, float]) -> List[float]:
        return [base_freq * ratio for ratio in ratios.values()]

    def select_random_ratio_set(self, stop_event: Optional[threading.Event] = None) -> Dict[str, float]:
        if stop_event and stop_event.is_set():
            logging.debug("Ratio set selection stopped early due to stop_event.")
            return {}
        selected_set = np.random.choice(self.ratio_set_names, p=self.ratio_set_weights)
        logging.debug(f"Selected ratio set: {selected_set}")
        return self.ratio_sets[selected_set]

    def get_frequencies(self, selection: int, base_freq_initial: float = 432) -> List[float]:
        if selection == 0:
            return [174, 396, 417, 528]  # Solfeggio base, includes 528 Hz
        elif selection == 1:
            return [852, 963]  # Solfeggio high
        elif selection == 2:
            return [136.10, 194.18, 211.44, 303]  # Planetary
        elif selection == 3:
            return (self.get_geometric_frequency_set(base_freq_initial, self.sacred_geometry_ratios) +
                    self.get_geometric_frequency_set(base_freq_initial, self.flower_of_life_ratios))  # Sacred geometry + flower of life
        elif selection == 4:
            return self.get_geometric_frequency_set(base_freq_initial, self.triple_helix_ratios)  # Triple helix
        elif selection == 5:
            base = 432
            taygetan_ratios = self.ratio_sets["taygetan"]
            # Filter out super-low ratios (<0.1) to avoid rattling; use them for modulation only
            filtered_taygetan_set = [base * ratio for ratio in taygetan_ratios.values() if ratio >= 0.1]
            delta = 7.7  # Taygetan sync beat
            return [(f + delta/2, f - delta/2) for f in filtered_taygetan_set]  # List of tuples for stereo generation
        else:
            return []  # Default for invalid selections