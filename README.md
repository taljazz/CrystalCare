# CrystalCare

A dynamic sound healing application that generates evolving, spiral-based tones to balance, empower, protect, and energetically heal. Inspired by Lemurian crystalline healing traditions and built with modern mathematical modeling, CrystalCare creates immersive soundscapes that go beyond static binaural beats or isochronic tones.

## Disclaimer

CrystalCare operates outside the framework of conventional linear-based science. It is a non-linear, energetic tool designed to balance, empower, protect, and energetically heal through vibrational frequencies and sacred geometry. It is not intended to diagnose, treat, cure, or prevent any medical condition. CrystalCare is offered as a complementary practice rooted in ancient sound healing traditions and should not be used as a substitute for professional medical advice or treatment.

## How It Works

CrystalCare fuses ancient sound healing principles with advanced mathematical and audio processing:

- **Mathematical Spirals** - Logarithmic spirals for harmonious natural growth patterns combined with Rossler attractor perturbations for chaotic, non-linear depth
- **Golden Ratio & Sacred Geometry** - Frequencies incorporate PHI (1.618), Solfeggio tunings, and sacred ratios for natural harmonic resonance
- **Six Sacred Healing Layers** - Sub-perceptual layers activate in longer sessions: Pleroma Mercy, Silent Solfeggio Grid, Archon Dissolution, Crystalline Resonance, Lemurian Merkaba, and Water Element
- **Nine Crystal Profiles** - Harmonic ratios derived from real Raman spectroscopy data for nine minerals, each with unique acoustic signatures
- **3D Spatial Audio** - Hybrid spiral panning and spatialized triple helix create immersive, evolving soundscapes
- **Streaming Audio Engine** - Real-time chunk-by-chunk generation allows sessions of unlimited length with ~50 MB constant memory usage
- **Lemurian Heart-Based Philosophy** - Heart coherence, divine feminine warmth, 432 Hz keynote, and water-element consciousness

## Modes

CrystalCare offers seven frequency modes:

| Mode | Frequencies | Purpose |
|------|------------|---------|
| **Lower Frequencies** | 174, 396, 417, 528 Hz | Grounding, clearing negativity, emotional release |
| **Higher Frequencies** | 852, 963 Hz | Spiritual awakening, intuition |
| **Atlantean Cosmic** | 136.10, 194.18, 211.44, 303 Hz | Planetary resonance, cosmic alignment |
| **Combined Mode** | Sacred Geometry + Flower of Life ratios | Holistic balance, comprehensive healing |
| **Triple Helix DNA Activation** | DNA-inspired ratios (1.0, 1.2, 1.4) | Deep activation, unconscious rewiring |
| **Taygetan Resonances** | Binaural sync pairs at 432 Hz base | DNA activation with stereo binaural beating |
| **Dimensional Journey** | Cycles through all frequency sets in phases | 1D-9D realignment, multidimensional traversal |

## Sacred Healing Layers

For sessions longer than 60 seconds (or any length in Dimensional Journey mode), six sacred layers blend beneath the primary tone:

1. **Pleroma Mercy Layer** - 13-step Aeonic ladder from Schumann resonance (7.83 Hz) ascending by PHI, with Ogdoad gateway and Archon mercy frequencies
2. **Silent Solfeggio Grid** - 12-tone Solfeggio scale interwoven with Tesla's 3-6-9 vortex mathematics (111-999 Hz)
3. **Archon Dissolution Layer** - Targeted mercy for the seven planetary Archons using Acknowledge-Elevate-Ground pattern
4. **Crystalline Resonance Layer** - Nine crystal profiles from Raman spectroscopy with PHI-timed crossfade evolution, always beginning with Lemurian Quartz
5. **Lemurian Merkaba Layer** - Sonic Merkaba from the Lemurian Frequency Quartet (324/432/540/698.4 Hz) with 0.1 Hz heart coherence breath
6. **Water Element Layer** - Seven-source hexagonal ripple field with lemniscate observer path and tidal simplex modulation

## Crystalline Resonance Profiles

| Crystal | Character |
|---------|-----------|
| Lemurian Quartz | Warm, heart-centered with 12 harmonics and 3 heart chakra bridges. Divine feminine warmth. Always first. |
| Clear Quartz | Pure, precise. 9 harmonics from quartz Raman spectroscopy. Master healer. |
| Amethyst | Quartz base with organic pitch drift. Meditative, dreamy. |
| Rose Quartz | Extended quartz harmonics with softer, loving tone. |
| Citrine | Quartz base with warm, solar energy character. |
| Black Tourmaline | Tourmaline structure with beating harmonic pairs. Protective, grounding. |
| Selenite | Monoclinic crystal system. Ethereal, high-vibration. |
| Lapis Lazuli | Composite mineral blending lazurite, calcite, and pyrite signatures. |
| Crystal Singing Bowl | Wide harmonic spacing evoking a physical singing bowl. |

## Features

- Real-time streaming playback with instant audio start (~0.2s to first sound)
- WAV export with streaming two-pass normalization
- Batch save for generating multiple sessions
- Accessible interface with screen reader support (wxPython + NVDA compatible)
- Optimized performance (30-34x faster than realtime generation)
- Constant ~50 MB memory usage regardless of session length
- Numba JIT compilation and NumPy vectorization for critical paths
- C++ simplex noise extension with GIL release for parallel threading

## Requirements

- Python 3.10
- NumPy, SciPy, Numba, sounddevice, wxPython, pybind11

### Setup

```bash
conda create -n cc python=3.10 -y
conda activate cc
pip install numpy scipy numba sounddevice wxPython pybind11
python setup.py build_ext --inplace
```

### Run

```bash
python main.py
```

Or use the included `run.bat` on Windows.

### Build Standalone Executable

```bash
pip install nuitka
build.bat
```

Produces `CrystalCare.zip` containing the standalone executable and user guide.

## System Requirements for Saving

When saving WAV files, the full session is generated in memory before writing to disk. This is computationally intensive — expect 40-55% CPU usage during generation, which is normal. Recommended maximum save durations by system RAM:

| System RAM | Recommended Max Save Duration |
|-----------|-------------------------------|
| 16 GB | 15-20 minutes |
| 32 GB | 30 minutes |
| 64 GB+ | 60 minutes |

Streaming playback (Play button) is **not** limited by RAM — sessions of any length are supported with ~50 MB constant memory usage. If you need longer saved files, consider saving multiple shorter sessions.

## Usage Tips

- Use headphones for full 3D immersion (essential for Taygetan Resonances)
- Set intentions before sessions
- Sessions of 60+ seconds activate all six sacred healing layers
- Dimensional Journey mode activates sacred layers regardless of session length
- Longer sessions (10-60 minutes) recommended for deep healing work
- Combine with breathwork or visualization for deeper effects
- Pair with guided hypnosis scripts for unconscious rewiring

## License

MIT License - see [LICENSE](LICENSE) for details.
