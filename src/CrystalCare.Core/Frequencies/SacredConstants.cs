namespace CrystalCare.Core.Frequencies;

/// <summary>
/// All sacred frequency constants, mathematical ratios, and crystal profile data.
/// These are the healing frequencies at the heart of CrystalCare.
/// Every irrational constant is derived at full precision — never truncated.
/// Single source of truth for all sacred mathematics in the application.
/// </summary>
public static class SacredConstants
{
    // The core mathematical constants that govern all of CrystalCare's sacred mathematics.
    // PHI (golden ratio) is the fundamental organizing principle.
    // Schumann resonance (7.83 Hz) is Earth's electromagnetic heartbeat.
    // 432 Hz is the Lemurian tuning keynote, the center of the Sonic Merkaba.
    #region Fundamental Constants

    public const float PHI = 1.618033988749895f;             // Golden ratio (φ)
    public const float PHI_INVERSE = 1.0f / PHI;             // 1/φ = 0.618... (torus major radius)
    public const float PHI_SQ_INVERSE = 1.0f / (PHI * PHI);  // 1/φ² = 0.382... (torus minor radius)
    public const float TWO_PI = 2.0f * MathF.PI;             // Full circle in radians
    public const float SCHUMANN = 7.83f;                     // Earth's electromagnetic heartbeat (Hz)
    public const float OGDOAD_FREQ = SCHUMANN * 8;           // 62.64 Hz — 8th sphere threshold to Pleroma
    public const float MERKABA_KEYNOTE = 432.0f;             // Lemurian tuning keynote (Hz)

    #endregion

    // Irrational constants derived at full float precision from MathF.
    // These are the single source of truth — all ratio sets reference these
    // instead of typing truncated approximations like 1.414 or 2.718.
    #region Derived Irrational Constants

    public static readonly float SQRT_2 = MathF.Sqrt(2);    // √2 = 1.41421356... (Metatron's Cube)
    public static readonly float SQRT_3 = MathF.Sqrt(3);    // √3 = 1.73205081... (Vesica Piscis)
    public static readonly float SQRT_5 = MathF.Sqrt(5);    // √5 = 2.23606798... (Spiral ratio)
    public static readonly float EULER_E = MathF.E;          // e = 2.71828183... (Julia set)
    public static readonly float PI = MathF.PI;              // π = 3.14159265... (Mandelbrot set)

    #endregion

    // The golden angle (137.5°) — the angle between successive seeds in a sunflower,
    // leaves on a stem, scales on a pinecone. It's nature's solution for optimal packing.
    // Used for sacred layer phase offsets and crystal crossfade timing.
    #region Golden Angle

    // 360° × (1 − 1/φ) = 360° × (2 − φ) ≈ 137.5077°
    public static readonly float GOLDEN_ANGLE_DEG = 360.0f * (2.0f - PHI);
    public static readonly float GOLDEN_ANGLE_RAD = TWO_PI * (2.0f - PHI) / 2.0f;

    #endregion

    // LFO and panning drift constants — derived from the same Schumann root and PHI
    // ladder as the breath frequencies. This connects the entire sub-perceptual
    // modulation system into one unified organism: breath, LFO, and spatial drift
    // all descend from Earth's heartbeat through golden ratio harmonics.
    #region LFO and Drift Constants

    // LFO base frequency: BREATH_ROOT × PHI^4 — the breath ladder extended 4 PHI steps.
    // This is the amplitude modulation rhythm, faster than breath but PHI-connected.
    public static readonly float LFO_BASE_FREQ = SCHUMANN / 1000f * MathF.Pow(PHI, 4f);  // ~0.0537 Hz

    // Panning drift frequency center: BREATH_ROOT / PHI^3 — a PHI sub-harmonic of breath.
    // Extremely slow spatial drift (~540s period), connecting movement to Earth's rhythm.
    public static readonly float DRIFT_FREQ_CENTER = SCHUMANN / 1000f / MathF.Pow(PHI, 3f);  // ~0.00185 Hz

    // Panning drift amplitude bounds: Fibonacci reciprocals (1/89 to 1/55).
    // Drift magnitude bounded by Fibonacci, not arbitrary round numbers.
    public const float DRIFT_AMP_MIN = 1f / 89f;   // ~0.01124 (Fibonacci)
    public const float DRIFT_AMP_MAX = 1f / 55f;    // ~0.01818 (Fibonacci)

    // LFO internal drift amplitude: 1/89 (Fibonacci reciprocal)
    public const float LFO_DRIFT_AMP = 1f / 89f;    // ~0.01124

    // LFO inner modulation depth: 1/PHI² — golden ratio squared reciprocal
    public const float LFO_INNER_MOD_DEPTH = PHI_SQ_INVERSE;  // 0.382

    // Pan smoother cutoff: same as drift center — connects pan smoothing to Earth's breath
    public static readonly float PAN_SMOOTHER_CUTOFF = DRIFT_FREQ_CENTER;  // ~0.00185 Hz

    // Noise scale bounds: Fibonacci reciprocals 1/21 to 1/13
    public const float NOISE_SCALE_MIN = 1f / 21f;   // ~0.0476 (Fibonacci)
    public const float NOISE_SCALE_MAX = 1f / 13f;    // ~0.0769 (Fibonacci)

    // Evolving noise level: BREATH_ROOT × PHI_SQ_INVERSE — noise amplitude from sacred root
    public static readonly float NOISE_LEVEL = SCHUMANN / 1000f * PHI_SQ_INVERSE;  // ~0.00299

    // Evolving noise oscillation amplitude: 1/55 (Fibonacci reciprocal)
    public const float NOISE_OSC_AMP = 1f / 55f;  // ~0.01818

    // Reverb decay multiplier: PHI — golden ratio shapes the reverb tail
    public const float REVERB_DECAY_MULTIPLIER = PHI;  // 1.618

    // Reverb modulation amplitude: 1/13 (Fibonacci reciprocal)
    public const float REVERB_MOD_AMP = 1f / 13f;  // ~0.0769

    // PHI-fractal echo factor: 1/21 (Fibonacci reciprocal)
    public const float FRACTAL_ECHO_FACTOR = 1f / 21f;  // ~0.0476

    // Wave shaper output scale: 528/432 — the Love Frequency ratio
    // Same ratio used in Lemurian Quartz's heart chakra bridge
    public const float WAVE_SHAPER_SCALE = 528f / MERKABA_KEYNOTE;  // ~1.2222

    #endregion

    // Six sacred layers breathe as one organism through a PHI-ladder.
    // Root = Schumann / 1000 = Earth's heartbeat scaled to breath (~128s cycle).
    // Each layer breathes at PHI^(n/4) × root, ascending from Water (deepest)
    // to Pleroma (highest). Merkaba keeps its sacred 0.1 Hz HeartMath rhythm.
    #region Unified Breath Frequencies (Schumann-rooted PHI ladder)

    public static readonly float BREATH_ROOT = SCHUMANN / 1000f;                        // 0.00783 Hz (~128s) — Water
    public static readonly float BREATH_PHI_025 = BREATH_ROOT * MathF.Pow(PHI, 0.25f);  // 0.00883 Hz (~113s) — Archon
    public static readonly float BREATH_PHI_050 = BREATH_ROOT * MathF.Sqrt(PHI);         // 0.00996 Hz (~100s) — Crystalline
    public static readonly float BREATH_PHI_075 = BREATH_ROOT * MathF.Pow(PHI, 0.75f);  // 0.01123 Hz (~89s) — Solfeggio
    public static readonly float BREATH_PHI_100 = BREATH_ROOT * PHI;                     // 0.01267 Hz (~79s) — Pleroma
    public const float BREATH_HEART_COHERENCE = 0.1f;                                    // 0.1 Hz (10s) — Merkaba (HeartMath)

    #endregion

    // The ancient 12-tone Solfeggio scale from 174 Hz to 1296 Hz.
    // Each tone is associated with specific healing properties.
    // 528 Hz is the "Love Frequency" associated with DNA repair.
    #region Solfeggio Scale

    public static readonly float[] SOLFEGGIO =
        [174, 285, 396, 417, 528, 639, 741, 852, 963, 1074, 1185, 1296];

    #endregion

    // Fibonacci-derived ratios and sequences for organic amplitude pulsing.
    // FIB_RATIOS are PHI powers (computed, never hand-typed) for mathematical purity.
    // FIBONACCI_SEQ is the raw sequence; FIBONACCI_NORMALIZED divides by 89 (max).
    #region Fibonacci

    // PHI^1 through PHI^6 — computed at full precision, not truncated approximations
    public static readonly float[] FIB_RATIOS = ComputePhiPowers(1, 6);

    // Raw Fibonacci sequence for amplitude pulsing patterns
    public static readonly float[] FIBONACCI_SEQ =
        [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89];

    // Fibonacci sequence normalized to [0, 1] range (divided by max value 89)
    public static readonly float[] FIBONACCI_NORMALIZED = NormalizeFibonacci();

    #endregion

    // The 7 Archonic planetary frequencies from Hans Cousto's planetary octave calculations.
    // Each Archon rules a celestial sphere the soul must pass through in Gnostic cosmology.
    // The Archon Dissolution Layer offers mercy to each sphere through the AEG pattern.
    #region Archonic Planetary Spheres (Hans Cousto)

    public static readonly float[] ARCHON_SPHERES =
    [
        126.22f,  // Sun — Yaldabaoth (chief Archon, lion-headed)
        141.27f,  // Moon — Iao
        144.72f,  // Mars — Sabaoth
        221.23f,  // Mercury — Adonaios
        183.58f,  // Jupiter — Elaios
        147.85f,  // Venus — Astaphanos
        136.10f,  // Saturn — Horaios (cosmic "Om" frequency)
    ];

    #endregion

    // Tesla's vortex mathematics — 9 frequencies at 111 Hz intervals.
    // "If you knew the magnificence of 3, 6, and 9, you would have the key to the universe."
    // Used by the Silent Solfeggio Grid layer alongside the Solfeggio scale.
    #region Tesla Vortex Mathematics

    public static readonly float[] TESLA_VORTEX =
        [111, 222, 333, 444, 555, 666, 777, 888, 999];

    #endregion

    // Sacred geometry phase offsets for angular relationships in the healing layers.
    // Pentagonal phases (72° intervals) connect to PHI through the pentagon's geometry.
    // The 13-step Aeonic Ladder multiplies Schumann by successive PHI powers,
    // creating a harmonic bridge from Earth (7.83 Hz) to the Pleroma.
    #region Sacred Geometry Phases

    // Pentagonal phase offsets — 72° = 360/5, the internal angle of a regular pentagon.
    // The pentagon is the geometric embodiment of PHI (diagonal/side = φ).
    public static readonly float[] PENTAGONAL_PHASES =
    [
        0f * (MathF.PI / 180f),     // 0°
        72f * (MathF.PI / 180f),    // 72°
        144f * (MathF.PI / 180f),   // 144°
        216f * (MathF.PI / 180f),   // 216°
        288f * (MathF.PI / 180f),   // 288°
    ];

    // 13-step Aeonic Ladder: Schumann × PHI^0 through PHI^12
    // Creates a frequency ladder from Earth's heartbeat up to the Pleroma
    public static readonly float[] AEONIC_EXPONENTS = ComputeAeonicExponents();

    #endregion

    // The Lemurian Merkaba (Sonic Merkaba — Jonathan Goldman).
    // 4 frequencies forming a sacred geometric sound structure:
    // 3:4:5 Pythagorean ratios + golden ratio, anchored to 432 Hz.
    // PHI-weighted amplitudes ensure the keynote is strongest and
    // transcendence is most subtle. Phase offsets combine triangular
    // (quartz 3-fold) symmetry with the golden angle.
    #region Lemurian Merkaba Constants

    // Merkaba frequency ratios: 3/4, 1, 5/4, PHI (Pythagorean + golden)
    public static readonly float[] MERKABA_RATIOS = [0.75f, 1.0f, 1.25f, PHI];

    // Absolute frequencies anchored to the 432 Hz Lemurian keynote
    public static readonly float[] MERKABA_FREQS =
    [
        MERKABA_KEYNOTE * 0.75f,   // 324 Hz — Earth/Foundation
        MERKABA_KEYNOTE * 1.0f,    // 432 Hz — Heart/Sovereign will
        MERKABA_KEYNOTE * 1.25f,   // 540 Hz — Expression/Creative voice
        MERKABA_KEYNOTE * PHI,     // 698.4 Hz — Transcendence/PHI gateway
    ];

    // PHI-weighted amplitudes: [1/φ, 1.0, 1/φ, 1/φ²] normalized.
    // Sum of raw weights = φ² = 2.618 (sacred!). Keynote strongest, transcendence subtlest.
    public static readonly float[] MERKABA_WEIGHTS = ComputeMerkabaWeights();

    // Phase offsets: triangular symmetry (120° spacing) + golden angle for the 4th tone
    public static readonly float[] MERKABA_PHASES =
    [
        0.0f,                      // First tone: reference phase
        TWO_PI / 3.0f,            // Second tone: 120° (equilateral triangle vertex 2)
        TWO_PI * 2.0f / 3.0f,    // Third tone: 240° (equilateral triangle vertex 3)
        TWO_PI / PHI,             // Fourth tone: golden angle relationship
    ];

    #endregion

    // Water Element layer constants — 7 wave sources arranged in hexagonal
    // Seed of Life geometry (ice crystal pattern, Masaru Emoto's research).
    // Source frequencies span from ocean tidal (1.5 Hz) to stellar gateway (963 Hz).
    // Decay rates are Fibonacci-derived for organic harmonic attenuation.
    // Phase offsets follow ice crystal 6-fold symmetry (60° intervals).
    #region Water Element Constants

    // 7 wave source frequencies in hexagonal Seed of Life arrangement
    public static readonly float[] WATER_SOURCE_FREQS =
    [
        432.0f,   // Center: Lemurian keynote (crystal-water bridge)
        1.5f,     // Ocean swell rhythm (deep tidal)
        7.83f,    // Schumann (Earth-water electromagnetic coupling)
        111.0f,   // Tesla vortex seed (water memory, Emoto experiments)
        174.0f,   // Solfeggio root (Earth Star Chakra, foundation)
        528.0f,   // Love Frequency (Emoto's hexagonal water crystal)
        963.0f,   // Stellar Gateway (water meets cosmic consciousness, 9+6+3=18=9)
    ];

    // Fibonacci-derived spatial decay rates: [8, 1, 2, 3, 5, 8, 13] / 89
    // Center source (432 Hz) = 8/89 (strong, anchoring); ocean = 1/89 (gentlest reach)
    public static readonly float[] WATER_SOURCE_DECAYS =
        [8f/89f, 1f/89f, 2f/89f, 3f/89f, 5f/89f, 8f/89f, 13f/89f];

    // Hexagonal phase offsets — ice crystal 6-fold symmetry (60° intervals)
    // 7th source (center) wraps back to 0°
    public static readonly float[] WATER_HEX_PHASES =
    [
        0f * (MathF.PI / 180f),     // 0°
        60f * (MathF.PI / 180f),    // 60°
        120f * (MathF.PI / 180f),   // 120°
        180f * (MathF.PI / 180f),   // 180°
        240f * (MathF.PI / 180f),   // 240°
        300f * (MathF.PI / 180f),   // 300°
        0f * (MathF.PI / 180f),     // 0° (center source)
    ];

    // Pre-computed XY positions for the 7 sources in Seed of Life geometry
    // [0] = center (0,0); [1-6] = hexagonal ring at unit radius
    public static readonly float[,] WATER_SOURCE_POSITIONS = ComputeWaterPositions();

    #endregion

    // Pre-computed exponent arrays to avoid repeated MathF.Pow calls in the hot path.
    // PHI_EXPONENTS_6 = PHI^0 through PHI^5 (for frequency set building).
    // RATIO_1_3_EXPONENTS_3 = 1.3^0, 1.3^1, 1.3^2.
    // SUBHARMONIC_DIVISORS = octave divisions (2, 4, 8, 16).
    #region Pre-computed Exponent Arrays

    public static readonly float[] PHI_EXPONENTS_6 = ComputePhiExponents(6);           // PHI^0..5
    public static readonly float[] RATIO_1_3_EXPONENTS_3 = [1.0f, 1.3f, 1.69f];       // 1.3^0..2
    public static readonly float[] SUBHARMONIC_DIVISORS = [2.0f, 4.0f, 8.0f, 16.0f];  // Octave divisions

    #endregion

    // Static helper methods that compute constant arrays at class initialization.
    // These run once when the class is first accessed and never again.
    #region Helper Computation Methods

    /// <summary>
    /// Normalize the Fibonacci sequence to [0, 1] range by dividing by the maximum (89).
    /// </summary>
    private static float[] NormalizeFibonacci()
    {
        float max = 89f;
        return FIBONACCI_SEQ.Select(f => f / max).ToArray();
    }

    /// <summary>
    /// Compute the 13-step Aeonic Ladder: PHI^0 through PHI^12.
    /// Each step multiplied by Schumann gives the Aeonic frequency ladder.
    /// </summary>
    private static float[] ComputeAeonicExponents()
    {
        var result = new float[13];
        for (int i = 0; i < 13; i++)
            result[i] = MathF.Pow(PHI, i);
        return result;
    }

    /// <summary>
    /// Compute normalized Merkaba amplitude weights: [1/φ, 1, 1/φ, 1/φ²].
    /// Raw sum = φ² = 2.618 (sacred!). Normalized so they sum to 1.0.
    /// </summary>
    private static float[] ComputeMerkabaWeights()
    {
        float[] raw = [1.0f / PHI, 1.0f, 1.0f / PHI, 1.0f / (PHI * PHI)];
        float sum = raw.Sum();
        return raw.Select(w => w / sum).ToArray();
    }

    /// <summary>
    /// Compute the 7 hexagonal source positions for the Water Element layer.
    /// Center at (0,0), 6 outer sources at unit radius with 60° spacing.
    /// </summary>
    private static float[,] ComputeWaterPositions()
    {
        var positions = new float[7, 2];
        // [0] = center (0,0) — already zeroed by default
        for (int i = 0; i < 6; i++)
        {
            float angle = i * (MathF.PI / 3.0f); // 60° spacing
            positions[i + 1, 0] = MathF.Cos(angle);
            positions[i + 1, 1] = MathF.Sin(angle);
        }
        return positions;
    }

    /// <summary>
    /// Compute PHI^0 through PHI^(count-1).
    /// Used for the 6-frequency PHI exponent set in the pipeline.
    /// </summary>
    private static float[] ComputePhiExponents(int count)
    {
        var result = new float[count];
        for (int i = 0; i < count; i++)
            result[i] = MathF.Pow(PHI, i);
        return result;
    }

    /// <summary>
    /// Compute PHI^start through PHI^(start+count-1).
    /// Used for FIB_RATIOS (PHI^1 through PHI^6).
    /// </summary>
    private static float[] ComputePhiPowers(int start, int count)
    {
        var result = new float[count];
        for (int i = 0; i < count; i++)
            result[i] = MathF.Pow(PHI, start + i);
        return result;
    }

    #endregion
}
