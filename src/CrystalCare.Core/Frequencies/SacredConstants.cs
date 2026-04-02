namespace CrystalCare.Core.Frequencies;

/// <summary>
/// All sacred frequency constants, mathematical ratios, and crystal profile data.
/// These are the healing frequencies at the heart of CrystalCare.
/// </summary>
public static class SacredConstants
{
    // === Fundamental Constants ===
    public const float PHI = 1.618033988749895f;       // Golden ratio
    public const float TWO_PI = 2.0f * MathF.PI;
    public const float SCHUMANN = 7.83f;               // Earth's electromagnetic heartbeat (Hz)
    public const float OGDOAD_FREQ = SCHUMANN * 8;     // 62.64 Hz — 8th sphere threshold to Pleroma
    public const float MERKABA_KEYNOTE = 432.0f;       // Lemurian tuning keynote (Hz)

    // === Solfeggio Scale (12 tones) ===
    public static readonly float[] SOLFEGGIO =
        [174, 285, 396, 417, 528, 639, 741, 852, 963, 1074, 1185, 1296];

    // === Fibonacci Ratios ===
    public static readonly float[] FIB_RATIOS =
        [1.618f, 2.618f, 4.236f, 6.854f, 11.090f, 17.944f];

    // === The 7 Archonic Planetary Spheres (Hans Cousto frequencies) ===
    // Each Archon rules a celestial sphere the soul must pass through
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

    // === Tesla 3-6-9 Vortex Mathematics ===
    // "If you knew the magnificence of 3, 6, and 9, you would have the key to the universe"
    public static readonly float[] TESLA_VORTEX =
        [111, 222, 333, 444, 555, 666, 777, 888, 999];

    // === Fibonacci Sequence (normalized) for amplitude pulsing ===
    public static readonly float[] FIBONACCI_SEQ =
        [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89];

    public static readonly float[] FIBONACCI_NORMALIZED = NormalizeFibonacci();

    // === Pentagonal Phase Offsets (sacred geometry — connected to PHI) ===
    // 72 degrees = 360/5, the internal angle of a regular pentagon
    public static readonly float[] PENTAGONAL_PHASES =
    [
        0f * (MathF.PI / 180f),
        72f * (MathF.PI / 180f),
        144f * (MathF.PI / 180f),
        216f * (MathF.PI / 180f),
        288f * (MathF.PI / 180f),
    ];

    // === 13-step Aeonic Ladder (Schumann x PHI^n) ===
    public static readonly float[] AEONIC_EXPONENTS = ComputeAeonicExponents();

    // === Lemurian Merkaba (Sonic Merkaba — Jonathan Goldman) ===
    // 3:4:5 Pythagorean ratios + PHI, anchored to 432 Hz
    public static readonly float[] MERKABA_RATIOS = [0.75f, 1.0f, 1.25f, PHI];
    public static readonly float[] MERKABA_FREQS =
    [
        MERKABA_KEYNOTE * 0.75f,   // 324 Hz — Earth/Foundation
        MERKABA_KEYNOTE * 1.0f,    // 432 Hz — Heart/Sovereign will
        MERKABA_KEYNOTE * 1.25f,   // 540 Hz — Expression/Creative voice
        MERKABA_KEYNOTE * PHI,     // 698.4 Hz — Transcendence
    ];

    // PHI-weighted amplitudes: keynote strongest, transcendence most subtle
    // Weights: [1/PHI, 1.0, 1/PHI, 1/PHI^2] — sum = PHI^2 = 2.618 (sacred!)
    public static readonly float[] MERKABA_WEIGHTS = ComputeMerkabaWeights();

    // Triangular + PHI phase offsets (quartz 3-fold symmetry + golden angle)
    public static readonly float[] MERKABA_PHASES =
    [
        0.0f,
        TWO_PI / 3.0f,
        TWO_PI * 2.0f / 3.0f,
        TWO_PI / PHI,
    ];

    // === Water Element Sacred Constants (6th sacred layer) ===
    // 7 wave sources in hexagonal Seed of Life arrangement (ice crystal geometry)
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

    public static readonly float[] WATER_SOURCE_DECAYS =
        [0.08f, 0.01f, 0.015f, 0.04f, 0.05f, 0.10f, 0.12f];

    // Hexagonal phase offsets (ice crystal 6-fold symmetry)
    public static readonly float[] WATER_HEX_PHASES =
    [
        0f * (MathF.PI / 180f),
        60f * (MathF.PI / 180f),
        120f * (MathF.PI / 180f),
        180f * (MathF.PI / 180f),
        240f * (MathF.PI / 180f),
        300f * (MathF.PI / 180f),
        0f * (MathF.PI / 180f),
    ];

    // Pre-computed hexagonal source positions (Seed of Life geometry)
    // [0] = center (0,0); [1-6] = hexagonal ring at radius 1.0
    public static readonly float[,] WATER_SOURCE_POSITIONS = ComputeWaterPositions();

    // === Pre-computed Exponent Arrays (SoundGenerator class-level) ===
    // Avoids repeated exponentiation in hot paths
    public static readonly float[] PHI_EXPONENTS_6 = ComputePhiExponents(6);
    public static readonly float[] RATIO_1_3_EXPONENTS_3 = [1.0f, 1.3f, 1.69f]; // 1.3^0, 1.3^1, 1.3^2
    public static readonly float[] SUBHARMONIC_DIVISORS = [2.0f, 4.0f, 8.0f, 16.0f];

    // ========================================
    // Helper computation methods
    // ========================================

    private static float[] NormalizeFibonacci()
    {
        float max = 89f; // max of FIBONACCI_SEQ
        return FIBONACCI_SEQ.Select(f => f / max).ToArray();
    }

    private static float[] ComputeAeonicExponents()
    {
        var result = new float[13];
        for (int i = 0; i < 13; i++)
            result[i] = MathF.Pow(PHI, i);
        return result;
    }

    private static float[] ComputeMerkabaWeights()
    {
        float[] raw = [1.0f / PHI, 1.0f, 1.0f / PHI, 1.0f / (PHI * PHI)];
        float sum = raw.Sum();
        return raw.Select(w => w / sum).ToArray();
    }

    private static float[,] ComputeWaterPositions()
    {
        var positions = new float[7, 2];
        // [0] = center (0,0) — already zeroed
        for (int i = 0; i < 6; i++)
        {
            float angle = i * (MathF.PI / 3.0f);
            positions[i + 1, 0] = MathF.Cos(angle);
            positions[i + 1, 1] = MathF.Sin(angle);
        }
        return positions;
    }

    private static float[] ComputePhiExponents(int count)
    {
        var result = new float[count];
        for (int i = 0; i < count; i++)
            result[i] = MathF.Pow(PHI, i);
        return result;
    }
}
