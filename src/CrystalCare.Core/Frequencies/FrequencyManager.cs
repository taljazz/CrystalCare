namespace CrystalCare.Core.Frequencies;

/// <summary>
/// Manages frequency sets and handles random selection based on predefined weights.
/// Ported from frequencies.py — all ratio sets and frequency modes preserved exactly.
/// </summary>
public sealed class FrequencyManager
{
    private readonly Random _rng = new();

    // All 9 sacred geometry ratio sets used for geometric modulation.
    // Sacred geometry, Flower of Life, Triple Helix, Taygetan, and derived combinations.
    // Each ratio set defines frequency relationships rooted in sacred mathematics.
    #region Ratio Sets

    public static readonly Dictionary<string, float> SacredGeometryRatios = new()
    {
        ["metatron_ratio"] = SacredConstants.SQRT_2,            // √2 — Metatron's Cube diagonal
        ["vesica_piscis_ratio"] = SacredConstants.SQRT_3,       // √3 — Vesica Piscis height
        ["hexagon_ratio"] = 2.0f,
        ["flower_of_life_ratio"] = SacredConstants.PHI,         // φ — Golden ratio
        ["circle_ratio"] = 3.0f,
    };

    public static readonly Dictionary<string, float> FlowerOfLifeRatios = new()
    {
        ["petal_ratio"] = 1.3f,
        ["intersect_ratio"] = 1.5f,
        ["symmetry_ratio"] = 2.5f,
    };

    public static readonly Dictionary<string, float> TripleHelixRatios = new()
    {
        ["strand_1"] = 1.0f,
        ["strand_2"] = 1.2f,
        ["strand_3"] = 1.4f,
    };

    // All 9 ratio sets with their names
    public static readonly Dictionary<string, Dictionary<string, float>> RatioSets = new()
    {
        ["sacred_geometry"] = SacredGeometryRatios,
        ["flower_of_life"] = FlowerOfLifeRatios,
        ["triple_helix"] = TripleHelixRatios,
        ["combined"] = Merge(SacredGeometryRatios, FlowerOfLifeRatios),
        ["minimal"] = new() { ["metatron_ratio"] = SacredConstants.SQRT_2 },
        ["enhanced_geometry"] = new() { ["octagon_ratio"] = 2.0f * SacredConstants.SQRT_2, ["spiral_ratio"] = SacredConstants.SQRT_5 },
        ["fibonacci_set"] = new()
        {
            ["fibonacci_ratio_1"] = SacredConstants.PHI,
            ["fibonacci_ratio_2"] = SacredConstants.PHI * SacredConstants.PHI,
            ["fibonacci_ratio_3"] = SacredConstants.PHI * SacredConstants.PHI * SacredConstants.PHI,
        },
        ["fractal_set"] = new() { ["mandelbrot_ratio"] = SacredConstants.PI, ["julia_ratio"] = SacredConstants.EULER_E },
        ["taygetan"] = new()
        {
            ["root"] = 1.0f,
            ["etheric_body"] = SacredConstants.SQRT_2,                                    // √2
            ["astral_bridge"] = SacredConstants.SQRT_3,                                   // √3
            ["natural_log"] = SacredConstants.EULER_E,                                    // e
            ["crown_portal"] = SacredConstants.PI,                                        // π
            ["zero_point"] = SacredConstants.PHI * SacredConstants.PHI * SacredConstants.PHI, // φ³
            ["remembrance"] = SacredConstants.PHI,                                        // φ — one golden ratio, one truth
        },
    };

    #endregion

    // Selects a random ratio set using weighted probability distribution.
    // Sacred geometry has the highest weight (0.25), with decreasing weights
    // for less common sets. Ensures variety with controlled distribution.
    #region Weighted Random Selection

    // Weighted probability distribution for random selection
    private static readonly string[] RatioSetNames = RatioSets.Keys.ToArray();
    private static readonly float[] RatioSetWeights = NormalizeWeights(
        [0.25f, 0.15f, 0.12f, 0.10f, 0.08f, 0.07f, 0.06f, 0.05f, 0.12f]);

    /// <summary>
    /// Multiply base frequency by each ratio in the set.
    /// </summary>
    public static float[] GetGeometricFrequencySet(float baseFreq, Dictionary<string, float> ratios)
    {
        return ratios.Values.Select(r => baseFreq * r).ToArray();
    }

    /// <summary>
    /// Select a random ratio set using weighted probability distribution.
    /// </summary>
    public Dictionary<string, float> SelectRandomRatioSet(CancellationToken ct = default)
    {
        if (ct.IsCancellationRequested)
            return new Dictionary<string, float>();

        float roll = (float)_rng.NextDouble();
        float cumulative = 0f;
        for (int i = 0; i < RatioSetNames.Length; i++)
        {
            cumulative += RatioSetWeights[i];
            if (roll <= cumulative)
                return RatioSets[RatioSetNames[i]];
        }
        return RatioSets[RatioSetNames[^1]];
    }

    #endregion

    // Returns the frequency array (or binaural pairs) for a given FrequencyMode.
    // Modes 0-4 return mono frequency arrays; mode 5 returns stereo binaural pairs;
    // mode 6 (Dimensional) is handled by the pipeline itself.
    #region Frequency Mode Selection

    /// <summary>
    /// Get frequencies for a given mode selection.
    /// Returns float[] for most modes, or (float, float)[] encoded as flat array for binaural (mode 5).
    /// </summary>
    public FrequencyResult GetFrequencies(FrequencyMode mode, float baseFreqInitial = 432f)
    {
        return mode switch
        {
            FrequencyMode.Standard => new FrequencyResult([174f, 396f, 417f, 528f]),
            FrequencyMode.Solfeggio => new FrequencyResult([852f, 963f]),
            FrequencyMode.Fibonacci => new FrequencyResult([136.10f, 194.18f, 211.44f, 303f]),
            FrequencyMode.Pythagorean => new FrequencyResult(
                GetGeometricFrequencySet(baseFreqInitial, SacredGeometryRatios)
                .Concat(GetGeometricFrequencySet(baseFreqInitial, FlowerOfLifeRatios))
                .ToArray()),
            FrequencyMode.TripleHelixDna => new FrequencyResult(
                GetGeometricFrequencySet(baseFreqInitial, TripleHelixRatios)),
            FrequencyMode.TaygetanBinaural => CreateTaygetanBinaural(baseFreqInitial),
            _ => new FrequencyResult(Array.Empty<float>()),
        };
    }

    #endregion

    // Internal helpers: Taygetan binaural pair creation, weight normalization,
    // and dictionary merging for combined ratio sets.
    #region Helper Methods

    private static FrequencyResult CreateTaygetanBinaural(float baseFreq)
    {
        const float delta = 7.7f; // Taygetan sync beat
        var tayRatios = RatioSets["taygetan"];
        var filtered = tayRatios.Values.Where(r => r >= 0.1f).Select(r => baseFreq * r).ToArray();
        var pairs = filtered.Select(f => (f + delta / 2f, f - delta / 2f)).ToArray();
        return new FrequencyResult(pairs);
    }

    private static float[] NormalizeWeights(float[] weights)
    {
        float sum = weights.Sum();
        return weights.Select(w => w / sum).ToArray();
    }

    private static Dictionary<string, float> Merge(
        Dictionary<string, float> a, Dictionary<string, float> b)
    {
        var result = new Dictionary<string, float>(a);
        foreach (var kvp in b)
            result[kvp.Key] = kvp.Value;
        return result;
    }

    #endregion
}

/// <summary>
/// Result from GetFrequencies — either mono frequencies or binaural pairs.
/// </summary>
public sealed class FrequencyResult
{
    public float[] Frequencies { get; }
    public (float Left, float Right)[]? BinauralPairs { get; }
    public bool IsBinaural => BinauralPairs != null;

    public FrequencyResult(float[] frequencies)
    {
        Frequencies = frequencies;
        BinauralPairs = null;
    }

    public FrequencyResult((float Left, float Right)[] pairs)
    {
        Frequencies = pairs.Select(p => (p.Left + p.Right) / 2f).ToArray();
        BinauralPairs = pairs;
    }
}
