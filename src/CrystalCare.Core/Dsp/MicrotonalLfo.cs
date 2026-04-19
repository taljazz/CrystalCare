using CrystalCare.Core.Frequencies;

namespace CrystalCare.Core.Dsp;

/// <summary>
/// Microtonal LFO with PHI-modulated frequency and inner modulation.
/// Creates organic, breathing amplitude variation.
/// Port of AudioProcessor.microtonal_lfo() and batch_microtonal_lfo() from SoundGenerator.py.
/// </summary>
public static class MicrotonalLfo
{
    // LFO parameter container — drawn once at pipeline start for session consistency.
    // Two frequencies (lfo1 and lfo2) with PHI-modulated relationship.
    #region Parameters

    public sealed class LfoParams
    {
        public float Lfo1Freq { get; init; }
        public float Lfo2Freq { get; init; }
        public float Depth { get; init; }
    }

    #endregion

    // DrawParams: randomizes LFO frequencies and depth with PHI-power modulation.
    // Compute: generates LFO values for a time array using the pre-drawn params.
    // Generate: convenience one-shot method combining DrawParams + Compute.
    #region Parameter Drawing and Computation

    public static LfoParams DrawParams(float baseFrequency, Random? rng = null)
    {
        rng ??= Random.Shared;
        float phiPower = MathF.Pow(SacredConstants.PHI, (float)(rng.NextDouble() * 0.1 - 0.05));
        float lfo1 = baseFrequency * phiPower;
        float lfo2 = lfo1 * (float)(rng.NextDouble() * 0.02 + 0.99);
        float depth = (float)(rng.NextDouble() * 0.004 + 0.002);
        return new LfoParams { Lfo1Freq = lfo1, Lfo2Freq = lfo2, Depth = depth };
    }

    /// <summary>
    /// Compute LFO values for a time array using pre-drawn parameters.
    /// </summary>
    public static float[] Compute(ReadOnlySpan<double> t, LfoParams p)
    {
        var result = new float[t.Length];
        for (int i = 0; i < t.Length; i++)
        {
            // Drift: 1/89 (Fibonacci) amplitude at 0.1 Hz (HeartMath heart coherence)
            // Double precision for phase, cast sin result to float
            float drift = SacredConstants.LFO_DRIFT_AMP *
                (float)System.Math.Sin(SacredConstants.BREATH_HEART_COHERENCE * System.Math.PI * t[i]);

            // Inner modulation: 1/PHI² depth — golden ratio squared reciprocal
            float innerMod = SacredConstants.LFO_INNER_MOD_DEPTH *
                (float)System.Math.Sin(SacredConstants.TWO_PI_D * p.Lfo2Freq * t[i]);

            // Combined LFO: depth + drift, modulated by lfo1 with inner phase modulation
            result[i] = (p.Depth + drift) *
                (1.0f + (float)System.Math.Sin(SacredConstants.TWO_PI_D * p.Lfo1Freq * t[i] + innerMod));
        }
        return result;
    }

    /// <summary>
    /// Single-call LFO generation (for batch/non-streaming use).
    /// </summary>
    public static float[] Generate(ReadOnlySpan<double> t, float baseFrequency, Random? rng = null)
    {
        return Compute(t, DrawParams(baseFrequency, rng));
    }

    #endregion
}
