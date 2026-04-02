using CrystalCare.Core.Frequencies;

namespace CrystalCare.Core.Dsp;

/// <summary>
/// Microtonal LFO with PHI-modulated frequency and inner modulation.
/// Creates organic, breathing amplitude variation.
/// Port of AudioProcessor.microtonal_lfo() and batch_microtonal_lfo() from SoundGenerator.py.
/// </summary>
public static class MicrotonalLfo
{
    /// <summary>
    /// Pre-computed LFO parameters (drawn once at pipeline start).
    /// </summary>
    public sealed class LfoParams
    {
        public float Lfo1Freq { get; init; }
        public float Lfo2Freq { get; init; }
        public float Depth { get; init; }
    }

    /// <summary>
    /// Draw random LFO parameters for a given base frequency.
    /// </summary>
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
    public static float[] Compute(ReadOnlySpan<float> t, LfoParams p)
    {
        var result = new float[t.Length];
        for (int i = 0; i < t.Length; i++)
        {
            float drift = 0.01f * MathF.Sin(0.1f * MathF.PI * t[i]);
            float innerMod = 0.3f * MathF.Sin(SacredConstants.TWO_PI * p.Lfo2Freq * t[i]);
            result[i] = (p.Depth + drift) *
                (1.0f + MathF.Sin(SacredConstants.TWO_PI * p.Lfo1Freq * t[i] + innerMod));
        }
        return result;
    }

    /// <summary>
    /// Single-call LFO generation (for batch/non-streaming use).
    /// </summary>
    public static float[] Generate(ReadOnlySpan<float> t, float baseFrequency, Random? rng = null)
    {
        return Compute(t, DrawParams(baseFrequency, rng));
    }
}
