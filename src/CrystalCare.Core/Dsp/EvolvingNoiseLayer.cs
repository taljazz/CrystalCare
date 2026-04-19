using CrystalCare.Core.Frequencies;

namespace CrystalCare.Core.Dsp;

/// <summary>
/// Evolving noise layer: low-frequency modulated Gaussian noise.
/// Port of AudioProcessor.evolving_noise_layer() from SoundGenerator.py.
/// </summary>
public static class EvolvingNoiseLayer
{
    /// <summary>
    /// Generate an evolving noise layer with low-frequency oscillation modulation.
    /// Noise level = BREATH_ROOT × PHI_SQ_INVERSE (sacred root derivative).
    /// Oscillation amplitude = 1/55 (Fibonacci reciprocal).
    /// Frequency range spans the breath ladder: BREATH_ROOT to BREATH_PHI_100.
    /// </summary>
    public static float[] Generate(ReadOnlySpan<double> t,
        float noiseLevel = 0f, // 0 = use sacred default
        Random? rng = null)
    {
        rng ??= Random.Shared;

        // Sacred noise level: BREATH_ROOT × PHI_SQ_INVERSE (~0.00299)
        if (noiseLevel == 0f)
            noiseLevel = SacredConstants.NOISE_LEVEL;

        var result = new float[t.Length];

        // Frequency range spans the breath ladder: BREATH_ROOT (~0.00783) to BREATH_PHI_100 (~0.01267)
        double freqRange = SacredConstants.BREATH_PHI_100 - SacredConstants.BREATH_ROOT;
        double freq = SacredConstants.BREATH_ROOT + rng.NextDouble() * freqRange;

        for (int i = 0; i < t.Length; i++)
        {
            // Box-Muller transform for Gaussian noise
            float u1 = (float)rng.NextDouble();
            float u2 = (float)rng.NextDouble();
            float gaussian = MathF.Sqrt(-2f * MathF.Log(MathF.Max(u1, 1e-10f))) *
                             MathF.Cos(SacredConstants.TWO_PI * u2);

            float noise = gaussian * noiseLevel;
            // Oscillation amplitude = 1/55 (Fibonacci reciprocal)
            // Double precision phase for long-session stability
            float lowFreqOsc = SacredConstants.NOISE_OSC_AMP *
                (float)System.Math.Sin(SacredConstants.TWO_PI_D * freq * t[i]);
            result[i] = noise * (1.0f + lowFreqOsc);
        }

        return result;
    }
}
