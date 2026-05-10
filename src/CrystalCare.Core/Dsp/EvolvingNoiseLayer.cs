using CrystalCare.Core.Frequencies;

namespace CrystalCare.Core.Dsp;

/// <summary>
/// Evolving noise layer: low-frequency modulated Gaussian noise.
/// Port of AudioProcessor.evolving_noise_layer() from SoundGenerator.py.
/// </summary>
public static class EvolvingNoiseLayer
{
    // Generates Gaussian noise modulated by a slow low-frequency oscillation.
    // The Box-Muller transform produces unit-Gaussian samples; the slow oscillation
    // (BREATH_ROOT to BREATH_PHI_100, ~0.008-0.013 Hz) breathes the noise level.
    // Optional pre-drawn frequency parameter prevents the slow envelope from
    // discontinuously jumping at every chunk boundary when called from a streaming
    // pipeline — letting the noise floor evolve smoothly across the entire session.
    #region Noise Generation

    /// <summary>
    /// Generate an evolving noise layer with low-frequency oscillation modulation.
    /// Noise level = BREATH_ROOT × PHI_SQ_INVERSE (sacred root derivative).
    /// Oscillation amplitude = 1/55 (Fibonacci reciprocal).
    /// Frequency range spans the breath ladder: BREATH_ROOT to BREATH_PHI_100.
    /// </summary>
    /// <param name="frequency">
    /// Optional pre-drawn oscillation frequency (Hz). When 0 (default), the function
    /// draws a fresh random frequency in the breath-ladder range each call — this is
    /// the legacy behavior, but in streaming pipelines it creates a discontinuity in
    /// the slow envelope at every chunk boundary. Pass a single session-level value
    /// (drawn once before the chunk loop) to fix the discontinuity and let the noise
    /// floor breathe smoothly across the whole session.
    /// </param>
    public static float[] Generate(ReadOnlySpan<double> t,
        float noiseLevel = 0f, // 0 = use sacred default
        Random? rng = null,
        float frequency = 0f)
    {
        rng ??= Random.Shared;

        // Sacred noise level: BREATH_ROOT × PHI_SQ_INVERSE (~0.00299)
        if (noiseLevel == 0f)
            noiseLevel = SacredConstants.NOISE_LEVEL;

        var result = new float[t.Length];

        // Resolve the oscillation frequency. If the caller supplied a non-zero value,
        // use it (smooth session-wide envelope); otherwise fall back to legacy
        // behavior of drawing a fresh frequency in the breath-ladder range each call.
        double freqRange = SacredConstants.BREATH_PHI_100 - SacredConstants.BREATH_ROOT;
        double freq = frequency > 0f
            ? frequency
            : SacredConstants.BREATH_ROOT + rng.NextDouble() * freqRange;

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

    #endregion
}
