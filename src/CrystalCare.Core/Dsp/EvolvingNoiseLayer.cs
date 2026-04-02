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
    /// </summary>
    public static float[] Generate(ReadOnlySpan<float> t, float noiseLevel = 0.003f, Random? rng = null)
    {
        rng ??= Random.Shared;
        var result = new float[t.Length];

        float freq = (float)(rng.NextDouble() * 0.013 + 0.002); // [0.002, 0.015]

        for (int i = 0; i < t.Length; i++)
        {
            // Box-Muller transform for Gaussian noise
            float u1 = (float)rng.NextDouble();
            float u2 = (float)rng.NextDouble();
            float gaussian = MathF.Sqrt(-2f * MathF.Log(MathF.Max(u1, 1e-10f))) *
                             MathF.Cos(SacredConstants.TWO_PI * u2);

            float noise = gaussian * noiseLevel;
            float lowFreqOsc = 0.015f * MathF.Sin(SacredConstants.TWO_PI * freq * t[i]);
            result[i] = noise * (1.0f + lowFreqOsc);
        }

        return result;
    }
}
