using CrystalCare.Core.Frequencies;

namespace CrystalCare.Core.Dsp;

/// <summary>
/// Vectorized harmonic generation with cross-modulation.
/// Port of jit_generate_harmonics_vectorized() and jit_cross_modulate_wave().
/// </summary>
public static class HarmonicGenerator
{
    /// <summary>
    /// Generate harmonics for all frequencies with envelope, LFO, and modulation depth.
    /// Port of jit_generate_harmonics_vectorized — the hot inner loop of sound generation.
    /// </summary>
    public static float[] GenerateHarmonics(ReadOnlySpan<float> t, float[] frequencies,
        ReadOnlySpan<float> envelope, ReadOnlySpan<float> lfo, float[] modDepths)
    {
        int nSamples = t.Length;
        int nFreqs = frequencies.Length;
        var result = new float[nSamples];

        for (int f = 0; f < nFreqs; f++)
        {
            float freq = frequencies[f];
            float modFreq = freq * 0.5f;
            float modDepth = modDepths[f];
            float scale = 0.015f / (f + 1);

            for (int i = 0; i < nSamples; i++)
            {
                float modSignal = MathF.Sin(SacredConstants.TWO_PI * modFreq * t[i]) * modDepth;
                float wave = MathF.Sin(SacredConstants.TWO_PI * freq * t[i] + modSignal);
                result[i] += wave * envelope[i] * scale * lfo[i];
            }
        }

        return result;
    }

    /// <summary>
    /// Cross-modulation wave generation.
    /// Port of jit_cross_modulate_wave().
    /// </summary>
    public static float[] CrossModulate(float baseFreq, float modFreq,
        ReadOnlySpan<float> t, float modDepth)
    {
        var result = new float[t.Length];
        for (int i = 0; i < t.Length; i++)
        {
            float mod = MathF.Sin(SacredConstants.TWO_PI * modFreq * t[i]) * modDepth;
            result[i] = MathF.Sin(SacredConstants.TWO_PI * baseFreq * t[i] + mod);
        }
        return result;
    }

    /// <summary>
    /// Quantum harmonic interference with Simplex-driven gamma.
    /// Port of jit_quantum_harmonic().
    /// </summary>
    public static float[] QuantumHarmonic(ReadOnlySpan<float> t, float baseFreq,
        ReadOnlySpan<float> gamma)
    {
        float f1 = baseFreq;
        float f2 = baseFreq * 1.41421356237f;   // sqrt(2)
        float f3 = baseFreq * 2.71828182846f;   // e
        float f4 = baseFreq * 3.14159265359f;   // pi

        var result = new float[t.Length];
        for (int i = 0; i < t.Length; i++)
        {
            float time = t[i];
            float alpha = 0.25f * MathF.Sin(0.03f * MathF.PI * time);
            float beta = 0.2f * MathF.Cos(0.02f * MathF.PI * time);

            float w1 = MathF.Sin(SacredConstants.TWO_PI * f1 * time + alpha);
            float w2 = MathF.Sin(SacredConstants.TWO_PI * f2 * time + beta);
            float w3 = MathF.Sin(SacredConstants.TWO_PI * f3 * time + gamma[i]);
            float w4 = MathF.Sin(SacredConstants.TWO_PI * f4 * time + gamma[i] * 0.7f);

            result[i] = (w1 + w2 + w3 + w4) / 3.8f +
                         0.15f * MathF.Sin(SacredConstants.TWO_PI * 7.83f * time);
        }
        return result;
    }
}
