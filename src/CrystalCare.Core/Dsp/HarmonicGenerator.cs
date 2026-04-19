using CrystalCare.Core.Frequencies;

namespace CrystalCare.Core.Dsp;

/// <summary>
/// Vectorized harmonic generation with cross-modulation.
/// Time input is double precision to preserve phase accuracy over long sessions.
/// </summary>
public static class HarmonicGenerator
{
    // The hot inner loop — generates harmonics for all 13 frequencies with
    // envelope, LFO, cross-modulation, and per-harmonic amplitude scaling.
    // Cross-modulation at freq × PHI_INVERSE creates golden-ratio harmonic relationships.
    #region Harmonic Generation

    public static float[] GenerateHarmonics(ReadOnlySpan<double> t, float[] frequencies,
        ReadOnlySpan<float> envelope, ReadOnlySpan<float> lfo, float[] modDepths)
    {
        int nSamples = t.Length;
        int nFreqs = frequencies.Length;
        var result = new float[nSamples];

        for (int f = 0; f < nFreqs; f++)
        {
            // Use double precision for freq * t multiplication to preserve phase accuracy
            double freq = frequencies[f];
            double modFreq = freq * SacredConstants.PHI_INVERSE; // 1/φ cross-modulation
            float modDepth = modDepths[f];
            float scale = 0.015f / (f + 1);

            for (int i = 0; i < nSamples; i++)
            {
                // Compute phase in double, call Sin in double, cast result to float
                float modSignal = (float)System.Math.Sin(SacredConstants.TWO_PI_D * modFreq * t[i]) * modDepth;
                float wave = (float)System.Math.Sin(SacredConstants.TWO_PI_D * freq * t[i] + modSignal);
                result[i] += wave * envelope[i] * scale * lfo[i];
            }
        }

        return result;
    }

    #endregion

    // FM synthesis cross-modulation — one frequency modulates another's phase.
    // Creates complex harmonic sidebands for richer tonal character.
    #region Cross-Modulation

    public static float[] CrossModulate(float baseFreq, float modFreq,
        ReadOnlySpan<double> t, float modDepth)
    {
        var result = new float[t.Length];
        for (int i = 0; i < t.Length; i++)
        {
            // Double precision through the phase computation
            float mod = (float)System.Math.Sin(SacredConstants.TWO_PI_D * modFreq * t[i]) * modDepth;
            result[i] = (float)System.Math.Sin(SacredConstants.TWO_PI_D * baseFreq * t[i] + mod);
        }
        return result;
    }

    #endregion

    // Quantum harmonic interference — 4 waves at irrational frequency ratios
    // (√2, e, π) with simplex-driven phase modulation. Creates non-repeating
    // interference patterns. Includes Schumann resonance undertone.
    #region Quantum Harmonic

    public static float[] QuantumHarmonic(ReadOnlySpan<double> t, float baseFreq,
        ReadOnlySpan<float> gamma)
    {
        double f1 = baseFreq;
        double f2 = baseFreq * 1.41421356237;   // sqrt(2)
        double f3 = baseFreq * 2.71828182846;   // e
        double f4 = baseFreq * 3.14159265359;   // pi

        var result = new float[t.Length];
        for (int i = 0; i < t.Length; i++)
        {
            double time = t[i];
            // alpha/beta are slow modulations — precision is less critical here but keep double
            float alpha = 0.25f * (float)System.Math.Sin(0.03 * System.Math.PI * time);
            float beta = 0.2f * (float)System.Math.Cos(0.02 * System.Math.PI * time);

            float w1 = (float)System.Math.Sin(SacredConstants.TWO_PI_D * f1 * time + alpha);
            float w2 = (float)System.Math.Sin(SacredConstants.TWO_PI_D * f2 * time + beta);
            float w3 = (float)System.Math.Sin(SacredConstants.TWO_PI_D * f3 * time + gamma[i]);
            float w4 = (float)System.Math.Sin(SacredConstants.TWO_PI_D * f4 * time + gamma[i] * 0.7f);

            result[i] = (w1 + w2 + w3 + w4) / 3.8f +
                         0.15f * (float)System.Math.Sin(SacredConstants.TWO_PI_D * 7.83 * time);
        }
        return result;
    }

    #endregion
}
