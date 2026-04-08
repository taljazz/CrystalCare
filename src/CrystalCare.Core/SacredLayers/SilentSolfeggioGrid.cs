using CrystalCare.Core.Frequencies;

namespace CrystalCare.Core.SacredLayers;

/// <summary>
/// Silent Solfeggio Grid — 2nd sacred layer.
///
/// Combines the ancient 12-tone Solfeggio scale with Tesla's 3-6-9 vortex
/// mathematics. Fibonacci amplitude pulsing creates organic, living modulation.
///
/// Scale: 0.00165 (sub-perceptual)
/// Fade: 40 seconds
/// Breath: 15% at 0.01 Hz (~100 second cycle)
///
/// Port of AudioProcessor.silent_solfeggio_grid_chunk() from SoundGenerator.py.
/// </summary>
public sealed class SilentSolfeggioGrid : ISacredLayer
{
    public float[] ComputeChunk(ReadOnlySpan<float> tChunk, float totalDuration)
    {
        if (totalDuration < 60f)
            return new float[tChunk.Length];

        int n = tChunk.Length;

        // Solfeggio grid: sum of 12 solfeggio frequencies
        var solfeggioGrid = new float[n];
        var solfeggio = SacredConstants.SOLFEGGIO;
        for (int s = 0; s < solfeggio.Length; s++)
        {
            float freq = solfeggio[s];
            for (int i = 0; i < n; i++)
                solfeggioGrid[i] += MathF.Sin(SacredConstants.TWO_PI * freq * tChunk[i]);
        }

        // Tesla 3-6-9 vortex grid
        var teslaGrid = new float[n];
        var tesla = SacredConstants.TESLA_VORTEX;
        for (int s = 0; s < tesla.Length; s++)
        {
            float freq = tesla[s];
            for (int i = 0; i < n; i++)
                teslaGrid[i] += MathF.Sin(SacredConstants.TWO_PI * freq * tChunk[i]);
        }

        // Fibonacci amplitude (smooth sine modulation)
        const float fibCycleFreq = 0.004f;
        var combined = new float[n];
        for (int i = 0; i < n; i++)
        {
            float fibAmp = 0.95f + 0.05f * MathF.Sin(SacredConstants.TWO_PI * fibCycleFreq * tChunk[i]);
            fibAmp += 0.02f * MathF.Sin(SacredConstants.TWO_PI * fibCycleFreq * SacredConstants.PHI * tChunk[i]);
            fibAmp = global::System.Math.Clamp(fibAmp, 0.88f, 1.0f);

            combined[i] = (0.6f * solfeggioGrid[i] + 0.4f * teslaGrid[i]) * fibAmp;
        }

        // Fade envelope + breathing + scale
        var fade = SacredFadeEnvelope.Compute(tChunk, totalDuration, fadeSeconds: 34.0f);
        for (int i = 0; i < n; i++)
        {
            float breath = 0.925f + 0.075f * MathF.Sin(SacredConstants.TWO_PI * 0.01f * tChunk[i]);
            combined[i] *= fade[i] * breath * 0.00165f;
        }

        return combined;
    }
}
