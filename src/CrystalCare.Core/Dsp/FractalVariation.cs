using CrystalCare.Core.Frequencies;
using CrystalCare.Core.Noise;

namespace CrystalCare.Core.Dsp;

/// <summary>
/// Fractal frequency variation using Simplex noise and LFO modulation.
/// Creates organic, ever-changing pitch micro-variations.
/// Time input is double precision; simplex noise receives scaled-down float values
/// (simplex doesn't need nanosecond precision; scaling keeps values small).
/// </summary>
public static class FractalVariation
{
    // Full-signal fractal variation using simplex noise with LFO modulation.
    #region Batch Computation

    /// <summary>
    /// Compute fractal frequency variation for a time array.
    /// Uses simplex noise with octave-like layering for rich variation.
    /// </summary>
    public static float[] Compute(ReadOnlySpan<double> t, float baseFreq,
        Simplex5D simplex, CancellationToken ct = default)
    {
        if (ct.IsCancellationRequested)
            return new float[t.Length];

        // LFO for modulation
        var lfo = MicrotonalLfo.Generate(t, 0.01f);

        // Simplex noise scaled by time — cast to float after scaling (small values OK)
        var tScaled = new float[t.Length];
        for (int i = 0; i < t.Length; i++)
            tScaled[i] = (float)(t[i] * 0.02);

        // Use baseFreq * 0.01 as y-offset to avoid degenerate seeds
        var baseNoise = simplex.GenerateNoise(tScaled, baseFreq * 0.01f);

        // Create "octaves" by scaling and phase-shifting
        var variation = new float[t.Length];
        int shift = t.Length / 8;

        for (int i = 0; i < t.Length; i++)
        {
            float n = baseNoise[i];
            int shiftedIdx = (i + shift) % t.Length;
            float shifted = baseNoise[shiftedIdx];

            variation[i] = n * 0.5f + shifted * 0.3f + n * n * 0.2f;
            variation[i] *= 12.0f;
            variation[i] *= (1.0f + 0.1f * lfo[i]);
        }

        return variation;
    }

    #endregion

    // Per-chunk fractal variation using shifted index trick on single noise array.
    // Simpler than dual-simplex but still produces rich organic variation.
    // Used by CrystallineResonanceLayer for crystal micro-variation.
    #region Streaming Chunk Computation

    public static float[] ComputeChunk(ReadOnlySpan<double> tChunk, float baseFreq,
        Simplex5D simplex)
    {
        // Simplex input scaled to small values — float is fine after scaling
        var tScaled = new float[tChunk.Length];
        for (int i = 0; i < tChunk.Length; i++)
            tScaled[i] = (float)(tChunk[i] * 0.02);

        var baseNoise = simplex.GenerateNoise(tScaled, baseFreq * 0.01f);

        var variation = new float[tChunk.Length];
        int shift = tChunk.Length / 8;

        for (int i = 0; i < tChunk.Length; i++)
        {
            float n = baseNoise[i];
            int shiftedIdx = (i + shift) % tChunk.Length;
            float shifted = baseNoise[shiftedIdx];

            variation[i] = (n * 0.5f + shifted * 0.3f + n * n * 0.2f) * 12.0f;
        }

        return variation;
    }

    #endregion

    // Dual-simplex fractal variation — the version used by the main pipeline.
    // Two independent simplex noise calls provide richer variation than single-noise.
    #region Dual Simplex Computation (Pipeline)

    /// <summary>
    /// Compute fractal variation using dual simplex noise layers.
    /// Matches the inline pipeline logic exactly — two independent noise calls
    /// for richer variation than the single-noise shifted approach.
    /// </summary>
    public static float[] ComputeChunkDual(ReadOnlySpan<double> tChunk, float baseFreq,
        Simplex5D simplex)
    {
        // Simplex input scaled to small values — float precision is sufficient
        var tScaled = new float[tChunk.Length];
        for (int i = 0; i < tChunk.Length; i++)
            tScaled[i] = (float)(tChunk[i] * 0.02);

        var baseNoise = simplex.GenerateNoise(tScaled, baseFreq * 0.01f);
        var variation2 = simplex.GenerateNoise(tScaled, baseFreq * 0.01f, 1.0f);

        var result = new float[tChunk.Length];
        for (int i = 0; i < tChunk.Length; i++)
            result[i] = (baseNoise[i] * 0.5f + variation2[i] * 0.3f +
                         baseNoise[i] * baseNoise[i] * 0.2f) * 12.0f;
        return result;
    }

    #endregion
}
