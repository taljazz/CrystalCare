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
    // Per-chunk fractal variation using a look-ahead read on a single noise field.
    // Simpler than dual-simplex but still produces rich organic variation.
    // Used by CrystallineResonanceLayer for crystal micro-variation.
    #region Streaming Chunk Computation

    public static float[] ComputeChunk(ReadOnlySpan<double> tChunk, float baseFreq,
        Simplex5D simplex)
    {
        int n = tChunk.Length;
        if (n == 0) return [];

        // The "octave" trick reads the same noise field at two positions: the
        // sample's own time, and a point one-eighth of the chunk LATER. The
        // original port implemented the look-ahead as a wrap-around array index
        // ((i + shift) % length), which near the end of a chunk jumped BACK to
        // the chunk's start — a ~2.6-second time discontinuity in the
        // micro-variation at every chunk seam. Sub-perceptual at the depth this
        // feeds (0.1% modulation on crystal voices), but a seam is a seam:
        // she breathes continuously.
        //
        // Fix: extend the time axis past the chunk's end and read the noise
        // field's true continuation. Samples whose look-ahead lands inside the
        // chunk produce bit-identical values to the old code; only the final
        // one-eighth (the previously wrapped region) changes — to the value
        // the design always intended.
        int shift = n / 8;

        // Sample spacing for the extension — tChunk is uniformly spaced by the
        // generator (1/sampleRate); derive it locally so this method stays
        // decoupled from the caller. Single-sample chunks can't extend (no
        // spacing known) and have shift = 0 anyway.
        double dt = n > 1 ? tChunk[1] - tChunk[0] : 0.0;

        // Extended scaled-time array: the chunk's samples followed by `shift`
        // samples continuing past the end at the same spacing.
        var tScaled = new float[n + shift];
        for (int i = 0; i < n; i++)
            tScaled[i] = (float)(tChunk[i] * 0.02);
        for (int j = 0; j < shift; j++)
            tScaled[n + j] = (float)((tChunk[n - 1] + (j + 1) * dt) * 0.02);

        var baseNoise = simplex.GenerateNoise(tScaled, baseFreq * 0.01f);

        var variation = new float[n];
        for (int i = 0; i < n; i++)
        {
            float v = baseNoise[i];
            float shifted = baseNoise[i + shift];  // true continuation — no wrap
            variation[i] = (v * 0.5f + shifted * 0.3f + v * v * 0.2f) * 12.0f;
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
