using CrystalCare.Core.Frequencies;
using CrystalCare.Core.Noise;

namespace CrystalCare.Core.Dsp;

/// <summary>
/// Fractal frequency variation using Simplex noise and LFO modulation.
/// Creates organic, ever-changing pitch micro-variations.
/// Port of AudioProcessor.fractal_frequency_variation() from SoundGenerator.py.
/// </summary>
public static class FractalVariation
{
    /// <summary>
    /// Compute fractal frequency variation for a time array.
    /// Uses simplex noise with octave-like layering for rich variation.
    /// </summary>
    public static float[] Compute(ReadOnlySpan<float> t, float baseFreq,
        Simplex5D simplex, CancellationToken ct = default)
    {
        if (ct.IsCancellationRequested)
            return new float[t.Length];

        // LFO for modulation
        var lfo = MicrotonalLfo.Generate(t, 0.01f);

        // Simplex noise scaled by time
        var tScaled = new float[t.Length];
        for (int i = 0; i < t.Length; i++)
            tScaled[i] = t[i] * 0.02f;

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

    /// <summary>
    /// Compute fractal variation for a streaming chunk (stateless).
    /// </summary>
    public static float[] ComputeChunk(ReadOnlySpan<float> tChunk, float baseFreq,
        Simplex5D simplex)
    {
        var tScaled = new float[tChunk.Length];
        for (int i = 0; i < tChunk.Length; i++)
            tScaled[i] = tChunk[i] * 0.02f;

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

    /// <summary>
    /// Compute fractal variation using dual simplex noise layers.
    /// Matches the inline pipeline logic exactly — two independent noise calls
    /// for richer variation than the single-noise shifted approach.
    /// </summary>
    public static float[] ComputeChunkDual(ReadOnlySpan<float> tChunk, float baseFreq,
        Simplex5D simplex)
    {
        var tScaled = new float[tChunk.Length];
        for (int i = 0; i < tChunk.Length; i++)
            tScaled[i] = tChunk[i] * 0.02f;

        var baseNoise = simplex.GenerateNoise(tScaled, baseFreq * 0.01f);
        var variation2 = simplex.GenerateNoise(tScaled, baseFreq * 0.01f, 1.0f);

        var result = new float[tChunk.Length];
        for (int i = 0; i < tChunk.Length; i++)
            result[i] = (baseNoise[i] * 0.5f + variation2[i] * 0.3f +
                         baseNoise[i] * baseNoise[i] * 0.2f) * 12.0f;
        return result;
    }
}
