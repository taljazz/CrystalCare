using CrystalCare.Core.Frequencies;

namespace CrystalCare.Core.Noise;

/// <summary>
/// 5D Simplex noise generator using FastNoiseLite.
/// Two OpenSimplex2 instances blended in a 3-layer pattern (0.5/0.3/0.2 weights)
/// to simulate higher-dimensional noise from 3D inputs.
///
/// Exact port of simplex5d.cpp — same blend formula, same frequencies, same seed offsets.
/// </summary>
public sealed class Simplex5D
{
    private readonly FastNoiseLite _noise1;
    private readonly FastNoiseLite _noise2;

    public Simplex5D(int seed)
    {
        _noise1 = new FastNoiseLite(seed);
        _noise1.SetNoiseType(FastNoiseLite.NoiseType.OpenSimplex2);
        _noise1.SetFrequency(0.02f);

        _noise2 = new FastNoiseLite(seed + 1);
        _noise2.SetNoiseType(FastNoiseLite.NoiseType.OpenSimplex2);
        _noise2.SetFrequency(0.015f);
    }

    /// <summary>
    /// Generate noise for a time array with 4 spatial offsets.
    /// Matches the C++ generate_noise() exactly:
    ///   result = 0.5 * noise1(x+t, y+t, z+t)
    ///          + 0.3 * noise2(x+w, y+w, z+w)
    ///          + 0.2 * noise1(x+t*PHI, y+t*PHI, z+w)
    /// </summary>
    public float[] GenerateNoise(ReadOnlySpan<float> t, float xOffset = 0f,
        float yOffset = 0f, float zOffset = 0f, float wOffset = 0f)
    {
        var result = new float[t.Length];

        // Layer 2: orthogonal variation (w parameter, independent of time).
        // This layer does NOT depend on t — it's the same value for every sample
        // of the call, so it's evaluated ONCE here instead of once per sample.
        // (Same arithmetic, same blend order, bit-identical output — the per-sample
        // evaluation in the original port wasted a third of every noise call.)
        float n2Weighted = 0.3f * _noise2.GetNoise(
            xOffset + wOffset, yOffset + wOffset, zOffset + wOffset);

        for (int i = 0; i < t.Length; i++)
        {
            float tVal = t[i];

            // Layer 1: temporal coherence (same offset added to all axes)
            float n1 = _noise1.GetNoise(
                xOffset + tVal, yOffset + tVal, zOffset + tVal);

            // Layer 3: PHI-modulated time + w coupling
            float n3 = _noise1.GetNoise(
                xOffset + tVal * SacredConstants.PHI,
                yOffset + tVal * SacredConstants.PHI,
                zOffset + wOffset);

            result[i] = 0.5f * n1 + n2Weighted + 0.2f * n3;
        }

        return result;
    }
}
