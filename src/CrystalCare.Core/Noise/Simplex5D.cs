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

        for (int i = 0; i < t.Length; i++)
        {
            float tVal = t[i];

            // Layer 1: temporal coherence (same offset added to all axes)
            float n1 = _noise1.GetNoise(
                xOffset + tVal, yOffset + tVal, zOffset + tVal);

            // Layer 2: orthogonal variation (w parameter, independent of time)
            float n2 = _noise2.GetNoise(
                xOffset + wOffset, yOffset + wOffset, zOffset + wOffset);

            // Layer 3: PHI-modulated time + w coupling
            float n3 = _noise1.GetNoise(
                xOffset + tVal * SacredConstants.PHI,
                yOffset + tVal * SacredConstants.PHI,
                zOffset + wOffset);

            result[i] = 0.5f * n1 + 0.3f * n2 + 0.2f * n3;
        }

        return result;
    }

    /// <summary>
    /// Generate multiple noise arrays in one call (reduces overhead).
    /// Each parameter tuple is (y, z, w, v) — matching the C++ batch_generate.
    /// </summary>
    public float[][] BatchGenerate(ReadOnlySpan<float> t, (float y, float z, float w, float v)[] paramSets)
    {
        var results = new float[paramSets.Length][];

        for (int j = 0; j < paramSets.Length; j++)
        {
            var p = paramSets[j];
            // In C++ code, params are mapped: y->x_off, z->y_off, w->z_off, v->w_off
            results[j] = GenerateNoise(t, p.y, p.z, p.w, p.v);
        }

        return results;
    }
}
