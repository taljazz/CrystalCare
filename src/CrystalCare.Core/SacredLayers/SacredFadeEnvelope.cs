using CrystalCare.Core.Frequencies;

namespace CrystalCare.Core.SacredLayers;

/// <summary>
/// Ken Perlin's smoother step fade envelope for sacred layers.
/// Ultra-smooth transitions: 6t^5 - 15t^4 + 10t^3
/// Port of AudioProcessor._sacred_fade_envelope_chunk() from SoundGenerator.py.
/// </summary>
public static class SacredFadeEnvelope
{
    /// <summary>
    /// Compute fade envelope for a chunk using absolute time positions.
    /// </summary>
    public static float[] Compute(ReadOnlySpan<float> tChunk, float totalDuration,
        float fadeSeconds = 45.0f)
    {
        var envelope = new float[tChunk.Length];

        if (totalDuration < fadeSeconds * 2)
        {
            // Short session: raised cosine
            for (int i = 0; i < tChunk.Length; i++)
                envelope[i] = 0.5f - 0.5f * MathF.Cos(SacredConstants.TWO_PI * tChunk[i] / totalDuration);
            return envelope;
        }

        float fadeOutThreshold = totalDuration - fadeSeconds;

        for (int i = 0; i < tChunk.Length; i++)
        {
            float t = tChunk[i];

            if (t < fadeSeconds)
            {
                // Fade in: Perlin smoother step
                float x = t / fadeSeconds;
                envelope[i] = x * x * x * (x * (x * 6.0f - 15.0f) + 10.0f);
            }
            else if (t > fadeOutThreshold)
            {
                // Fade out: Perlin smoother step (inverted)
                float x = global::System.Math.Clamp((totalDuration - t) / fadeSeconds, 0f, 1f);
                envelope[i] = x * x * x * (x * (x * 6.0f - 15.0f) + 10.0f);
            }
            else
            {
                // Plateau
                envelope[i] = 1.0f;
            }
        }

        return envelope;
    }
}
