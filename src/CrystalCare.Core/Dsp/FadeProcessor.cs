using CrystalCare.Core.Frequencies;

namespace CrystalCare.Core.Dsp;

/// <summary>
/// Fade in/out processing using exponential curves and Perlin smoother step.
/// Port of AudioProcessor.fade_in_out() and _sacred_fade_envelope().
/// </summary>
public static class FadeProcessor
{
    /// <summary>
    /// Apply fade in/out to a signal in-place. Uses t^1.5 curve.
    /// </summary>
    public static void FadeInOut(Span<float> signal, int fadeSamples)
    {
        if (2 * fadeSamples > signal.Length)
            fadeSamples = signal.Length / 2;

        // Fade in: t^1.5
        for (int i = 0; i < fadeSamples; i++)
        {
            float t = (float)i / fadeSamples;
            signal[i] *= MathF.Pow(t, 1.5f);
        }

        // Fade out: (1-t)^1.5
        int fadeOutStart = signal.Length - fadeSamples;
        for (int i = 0; i < fadeSamples; i++)
        {
            float t = (float)i / fadeSamples;
            signal[fadeOutStart + i] *= MathF.Pow(1f - t, 1.5f);
        }
    }

    /// <summary>
    /// Ken Perlin's smoother step: 6t^5 - 15t^4 + 10t^3
    /// Used for ultra-smooth sacred layer fades.
    /// </summary>
    public static float SmootherStep(float t)
    {
        t = global::System.Math.Clamp(t, 0f, 1f);
        return t * t * t * (t * (t * 6f - 15f) + 10f);
    }

    /// <summary>
    /// Create a sacred fade envelope for a full signal.
    /// Fade in over fadeSeconds, plateau, fade out over fadeSeconds.
    /// Uses Perlin smoother step for ultra-gentle transitions.
    /// </summary>
    public static float[] CreateSacredFadeEnvelope(int totalSamples, float totalDuration,
        float fadeSeconds, int sampleRate)
    {
        var envelope = new float[totalSamples];
        int fadeSamples = (int)(fadeSeconds * sampleRate);

        if (totalSamples > fadeSamples * 2)
        {
            // Fade in
            for (int i = 0; i < fadeSamples && i < totalSamples; i++)
            {
                float t = (float)i / fadeSamples;
                envelope[i] = SmootherStep(t);
            }

            // Plateau
            for (int i = fadeSamples; i < totalSamples - fadeSamples; i++)
                envelope[i] = 1.0f;

            // Fade out
            for (int i = 0; i < fadeSamples && (totalSamples - fadeSamples + i) < totalSamples; i++)
            {
                float t = 1.0f - (float)i / fadeSamples;
                envelope[totalSamples - fadeSamples + i] = SmootherStep(t);
            }
        }
        else
        {
            // Short session: raised cosine
            for (int i = 0; i < totalSamples; i++)
            {
                float t = (float)i / totalSamples;
                envelope[i] = 0.5f - 0.5f * MathF.Cos(SacredConstants.TWO_PI * t);
            }
        }

        return envelope;
    }

    /// <summary>
    /// Compute sacred fade for a chunk at a given position in the total duration.
    /// Used by the streaming pipeline.
    /// </summary>
    public static void ApplySacredFadeChunk(Span<float> chunk, int chunkOffset,
        float totalDuration, float fadeSeconds, int sampleRate)
    {
        int fadeSamples = (int)(fadeSeconds * sampleRate);
        int totalSamples = (int)(totalDuration * sampleRate);

        for (int i = 0; i < chunk.Length; i++)
        {
            int globalIdx = chunkOffset + i;
            float fade = 1.0f;

            if (totalSamples > fadeSamples * 2)
            {
                if (globalIdx < fadeSamples)
                    fade = SmootherStep((float)globalIdx / fadeSamples);
                else if (globalIdx >= totalSamples - fadeSamples)
                    fade = SmootherStep((float)(totalSamples - globalIdx) / fadeSamples);
            }
            else
            {
                float t = (float)globalIdx / totalSamples;
                fade = 0.5f - 0.5f * MathF.Cos(SacredConstants.TWO_PI * t);
            }

            chunk[i] *= fade;
        }
    }
}
