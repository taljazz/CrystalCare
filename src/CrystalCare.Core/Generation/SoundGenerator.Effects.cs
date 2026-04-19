using CrystalCare.Core.Dsp;
using CrystalCare.Core.Frequencies;
using CrystalCare.Core.Noise;
using Math = CrystalCare.Core.Math;

namespace CrystalCare.Core.Generation;

/// <summary>
/// SoundGenerator effects — master fade in/out and toroidal panning.
/// Split from the main file for readability.
/// </summary>
public sealed partial class SoundGenerator
{
    // Applies a t^1.5 power curve fade at the start and end of the session.
    // Fade duration is a Fibonacci pair (21 or 34 seconds, chosen randomly).
    // Works across chunk boundaries using global sample offset.
    #region Fade In/Out

    /// <summary>
    /// Apply master fade in/out across chunk boundaries.
    /// Uses t^1.5 power curve for smooth, organic transitions.
    /// </summary>
    private static void ApplyFade(float[] waveLeft, float[] waveRight,
        int chunkOffset, int chunkSamples, int fadeSamples, int totalSamples)
    {
        // Fade in: apply t^1.5 curve to samples within the fade-in region
        if (chunkOffset < fadeSamples)
        {
            int fadeEnd = global::System.Math.Min(chunkSamples, fadeSamples - chunkOffset);
            for (int i = 0; i < fadeEnd; i++)
            {
                float t = (float)(chunkOffset + i) / fadeSamples;
                float fade = MathF.Pow(t, 1.5f);
                waveLeft[i] *= fade;
                waveRight[i] *= fade;
            }
        }

        // Fade out: apply inverted t^1.5 curve to samples within the fade-out region
        int fadeOutStart = totalSamples - fadeSamples;
        if (chunkOffset + chunkSamples > fadeOutStart)
        {
            int startIdx = global::System.Math.Max(0, fadeOutStart - chunkOffset);
            for (int i = startIdx; i < chunkSamples; i++)
            {
                float remaining = (float)(totalSamples - (chunkOffset + i)) / fadeSamples;
                float fade = global::System.Math.Clamp(MathF.Pow(remaining, 1.5f), 0f, 1f);
                waveLeft[i] *= fade;
                waveRight[i] *= fade;
            }
        }
    }

    #endregion

    // Applies toroidal (donut-shaped) panning to create 3D-like spatial movement.
    // The pan path traces a torus with PHI-derived radii (0.618 + 0.382 = 1.0).
    // Simplex noise adds organic perturbation to the torus angles.
    // Rössler attractor adds genuine mathematical chaos (bounded, non-repeating).
    // The result is smoothed through a 0.002 Hz exponential filter for ultra-slow drift,
    // shaped with tanh for gentle bounds, and mixed with a slow sine drift.
    #region Toroidal Panning

    /// <summary>
    /// Apply toroidal panning with Rössler chaos and simplex perturbation.
    /// Uses pooled buffers for panCurve, tScaled1, and tScaled2 to avoid allocation.
    /// </summary>
    private static void ApplyToroidalPanning(float[] waveLeft, float[] waveRight,
        ReadOnlySpan<double> tChunk, int chunkSamples,
        float thetaFreq, float phiFreq, float R, float r,
        Simplex5D simplexPan, Math.RosslerAttractor.Trajectory rossler,
        ExponentialSmoother panSmoother, float driftFreq, float driftAmp,
        ChunkBufferPool pool)
    {
        // Use pooled buffers instead of allocating new arrays each chunk
        var panCurve = pool.PanCurve;
        var tScaled1 = pool.PanTScaled1;
        var tScaled2 = pool.PanTScaled2;

        // Scale time arrays for simplex noise at two different rates.
        // Scaled values stay small enough for float32 precision.
        for (int i = 0; i < chunkSamples; i++)
        {
            tScaled1[i] = (float)(tChunk[i] * 0.005);  // Slower simplex variation
            tScaled2[i] = (float)(tChunk[i] * 0.007);  // Slightly faster, orthogonal variation
        }

        // Generate simplex noise for organic angular perturbation
        var thetaPerturb = simplexPan.GenerateNoise(tScaled1);
        var phiPerturb = simplexPan.GenerateNoise(tScaled2, 1.0f);

        // Compute the raw toroidal pan curve with chaos and simplex perturbation.
        // Double precision phase for long-session stability.
        for (int i = 0; i < chunkSamples; i++)
        {
            double time = tChunk[i];

            // Simplex perturbation: ±0.10 radians of organic angular drift
            double tp = 0.10f * thetaPerturb[i];
            double pp = 0.10f * phiPerturb[i];

            // Rössler chaotic perturbation: ±0.08 radians of bounded chaos (Interpolate takes float)
            tp += 0.08f * Math.RosslerAttractor.Interpolate(rossler.X, rossler.T, (float)time);
            pp += 0.08f * Math.RosslerAttractor.Interpolate(rossler.Y, rossler.T, (float)time);

            // Compute torus position in double: theta (horizontal) and phi (depth)
            double theta = SacredConstants.TWO_PI_D * thetaFreq * time + tp;
            double phi = SacredConstants.TWO_PI_D * phiFreq * time + pp;

            // Map torus position to mono pan value: x-projection of torus surface point
            panCurve[i] = (float)((R + r * System.Math.Cos(phi)) * System.Math.Cos(theta) / (R + r));
        }

        // Ultra-slow smoothing (0.002 Hz) — only the slowest drift survives
        panSmoother.Process(panCurve);

        // Tanh soft-clipping to keep pan within [-0.8, 0.8] with gentle saturation
        WaveShaper.PanCurveTanh(panCurve, 0.6f, -0.8f, 0.8f);

        // Add slow sine drift for additional gentle stereo movement (double phase)
        for (int i = 0; i < chunkSamples; i++)
            panCurve[i] += driftAmp * (float)System.Math.Sin(SacredConstants.TWO_PI_D * driftFreq * tChunk[i]);

        // Apply the computed pan curve to the stereo signal
        for (int i = 0; i < chunkSamples; i++)
        {
            float panScaled = panCurve[i] * 0.6f;
            waveLeft[i] *= (1.0f - panScaled);   // Left gets louder as pan goes left
            waveRight[i] *= (1.0f + panScaled);  // Right gets louder as pan goes right
        }
    }

    #endregion
}
