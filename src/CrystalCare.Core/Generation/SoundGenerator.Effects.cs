using CrystalCare.Core.Dsp;
using CrystalCare.Core.Frequencies;
using CrystalCare.Core.Noise;
using Math = CrystalCare.Core.Math;

namespace CrystalCare.Core.Generation;

public sealed partial class SoundGenerator
{
    private static void ApplyFade(float[] waveLeft, float[] waveRight,
        int chunkOffset, int chunkSamples, int fadeSamples, int totalSamples)
    {
        // Fade in
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

        // Fade out
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

    private static void ApplyToroidalPanning(float[] waveLeft, float[] waveRight,
        ReadOnlySpan<float> tChunk, int chunkSamples,
        float thetaFreq, float phiFreq, float R, float r,
        Simplex5D simplexPan, Math.RosslerAttractor.Trajectory rossler,
        ExponentialSmoother panSmoother, float driftFreq, float driftAmp,
        ChunkBufferPool pool)
    {
        var panCurve = pool.PanCurve;

        var tScaled1 = pool.PanTScaled1;
        var tScaled2 = pool.PanTScaled2;
        for (int i = 0; i < chunkSamples; i++)
        {
            tScaled1[i] = tChunk[i] * 0.005f;
            tScaled2[i] = tChunk[i] * 0.007f;
        }

        var thetaPerturb = simplexPan.GenerateNoise(tScaled1);
        var phiPerturb = simplexPan.GenerateNoise(tScaled2, 1.0f);

        for (int i = 0; i < chunkSamples; i++)
        {
            float time = tChunk[i];

            // Simplex perturbation
            float tp = 0.10f * thetaPerturb[i];
            float pp = 0.10f * phiPerturb[i];

            // Rössler chaotic perturbation
            tp += 0.08f * Math.RosslerAttractor.Interpolate(rossler.X, rossler.T, time);
            pp += 0.08f * Math.RosslerAttractor.Interpolate(rossler.Y, rossler.T, time);

            float theta = SacredConstants.TWO_PI * thetaFreq * time + tp;
            float phi = SacredConstants.TWO_PI * phiFreq * time + pp;

            panCurve[i] = (R + r * MathF.Cos(phi)) * MathF.Cos(theta) / (R + r);
        }

        // Smooth pan curve (0.002 Hz — ultra-slow, organic drift)
        panSmoother.Process(panCurve);

        // Tanh + drift
        WaveShaper.PanCurveTanh(panCurve, 0.6f, -0.8f, 0.8f);
        for (int i = 0; i < chunkSamples; i++)
            panCurve[i] += driftAmp * MathF.Sin(SacredConstants.TWO_PI * driftFreq * tChunk[i]);

        // Apply stereo panning
        for (int i = 0; i < chunkSamples; i++)
        {
            float panScaled = panCurve[i] * 0.6f;
            waveLeft[i] *= (1.0f - panScaled);
            waveRight[i] *= (1.0f + panScaled);
        }
    }
}
