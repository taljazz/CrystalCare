using CrystalCare.Core.Frequencies;
using CrystalCare.Core.Noise;

namespace CrystalCare.Core.SacredLayers;

/// <summary>
/// Base class for all sacred healing layers.
/// Implements the common pattern: duration check → generate signal → fade + breath + scale.
///
/// Two breath application orders:
/// - Group A (default): fade * breath * scale in one pass (Pleroma, Solfeggio, Archon, Crystalline)
/// - Group B (BreathBeforeFade = true): breath first, then fade * scale (Merkaba, Water)
///
/// Override PostProcess() for additional processing after fade/breath/scale (e.g., Pleroma cosine nulling).
/// </summary>
public abstract class SacredLayerBase : ISacredLayer
{
    protected virtual float MinDuration => 60f;
    protected abstract float FadeSeconds { get; }
    protected abstract float BreathCenter { get; }
    protected abstract float BreathDepth { get; }
    protected abstract float BreathFreq { get; }
    protected abstract float OutputScale { get; }
    protected virtual bool BreathBeforeFade => false;

    protected ThreadLocal<Simplex5D> Simplex { get; } =
        new(() => new Simplex5D(Random.Shared.Next(100)));

    public float[] ComputeChunk(ReadOnlySpan<float> tChunk, float totalDuration)
    {
        if (totalDuration < MinDuration)
            return new float[tChunk.Length];

        int n = tChunk.Length;
        var signal = GenerateSignal(tChunk, totalDuration, n);

        var fade = SacredFadeEnvelope.Compute(tChunk, totalDuration, fadeSeconds: FadeSeconds);

        if (BreathBeforeFade)
        {
            // Group B: breath first, then fade * scale
            for (int i = 0; i < n; i++)
            {
                float breath = BreathCenter + BreathDepth *
                    MathF.Sin(SacredConstants.TWO_PI * BreathFreq * tChunk[i]);
                signal[i] *= breath;
            }
            for (int i = 0; i < n; i++)
                signal[i] *= fade[i] * OutputScale;
        }
        else
        {
            // Group A: fade * breath * scale together
            for (int i = 0; i < n; i++)
            {
                float breath = BreathCenter + BreathDepth *
                    MathF.Sin(SacredConstants.TWO_PI * BreathFreq * tChunk[i]);
                signal[i] *= fade[i] * breath * OutputScale;
            }
        }

        PostProcess(signal, tChunk, n);

        return signal;
    }

    protected abstract float[] GenerateSignal(ReadOnlySpan<float> tChunk,
        float totalDuration, int n);

    protected virtual void PostProcess(float[] signal, ReadOnlySpan<float> tChunk, int n) { }
}
