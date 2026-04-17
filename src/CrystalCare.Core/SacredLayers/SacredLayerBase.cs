using CrystalCare.Core.Frequencies;
using CrystalCare.Core.Noise;

namespace CrystalCare.Core.SacredLayers;

/// <summary>
/// Base class for all sacred healing layers.
/// Implements the template method pattern: duration check, signal generation,
/// fade envelope, breath modulation, and output scaling.
///
/// Two breath application orders handle different layer characteristics:
/// - Group A (default): fade * breath * scale in one pass (Pleroma, Solfeggio, Archon, Crystalline)
/// - Group B (BreathBeforeFade = true): breath first, then fade * scale (Merkaba, Water)
///
/// Each subclass provides its unique signal through GenerateSignal() and configures
/// its properties (fade duration, breath params, output scale). PostProcess() is an
/// optional hook for additional processing (used by Pleroma for cosine nulling).
/// </summary>
public abstract class SacredLayerBase : ISacredLayer
{
    // Configuration properties that each sacred layer overrides to define its character.
    // MinDuration: minimum session length to activate (60s for most, 30s for Archon).
    // FadeSeconds: Fibonacci-timed fade duration (34s or 55s).
    // BreathCenter/Depth/Freq: sine breath modulation parameters from the PHI ladder.
    // OutputScale: final amplitude scaling (sub-perceptual, typically 0.0003-0.00165).
    // BreathBeforeFade: false = Group A (most layers), true = Group B (Merkaba, Water).
    // Simplex: thread-local 5D simplex noise generator for organic variation.
    #region Abstract and Virtual Properties

    /// <summary>Minimum session duration to activate this layer (seconds).</summary>
    protected virtual float MinDuration => 60f;

    /// <summary>Fibonacci-timed fade envelope duration (34 or 55 seconds).</summary>
    protected abstract float FadeSeconds { get; }

    /// <summary>Breath modulation center value (0.92-0.98 range).</summary>
    protected abstract float BreathCenter { get; }

    /// <summary>Breath modulation depth (how much the breath varies).</summary>
    protected abstract float BreathDepth { get; }

    /// <summary>Breath frequency from the unified PHI ladder (Hz).</summary>
    protected abstract float BreathFreq { get; }

    /// <summary>Final amplitude scaling — keeps the layer sub-perceptual.</summary>
    protected abstract float OutputScale { get; }

    /// <summary>If true, breath is applied before fade (Group B: Merkaba, Water).</summary>
    protected virtual bool BreathBeforeFade => false;

    /// <summary>Thread-local simplex noise generator for organic variation.</summary>
    protected ThreadLocal<Simplex5D> Simplex { get; } =
        new(() => new Simplex5D(Random.Shared.Next(100)));

    #endregion

    // The template method that all sacred layers share.
    // 1. Check if the session is long enough to activate this layer.
    // 2. Call the subclass's GenerateSignal() for the unique computation.
    // 3. Apply Perlin smoother step fade envelope (Fibonacci-timed).
    // 4. Apply breath modulation (sine wave from PHI ladder frequency).
    // 5. Apply output scale (sub-perceptual amplitude).
    // 6. Call PostProcess() for any additional layer-specific processing.
    #region Template Method (ComputeChunk)

    /// <summary>
    /// Compute the sacred layer's audio for one chunk.
    /// Called by SoundGenerator for each 3-second chunk during sessions >60s.
    /// </summary>
    public float[] ComputeChunk(ReadOnlySpan<float> tChunk, float totalDuration)
    {
        // Return silence if the session is too short for this layer
        if (totalDuration < MinDuration)
            return new float[tChunk.Length];

        int n = tChunk.Length;

        // Call the subclass to generate its unique signal
        var signal = GenerateSignal(tChunk, totalDuration, n);

        // Compute Perlin smoother step fade envelope (6t^5 - 15t^4 + 10t^3)
        var fade = SacredFadeEnvelope.Compute(tChunk, totalDuration, fadeSeconds: FadeSeconds);

        if (BreathBeforeFade)
        {
            // Group B (Merkaba, Water): apply breath modulation first, then fade + scale.
            // This order gives the breath more presence before the fade attenuates it.
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
            // Group A (Pleroma, Solfeggio, Archon, Crystalline): fade * breath * scale together.
            // This is the standard path — fade, breath, and scale applied in one pass.
            for (int i = 0; i < n; i++)
            {
                float breath = BreathCenter + BreathDepth *
                    MathF.Sin(SacredConstants.TWO_PI * BreathFreq * tChunk[i]);
                signal[i] *= fade[i] * breath * OutputScale;
            }
        }

        // Optional post-processing hook (Pleroma uses this for cosine nulling)
        PostProcess(signal, tChunk, n);

        return signal;
    }

    #endregion

    // Override points for subclasses.
    // GenerateSignal(): the unique computation for each sacred layer (required).
    // PostProcess(): optional additional processing after fade/breath/scale.
    #region Override Points

    /// <summary>
    /// Generate the unique signal for this sacred layer.
    /// Called by the template method before fade/breath/scale are applied.
    /// </summary>
    protected abstract float[] GenerateSignal(ReadOnlySpan<float> tChunk,
        float totalDuration, int n);

    /// <summary>
    /// Optional post-processing after fade/breath/scale.
    /// Override in subclasses that need additional processing
    /// (e.g., Pleroma's cosine nulling scalar imprint).
    /// </summary>
    protected virtual void PostProcess(float[] signal, ReadOnlySpan<float> tChunk, int n) { }

    #endregion
}
