using CrystalCare.Core.Frequencies;

namespace CrystalCare.Core.SacredLayers;

/// <summary>
/// Silent Solfeggio Grid — 2nd sacred layer.
///
/// Combines the ancient 12-tone Solfeggio scale with Tesla's 3-6-9 vortex
/// mathematics. Fibonacci amplitude pulsing creates organic, living modulation.
///
/// Scale: 0.00165 (sub-perceptual)
/// Fade: 34 seconds (Fibonacci)
/// Breath: 15% at 0.01 Hz (~100 second cycle)
/// </summary>
public sealed class SilentSolfeggioGrid : SacredLayerBase
{
    // Configuration: 34s Fibonacci fade (lighter layer, arrives sooner),
    // 15% breath depth at PHI^0.75 × root, 0.00165 output scale.
    #region Layer Configuration

    protected override float FadeSeconds => 34.0f;
    protected override float BreathCenter => 0.925f;
    protected override float BreathDepth => 0.075f;
    protected override float BreathFreq => SacredConstants.BREATH_PHI_075; // PHI^0.75 × root
    protected override float OutputScale => 0.00165f;

    #endregion

    // Generates the Solfeggio grid: sum of 12 Solfeggio frequencies + 9 Tesla
    // vortex frequencies (111-999 Hz). Combined 60/40 with Fibonacci amplitude pulsing
    // for organic, breathing modulation. PHI-modulated secondary pulsing adds depth.
    #region Signal Generation

    protected override float[] GenerateSignal(ReadOnlySpan<double> tChunk,
        float totalDuration, int n)
    {
        // Solfeggio grid: sum of 12 solfeggio frequencies — double precision phase
        var solfeggioGrid = new float[n];
        var solfeggio = SacredConstants.SOLFEGGIO;
        for (int s = 0; s < solfeggio.Length; s++)
        {
            double freq = solfeggio[s];
            for (int i = 0; i < n; i++)
                solfeggioGrid[i] += (float)System.Math.Sin(SacredConstants.TWO_PI_D * freq * tChunk[i]);
        }

        // Tesla 3-6-9 vortex grid — double precision phase
        var teslaGrid = new float[n];
        var tesla = SacredConstants.TESLA_VORTEX;
        for (int s = 0; s < tesla.Length; s++)
        {
            double freq = tesla[s];
            for (int i = 0; i < n; i++)
                teslaGrid[i] += (float)System.Math.Sin(SacredConstants.TWO_PI_D * freq * tChunk[i]);
        }

        // Fibonacci amplitude (smooth sine modulation) — double precision
        const double fibCycleFreq = 0.004;
        var combined = new float[n];
        for (int i = 0; i < n; i++)
        {
            float fibAmp = 0.95f + 0.05f * (float)System.Math.Sin(SacredConstants.TWO_PI_D * fibCycleFreq * tChunk[i]);
            fibAmp += 0.02f * (float)System.Math.Sin(SacredConstants.TWO_PI_D * fibCycleFreq * SacredConstants.PHI * tChunk[i]);
            fibAmp = global::System.Math.Clamp(fibAmp, 0.88f, 1.0f);

            combined[i] = (0.6f * solfeggioGrid[i] + 0.4f * teslaGrid[i]) * fibAmp;
        }

        return combined;
    }

    #endregion
}
