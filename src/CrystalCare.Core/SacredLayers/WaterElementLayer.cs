using CrystalCare.Core.Frequencies;

namespace CrystalCare.Core.SacredLayers;

/// <summary>
/// Water Element Layer — 6th sacred layer.
///
/// 7-source hexagonal ripple field with lemniscate (figure-8) observer path.
/// Spatial wave interference (not sine stacking) — distance-based decay creates
/// organic movement as the observer traces the lemniscate.
///
/// Scale: 0.0012
/// Fade: 55 seconds (Fibonacci)
/// Breath: 11% at 0.007 Hz (~143 second cycle)
/// </summary>
public sealed class WaterElementLayer : SacredLayerBase
{
    // Configuration: 55s Fibonacci fade, 11% breath depth at Schumann/1000 root
    // (Earth's deepest breath), 0.0012 output scale. BreathBeforeFade = true (Group B).
    #region Layer Configuration

    protected override float FadeSeconds => 55.0f;
    protected override float BreathCenter => 0.945f;
    protected override float BreathDepth => 0.055f;
    protected override float BreathFreq => SacredConstants.BREATH_ROOT; // Schumann/1000 — Earth's breath
    protected override float OutputScale => 0.0012f;
    protected override bool BreathBeforeFade => true;

    #endregion

    // Generates a 7-source hexagonal ripple field with a lemniscate (figure-8)
    // observer path. Uses spatial wave interference (not sine stacking) —
    // distance-based exponential decay creates organic movement as the observer
    // traces the lemniscate. Fibonacci-derived decay rates per source.
    // Tidal simplex modulation adds slow breathing amplitude.
    // Standing wave resonance at Schumann × Schumann*PHI adds harmonic depth.
    #region Signal Generation

    protected override float[] GenerateSignal(ReadOnlySpan<double> tChunk,
        float totalDuration, int n)
    {
        var simplex = Simplex.Value!;

        // Simplex perturbation for organic observer drift — scaled time stays small
        var tScaled = new float[n];
        for (int i = 0; i < n; i++)
            tScaled[i] = (float)(tChunk[i] * 0.005);
        var thetaPerturb = simplex.GenerateNoise(tScaled, 0.4f);
        for (int i = 0; i < n; i++)
            thetaPerturb[i] *= 0.15f;

        var sourceFreqs = SacredConstants.WATER_SOURCE_FREQS;
        var sourceDecays = SacredConstants.WATER_SOURCE_DECAYS;
        var hexPhases = SacredConstants.WATER_HEX_PHASES;
        var positions = SacredConstants.WATER_SOURCE_POSITIONS;

        // Accumulate wave interference from 7 sources — double precision phase
        var result = new float[n];

        for (int s = 0; s < 7; s++)
        {
            float srcX = positions[s, 0];
            float srcY = positions[s, 1];
            double sourceFreq = sourceFreqs[s];

            for (int i = 0; i < n; i++)
            {
                // Lemniscate path (figure-8) — slow rate at 0.005 Hz, compute in double
                double theta = SacredConstants.TWO_PI_D * 0.005 * tChunk[i] + thetaPerturb[i];
                float sinT = (float)System.Math.Sin(theta);
                float cosT = (float)System.Math.Cos(theta);
                float denom = 1.0f + sinT * sinT;
                float obsX = 0.7f * cosT / denom;
                float obsY = 0.7f * sinT * cosT / denom;

                // Distance from observer to source
                float dx = obsX - srcX;
                float dy = obsY - srcY;
                float dist = MathF.Sqrt(dx * dx + dy * dy);

                // Spatial envelope: exponential decay with distance
                float envelope = MathF.Exp(-sourceDecays[s] * dist * 10.0f);

                // Wave from this source — double precision phase
                float wave = (float)System.Math.Sin(SacredConstants.TWO_PI_D * sourceFreq * tChunk[i] + hexPhases[s]);
                result[i] += wave * envelope;
            }
        }

        // Normalize by source count
        for (int i = 0; i < n; i++)
            result[i] *= 1.0f / 7.0f;

        // Tidal modulation: slow simplex-driven amplitude (~200s period)
        for (int i = 0; i < n; i++)
            tScaled[i] = (float)(tChunk[i] * 0.0005);
        var tidal = simplex.GenerateNoise(tScaled, 0.9f);
        for (int i = 0; i < n; i++)
            result[i] *= 0.85f + 0.15f * tidal[i];

        // Standing wave resonance: Schumann x Schumann*PHI — double precision phase
        const double schumannPhi = SacredConstants.SCHUMANN * SacredConstants.PHI;
        for (int i = 0; i < n; i++)
        {
            float standing = 0.1f *
                (float)System.Math.Sin(SacredConstants.TWO_PI_D * SacredConstants.SCHUMANN * tChunk[i]) *
                (float)System.Math.Cos(SacredConstants.TWO_PI_D * schumannPhi * tChunk[i]);
            result[i] += standing;
        }

        return result;
    }

    #endregion
}
