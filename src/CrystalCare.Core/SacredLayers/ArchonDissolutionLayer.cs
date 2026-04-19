using CrystalCare.Core.Frequencies;

namespace CrystalCare.Core.SacredLayers;

/// <summary>
/// Archon Dissolution Layer — 3rd sacred layer.
///
/// Targeted mercy frequencies for each of the 7 planetary Archons.
/// Uses the AEG pattern: Acknowledge, Elevate, Ground.
///
/// Scale: 0.000225 (sub-perceptual)
/// Fade: 55 seconds (Fibonacci)
/// Breath: 12% at 0.008 Hz (~125 second cycle)
/// </summary>
public sealed class ArchonDissolutionLayer : SacredLayerBase
{
    // Configuration: MinDuration 30s (internal guard), 55s Fibonacci fade,
    // 12% breath depth at PHI^0.25 × root, 0.000225 output scale.
    #region Layer Configuration

    protected override float MinDuration => 30f;
    protected override float FadeSeconds => 55.0f;
    protected override float BreathCenter => 0.94f;
    protected override float BreathDepth => 0.06f;
    protected override float BreathFreq => SacredConstants.BREATH_PHI_025; // PHI^0.25 × root
    protected override float OutputScale => 0.000225f;

    #endregion

    // Generates targeted mercy frequencies for each of the 7 planetary Archons.
    // Each Archon gets three waves: Acknowledge (planetary freq), Elevate (PHI harmonic
    // above, directing toward Pleroma), Ground (nearest Schumann harmonic, Earth's truth).
    // PHI-scaled amplitude ensures the chief Archon (Sun) is strongest.
    #region Signal Generation

    protected override float[] GenerateSignal(ReadOnlySpan<double> tChunk,
        float totalDuration, int n)
    {
        var simplex = Simplex.Value!;
        var archonFreqs = SacredConstants.ARCHON_SPHERES;
        var pentPhases = SacredConstants.PENTAGONAL_PHASES;
        int nArchons = archonFreqs.Length;

        // Pre-compute derived frequencies
        var elevateFreqs = new float[nArchons];
        var groundFreqs = new float[nArchons];
        var ampScales = new float[nArchons];
        for (int j = 0; j < nArchons; j++)
        {
            elevateFreqs[j] = archonFreqs[j] * SacredConstants.PHI;
            float divisor = MathF.Max(MathF.Round(archonFreqs[j] / SacredConstants.SCHUMANN), 1f);
            groundFreqs[j] = archonFreqs[j] / divisor;
            ampScales[j] = 1.0f / (1.0f + j * 0.1f);
        }

        // Sequential archon processing — double precision phase for long-session stability
        var dissolution = new float[n];
        for (int j = 0; j < nArchons; j++)
        {
            float basePhase = pentPhases[j % 5];

            // Phase variation from simplex — scaled time stays small enough for float
            var tScaled = new float[n];
            for (int i = 0; i < n; i++)
                tScaled[i] = (float)(tChunk[i] * 0.00002 * (j + 1));
            var phaseVar = simplex.GenerateNoise(tScaled, j, j, j, j);
            for (int i = 0; i < n; i++)
                phaseVar[i] *= 0.1f;

            // AEG: Acknowledge + Elevate + Ground — double precision phase
            double ackFreq = archonFreqs[j];
            double elevFreq = elevateFreqs[j];
            double gndFreq = groundFreqs[j];
            for (int i = 0; i < n; i++)
            {
                float ack = (float)System.Math.Sin(SacredConstants.TWO_PI_D * ackFreq * tChunk[i] +
                    basePhase + phaseVar[i]);
                float elev = (float)System.Math.Sin(SacredConstants.TWO_PI_D * elevFreq * tChunk[i] +
                    basePhase * SacredConstants.PHI + phaseVar[i]);
                float gnd = (float)System.Math.Sin(SacredConstants.TWO_PI_D * gndFreq * tChunk[i] +
                    phaseVar[i] * 0.5f);
                dissolution[i] += (0.25f * ack + 0.5f * elev + 0.25f * gnd) * ampScales[j];
            }
        }

        // Normalize by archon count
        for (int i = 0; i < n; i++)
            dissolution[i] /= 7.0f;

        return dissolution;
    }

    #endregion
}
