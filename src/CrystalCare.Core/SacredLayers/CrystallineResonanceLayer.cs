using CrystalCare.Core.Dsp;
using CrystalCare.Core.Frequencies;

namespace CrystalCare.Core.SacredLayers;

/// <summary>
/// Crystalline Resonance Layer — 4th sacred layer.
///
/// Dynamic crystal sequence: 9 crystal profiles with Raman harmonic ratios.
/// Lemurian Quartz always first — every session starts with divine feminine heart energy.
/// PHI-timed crossfade evolution between crystal types.
///
/// Scale: 0.0009
/// Fade: 34 seconds (Fibonacci)
/// Breath: 13% at 0.009 Hz
/// </summary>
public sealed class CrystallineResonanceLayer : SacredLayerBase
{
    // Crystal library reference, session crystal sequence (Lemurian first),
    // base frequency, and configuration: 34s Fibonacci fade, 13% breath depth
    // at √PHI × root, 0.0009 output scale.
    #region Fields and Configuration

    private readonly CrystalProfileLibrary _crystalLib;
    private readonly int[] _crystalSequence;
    private readonly float _baseFreq;

    protected override float FadeSeconds => 34.0f;
    protected override float BreathCenter => 0.935f;
    protected override float BreathDepth => 0.065f;
    protected override float BreathFreq => SacredConstants.BREATH_PHI_050; // √PHI × root
    protected override float OutputScale => 0.0009f;

    public CrystallineResonanceLayer(CrystalProfileLibrary crystalLib,
        int[] crystalSequence, float baseFreq)
    {
        _crystalLib = crystalLib;
        _crystalSequence = crystalSequence;
        _baseFreq = baseFreq;
    }

    #endregion

    // Generates crystal harmonic profiles from Raman spectroscopy data.
    // Adaptive crystal count based on duration (2 for short, up to 9 for long sessions).
    // Golden angle modulates segment boundaries for organic transition timing.
    // PHI-timed crossfade (Perlin smoother step) between crystal personalities.
    // 0.1% fractal micro-variation adds organic aliveness to each crystal.
    #region Signal Generation

    protected override float[] GenerateSignal(ReadOnlySpan<float> tChunk,
        float totalDuration, int n)
    {
        var simplex = Simplex.Value!;
        var profiles = _crystalLib.Profiles;
        int numProfiles = profiles.Length;

        // Adaptive crystal count based on duration
        int numCrystals;
        if (totalDuration < 120) numCrystals = 2;
        else if (totalDuration < 300) numCrystals = 3;
        else if (totalDuration < 900) numCrystals = 5;
        else if (totalDuration < 1800) numCrystals = 7;
        else numCrystals = numProfiles;

        // Golden angle crystal timing — each crystal's segment boundary is placed
        // at golden-angle fractions of the total duration, like seeds on a sunflower.
        // This prevents the mind from predicting transitions.
        float avgSegment = totalDuration / numCrystals;
        var segBoundaries = new float[numCrystals + 1];
        segBoundaries[0] = 0;
        for (int ci = 1; ci < numCrystals; ci++)
        {
            // Golden angle fraction modulates the midpoint of each boundary
            float goldenOffset = 0.15f * avgSegment *
                MathF.Sin(ci * SacredConstants.GOLDEN_ANGLE_RAD);
            segBoundaries[ci] = ci * avgSegment + goldenOffset;
        }
        segBoundaries[numCrystals] = totalDuration;

        var result = new float[n];

        for (int ci = 0; ci < numCrystals; ci++)
        {
            int crystalIdx = _crystalSequence[ci % numProfiles];
            var profile = profiles[crystalIdx];

            float segStart = segBoundaries[ci];
            float segEnd = segBoundaries[ci + 1];
            float segmentDuration = segEnd - segStart;
            float crossfadeDur = segmentDuration / SacredConstants.PHI;
            float halfXfade = crossfadeDur / 2.0f;

            float extStart = ci > 0 ? MathF.Max(0, segStart - halfXfade) : 0;
            float extEnd = ci < numCrystals - 1 ? MathF.Min(totalDuration, segEnd + halfXfade) : totalDuration;
            float soloStart = ci > 0 ? segStart + halfXfade : 0;
            float soloEnd = ci < numCrystals - 1 ? segEnd - halfXfade : totalDuration;

            // Find samples in this crystal's active range
            int firstIdx = -1, lastIdx = -1;
            for (int i = 0; i < n; i++)
            {
                if (tChunk[i] >= extStart && tChunk[i] < extEnd)
                {
                    if (firstIdx < 0) firstIdx = i;
                    lastIdx = i;
                }
            }
            if (firstIdx < 0) continue;

            int segLen = lastIdx - firstIdx + 1;
            var tSeg = new float[segLen];
            for (int i = 0; i < segLen; i++)
                tSeg[i] = tChunk[firstIdx + i];

            // Generate crystal harmonics for this segment
            var wave = CrystalProfileLibrary.GenerateHarmonics(tSeg, _baseFreq, profile, simplex);

            // Fractal micro-variation: subtle organic aliveness (0.1% modulation)
            var microVar = FractalVariation.ComputeChunk(tSeg, _baseFreq, simplex);
            for (int i = 0; i < segLen; i++)
                wave[i] *= (1.0f + microVar[i] * 0.001f);

            // Crossfade weights
            for (int i = 0; i < segLen; i++)
            {
                float t = tSeg[i];
                float weight = 1.0f;

                if (ci > 0 && crossfadeDur > 0 && t < soloStart)
                {
                    float x = global::System.Math.Clamp((t - extStart) / crossfadeDur, 0f, 1f);
                    weight = x * x * x * (x * (x * 6f - 15f) + 10f);
                }
                else if (ci < numCrystals - 1 && crossfadeDur > 0 && t >= soloEnd)
                {
                    float x = global::System.Math.Clamp((extEnd - t) / crossfadeDur, 0f, 1f);
                    weight = x * x * x * (x * (x * 6f - 15f) + 10f);
                }

                result[firstIdx + i] += wave[i] * weight;
            }
        }

        return result;
    }

    #endregion
}
