using CrystalCare.Core.Dsp;
using CrystalCare.Core.Frequencies;
using CrystalCare.Core.Noise;

namespace CrystalCare.Core.SacredLayers;

/// <summary>
/// Crystalline Resonance Layer — 4th sacred layer.
///
/// Dynamic crystal sequence: 9 crystal profiles with Raman harmonic ratios.
/// Lemurian Quartz always first — every session starts with divine feminine heart energy.
/// PHI-timed crossfade evolution between crystal types.
///
/// Scale: 0.0009
/// Fade: 42 seconds
/// Breath: 13% at 0.009 Hz
///
/// Port of AudioProcessor.crystalline_resonance_layer_chunk() from SoundGenerator.py.
/// </summary>
public sealed class CrystallineResonanceLayer : ISacredLayer
{
    private readonly CrystalProfileLibrary _crystalLib;
    private readonly int[] _crystalSequence;
    private readonly float _baseFreq;
    private readonly ThreadLocal<Simplex5D> _simplex = new(() => new Simplex5D(Random.Shared.Next(100)));

    public CrystallineResonanceLayer(CrystalProfileLibrary crystalLib,
        int[] crystalSequence, float baseFreq)
    {
        _crystalLib = crystalLib;
        _crystalSequence = crystalSequence;
        _baseFreq = baseFreq;
    }

    public float[] ComputeChunk(ReadOnlySpan<float> tChunk, float totalDuration)
    {
        if (totalDuration < 60f)
            return new float[tChunk.Length];

        var simplex = _simplex.Value!;
        int n = tChunk.Length;
        var profiles = _crystalLib.Profiles;
        int numProfiles = profiles.Length;

        // Adaptive crystal count based on duration
        int numCrystals;
        if (totalDuration < 120) numCrystals = 2;
        else if (totalDuration < 300) numCrystals = 3;
        else if (totalDuration < 900) numCrystals = 5;
        else if (totalDuration < 1800) numCrystals = 7;
        else numCrystals = numProfiles;

        float segmentDuration = totalDuration / numCrystals;
        float crossfadeDur = segmentDuration / SacredConstants.PHI;
        float halfXfade = crossfadeDur / 2.0f;

        var result = new float[n];

        for (int ci = 0; ci < numCrystals; ci++)
        {
            int crystalIdx = _crystalSequence[ci % numProfiles];
            var profile = profiles[crystalIdx];

            float segStart = ci * segmentDuration;
            float segEnd = (ci + 1) * segmentDuration;

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

        // Fade envelope + breathing + scale
        var fade = SacredFadeEnvelope.Compute(tChunk, totalDuration, fadeSeconds: 42.0f);
        for (int i = 0; i < n; i++)
        {
            float breath = 0.935f + 0.065f * MathF.Sin(SacredConstants.TWO_PI * 0.009f * tChunk[i]);
            result[i] *= fade[i] * breath * 0.0009f;
        }

        return result;
    }
}
