using CrystalCare.Core.Frequencies;
using CrystalCare.Core.Noise;

namespace CrystalCare.Core.SacredLayers;

/// <summary>
/// Water Element Layer — 6th sacred layer.
///
/// 7-source hexagonal ripple field with lemniscate (figure-8) observer path.
/// Spatial wave interference (not sine stacking) — distance-based decay creates
/// organic movement as the observer traces the lemniscate.
///
/// Sources arranged in Seed of Life geometry (hexagonal, ice crystal pattern).
/// Tidal simplex modulation for organic, breathing amplitude.
/// Standing wave resonance at Schumann x PHI.
///
/// Fully stateless: observer position determined from absolute time.
///
/// Scale: 0.0012
/// Fade: 46 seconds
/// Breath: 11% at 0.007 Hz (~143 second cycle)
///
/// Port of AudioProcessor.water_element_layer_chunk() from SoundGenerator.py.
/// </summary>
public sealed class WaterElementLayer : ISacredLayer
{
    private readonly ThreadLocal<Simplex5D> _simplex = new(() => new Simplex5D(Random.Shared.Next(100)));

    public float[] ComputeChunk(ReadOnlySpan<float> tChunk, float totalDuration)
    {
        if (totalDuration < 60f)
            return new float[tChunk.Length];

        var simplex = _simplex.Value!;
        int n = tChunk.Length;

        // Simplex perturbation for organic observer drift
        var tScaled = new float[n];
        for (int i = 0; i < n; i++)
            tScaled[i] = tChunk[i] * 0.005f;
        var thetaPerturb = simplex.GenerateNoise(tScaled, 0.4f);
        for (int i = 0; i < n; i++)
            thetaPerturb[i] *= 0.15f;

        var sourceFreqs = SacredConstants.WATER_SOURCE_FREQS;
        var sourceDecays = SacredConstants.WATER_SOURCE_DECAYS;
        var hexPhases = SacredConstants.WATER_HEX_PHASES;
        var positions = SacredConstants.WATER_SOURCE_POSITIONS;

        // Accumulate wave interference from 7 sources
        var result = new float[n];

        for (int s = 0; s < 7; s++)
        {
            float srcX = positions[s, 0];
            float srcY = positions[s, 1];

            for (int i = 0; i < n; i++)
            {
                // Lemniscate path (figure-8)
                float theta = SacredConstants.TWO_PI * 0.005f * tChunk[i] + thetaPerturb[i];
                float sinT = MathF.Sin(theta);
                float cosT = MathF.Cos(theta);
                float denom = 1.0f + sinT * sinT;
                float obsX = 0.7f * cosT / denom;
                float obsY = 0.7f * sinT * cosT / denom;

                // Distance from observer to source
                float dx = obsX - srcX;
                float dy = obsY - srcY;
                float dist = MathF.Sqrt(dx * dx + dy * dy);

                // Spatial envelope: exponential decay with distance
                float envelope = MathF.Exp(-sourceDecays[s] * dist * 10.0f);

                // Wave from this source
                float wave = MathF.Sin(SacredConstants.TWO_PI * sourceFreqs[s] * tChunk[i] + hexPhases[s]);
                result[i] += wave * envelope;
            }
        }

        // Normalize by source count
        for (int i = 0; i < n; i++)
            result[i] *= 1.0f / 7.0f;

        // Tidal modulation: slow simplex-driven amplitude (~200s period)
        for (int i = 0; i < n; i++)
            tScaled[i] = tChunk[i] * 0.0005f;
        var tidal = simplex.GenerateNoise(tScaled, 0.9f);
        for (int i = 0; i < n; i++)
            result[i] *= 0.85f + 0.15f * tidal[i];

        // Standing wave resonance: Schumann x Schumann*PHI
        const float schumannPhi = SacredConstants.SCHUMANN * SacredConstants.PHI;
        for (int i = 0; i < n; i++)
        {
            float standing = 0.1f *
                MathF.Sin(SacredConstants.TWO_PI * SacredConstants.SCHUMANN * tChunk[i]) *
                MathF.Cos(SacredConstants.TWO_PI * schumannPhi * tChunk[i]);
            result[i] += standing;
        }

        // Breath modulation: 0.007 Hz (~143s period), 11% depth
        for (int i = 0; i < n; i++)
        {
            float breath = 0.945f + 0.055f * MathF.Sin(SacredConstants.TWO_PI * 0.007f * tChunk[i]);
            result[i] *= breath;
        }

        // Sacred fade envelope + scale
        var fade = SacredFadeEnvelope.Compute(tChunk, totalDuration, fadeSeconds: 46.0f);
        for (int i = 0; i < n; i++)
            result[i] *= fade[i] * 0.0012f;

        return result;
    }
}
