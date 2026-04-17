using CrystalCare.Core.Frequencies;
using CrystalCare.Core.Generation;
using CrystalCare.Core.Noise;

namespace CrystalCare.Core.Dsp;

/// <summary>
/// Organic ADSR envelope with Fibonacci timing and Simplex noise chaos.
/// Each phase uses golden-ratio-based durations with subtle noise modulation
/// for breathing, alive envelope shapes.
/// Port of AudioProcessor.organic_adsr() from SoundGenerator.py.
/// </summary>
public sealed class OrganicAdsrEnvelope
{
    // Single source of truth — derived from PHI, not hardcoded truncations
    private static readonly float[] FibRatios = SacredConstants.FIB_RATIOS;

    // Pre-computed ADSR parameters drawn once at pipeline start.
    // Fibonacci-based durations with chaotic variation for organic uniqueness.
    #region ADSR Parameters

    public sealed class AdsrParams
    {
        public int AttackSamples { get; init; }
        public int DecaySamples { get; init; }
        public int SustainSamples { get; init; }
        public int ReleaseSamples { get; init; }
        public float SustainLevel { get; init; }
        public float Chaos { get; init; }
        public float DecayWobbleAmp { get; init; }
        public float DecayWobbleFreqScale { get; init; }
        public float SustainWobbleAmp { get; init; }
        public float SustainWobbleFreqScale { get; init; }
        public float OverallScale { get; init; }
    }

    #endregion

    // Draws random ADSR parameters using Fibonacci ratios and chaotic variation.
    // Called once before the pipeline loop for session-wide consistency.
    #region Parameter Computation

    public static AdsrParams ComputeParams(int totalSamples, int sampleRate,
        ChaoticSelector chaotic, Random? rng = null)
    {
        rng ??= Random.Shared;
        float chaos = chaotic.NextValue();
        float baseScale = 6.0f + chaos * 0.5f;

        float attack = FibRatios[rng.Next(FibRatios.Length)] / baseScale;
        float decay = FibRatios[rng.Next(FibRatios.Length)] / baseScale;
        float sustain = FibRatios[rng.Next(FibRatios.Length)] / baseScale;
        float release = FibRatios[rng.Next(FibRatios.Length)] / baseScale;

        int aSamples = (int)(attack * sampleRate * (1 + chaos * 0.2f));
        int dSamples = (int)(decay * sampleRate * (1 + chaos * 0.2f));
        int rSamples = (int)(release * sampleRate * (1 + chaos * 0.2f));
        int sSamples = totalSamples - (aSamples + dSamples + rSamples);

        if (sSamples < 0)
        {
            sSamples = 0;
            rSamples = totalSamples - (aSamples + dSamples);
            if (rSamples < 0) rSamples = 0;
        }

        return new AdsrParams
        {
            AttackSamples = aSamples,
            DecaySamples = dSamples,
            SustainSamples = sSamples,
            ReleaseSamples = rSamples,
            SustainLevel = global::System.Math.Min(sustain, 0.95f),  // Cap to valid range — Python's np.clip silently handles min>max
            Chaos = chaos,
            DecayWobbleAmp = (float)(rng.NextDouble() * 0.02 + 0.015),
            DecayWobbleFreqScale = 1.0f + chaos * 0.3f,
            SustainWobbleAmp = (float)(rng.NextDouble() * 0.015 + 0.01),
            SustainWobbleFreqScale = 1.5f + chaos * 0.5f,
            OverallScale = (float)(rng.NextDouble() * 0.1 + 0.95),
        };
    }

    #endregion

    // Full ADSR envelope generation for batch mode — produces the complete
    // envelope array for the entire session in one call.
    #region Batch Envelope Generation

    public static float[] Generate(int totalSamples, int sampleRate,
        ChaoticSelector chaotic, Simplex5D simplex, Random? rng = null)
    {
        var p = ComputeParams(totalSamples, sampleRate, chaotic, rng);
        return GenerateFromParams(totalSamples, sampleRate, p, simplex);
    }

    /// <summary>
    /// Generate full ADSR envelope from pre-computed parameters.
    /// </summary>
    public static float[] GenerateFromParams(int totalSamples, int sampleRate,
        AdsrParams p, Simplex5D simplex)
    {
        var envelope = new float[totalSamples];

        // Generate simplex noise for organic variation
        var tScaled = new float[totalSamples];
        for (int i = 0; i < totalSamples; i++)
            tScaled[i] = (float)i / totalSamples * 0.04f;
        var fullNoise = simplex.GenerateNoise(tScaled);

        // Attack phase: t^1.5 curve + 5% noise
        for (int i = 0; i < p.AttackSamples && i < totalSamples; i++)
        {
            float t = (float)i / p.AttackSamples;
            float curve = MathF.Pow(t, 1.5f);
            envelope[i] = System.Math.Clamp(curve + fullNoise[i] * 0.05f, 0f, 1f);
        }

        // Decay phase: t^1.2 curve + wobble + 3% noise
        int dStart = p.AttackSamples;
        for (int i = 0; i < p.DecaySamples && (dStart + i) < totalSamples; i++)
        {
            float t = (float)i / System.Math.Max(p.DecaySamples, 1);
            float curve = MathF.Pow(1f - t * (1f - p.SustainLevel), 1.2f);
            float wobble = p.DecayWobbleAmp *
                MathF.Sin(MathF.PI * p.DecayWobbleFreqScale * t);
            int idx = dStart + i;
            envelope[idx] = System.Math.Clamp(curve + wobble + fullNoise[idx] * 0.03f,
                p.SustainLevel, 1.0f);
        }

        // Sustain phase: constant level + wobble + 4% noise
        int sStart = dStart + p.DecaySamples;
        for (int i = 0; i < p.SustainSamples && (sStart + i) < totalSamples; i++)
        {
            float t = (float)i / System.Math.Max(p.SustainSamples, 1);
            float wobble = p.SustainWobbleAmp *
                MathF.Sin(SacredConstants.TWO_PI * p.SustainWobbleFreqScale * t);
            int idx = sStart + i;
            envelope[idx] = System.Math.Clamp(p.SustainLevel + wobble + fullNoise[idx] * 0.04f,
                p.SustainLevel * 0.8f, 1.0f);
        }

        // Release phase: t^1.5 decay + 5% noise
        if (p.ReleaseSamples > 0)
        {
            int rStart = totalSamples - p.ReleaseSamples;
            for (int i = 0; i < p.ReleaseSamples && (rStart + i) < totalSamples; i++)
            {
                float t = (float)i / p.ReleaseSamples;
                float curve = MathF.Pow(p.SustainLevel * (1f - t), 1.5f);
                int idx = rStart + i;
                envelope[idx] = System.Math.Clamp(curve + fullNoise[idx] * 0.05f, 0f, p.SustainLevel);
            }
        }

        // Overall scale (±5% variation)
        for (int i = 0; i < totalSamples; i++)
            envelope[i] *= p.OverallScale;

        return envelope;
    }

    #endregion

    // Per-chunk envelope computation for streaming mode. Uses global sample offset
    // to determine which ADSR phase (attack/decay/sustain/release) each sample falls in.
    // Simplex noise adds organic variation; wobble adds breathing to decay/sustain phases.
    #region Streaming Chunk Computation

    public static void ComputeChunk(Span<float> output, int chunkOffset, int chunkSamples,
        AdsrParams p, Simplex5D simplex, int totalSamples)
    {
        // Generate noise for this chunk's range
        var tScaled = new float[chunkSamples];
        for (int i = 0; i < chunkSamples; i++)
            tScaled[i] = (float)(chunkOffset + i) / totalSamples * 0.04f;
        var noise = simplex.GenerateNoise(tScaled);

        int dStart = p.AttackSamples;
        int sStart = dStart + p.DecaySamples;
        int rStart = totalSamples - p.ReleaseSamples;

        for (int i = 0; i < chunkSamples; i++)
        {
            int globalIdx = chunkOffset + i;
            float value;

            if (globalIdx < p.AttackSamples)
            {
                float t = (float)globalIdx / System.Math.Max(p.AttackSamples, 1);
                value = System.Math.Clamp(MathF.Pow(t, 1.5f) + noise[i] * 0.05f, 0f, 1f);
            }
            else if (globalIdx < sStart)
            {
                float t = (float)(globalIdx - dStart) / System.Math.Max(p.DecaySamples, 1);
                float curve = MathF.Pow(1f - t * (1f - p.SustainLevel), 1.2f);
                float wobble = p.DecayWobbleAmp * MathF.Sin(MathF.PI * p.DecayWobbleFreqScale * t);
                value = System.Math.Clamp(curve + wobble + noise[i] * 0.03f, p.SustainLevel, 1.0f);
            }
            else if (globalIdx < rStart || p.ReleaseSamples == 0)
            {
                float t = (float)(globalIdx - sStart) / System.Math.Max(p.SustainSamples, 1);
                float wobble = p.SustainWobbleAmp * MathF.Sin(SacredConstants.TWO_PI * p.SustainWobbleFreqScale * t);
                value = System.Math.Clamp(p.SustainLevel + wobble + noise[i] * 0.04f, p.SustainLevel * 0.8f, 1.0f);
            }
            else
            {
                float t = (float)(globalIdx - rStart) / System.Math.Max(p.ReleaseSamples, 1);
                value = System.Math.Clamp(MathF.Pow(p.SustainLevel * (1f - t), 1.5f) + noise[i] * 0.05f, 0f, p.SustainLevel);
            }

            output[i] = value * p.OverallScale;
        }
    }

    #endregion
}
