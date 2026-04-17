using CrystalCare.Core.Dsp;
using CrystalCare.Core.Frequencies;

namespace CrystalCare.Core.Generation;

/// <summary>
/// SoundGenerator helper methods — frequency set construction, modulation scheduling,
/// and normalization gain estimation. Split from the main file for readability.
/// </summary>
public sealed partial class SoundGenerator
{
    // Builds the 13-frequency set from a base frequency using three different
    // mathematical relationships: 6 PHI exponents, 3 ratio-1.3 exponents,
    // and 4 subharmonics. Each frequency gets slight random jitter (±2-5%)
    // so every session has a unique tonal character.
    #region Frequency Set Building

    /// <summary>
    /// Build the 13-frequency set from the base frequency.
    /// 6 PHI exponents + 3 ratio-1.3 exponents + 4 subharmonics = 13 total.
    /// Each gets random jitter for organic uniqueness.
    /// </summary>
    private float[] BuildFrequencySet(float baseFreq)
    {
        var freqs = new List<float>();

        // 6 PHI exponents: base * PHI^0 through PHI^5 (with ±2% jitter)
        for (int i = 0; i < 6; i++)
            freqs.Add(baseFreq * SacredConstants.PHI_EXPONENTS_6[i] *
                (float)(_rng.NextDouble() * 0.04 + 0.98));

        // 3 ratio-1.3 exponents: base * 1.3^0, 1.3^1, 1.3^2 (with ±2% jitter)
        for (int i = 0; i < 3; i++)
            freqs.Add(baseFreq * SacredConstants.RATIO_1_3_EXPONENTS_3[i] *
                (float)(_rng.NextDouble() * 0.04 + 0.98));

        // 4 subharmonics: base / 2, 4, 8, 16 (with ±5% jitter)
        for (int i = 0; i < 4; i++)
            freqs.Add(baseFreq / SacredConstants.SUBHARMONIC_DIVISORS[i] *
                (float)(_rng.NextDouble() * 0.1 + 0.95));

        return freqs.ToArray();
    }

    #endregion

    // Builds the modulation schedule that determines which sacred geometry ratio sets
    // are active during which time segments, and at what modulation depth.
    // Dimensional mode cycles through 7 phases then an "all ratios" finale.
    // Normal mode uses Fibonacci-timed intervals cycling through random ratio sets.
    #region Modulation Schedule

    /// <summary>
    /// Build the modulation schedule — time segments paired with ratio sets and depths.
    /// Dimensional mode: 7 phases + all-ratios finale.
    /// Normal mode: Fibonacci-timed intervals with random ratio set selection.
    /// </summary>
    private List<(int start, int end, float[] ratioValues, float modIndex)>
        BuildModulationSchedule(float duration, int totalSamples, int sampleRate,
            int[] intervalDurations, bool dimensionalMode, CancellationToken ct)
    {
        var schedule = new List<(int, int, float[], float)>();
        int current = 0;

        if (dimensionalMode)
        {
            // Dimensional Journey: cycle through 7 frequency dimension phases
            int[] phases = [0, 2, 1, 3, 4, 3, 5];
            int phaseLen = totalSamples / (phases.Length + 1);
            foreach (int sel in phases)
            {
                if (ct.IsCancellationRequested) break;
                var ratioSet = _frequencyManager.SelectRandomRatioSet(ct);
                float modIndex = (float)(_rng.NextDouble() * 0.05 + 0.2);
                int end = global::System.Math.Min(current + phaseLen, totalSamples);
                schedule.Add((current, end, ratioSet.Values.ToArray(), modIndex));
                current = end;
            }
            // Final "all ratios" phase — every sacred geometry ratio active simultaneously
            if (current < totalSamples)
            {
                var allRatios = FrequencyManager.RatioSets.Values
                    .SelectMany(d => d.Values).Distinct().ToArray();
                float modIndex = (float)(_rng.NextDouble() * 0.05 + 0.2);
                schedule.Add((current, totalSamples, allRatios, modIndex));
            }
        }
        else
        {
            // Normal mode: cycle through Fibonacci-timed intervals [34, 55, 89, 144]
            float remaining = duration;
            int intervalCount = 0;
            while (remaining > 0 && !ct.IsCancellationRequested)
            {
                // Select interval duration from the Fibonacci list (cycles)
                float interval = global::System.Math.Min(
                    intervalDurations[intervalCount % intervalDurations.Length], remaining);
                int segSamples = (int)(sampleRate * interval);

                // Pick a random ratio set with weighted probability
                var ratioSet = _frequencyManager.SelectRandomRatioSet(ct);
                float modIndex = (float)(_rng.NextDouble() * 0.05 + 0.2);
                int end = global::System.Math.Min(current + segSamples, totalSamples);
                schedule.Add((current, end, ratioSet.Values.ToArray(), modIndex));
                current = end;
                remaining -= interval;
                intervalCount++;
            }
        }

        return schedule;
    }

    /// <summary>
    /// Compute modulation values for a single chunk from the pre-built schedule.
    /// Only processes schedule segments that overlap with this chunk's sample range.
    /// Each active ratio contributes a sine wave at modulationIndex amplitude.
    /// </summary>
    private static float[] ComputeModulationChunk(ReadOnlySpan<float> tChunk,
        int chunkOffset, int chunkSamples,
        List<(int start, int end, float[] ratioValues, float modIndex)> schedule)
    {
        var result = new float[chunkSamples];
        int chunkEnd = chunkOffset + chunkSamples;

        // Only process schedule segments that overlap with this chunk
        foreach (var (start, end, ratioValues, modIndex) in schedule)
        {
            if (start >= chunkEnd || end <= chunkOffset) continue;

            // Compute local sample range within this chunk
            int localStart = global::System.Math.Max(0, start - chunkOffset);
            int localEnd = global::System.Math.Min(chunkSamples, end - chunkOffset);

            // Sum sine waves for each ratio in this segment
            for (int r = 0; r < ratioValues.Length; r++)
            {
                float ratio = ratioValues[r];
                for (int i = localStart; i < localEnd; i++)
                    result[i] += modIndex * MathF.Sin(SacredConstants.TWO_PI * ratio * tChunk[i]);
            }
        }

        return result;
    }

    #endregion

    // Estimates the peak amplitude of the generated harmonics to compute
    // a normalization gain factor. Uses a quarter-period LFO sample to catch
    // the peak, then applies 1.3x headroom for safety margin.
    #region Normalization

    /// <summary>
    /// Estimate normalization gain by generating a short test signal.
    /// Runs harmonics through wave shaping at full envelope to find the peak,
    /// then returns 1/(peak * 1.3) for headroom.
    /// </summary>
    private float EstimateNormGain(float[] frequencies, float[] modDepths,
        MicrotonalLfo.LfoParams lfoParams, int sampleRate)
    {
        // Generate a half-second test signal starting at the LFO quarter-period (peak)
        int estSamples = global::System.Math.Min(sampleRate / 2, 24000);
        float quarterPeriod = 1.0f / (4.0f * MathF.Max(lfoParams.Lfo1Freq, 0.001f));

        // Build test time array offset to the LFO peak
        var estT = new float[estSamples];
        for (int i = 0; i < estSamples; i++)
            estT[i] = quarterPeriod + (float)i / sampleRate;

        // Generate test harmonics at full envelope with LFO modulation
        var estEnv = new float[estSamples];
        Array.Fill(estEnv, 1.0f);
        var estLfo = MicrotonalLfo.Compute(estT, lfoParams);
        var estWave = HarmonicGenerator.GenerateHarmonics(estT, frequencies, estEnv, estLfo, modDepths);
        WaveShaper.Process(estWave, 2.5f);

        // Find peak amplitude
        float peak = 0.001f;
        for (int i = 0; i < estSamples; i++)
            peak = MathF.Max(peak, MathF.Abs(estWave[i]));

        // Return gain with 1.3x headroom (conservative, aims for ~0.77 peak)
        return 1.0f / (peak * 1.3f);
    }

    #endregion
}
